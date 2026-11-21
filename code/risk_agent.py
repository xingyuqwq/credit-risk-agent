# -*- coding: utf-8 -*-
"""信用卡违约风险审查 Agent。

输入一份申请件（JSON），输出违约概率、风险等级、处置建议、主要影响因素和
命中的业务规则。灰区与高风险一律转人工，不做自动拒贷。

用法：
    python code/risk_agent.py --self-test
    python code/risk_agent.py --input examples/sample_low_risk.json
    python code/risk_agent.py --input examples/sample_batch.json --batch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "credit_risk_model.joblib"
RULES_PATH = ROOT / "data" / "knowledge_rules.json"

# 字段中文名，用于生成可读的解释文本
FEATURE_CN = {
    "LIMIT_BAL": "授信额度",
    "SEX": "性别",
    "EDUCATION": "教育程度",
    "MARRIAGE": "婚姻状况",
    "AGE": "年龄",
    "PAY_0": "最近一期还款状态",
    "PAY_2": "近第二期还款状态",
    "PAY_3": "近第三期还款状态",
    "PAY_4": "近第四期还款状态",
    "PAY_5": "近第五期还款状态",
    "PAY_6": "近第六期还款状态",
    "BILL_AMT1": "最近一期账单金额",
    "BILL_AMT2": "近第二期账单金额",
    "BILL_AMT3": "近第三期账单金额",
    "BILL_AMT4": "近第四期账单金额",
    "BILL_AMT5": "近第五期账单金额",
    "BILL_AMT6": "近第六期账单金额",
    "PAY_AMT1": "最近一期还款金额",
    "PAY_AMT2": "近第二期还款金额",
    "PAY_AMT3": "近第三期还款金额",
    "PAY_AMT4": "近第四期还款金额",
    "PAY_AMT5": "近第五期还款金额",
    "PAY_AMT6": "近第六期还款金额",
}

# 敏感属性：可以入模，但不能作为拒绝授信的唯一依据
SENSITIVE_FIELDS = {"SEX", "MARRIAGE", "EDUCATION", "AGE"}

# 取值范围按真实数据分布标定：EDUCATION 与 MARRIAGE 存在 0（未知），
# 账单金额存在负值（溢缴款），照数据字典硬卡会把合法申请件判成异常
VALUE_RANGES = {
    "LIMIT_BAL": (0.0, 1e9),
    "SEX": (1, 2),
    "EDUCATION": (0, 6),
    "MARRIAGE": (0, 3),
    "AGE": (18, 100),
    "PAY_0": (-2, 9),
    "PAY_2": (-2, 9),
    "PAY_3": (-2, 9),
    "PAY_4": (-2, 9),
    "PAY_5": (-2, 9),
    "PAY_6": (-2, 9),
}
for _i in range(1, 7):
    VALUE_RANGES[f"BILL_AMT{_i}"] = (-1e9, 1e9)
    VALUE_RANGES[f"PAY_AMT{_i}"] = (0.0, 1e9)


class InputValidationError(ValueError):
    """申请件字段不完整、类型错误或超出业务取值范围。"""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("；".join(problems))


class ModelNotReadyError(RuntimeError):
    """模型工件缺失或损坏，Agent 无法进入推演环节。"""


@dataclass
class RuleHit:
    """知识库规则的命中记录。"""
    id: str
    tag: str
    level: str
    message: str


@dataclass
class RiskDecision:
    """Agent 对单个申请件的完整审查结论。"""
    request_id: str
    probability: float
    risk_level: str
    action: str
    needs_human_review: bool
    top_factors: list[dict[str, Any]] = field(default_factory=list)
    rule_hits: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    explanation: str = ""
    latency_ms: float = 0.0


class RiskKnowledgeBase:
    """本地规则知识库。

    规则以 JSON 声明，条件支持单字段比较和 all 组合，不使用 eval，
    避免规则文件被污染后产生代码注入风险。
    """

    def __init__(self, path: Path = RULES_PATH):
        if not path.exists():
            raise FileNotFoundError(f"规则库缺失：{path}")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.version = payload.get("version", "unknown")
        self.rules = payload.get("rules", [])

    @staticmethod
    def _compare(left: float, op: str, right: float) -> bool:
        if op == ">=":
            return left >= right
        if op == ">":
            return left > right
        if op == "<=":
            return left <= right
        if op == "<":
            return left < right
        if op == "==":
            return left == right
        raise ValueError(f"不支持的比较运算符：{op}")

    def _match(self, cond: dict, facts: dict[str, float]) -> bool:
        if "all" in cond:
            return all(self._match(c, facts) for c in cond["all"])
        fld = cond["field"]
        if fld not in facts:
            return False
        return self._compare(facts[fld], cond["op"], cond["value"])

    def evaluate(self, facts: dict[str, float]) -> list[RuleHit]:
        hits: list[RuleHit] = []
        for rule in self.rules:
            try:
                if self._match(rule["condition"], facts):
                    hits.append(RuleHit(
                        id=rule["id"], tag=rule["tag"],
                        level=rule["level"], message=rule["message"],
                    ))
            except (KeyError, TypeError, ValueError):
                # 单条规则格式异常不影响整体审查，跳过并继续
                continue
        return hits


class CreditRiskAgent:
    """信用风险审查 Agent。"""

    def __init__(self, model_path: Path = MODEL_PATH, rules_path: Path = RULES_PATH):
        self.model_path = model_path
        self._bundle: dict | None = None
        self.kb = RiskKnowledgeBase(rules_path)

    # ---- 模型懒加载：只有在真正需要推理时才读盘 ----
    @property
    def bundle(self) -> dict:
        if self._bundle is None:
            if not self.model_path.exists():
                raise ModelNotReadyError(
                    f"模型工件不存在：{self.model_path}，请先运行 code/train_model.py"
                )
            try:
                self._bundle = joblib.load(self.model_path)
            except Exception as exc:  # 损坏、版本不兼容等
                raise ModelNotReadyError(f"模型加载失败：{exc}") from exc
            for key in ("scaler", "pipeline", "features", "coefficients", "thresholds"):
                if key not in self._bundle:
                    raise ModelNotReadyError(f"模型工件缺少字段：{key}")
        return self._bundle

    # ---- 第 2 步：输入校验 ----
    def validate(self, payload: dict) -> tuple[str, dict[str, float]]:
        """校验申请件并返回 (request_id, 干净特征字典)。"""
        if not isinstance(payload, dict):
            raise InputValidationError(["申请件必须是一个 JSON 对象"])

        request_id = str(payload.get("request_id") or "UNKNOWN")
        raw = payload.get("applicant", payload)
        if not isinstance(raw, dict):
            raise InputValidationError(["applicant 字段必须是 JSON 对象"])

        features = self.bundle["features"]
        problems: list[str] = []
        clean: dict[str, float] = {}

        for name in features:
            if name not in raw:
                problems.append(f"缺少字段 {name}（{FEATURE_CN.get(name, name)}）")
                continue
            value = raw[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(f"字段 {name} 必须是数值，实际为 {type(value).__name__}")
                continue
            value = float(value)
            if not np.isfinite(value):
                problems.append(f"字段 {name} 不是有限数值")
                continue
            low, high = VALUE_RANGES.get(name, (-1e12, 1e12))
            if not (low <= value <= high):
                problems.append(f"字段 {name}={value:g} 超出合理范围 [{low:g}, {high:g}]")
                continue
            clean[name] = value

        extra = set(raw) - set(features) - {"request_id", "applicant"}
        if extra:
            problems.append(f"存在无法识别的字段：{sorted(extra)}")

        if problems:
            raise InputValidationError(problems)
        return request_id, clean

    # ---- 第 3 步：派生特征 ----
    @staticmethod
    def build_facts(features: dict[str, float]) -> dict[str, float]:
        """在原始字段基础上计算业务派生指标，供规则库使用。"""
        pays = [features[f"PAY_{i}"] if i == 0 else features[f"PAY_{i}"]
                for i in [0, 2, 3, 4, 5, 6]]
        bills = [features[f"BILL_AMT{i}"] for i in range(1, 7)]
        amts = [features[f"PAY_AMT{i}"] for i in range(1, 7)]

        limit = max(features["LIMIT_BAL"], 1.0)
        total_bill = max(sum(bills), 1.0)
        total_pay = sum(amts)

        facts = dict(features)
        facts["max_delay"] = float(max(pays))
        facts["delay_count"] = float(sum(1 for p in pays if p >= 1))
        facts["utilization"] = float(features["BILL_AMT1"] / limit)
        facts["repay_ratio"] = float(total_pay / total_bill)
        facts["bill_growth"] = float(
            features["BILL_AMT1"] / max(features["BILL_AMT2"], 1.0) - 1.0
        )
        facts["repay_cover"] = float(
            features["PAY_AMT1"] / max(features["BILL_AMT2"], 1.0)
        )
        return facts

    # ---- 第 4~6 步：推演、解释、分流 ----
    def decide(self, payload: dict) -> RiskDecision:
        t0 = time.perf_counter()

        request_id, clean = self.validate(payload)

        b = self.bundle
        # pipeline 是在标准化后的特征上训练的，推理前必须先过同一个 scaler
        row = np.array([[clean[f] for f in b["features"]]], dtype=float)
        z = b["scaler"].transform(row)[0]
        raw = float(b["pipeline"].predict_proba(z.reshape(1, -1))[0, 1])

        # SMOTE 让原始概率整体偏高，用训练阶段的 Platt 校准器拉回真实水平
        prob = raw
        calibrator = b.get("calibrator")
        if calibrator is not None:
            eps = 1e-6
            p_clip = float(np.clip(raw, eps, 1 - eps))
            logit_raw = np.log(p_clip / (1 - p_clip))
            prob = float(calibrator.predict_proba([[logit_raw]])[0, 1])

        th = b["thresholds"]
        low, high = th["auto_low"], th["auto_high"]

        if prob < low:
            risk_level = "低"
            action = "自动通过：建议维持现有授信策略"
            needs_review = False
        elif prob <= high:
            risk_level = "中"
            action = "转人工复核：模型输出位于灰区，自动结论不可靠"
            needs_review = True
        else:
            risk_level = "高"
            action = "高风险：提请人工优先核查，不建议自动拒绝"
            needs_review = True

        # 线性模型的局部解释：贡献 = 系数 x 标准化取值，按绝对值取前五
        coef = np.array([b["coefficients"][f] for f in b["features"]])
        contrib = coef * z
        order = np.argsort(np.abs(contrib))[::-1][:5]
        factors: list[dict[str, Any]] = []
        for idx in order:
            factors.append({
                "feature": b["features"][idx],
                "name": FEATURE_CN.get(b["features"][idx], b["features"][idx]),
                "value": clean[b["features"][idx]],
                "contribution": round(float(contrib[idx]), 4),
                "direction": "推高违约风险" if contrib[idx] > 0 else "降低违约风险",
                "sensitive": b["features"][idx] in SENSITIVE_FIELDS,
            })

        facts = self.build_facts(clean)
        hits = self.kb.evaluate(facts)

        warnings: list[str] = []
        if any(f["sensitive"] for f in factors):
            warnings.append(
                "主要影响因素中包含敏感属性（性别/婚姻/教育/年龄），"
                "复核时不应将其作为拒绝授信的唯一依据。"
            )
        if prob > high:
            warnings.append("该结论属于高影响决策，必须经人工确认后方可使用。")
        if abs(prob - th["center"]) < 0.05:
            warnings.append("概率接近标定阈值中心，结论对阈值设定较敏感。")

        explanation = self._render(
            request_id, prob, risk_level, action, factors, hits, th
        )

        return RiskDecision(
            request_id=request_id,
            probability=round(prob, 4),
            risk_level=risk_level,
            action=action,
            needs_human_review=needs_review,
            top_factors=factors,
            rule_hits=[asdict(h) for h in hits],
            warnings=warnings,
            explanation=explanation,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    @staticmethod
    def _render(request_id, prob, level, action, factors, hits, th) -> str:
        lines = [
            f"申请件 {request_id}",
            f"  违约概率：{prob:.3f}（已做概率校准）    风险等级：{level}",
            f"  处置建议：{action}",
            f"  判定依据：标定阈值中心 {th['center']:.2f}，灰区 [{th['auto_low']:.2f}, {th['auto_high']:.2f}]",
        ]
        if factors:
            lines.append("  主要影响因素：")
            for i, f in enumerate(factors, 1):
                sign = "+" if f["contribution"] > 0 else ""
                lines.append(
                    f"    {i}. {f['name']} = {f['value']:g}"
                    f"（{f['direction']}，对数几率贡献 {sign}{f['contribution']:.3f}）"
                )
        if hits:
            lines.append("  规则命中：")
            for h in hits:
                lines.append(f"    - [{h.level}] {h.id} {h.tag}：{h.message}")
        lines.append("  免责声明：本结论仅用于风险排序与人工辅助，不构成自动拒贷依据。")
        return "\n".join(lines)


def _self_test_cases() -> list[dict]:
    """内置边界用例：正常低风险、明显高风险、以及一个缺字段的异常件。"""
    base = {
        "LIMIT_BAL": 200000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 48000, "BILL_AMT2": 45000, "BILL_AMT3": 42000, "BILL_AMT4": 40000,
        "BILL_AMT5": 38000, "BILL_AMT6": 35000,
        "PAY_AMT1": 48000, "PAY_AMT2": 45000, "PAY_AMT3": 42000, "PAY_AMT4": 40000,
        "PAY_AMT5": 38000, "PAY_AMT6": 35000,
    }
    low_risk = {"request_id": "SELF-LOW", "applicant": dict(base)}

    high = dict(base)
    high.update({
        "LIMIT_BAL": 20000, "PAY_0": 3, "PAY_2": 2, "PAY_3": 2,
        "BILL_AMT1": 19500, "PAY_AMT1": 200, "PAY_AMT2": 300,
    })
    high_risk = {"request_id": "SELF-HIGH", "applicant": high}

    broken = {"request_id": "SELF-BAD", "applicant": dict(base)}
    broken["applicant"].pop("PAY_3")
    broken["applicant"]["AGE"] = 200

    return [low_risk, high_risk, broken]


def _print_decision(d: RiskDecision) -> None:
    print(d.explanation)
    if d.warnings:
        print("  提示：")
        for w in d.warnings:
            print(f"    ! {w}")
    print(f"  处理耗时：{d.latency_ms:.2f} ms")
    print("-" * 66)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="信用卡违约风险审查 Agent（人工授信复核辅助）"
    )
    parser.add_argument("--input", type=str, help="申请件 JSON 文件路径")
    parser.add_argument("--batch", action="store_true", help="批量模式：文件包含多个申请件")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出结构化结果")
    parser.add_argument("--self-test", action="store_true", help="运行内置自检用例")
    parser.add_argument("--output", type=str, help="将结构化结果写入指定 JSON 文件")
    args = parser.parse_args(argv)

    try:
        agent = CreditRiskAgent()
        _ = agent.bundle  # 预热：把模型加载耗时与单件处理耗时分开统计
    except (ModelNotReadyError, FileNotFoundError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    # 收集待处理申请件
    if args.self_test:
        cases = _self_test_cases()
    elif args.input:
        path = Path(args.input)
        if not path.exists():
            print(f"[错误] 输入文件不存在：{path}", file=sys.stderr)
            return 2
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        cases = payload.get("records", []) if args.batch and isinstance(payload, dict) else \
            (payload if isinstance(payload, list) else [payload])
    else:
        parser.print_help()
        return 1

    results: list[dict[str, Any]] = []
    print("=" * 66)
    print("信用风险审查 Agent 运行记录")
    print("=" * 66)

    exit_code = 0
    for case in cases:
        try:
            decision = agent.decide(case)
            results.append(asdict(decision))
            if not args.json:
                _print_decision(decision)
        except InputValidationError as exc:
            # 输入异常：拒答，不产生任何风险结论
            rid = str(case.get("request_id", "UNKNOWN")) if isinstance(case, dict) else "UNKNOWN"
            print(f"申请件 {rid}")
            print("  输入校验未通过，已拒答（不进入模型）：")
            for p in exc.problems:
                print(f"    - {p}")
            print(f"  处理耗时：0.00 ms")
            print("-" * 66)
            results.append({
                "request_id": rid, "status": "rejected",
                "problems": exc.problems,
            })
            exit_code = max(exit_code, 3)
        except Exception as exc:  # 兜底：任何意外都不静默失败
            rid = str(case.get("request_id", "UNKNOWN")) if isinstance(case, dict) else "UNKNOWN"
            print(f"申请件 {rid}")
            print(f"  处理异常：{type(exc).__name__}: {exc}")
            print("-" * 66)
            results.append({"request_id": rid, "status": "error", "error": str(exc)})
            exit_code = max(exit_code, 4)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))

    if args.output:
        Path(args.output).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[输出] 结构化结果已写入 {args.output}")

    ok = [r for r in results if r.get("status") not in ("rejected", "error")]
    if ok:
        review = sum(1 for r in ok if r["needs_human_review"])
        print(f"\n[汇总] 共处理 {len(ok)} 件，转人工复核 {review} 件，"
              f"自动通过 {len(ok) - review} 件")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
