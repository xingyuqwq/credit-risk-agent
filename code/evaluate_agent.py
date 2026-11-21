# -*- coding: utf-8 -*-
"""Agent 端到端评估与消融实验。

在测试集上逐条调用 CreditRiskAgent.decide，统计分流结构、误放行率、
复核覆盖率与判别力；再跑四组消融，量化灰区、校准与类别平衡的贡献。
结果写入 results/ 下的 CSV。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from risk_agent import CreditRiskAgent
from train_model import (FEATURES, RANDOM_STATE, TARGET, TEST_SIZE,
                         _logit, calibrate_threshold, load_dataset)

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"


def build_splits():
    """按训练时完全一致的划分重建训练/测试集，保证评估与训练不交叉。"""
    df = load_dataset()
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_tr, X_te, y_tr, y_te


def calibrated_prob(bundle: dict, X: np.ndarray) -> np.ndarray:
    """按 Agent 推理路径计算校准后概率。"""
    z = bundle["scaler"].transform(X)
    raw = bundle["pipeline"].predict_proba(z)[:, 1]
    cal = bundle.get("calibrator")
    if cal is None:
        return raw
    eps = 1e-6
    p = np.clip(raw, eps, 1 - eps)
    return cal.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]


def summarize(name: str, y_true: np.ndarray, prob: np.ndarray,
              low: float, high: float, cut: float | None = None,
              full_auto: bool = False) -> dict:
    """统计 Agent 在给定分流策略下的结构指标与安全性。

    参数
    ----
    low, high : float
        灰区上下界。full_auto=True 时忽略。
    cut : float | None
        二分类阈值，缺省取灰区中点。
    full_auto : bool
        True 表示关闭灰区分流：所有样本都直接给出结论，没有人工复核环节。
    """
    if cut is None:
        cut = (low + high) / 2
    y_pred = (prob >= cut).astype(int)

    if full_auto:
        # 没有灰区：判 0 即放行，判 1 即标记
        auto_pass = y_pred == 0
        review = np.zeros(len(y_true), dtype=bool)
        high_risk = y_pred == 1
    else:
        auto_pass = prob < low
        review = ~auto_pass                       # 灰区 + 高风险都进入人工
        high_risk = prob > high

    n = len(y_true)
    pos = int(y_true.sum())
    auto_n = int(auto_pass.sum())
    review_n = int(review.sum())
    # 自动放行里实际违约的样本：这是最危险的一类错误
    auto_false_pass = int((auto_pass & (y_true == 1)).sum())
    # 转人工样本覆盖的违约客户数
    review_caught = int((review & (y_true == 1)).sum())

    return {
        "配置": name,
        "AUC": round(float(roc_auc_score(y_true, prob)), 4),
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "自动通过率": round(auto_n / n, 4),
        "转人工率": round(review_n / n, 4),
        "高风险标记率": round(float(high_risk.sum()) / n, 4),
        "自动通过中实际违约率": round(auto_false_pass / max(auto_n, 1), 4),
        "复核覆盖违约客户比例": round(review_caught / max(pos, 1), 4),
        "自动放行漏判数": auto_false_pass,
    }


def threshold_scan(y_true: np.ndarray, prob: np.ndarray,
                   center: float, margin: float,
                   actual_low: float | None = None) -> pd.DataFrame:
    """扫描灰区下界，观察自动通过率与误放行率的此消彼长。"""
    rows = []
    pos = int(y_true.sum())
    lows = [float(x) for x in np.round(np.arange(0.02, 0.46, 0.02), 2)]
    if actual_low is not None and all(abs(actual_low - x) > 1e-9 for x in lows):
        # 把真实标定的下界插进扫描序列，保证图上的标定点与评估表一致
        lows.append(float(actual_low))
    for low in sorted(lows):
        high = min(0.95, low + 2 * margin)
        auto = prob < low
        review = ~auto
        caught = int((review & (y_true == 1)).sum())
        rows.append({
            "灰区下界": round(float(low), 3),
            "灰区上界": round(float(high), 3),
            "自动通过率": round(float(auto.mean()), 4),
            "自动通过中实际违约率": round(
                float((auto & (y_true == 1)).sum()) / max(int(auto.sum()), 1), 4
            ),
            "复核覆盖违约客户比例": round(caught / max(pos, 1), 4),
        })
    df = pd.DataFrame(rows)
    if actual_low is not None:
        df["是否为标定方案"] = np.isclose(df["灰区下界"], actual_low, atol=1e-6)
    else:
        df["是否为标定方案"] = False
    return df


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Agent 端到端评估与消融实验")
    print("=" * 70)

    agent = CreditRiskAgent()
    bundle = agent.bundle          # 预热，避免把模型加载时间算进处理耗时
    th = bundle["thresholds"]
    low, high = th["auto_low"], th["auto_high"]

    X_tr, X_test, y_tr, y_test = build_splits()
    y_arr = y_test.to_numpy()

    # ---------- 1. 逐条调用完整 Agent 流程 ----------
    print(f"[评估] 测试集 {len(X_test)} 条，逐条走完整 Agent 决策流程 ...")
    rows = []
    t_start = time.perf_counter()
    for i, (_, rec) in enumerate(X_test.iterrows()):
        payload = {
            "request_id": f"T{i:05d}",
            "applicant": {k: float(rec[k]) for k in FEATURES},
        }
        d = agent.decide(payload)
        rows.append({
            "request_id": d.request_id,
            "y_true": int(y_arr[i]),
            "probability": d.probability,
            "risk_level": d.risk_level,
            "action": d.action,
            "needs_human_review": d.needs_human_review,
            "latency_ms": d.latency_ms,
            "rule_hits": len(d.rule_hits),
            "top_factor": d.top_factors[0]["name"] if d.top_factors else "",
        })
    total_time = time.perf_counter() - t_start
    cases = pd.DataFrame(rows)
    cases.to_csv(RESULT_DIR / "agent_cases.csv", index=False, encoding="utf-8-sig")

    lat = cases["latency_ms"]
    print(f"[评估] 完成，总耗时 {total_time:.2f} s，"
          f"单条平均 {lat.mean():.2f} ms（P95 {lat.quantile(0.95):.2f} ms）")

    # ---------- 2. 分流结构与安全性 ----------
    prob_full = cases["probability"].to_numpy()
    full = summarize("A. 完整 Agent（SMOTE+校准+灰区）", y_arr, prob_full, low, high)

    # 消融 B：关闭灰区分流，在标定中心直接二分类，所有样本都自动决策
    no_grey = summarize(
        "B. 关闭灰区分流（全自动）", y_arr, prob_full,
        low=0.0, high=0.0, cut=th["center"], full_auto=True,
    )

    # 消融 C：关闭概率校准，用未校准的原始概率，阈值回到未校准标定点
    z = bundle["scaler"].transform(X_test.to_numpy(dtype=float))
    raw_prob = bundle["pipeline"].predict_proba(z)[:, 1]
    raw_center = th.get("raw_center", 0.58)
    no_cal = summarize(
        "C. 关闭概率校准", y_arr, raw_prob,
        max(0.05, raw_center - th["grey_margin"]),
        min(0.95, raw_center + th["grey_margin"]),
    )

    # 消融 D：关闭类别平衡，改用不做 SMOTE 的普通逻辑回归，并重新标定阈值
    sc2 = StandardScaler()
    X_tr_s = sc2.fit_transform(X_tr.to_numpy(dtype=float))
    base_clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    base_clf.fit(X_tr_s, y_tr.to_numpy())
    base_prob = base_clf.predict_proba(sc2.transform(X_test.to_numpy(dtype=float)))[:, 1]
    base_tau = calibrate_threshold(
        base_clf.predict_proba(X_tr_s)[:, 1], y_tr.to_numpy()
    )
    no_balance = summarize(
        "D. 关闭类别平衡（普通逻辑回归）", y_arr, base_prob,
        max(0.05, base_tau - th["grey_margin"]),
        min(0.95, base_tau + th["grey_margin"]),
    )

    eval_df = pd.DataFrame([full, no_grey, no_cal, no_balance])
    eval_df.to_csv(RESULT_DIR / "agent_eval.csv", index=False, encoding="utf-8-sig")
    eval_df.to_csv(RESULT_DIR / "agent_ablation.csv", index=False, encoding="utf-8-sig")

    print("\n" + "-" * 70)
    print(eval_df.to_string(index=False))

    # ---------- 3. 灰区下界扫描 ----------
    scan = threshold_scan(y_arr, prob_full, th["center"], th["grey_margin"],
                          actual_low=low)
    scan.to_csv(RESULT_DIR / "agent_threshold.csv", index=False, encoding="utf-8-sig")

    # ---------- 4. 失败案例分析 ----------
    auto_pass = cases["probability"] < low
    fp = cases[(auto_pass) & (cases["y_true"] == 1)].nlargest(8, "probability")
    miss = cases[(~auto_pass) & (cases["y_true"] == 0)].nsmallest(8, "probability")
    fp.assign(错误类型="自动放行但实际违约").to_csv(
        RESULT_DIR / "failure_false_pass.csv", index=False, encoding="utf-8-sig")
    miss.assign(错误类型="转人工但实际未违约").to_csv(
        RESULT_DIR / "failure_false_alarm.csv", index=False, encoding="utf-8-sig")

    summary = {
        "n_test": int(len(cases)),
        "latency_mean_ms": round(float(lat.mean()), 3),
        "latency_p95_ms": round(float(lat.quantile(0.95)), 3),
        "throughput_per_sec": round(len(cases) / total_time, 1),
        "thresholds": th,
        "metrics": bundle["metrics"],
        "full_agent": full,
        "cases_csv": "results/agent_cases.csv",
    }
    with open(RESULT_DIR / "agent_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[输出] results/agent_eval.csv")
    print("[输出] results/agent_ablation.csv（= agent_eval.csv）")
    print("[输出] results/agent_threshold.csv")
    print("[输出] results/failure_false_pass.csv / failure_false_alarm.csv")
    print("[输出] results/agent_summary.json")


if __name__ == "__main__":
    main()
