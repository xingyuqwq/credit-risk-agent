# -*- coding: utf-8 -*-
"""按命令行顺序跑一遍批量样例与异常样例，把完整输出写入
results/demo_run.log，供报告中的运行截图使用。
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from risk_agent import CreditRiskAgent, InputValidationError

ROOT = Path(__file__).resolve().parents[1]
EX_DIR = ROOT / "examples"
RESULT_DIR = ROOT / "results"

FEATURE_HINT = ["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "BILL_AMT1", "PAY_AMT1"]


def dump(payload: dict) -> str:
    """只打印关键字段，避免整份 23 维 JSON 占满屏幕。"""
    app = payload.get("applicant", {})
    items = [f"{k}={app.get(k)}" for k in FEATURE_HINT if k in app]
    return f"  request_id={payload.get('request_id')}  " + "  ".join(items) + "  ..."


def main() -> None:
    buf = io.StringIO()
    agent = CreditRiskAgent()
    _ = agent.bundle

    with redirect_stdout(buf):
        print("$ python code/risk_agent.py --input examples/sample_batch.json --batch")
        print("=" * 74)
        print("【输入】examples/sample_batch.json（3 个申请件）")
        batch = json.loads((EX_DIR / "sample_batch.json").read_text(encoding="utf-8"))
        for rec in batch["records"]:
            print(dump(rec))
        print("=" * 74)
        print("【Agent 处理过程与输出】")
        print()

        done = []
        for rec in batch["records"]:
            d = agent.decide(rec)
            print(d.explanation)
            if d.warnings:
                print("  提示：")
                for w in d.warnings:
                    print(f"    ! {w}")
            print(f"  处理耗时：{d.latency_ms:.2f} ms")
            print("-" * 74)
            done.append(d)

        review = sum(1 for d in done if d.needs_human_review)
        print(f"[汇总] 共处理 {len(done)} 件，转人工复核 {review} 件，"
              f"自动通过 {len(done) - review} 件")
        print()

        print("$ python code/risk_agent.py --input examples/sample_invalid.json")
        print("=" * 74)
        print("【输入】examples/sample_invalid.json（故意构造的异常申请件）")
        bad = json.loads((EX_DIR / "sample_invalid.json").read_text(encoding="utf-8"))
        print(dump(bad))
        print("  缺失字段：PAY_4；越界：AGE=17；类型错误：BILL_AMT2='unknown'")
        print("=" * 74)
        print("【Agent 处理过程与输出】")
        print()
        try:
            agent.decide(bad)
        except InputValidationError as exc:
            print(f"申请件 {bad['request_id']}")
            print("  输入校验未通过，已拒答（不进入模型）：")
            for p in exc.problems:
                print(f"    - {p}")
            print("  处理耗时：0.00 ms")
        print("-" * 74)
        print("[说明] 异常输入不会进入模型，也不会产生任何风险结论。")

    text = buf.getvalue()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "demo_run.log").write_text(text, encoding="utf-8")
    print(text)
    print(f"[输出] 完整运行日志已写入 {RESULT_DIR / 'demo_run.log'}")


if __name__ == "__main__":
    main()
