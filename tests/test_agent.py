# -*- coding: utf-8 -*-
"""Agent 关键行为的单元测试：输入校验、派生特征、规则库、分流边界。

运行前需先执行 code/train_model.py 生成模型工件。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from risk_agent import (CreditRiskAgent, InputValidationError,  # noqa: E402
                        RiskKnowledgeBase)

EX = ROOT / "examples"


def load_sample(name: str) -> dict:
    return json.loads((EX / name).read_text(encoding="utf-8"))


class TestInputValidation(unittest.TestCase):
    """校验层：宁可拒答，也不让异常数据进入模型。"""

    @classmethod
    def setUpClass(cls):
        cls.agent = CreditRiskAgent()

    def test_valid_payload_passes(self):
        rid, clean = self.agent.validate(load_sample("sample_low_risk.json"))
        self.assertTrue(rid)
        self.assertEqual(len(clean), 23)

    def test_missing_field_rejected(self):
        payload = load_sample("sample_low_risk.json")
        payload["applicant"].pop("PAY_3")
        with self.assertRaises(InputValidationError) as ctx:
            self.agent.validate(payload)
        self.assertTrue(any("PAY_3" in p for p in ctx.exception.problems))

    def test_out_of_range_rejected(self):
        payload = load_sample("sample_low_risk.json")
        payload["applicant"]["AGE"] = 200
        with self.assertRaises(InputValidationError) as ctx:
            self.agent.validate(payload)
        self.assertTrue(any("AGE" in p for p in ctx.exception.problems))

    def test_wrong_type_rejected(self):
        payload = load_sample("sample_low_risk.json")
        payload["applicant"]["BILL_AMT1"] = "很多"
        with self.assertRaises(InputValidationError):
            self.agent.validate(payload)

    def test_all_problems_reported_at_once(self):
        """一次把所有问题报全，避免用户改一个错试一轮。"""
        payload = load_sample("sample_invalid.json")
        with self.assertRaises(InputValidationError) as ctx:
            self.agent.validate(payload)
        self.assertGreaterEqual(len(ctx.exception.problems), 3)

    def test_negative_bill_amount_allowed(self):
        """账单为负表示溢缴款，属于合法输入，不应被拒。"""
        payload = load_sample("sample_low_risk.json")
        payload["applicant"]["BILL_AMT2"] = -1
        _, clean = self.agent.validate(payload)
        self.assertEqual(clean["BILL_AMT2"], -1.0)


class TestDerivedFacts(unittest.TestCase):
    """派生指标：规则库依赖这些量，算错会导致规则误命中。"""

    def test_utilization_and_delay(self):
        payload = load_sample("sample_high_risk.json")
        facts = CreditRiskAgent.build_facts(
            {k: float(v) for k, v in payload["applicant"].items()}
        )
        app = payload["applicant"]
        self.assertAlmostEqual(
            facts["utilization"], app["BILL_AMT1"] / max(app["LIMIT_BAL"], 1), places=6
        )
        pays = [app[f"PAY_{i}"] for i in [0, 2, 3, 4, 5, 6]]
        self.assertEqual(facts["max_delay"], max(pays))
        self.assertEqual(facts["delay_count"], sum(1 for p in pays if p >= 1))


class TestKnowledgeBase(unittest.TestCase):
    """规则库：条件判断正确，且对缺失字段不抛异常。"""

    @classmethod
    def setUpClass(cls):
        cls.kb = RiskKnowledgeBase()

    def test_recent_delay_triggers(self):
        hits = self.kb.evaluate({"PAY_0": 3})
        self.assertIn("R01", [h.id for h in hits])

    def test_no_delay_triggers_protective(self):
        # R10 依赖派生字段 delay_count（近半年逾期期数），由 build_facts 计算
        hits = self.kb.evaluate({"delay_count": 0})
        self.assertIn("R10", [h.id for h in hits])

    def test_missing_field_is_safe(self):
        self.assertEqual(self.kb.evaluate({}), [])

    def test_rule_count(self):
        self.assertGreaterEqual(len(self.kb.rules), 10)


class TestDecisionRouting(unittest.TestCase):
    """分流：低风险自动通过，灰区与高风险必须转人工。"""

    @classmethod
    def setUpClass(cls):
        cls.agent = CreditRiskAgent()

    def test_low_risk_auto_pass(self):
        d = self.agent.decide(load_sample("sample_low_risk.json"))
        self.assertEqual(d.risk_level, "低")
        self.assertFalse(d.needs_human_review)
        self.assertGreater(d.probability, 0.0)
        self.assertLess(d.probability, 1.0)

    def test_grey_zone_requires_review(self):
        d = self.agent.decide(load_sample("sample_grey_zone.json"))
        self.assertEqual(d.risk_level, "中")
        self.assertTrue(d.needs_human_review)

    def test_high_risk_not_auto_rejected(self):
        d = self.agent.decide(load_sample("sample_high_risk.json"))
        self.assertEqual(d.risk_level, "高")
        self.assertTrue(d.needs_human_review)
        # 高风险也不允许自动拒绝，必须保留人工环节
        self.assertIn("人工", d.action)

    def test_explanation_contains_disclaimer(self):
        d = self.agent.decide(load_sample("sample_high_risk.json"))
        self.assertIn("不构成自动拒贷依据", d.explanation)

    def test_top_factors_have_direction(self):
        d = self.agent.decide(load_sample("sample_high_risk.json"))
        self.assertGreater(len(d.top_factors), 0)
        for f in d.top_factors:
            self.assertIn(f["direction"], ("推高违约风险", "降低违约风险"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
