from __future__ import annotations

import unittest

from src.papo.proactive_quality_gate import ProactiveQualityGate


class ProactiveQualityGateTest(unittest.TestCase):
    def test_blocks_prompt_leak_oracle_not_top1_and_weak_margin(self) -> None:
        rows = [
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "Previous intents:\n- 打开天气查看明天天气"},
                    {"role": "assistant", "content": "打开天气查看明天天气"},
                ],
                "papo_listwise_weight": 0.55,
                "metadata": {
                    "group_id": "g1",
                    "candidate_source": "same_user_history",
                    "target": "打开闹钟设置八点提醒",
                },
            },
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "Previous intents:\n- 打开天气查看明天天气"},
                    {"role": "assistant", "content": "打开闹钟设置八点提醒"},
                ],
                "papo_listwise_weight": 0.45,
                "metadata": {
                    "group_id": "g1",
                    "candidate_source": "oracle_target",
                    "target": "打开闹钟设置八点提醒",
                },
            },
        ]

        gate = ProactiveQualityGate(min_oracle_margin=0.10, progress_every=0)
        summary = gate.audit_listwise(rows, name="unit")
        decision = gate.decide([summary])

        self.assertEqual(decision.status, "failed")
        categories = {issue.category for issue in gate.issues}
        self.assertIn("prompt_leak_high_weight_candidate", categories)
        self.assertIn("oracle_not_top1", categories)
        self.assertIn("weak_oracle_margin", categories)

    def test_passes_clean_oracle_anchored_group(self) -> None:
        rows = [
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "Previous intents:\n- 查看天气"},
                    {"role": "assistant", "content": "打开闹钟设置八点提醒"},
                ],
                "papo_listwise_weight": 0.90,
                "metadata": {
                    "group_id": "g1",
                    "candidate_source": "oracle_target",
                    "target": "打开闹钟设置八点提醒",
                },
            },
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "Previous intents:\n- 查看天气"},
                    {"role": "assistant", "content": "打开视频软件签到"},
                ],
                "papo_listwise_weight": 0.10,
                "metadata": {
                    "group_id": "g1",
                    "candidate_source": "cross_user_hard",
                    "target": "打开闹钟设置八点提醒",
                },
            },
        ]

        gate = ProactiveQualityGate(min_oracle_margin=0.10, progress_every=0)
        summary = gate.audit_listwise(rows, name="unit")
        decision = gate.decide([summary])

        self.assertEqual(decision.status, "passed")
        self.assertEqual(gate.issues, [])


if __name__ == "__main__":
    unittest.main()
