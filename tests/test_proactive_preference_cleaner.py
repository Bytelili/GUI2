from __future__ import annotations

import unittest

from src.papo.proactive_preference_cleaner import CleanConfig, clean_preference_split
from src.papo.proactive_quality_gate import assistant_text, row_source, row_weight


class ProactivePreferenceCleanerTest(unittest.TestCase):
    def test_cleaner_removes_prompt_copy_and_reweights_oracle(self) -> None:
        prompt = "Previous intents:\n- 打开天气查看明天天气"
        rows = [
            _row("g1", "oracle_target", "打开闹钟设置八点提醒", 0.40, "打开闹钟设置八点提醒", prompt),
            _row("g1", "same_user_history", "打开天气查看明天天气", 0.35, "打开闹钟设置八点提醒", prompt),
            _row("g1", "sft_sample", "打开视频软件签到", 0.25, "打开闹钟设置八点提醒", prompt),
        ]

        artifacts = clean_preference_split(
            rows,
            split="train",
            config=CleanConfig(oracle_weight=0.80, min_oracle_margin=0.10, max_negatives_per_group=2),
        )

        self.assertEqual(len(artifacts.listwise_rows), 2)
        self.assertEqual(len(artifacts.dpo_rows), 1)
        self.assertEqual({row_source(row) for row in artifacts.listwise_rows}, {"oracle_target", "sft_sample"})
        self.assertIn("prompt_history_copy", artifacts.report["rejected"]["reason_counts"])

        oracle = next(row for row in artifacts.listwise_rows if row_source(row) == "oracle_target")
        negative = next(row for row in artifacts.listwise_rows if row_source(row) == "sft_sample")
        self.assertAlmostEqual(row_weight(oracle), 0.80)
        self.assertAlmostEqual(row_weight(negative), 0.20)
        self.assertEqual(artifacts.dpo_rows[0]["chosen"]["value"], "打开闹钟设置八点提醒")


def _row(group: str, source: str, answer: str, weight: float, target: str, prompt: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "images": [],
        "papo_listwise_weight": weight,
        "metadata": {
            "group_id": group,
            "candidate_source": source,
            "target": target,
        },
    }


if __name__ == "__main__":
    unittest.main()
