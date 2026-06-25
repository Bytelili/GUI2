from __future__ import annotations

import unittest

from src.papo.proactive_fixed_export import (
    DPOExportConfig,
    RerankExportConfig,
    WeightedListwiseExportConfig,
    audit_wide_rows,
    export_dpo_rows,
    export_rerank_rows,
    export_weighted_listwise_rows,
)


class ProactiveFixedExportTest(unittest.TestCase):
    def test_dpo_pair_generation(self) -> None:
        rows, report = export_dpo_rows([_sample_row()], DPOExportConfig())
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["chosen"]["content"], "打开淘宝搜索蓝牙耳机")
        self.assertNotEqual(rows[0]["chosen"]["content"], rows[0]["rejected"]["content"])
        self.assertIn("negative_type", rows[0]["metadata"])
        self.assertGreater(rows[0]["papo_weight"], 0.0)
        self.assertEqual(report["status"], "passed")

    def test_rerank_export(self) -> None:
        rows, _ = export_rerank_rows([_sample_row()], RerankExportConfig(seed=7))
        self.assertEqual(len(rows), 1)
        metadata = rows[0]["metadata"]
        candidate_order = metadata["candidate_order"]
        self.assertGreaterEqual(len(candidate_order), 2)
        self.assertIn(rows[0]["messages"][-1]["content"], {"A", "B", "C", "D"})
        self.assertIn("oracle", candidate_order)
        correct_letter = rows[0]["messages"][-1]["content"]
        self.assertEqual(metadata["correct_letter"], correct_letter)

    def test_weighted_listwise_export(self) -> None:
        rows, _ = export_weighted_listwise_rows(
            [_sample_row(context_text="打开高德地图搜索附近餐厅", context_reward_total=0.81)],
            WeightedListwiseExportConfig(min_context_prob=0.02),
        )
        self.assertTrue(rows)
        weights_by_group: dict[str, float] = {}
        oracle_weights: list[float] = []
        context_weights: list[float] = []
        for row in rows:
            group_id = row["metadata"]["group_id"]
            weights_by_group[group_id] = weights_by_group.get(group_id, 0.0) + row["papo_listwise_weight"]
            if row["metadata"]["candidate_source"] == "oracle":
                oracle_weights.append(row["papo_listwise_weight"])
            if row["metadata"]["candidate_source"] == "context":
                context_weights.append(row["papo_listwise_weight"])
        self.assertAlmostEqual(weights_by_group["papo_v4::train::suggestion__1"], 1.0, places=6)
        self.assertTrue(oracle_weights)
        self.assertTrue(context_weights)
        self.assertGreater(context_weights[0], 0.0)
        self.assertNotAlmostEqual(oracle_weights[0], 0.9, places=6)

    def test_audit_warning_for_fixed_prob_and_zero_rejected(self) -> None:
        report = audit_wide_rows([_sample_row(), _sample_row(task_id="suggestion__2")])
        warning_text = "\n".join(report["warnings"])
        self.assertIn("oracle_prob has a single unique value", warning_text)
        self.assertIn("dpo_rejected_count is all zero", warning_text)


def _sample_row(
    *,
    task_id: str = "suggestion__1",
    same_user_text: str = "打开淘宝搜索耳机",
    context_text: str = "打开高德地图搜索餐厅",
    context_reward_total: float = 0.32,
) -> dict[str, object]:
    return {
        "split": "train",
        "task_id": task_id,
        "group_id": f"papo_v4::train::{task_id}",
        "user_id": "1",
        "target_time": "20250309_120000",
        "intent_class": "购物",
        "target_app": "com.taobao.taobao",
        "image_count": 2,
        "history_count": 20,
        "oracle_margin_prob": 0.4,
        "oracle_margin_reward": 0.2,
        "prompt_text": "当前界面和历史记录如下，请输出当前意图。",
        "image_paths": ["a.jpg", "b.jpg"],
        "oracle_text": "打开淘宝搜索蓝牙耳机",
        "oracle_prob": 0.9,
        "oracle_reward_total": 1.0,
        "same_user_text": same_user_text,
        "same_user_prob": 0.1,
        "same_user_reward_total": 0.76,
        "same_user_source_app": "com.taobao.taobao",
        "same_user_source_task_id": "suggestion__hist__1",
        "same_user_source_time": "20250308_120000",
        "same_user_semantic_similarity": 0.60,
        "same_user_eligibility": "listwise",
        "context_text": context_text,
        "context_prob": 0.0,
        "context_reward_total": context_reward_total,
        "context_source_app": "com.autonavi.minimap",
        "context_source_task_id": "suggestion__ctx__1",
        "context_source_time": "20250308_110000",
        "context_semantic_similarity": 0.20,
        "context_eligibility": "contrast_only_zero_mass",
        "cross_user_analysis_count": 2,
        "dpo_rejected_count": 0,
        "release_eligibility": "retrieval_only_history_candidates",
    }


if __name__ == "__main__":
    unittest.main()
