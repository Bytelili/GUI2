from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GroupListwiseSmokeExperimentTest(unittest.TestCase):
    def test_oracle_control_uses_same_prompt_and_only_oracle_answer(self) -> None:
        module = load_script("31_prepare_papo_group_listwise_v4_smoke_control.py")
        groups = [
            {
                "task_id": "suggestion__1__20250101_120000",
                "group_id": "g1",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "prompt"},
                ],
                "images": ["image.jpg"],
                "candidates": [
                    {"source": "oracle_target", "text": "oracle"},
                    {"source": "same_user", "text": "negative"},
                ],
                "oracle_index": 0,
                "metadata": {"protocol_id": "test", "partition": "train"},
            }
        ]
        rows = module.oracle_only_rows(groups, "train")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["messages"][-1], {"role": "assistant", "content": "oracle"})
        self.assertNotIn("negative", str(rows[0]))
        self.assertEqual(rows[0]["metadata"]["source_group_id"], "g1")

    def test_oracle_control_verification_is_hash_bound(self) -> None:
        module = load_script("31_prepare_papo_group_listwise_v4_smoke_control.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_manifest = root / "source_manifest.json"
            source_manifest.write_text("{}", encoding="utf-8")
            data = root / "train.json"
            data.write_text("[]", encoding="utf-8")
            manifest = {
                "formal_full_v4_complete": False,
                "source_release_manifest_sha256": module.sha256_file(source_manifest),
                "dataset_hashes": {"train.json": module.sha256_file(data)},
                "group_counts": {"train": 0},
            }
            (root / "oracle_control_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(module.verify_control(root, source_manifest)["status"], "passed")
            data.write_text("[1]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                module.verify_control(root, source_manifest)

    def test_group_and_control_configs_are_isolated(self) -> None:
        group_path = ROOT / "configs" / "llamafactory" / (
            "ui_tars_7b_papo_group_listwise_v4_retrieval_smoke.yaml"
        )
        control_path = ROOT / "configs" / "llamafactory" / (
            "ui_tars_7b_papo_v4_oracle_control_smoke.yaml"
        )
        group = yaml.safe_load(group_path.read_text(encoding="utf-8"))
        control = yaml.safe_load(control_path.read_text(encoding="utf-8"))
        self.assertTrue(group["use_papo_group_listwise"])
        self.assertFalse(group["use_papo_listwise"])
        self.assertFalse(group["packing"])
        self.assertTrue(group["papo_allow_nonformal_smoke"])
        self.assertFalse(control["use_papo_group_listwise"])
        self.assertFalse(control["use_papo_listwise"])
        self.assertNotEqual(group["output_dir"], control["output_dir"])
        self.assertEqual(group["model_name_or_path"], control["model_name_or_path"])
        self.assertEqual(group["adapter_name_or_path"], control["adapter_name_or_path"])

    def test_report_detects_metrics_and_anomalies(self) -> None:
        module = load_script("33_report_papo_group_listwise_v4_smoke.py")
        history = [
            {"step": 5, "papo_oracle_margin": -0.1, "papo_policy_entropy": 0.6},
            {"step": 10, "eval_loss": 0.9, "eval_papo_oracle_top1_accuracy": 0.75},
        ]
        rows = module.metric_rows(history)
        summary = module.summarize_metrics(rows)
        self.assertEqual(summary["eval_loss"]["latest"], 0.9)
        self.assertEqual(summary["papo_oracle_margin"]["min_step"], 5)
        self.assertEqual(module.scan_anomalies("CUDA out of memory", rows), ["oom"])

    def test_find_state_uses_highest_step(self) -> None:
        module = load_script("33_report_papo_group_listwise_v4_smoke.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for step in (5, 20):
                checkpoint = root / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "trainer_state.json").write_text(
                    json.dumps({"global_step": step, "log_history": []}), encoding="utf-8"
                )
            path, state = module.find_state(root)
            self.assertEqual(path.parent.name, "checkpoint-20")
            self.assertEqual(state["global_step"], 20)


if __name__ == "__main__":
    unittest.main()
