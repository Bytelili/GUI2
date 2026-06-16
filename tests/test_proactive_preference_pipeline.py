from __future__ import annotations

import json
import csv
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = PROJECT_ROOT / "proactive_preference_pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppipeline.audit import audit_preference_sets  # noqa: E402
from ppipeline.candidates import build_candidate_sets  # noqa: E402
from ppipeline.export import export_preference_datasets, preference_dataset_info, proactive_prompt  # noqa: E402
from ppipeline.quality import QualityThresholds, audit_candidate_quality, build_quality_review_sample  # noqa: E402
from ppipeline.rewards import RewardWeights, score_candidate_sets  # noqa: E402
from papo.data_protocol import sha256_file  # noqa: E402


class ProactivePreferencePipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.train_tasks = [
            self.task("train-1", "1", "20250101_080000", "home", "打开天气查看今天温度", []),
            self.task(
                "train-2",
                "2",
                "20250102_080000",
                "home",
                "打开新闻查看今日热点",
                [],
            ),
        ]
        self.eval_tasks = [
            self.task(
                "eval-1",
                "1",
                "20250103_080000",
                "home",
                "打开天气查看明天温度",
                [
                    {
                        "episode_id": "1__20250101_080000",
                        "user_id": "1",
                        "time": "20250101_080000",
                        "scenario": "home",
                        "intent": "打开天气查看今天温度",
                    }
                ],
                partition="eval",
            )
        ]
        self.allowed = {"1__20250101_080000", "2__20250102_080000"}

    def test_build_score_export_and_audit(self) -> None:
        train, evaluation = self.build_sets()
        report = audit_preference_sets(train, evaluation, train_reference_ids=self.allowed)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["train_eval_target_overlap"], 0)
        self.assertTrue(train)
        self.assertTrue(evaluation)
        for row in train + evaluation:
            probabilities = [candidate["target_policy_probability"] for candidate in row["candidates"]]
            self.assertAlmostEqual(sum(probabilities), 1.0)
            self.assertTrue(row["pairs"])
            self.assertTrue(all(pair["chosen_source"] == "oracle_target" for pair in row["pairs"]))

        listwise, dpo = export_preference_datasets(
            evaluation,
            raw_root="/raw",
            asset_prefix="RawDataset",
        )
        self.assertGreater(len(listwise), len(evaluation))
        self.assertTrue(dpo)
        self.assertEqual(dpo[0]["metadata"]["partition"], "eval")
        self.assertIn("papo_proactive_train_listwise", preference_dataset_info())

    def test_eval_candidates_never_use_eval_targets_as_references(self) -> None:
        _, evaluation = self.build_sets()
        sources = {
            str(candidate.get("source_episode_id") or "")
            for row in evaluation
            for candidate in row["candidates"]
            if candidate["source"] not in {"oracle_target", "sft_sample"}
        }
        self.assertTrue(sources <= self.allowed)
        self.assertNotIn("1__20250103_080000", sources)

    def test_audit_rejects_reference_outside_strict_train(self) -> None:
        train, evaluation = self.build_sets()
        evaluation[0]["candidates"][1]["source_episode_id"] = "99__20240101_000000"
        with self.assertRaisesRegex(ValueError, "outside_train_references"):
            audit_preference_sets(train, evaluation, train_reference_ids=self.allowed)

    def test_prompt_does_not_include_hidden_target_field(self) -> None:
        prompt = proactive_prompt(self.eval_tasks[0]["input"])
        self.assertNotIn("打开天气查看明天温度", prompt)
        self.assertIn("打开天气查看今天温度", prompt)

    def test_preference_dataset_info_matches_llamafactory_extensions(self) -> None:
        info = preference_dataset_info()
        listwise = info["papo_proactive_train_listwise"]
        dpo = info["papo_proactive_train_dpo"]
        self.assertEqual(listwise["columns"]["listwise_weight"], "papo_listwise_weight")
        self.assertEqual(dpo["columns"]["preference_weight"], "papo_weight")
        self.assertEqual(dpo["columns"]["preference_target"], "papo_target_probability")

    def test_reward_weights_must_sum_to_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to one"):
            RewardWeights(task=1.0, user=1.0, context=0.0, specificity=0.0).validate()

    def test_quality_gate_classifies_candidates_and_removes_unsafe_dpo_pairs(self) -> None:
        train = [self.quality_row("train", "train-quality")]
        evaluation = [self.quality_row("eval", "eval-quality")]
        report, flags = audit_candidate_quality(train, evaluation)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["hard_failures"])
        self.assertEqual(
            report["partitions"]["train"]["classification_counts"],
            {
                "easy_negative": 1,
                "invalid": 1,
                "pseudo_negative": 1,
                "valid_hard_negative": 1,
            },
        )
        self.assertEqual(len(train[0]["pairs"]), 2)
        self.assertEqual(
            {pair["rejected_candidate_id"] for pair in train[0]["pairs"]},
            {"easy", "hard"},
        )
        self.assertEqual(
            {item["quality_class"] for item in flags},
            {"invalid", "pseudo_negative", "easy_negative"},
        )
        review = build_quality_review_sample(train, evaluation, per_bucket=1)
        self.assertEqual(
            {item["quality"]["class"] for item in review},
            {"invalid", "pseudo_negative", "easy_negative", "valid_hard_negative"},
        )
        self.assertTrue(all(item["target"] for item in review))
        listwise, dpo = export_preference_datasets(train, raw_root="/raw", asset_prefix="RawDataset")
        self.assertNotIn("x", {row["messages"][-1]["content"] for row in listwise})
        self.assertAlmostEqual(sum(row["papo_listwise_weight"] for row in listwise), 1.0)
        self.assertEqual({row["metadata"]["rejected_candidate_id"] for row in dpo}, {"easy", "hard"})

    def test_quality_gate_can_make_proxy_warnings_without_hard_failure(self) -> None:
        train = self.quality_row("train", "train-warning")
        evaluation = self.quality_row("eval", "eval-warning")
        for row in [train, evaluation]:
            row["candidates"] = [
                candidate for candidate in row["candidates"] if candidate["candidate_id"] != "invalid"
            ]
            row["pairs"] = [
                pair for pair in row["pairs"] if pair["rejected_candidate_id"] != "invalid"
            ]
        report, _ = audit_candidate_quality([train], [evaluation], thresholds=QualityThresholds())
        self.assertEqual(report["status"], "warning")
        self.assertFalse(report["hard_failures"])

    def test_candidate_generation_resume_retries_empty_and_truncated_rows(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "generate_model_candidates",
            PIPELINE_ROOT / "generate_model_candidates.py",
        )
        if spec is None or spec.loader is None:
            self.fail("Cannot load candidate generator")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidates.jsonl"
            path.write_text(
                json.dumps({"task_id": "one", "candidates": ["valid"]})
                + "\n"
                + json.dumps({"task_id": "two", "candidates": []})
                + "\n"
                + '{"task_id": "three"',
                encoding="utf-8",
            )
            rows, removed = module._prepare_resume(path, {"one", "two", "three"})
            self.assertEqual([row["task_id"] for row in rows], ["one"])
            self.assertEqual(removed, 2)
            self.assertEqual(module._read_jsonl(path), rows)

    def test_build_cli_writes_audited_llamafactory_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            history_path = root / "history.csv"
            work_dir = root / "work"
            dataset_dir = root / "datasets"
            self.write_jsonl(train_path, self.train_tasks)
            self.write_jsonl(eval_path, self.eval_tasks)
            with history_path.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["user_id", "time"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"user_id": "1", "time": "20250101_080000"},
                        {"user_id": "2", "time": "20250102_080000"},
                    ]
                )
            subprocess.run(
                [
                    sys.executable,
                    str(PIPELINE_ROOT / "build_preferences.py"),
                    "--train-tasks",
                    str(train_path),
                    "--eval-tasks",
                    str(eval_path),
                    "--protocol-history",
                    str(history_path),
                    "--work-dir",
                    str(work_dir),
                    "--dataset-dir",
                    str(dataset_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads((work_dir / "preference_manifest.json").read_text(encoding="utf-8"))
            info = json.loads((dataset_dir / "dataset_info.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "passed")
            self.assertIn(manifest["candidate_quality"]["status"], {"passed", "warning"})
            self.assertTrue((work_dir / "candidate_quality_report.json").exists())
            self.assertTrue((work_dir / "candidate_quality_flags.jsonl").exists())
            self.assertTrue((work_dir / "candidate_quality_review_sample.jsonl").exists())
            self.assertEqual(len(manifest["datasets"]), 4)
            self.assertIn("papo_proactive_train_listwise", info)
            self.assertTrue(json.loads((dataset_dir / "papo_proactive_eval_dpo.json").read_text(encoding="utf-8")))
            training_path = root / "listwise.yaml"
            training_path.write_text(
                yaml.safe_dump(
                    {
                        "stage": "sft",
                        "dataset": "papo_proactive_train_listwise",
                        "eval_dataset": "papo_proactive_eval_listwise",
                        "output_dir": str(root / "proactive_preference_listwise_clean_v2"),
                        "use_papo_listwise": True,
                    }
                ),
                encoding="utf-8",
            )
            preflight = self.load_script(PIPELINE_ROOT / "preflight.py")
            report = preflight.validate_preference_training(
                work_dir / "preference_manifest.json",
                training_path,
            )
            self.assertEqual(report["status"], "passed")
            bad_manifest = dict(manifest)
            bad_manifest["candidate_quality"] = {
                **manifest["candidate_quality"],
                "status": "failed",
                "hard_failures": ["test failure"],
            }
            bad_manifest_path = work_dir / "preference_manifest_failed_quality.json"
            bad_manifest_path.write_text(json.dumps(bad_manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "quality hard gate"):
                preflight.validate_preference_training(bad_manifest_path, training_path)
            dataset = dataset_dir / "papo_proactive_train_listwise.json"
            dataset.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed after audited build"):
                preflight.validate_preference_training(
                    work_dir / "preference_manifest.json",
                    training_path,
                )

    def test_build_cli_excludes_invalid_train_oracle_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            history_path = root / "history.csv"
            work_dir = root / "work"
            dataset_dir = root / "datasets"
            train_tasks = self.train_tasks + [
                self.task("bad", "3", "20250104_080000", "home", "x", [])
            ]
            self.write_jsonl(train_path, train_tasks)
            self.write_jsonl(eval_path, self.eval_tasks)
            with history_path.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["user_id", "time"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"user_id": "1", "time": "20250101_080000"},
                        {"user_id": "2", "time": "20250102_080000"},
                        {"user_id": "3", "time": "20250104_080000"},
                    ]
                )
            subprocess.run(
                [
                    sys.executable,
                    str(PIPELINE_ROOT / "build_preferences.py"),
                    "--train-tasks",
                    str(train_path),
                    "--eval-tasks",
                    str(eval_path),
                    "--protocol-history",
                    str(history_path),
                    "--work-dir",
                    str(work_dir),
                    "--dataset-dir",
                    str(dataset_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads((work_dir / "preference_manifest.json").read_text(encoding="utf-8"))
            excluded = [
                json.loads(line)
                for line in (work_dir / "candidate_quality_excluded_targets.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(manifest["status"], "passed")
            self.assertEqual(manifest["candidate_quality"]["excluded_train_invalid_oracle_targets"]["count"], 1)
            self.assertEqual(excluded[0]["task_id"], "suggestion__bad")
            self.assertEqual(excluded[0]["reasons"], ["too_short"])
            self.assertEqual(manifest["partitions"]["train"]["targets"], 1)

    def test_rendered_configs_chain_sft_listwise_and_dpo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.yaml"
            out_dir = root / "configs"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "paths": {
                            "qwen_model_path": "/models/qwen",
                            "checkpoint_root": "/runs/checkpoints",
                            "logging_root": "/runs/logs",
                            "llamafactory_data_dir": "/datasets/papo",
                        },
                        "training": {
                            "image_max_pixels": 262144,
                            "lora_rank": 16,
                            "template": "qwen2_vl",
                            "cutoff_len": 4096,
                        },
                    }
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(PIPELINE_ROOT / "render_training_configs.py"),
                    "--config",
                    str(config_path),
                    "--out-dir",
                    str(out_dir),
                    "--sft-adapter",
                    "/runs/checkpoints/proactive_sft_clean_v2_best",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            listwise = yaml.safe_load((out_dir / "proactive_preference_listwise.yaml").read_text(encoding="utf-8"))
            dpo = yaml.safe_load((out_dir / "proactive_preference_dpo.yaml").read_text(encoding="utf-8"))
            self.assertEqual(listwise["adapter_name_or_path"], "/runs/checkpoints/proactive_sft_clean_v2_best")
            self.assertTrue(listwise["use_papo_listwise"])
            self.assertEqual(dpo["adapter_name_or_path"], "/runs/checkpoints/proactive_preference_listwise_clean_v2_best")
            self.assertEqual(dpo["pref_loss"], "papo")

    def test_official_predictor_accepts_clean_proactive_preference_adapter(self) -> None:
        predictor = self.load_script(PROJECT_ROOT / "scripts" / "18_run_proactive_predictions.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / "adapter"
            protocol = root / "protocol"
            datasets = root / "datasets"
            adapter.mkdir()
            protocol.mkdir()
            datasets.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            manifest = protocol / "protocol_manifest.json"
            manifest.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
            dataset_info = {}
            hashes = {}
            for name in ["papo_proactive_train_listwise", "papo_proactive_eval_listwise"]:
                filename = f"{name}.json"
                path = datasets / filename
                path.write_text("[]", encoding="utf-8")
                dataset_info[name] = {"file_name": filename}
                hashes[name] = sha256_file(path)
            (datasets / "dataset_info.json").write_text(json.dumps(dataset_info), encoding="utf-8")
            (adapter / "papo_training_provenance.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "protocol_id": "test_protocol",
                        "protocol_manifest_sha256": sha256_file(manifest),
                        "datasets": ["papo_proactive_train_listwise"],
                        "eval_datasets": ["papo_proactive_eval_listwise"],
                        "dataset_hashes": hashes,
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "_project_root": str(root),
                "paths": {
                    "protocol_dir": str(protocol),
                    "llamafactory_data_dir": str(datasets),
                },
                "data": {"protocol": {"protocol_id": "test_protocol"}},
            }
            predictor._validate_adapter(adapter, config)

    def test_preference_result_summary_collects_level_metrics(self) -> None:
        summary = self.load_script(PIPELINE_ROOT / "summarize_results.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics_path = root / "dpo" / "strict_holdout" / "level_3" / "metrics" / "benchmark_metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                json.dumps(
                    {
                        "proactive_suggestion": {
                            "level_3": {
                                "count": 10,
                                "official_similarity": {"mean": 0.8},
                                "edit_similarity": {"mean": 0.7},
                                "semantic_similarity": {"mean": 0.9},
                                "time": {"mean": 1.0},
                                "token": {"mean": 100.0},
                                "error_rate": 0.0,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            rows = summary.collect_results(root, "strict_holdout", ["dpo"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["level"], 3)
            self.assertEqual(rows[0]["official_similarity"], 0.8)

    def build_sets(self):
        train_raw = build_candidate_sets(
            self.train_tasks,
            reference_tasks=self.train_tasks,
            partition="train",
            model_candidates={"suggestion__train-2": ["打开浏览器查看新闻"]},
        )
        eval_raw = build_candidate_sets(
            self.eval_tasks,
            reference_tasks=self.train_tasks,
            partition="eval",
            model_candidates={"suggestion__eval-1": ["打开天气查看温度趋势"]},
        )
        weights = RewardWeights()
        return (
            score_candidate_sets(train_raw, weights=weights),
            score_candidate_sets(eval_raw, weights=weights),
        )

    @staticmethod
    def quality_row(partition: str, task_id: str) -> dict:
        candidates = [
            {
                "candidate_id": "oracle",
                "source": "oracle_target",
                "text": "open weather and show today's temperature",
                "reward": {"task_match": 1.0, "same_user_similarity": 0.8, "total": 1.0},
                "target_policy_probability": 0.2,
            },
            {
                "candidate_id": "pseudo",
                "source": "sft_sample",
                "text": "open weather and show today temperature",
                "reward": {"task_match": 0.95, "same_user_similarity": 0.8, "total": 0.9},
                "target_policy_probability": 0.2,
            },
            {
                "candidate_id": "easy",
                "source": "cross_user_hard",
                "text": "book a restaurant",
                "reward": {"task_match": 0.1, "same_user_similarity": 0.1, "total": 0.2},
                "target_policy_probability": 0.2,
            },
            {
                "candidate_id": "hard",
                "source": "same_user_history",
                "text": "open weather and show tomorrow's temperature",
                "reward": {"task_match": 0.7, "same_user_similarity": 0.8, "total": 0.7},
                "target_policy_probability": 0.2,
            },
            {
                "candidate_id": "invalid",
                "source": "sft_sample",
                "text": "x",
                "reward": {"task_match": 0.0, "same_user_similarity": 0.0, "total": 0.0},
                "target_policy_probability": 0.2,
            },
        ]
        return {
            "task_id": task_id,
            "partition": partition,
            "input": {"initial_screenshots": []},
            "target": {"intent": candidates[0]["text"]},
            "metadata": {"papo_episode_id": f"{task_id}__20250101_000000"},
            "candidates": candidates,
            "pairs": [
                {
                    "chosen_candidate_id": "oracle",
                    "rejected_candidate_id": candidate["candidate_id"],
                    "chosen": candidates[0]["text"],
                    "rejected": candidate["text"],
                    "chosen_source": "oracle_target",
                    "rejected_source": candidate["source"],
                    "reward_gap": 0.5,
                    "weight": 1.0,
                    "target_preference_probability": 0.9,
                }
                for candidate in candidates[1:]
            ],
        }

    @staticmethod
    def task(
        suffix: str,
        user_id: str,
        time: str,
        scenario: str,
        intent: str,
        history: list[dict],
        *,
        partition: str = "train",
    ) -> dict:
        return {
            "task_id": f"suggestion__{suffix}",
            "task_type": "proactive_suggestion",
            "input": {
                "user_id": user_id,
                "time": time,
                "scenario": scenario,
                "user_profile": {"age": "20"},
                "previous_intents": history,
                "initial_screenshots": [],
            },
            "target": {"intent": intent},
            "metadata": {
                "papo_episode_id": f"{user_id}__{time}",
                "partition": partition,
                "protocol_id": "test_protocol",
                "target_split": f"proactive_{partition}_targets.csv",
                "history_split": "proactive_history.csv",
                "history_policy": "same_user_strictly_before_target_time",
                "history_episode_ids": [item["episode_id"] for item in history],
            },
        }

    @staticmethod
    def write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    @staticmethod
    def load_script(path: Path):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


if __name__ == "__main__":
    unittest.main()
