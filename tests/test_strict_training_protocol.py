from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StrictTrainingProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.official = self.root / "official"
        self.protocol = self.root / "protocol"
        self.dataset_dir = self.root / "datasets"
        self.official.mkdir()
        self.dataset_dir.mkdir()
        self.rows = [
            self.row(user, f"2025010{index}_120000", f"intent-{user}-{index}")
            for user in ["1", "2"]
            for index in range(1, 5)
        ]
        self.write_csv(self.official / "train_set.csv", self.rows)
        self.write_csv(self.official / "test_suggestion.csv", [self.row("1", "20241201_120000", "test-suggestion")])
        self.write_csv(self.official / "test_execution.csv", [self.rows[1]])

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_protocol_is_temporal_and_excludes_same_track_tests(self) -> None:
        from papo.data_protocol import build_formal_protocol, episode_keys
        from papo.official_data import read_csv_rows

        manifest = build_formal_protocol(
            self.official,
            self.protocol,
            source_train_split="train_set.csv",
            proactive_test_split="test_suggestion.csv",
            execution_test_split="test_execution.csv",
            validation_fraction=0.25,
            min_validation_per_user=1,
            protocol_id="test_protocol",
        )
        proactive_train = read_csv_rows(self.protocol / "proactive_train_targets.csv")
        proactive_eval = read_csv_rows(self.protocol / "proactive_eval_targets.csv")
        execution_train = read_csv_rows(self.protocol / "execution_train_targets.csv")
        execution_eval = read_csv_rows(self.protocol / "execution_eval_targets.csv")
        execution_test = read_csv_rows(self.official / "test_execution.csv")

        self.assertEqual(manifest["status"], "passed")
        self.assertFalse(episode_keys(proactive_train) & episode_keys(proactive_eval))
        self.assertFalse(episode_keys(execution_train) & episode_keys(execution_eval))
        self.assertFalse((episode_keys(execution_train) | episode_keys(execution_eval)) & episode_keys(execution_test))
        for user in ["1", "2"]:
            train_times = [row["time"] for row in proactive_train if row["user_id"] == user]
            eval_times = [row["time"] for row in proactive_eval if row["user_id"] == user]
            self.assertLess(max(train_times), min(eval_times))

    def test_preflight_accepts_clean_data_and_rejects_test_history(self) -> None:
        from papo.config import load_config
        from papo.data_protocol import build_formal_protocol

        build_formal_protocol(
            self.official,
            self.protocol,
            source_train_split="train_set.csv",
            proactive_test_split="test_suggestion.csv",
            execution_test_split="test_execution.csv",
            validation_fraction=0.25,
            min_validation_per_user=1,
            protocol_id="test_protocol",
        )
        config_path = self.root / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "paths": {
                        "official_root": str(self.official),
                        "protocol_dir": str(self.protocol),
                    },
                    "data": {
                        "protocol": {
                            "protocol_id": "test_protocol",
                            "proactive_test_split": "test_suggestion.csv",
                            "execution_test_split": "test_execution.csv",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        project_config = load_config(config_path)
        test_id = "1__20241201_120000"
        self.write_dataset(
            "papo_proactive_train_sft",
            "train",
            ["1__20250101_120000", "2__20250101_120000"],
        )
        self.write_dataset(
            "papo_proactive_eval_sft",
            "eval",
            ["1__20250104_120000", "2__20250104_120000"],
        )
        training = {
            "dataset_dir": str(self.dataset_dir),
            "dataset": "papo_proactive_train_sft",
            "eval_dataset": "papo_proactive_eval_sft",
            "val_size": 0.0,
            "output_dir": str(self.root / "proactive_sft_clean_v2"),
            "save_steps": 10,
            "eval_steps": 10,
            "load_best_model_at_end": False,
        }
        training_path = self.root / "training.yaml"
        training_path.write_text(yaml.safe_dump(training), encoding="utf-8")
        preflight = load_script("15_training_preflight.py")
        report = preflight.validate_training(project_config, training_path, training)
        self.assertEqual(report["status"], "passed")

        path = self.dataset_dir / "papo_proactive_train_sft.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        rows[0]["metadata"]["history_episode_ids"] = [test_id]
        path.write_text(json.dumps(rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "context leakage"):
            preflight.validate_training(project_config, training_path, training)

    def test_best_checkpoint_selection_uses_lowest_eval_loss(self) -> None:
        finalizer = load_script("16_finalize_best_checkpoint.py")
        output = self.root / "run"
        for step, loss in [(10, 2.0), (20, 1.5), (30, 1.8)]:
            checkpoint = output / f"checkpoint-{step}"
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
            (checkpoint / "trainer_state.json").write_text(
                json.dumps({"log_history": [{"step": step, "eval_loss": loss}]}),
                encoding="utf-8",
            )
        selected, loss, step = finalizer.select_best_checkpoint(output)
        self.assertEqual(selected.name, "checkpoint-20")
        self.assertEqual(loss, 1.5)
        self.assertEqual(step, 20)
        target = self.root / "run_best"
        finalizer._replace_stable_directory(selected, target, output)
        self.assertEqual((target / "adapter_model.safetensors").read_bytes(), b"adapter")
        self.assertTrue((target / "trainer_state.json").exists())

        with self.assertRaisesRegex(ValueError, "Unsafe stable model target"):
            finalizer._replace_stable_directory(selected, self.root / "wrong_best", output)

    def test_proactive_pipeline_exports_explicit_train_and_eval(self) -> None:
        self.write_csv(
            self.official / "user_profile.csv",
            [{"user_id": "1", "age": "20"}, {"user_id": "2", "age": "30"}],
        )
        raw_root = self.root / "raw"
        for row in self.rows:
            self.write_raw_episode(raw_root, row)
        config = {
            "paths": {
                "official_root": str(self.official),
                "protocol_dir": str(self.protocol),
                "raw_root": str(raw_root),
                "work_dir": str(self.root / "work"),
                "task_dir": str(self.root / "tasks"),
                "llamafactory_data_dir": str(self.dataset_dir),
                "asset_prefix": "RawDataset",
            },
            "data": {
                "protocol": {
                    "protocol_id": "test_protocol",
                    "source_train_split": "train_set.csv",
                    "proactive_test_split": "test_suggestion.csv",
                    "execution_test_split": "test_execution.csv",
                    "validation_fraction": 0.25,
                    "min_validation_per_user": 1,
                },
                "suggestion_screenshot_level": 1,
                "suggestion_history_limit": 20,
                "require_complete": True,
            },
        }
        config_path = self.root / "pipeline_config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts/14_build_data_protocol.py"), "--config", str(config_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts/09_run_config_pipeline.py"),
                "--config",
                str(config_path),
                "--stages",
                "proactive_tasks,proactive_export",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        train_rows = json.loads((self.dataset_dir / "papo_proactive_train_sft.json").read_text(encoding="utf-8"))
        eval_rows = json.loads((self.dataset_dir / "papo_proactive_eval_sft.json").read_text(encoding="utf-8"))
        self.assertTrue(train_rows)
        self.assertTrue(eval_rows)
        self.assertTrue(all(row["metadata"]["partition"] == "train" for row in train_rows))
        self.assertTrue(all(row["metadata"]["partition"] == "eval" for row in eval_rows))
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts/08_validate_llamafactory_data.py"),
                "--dataset_dir",
                str(self.dataset_dir),
                "--datasets",
                "papo_proactive_train_sft,papo_proactive_eval_sft",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def write_dataset(self, name: str, partition: str, episode_ids: list[str]) -> None:
        info_path = self.dataset_dir / "dataset_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        filename = f"{name}.json"
        info[name] = {"file_name": filename}
        info_path.write_text(json.dumps(info), encoding="utf-8")
        rows = [
            {
                "messages": [],
                "images": [],
                "metadata": {
                    "partition": partition,
                    "protocol_id": "test_protocol",
                    "papo_episode_id": episode_id,
                    "history_episode_ids": [],
                },
            }
            for episode_id in episode_ids
        ]
        (self.dataset_dir / filename).write_text(json.dumps(rows), encoding="utf-8")

    @staticmethod
    def row(user_id: str, time: str, intent: str) -> dict[str, str]:
        return {
            "user_id": user_id,
            "time": time,
            "scenario": "home",
            "app": "example.app",
            "intentDescription": intent,
            "intentClass": "class",
        }

    @staticmethod
    def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def write_raw_episode(raw_root: Path, row: dict[str, str]) -> None:
        episode = raw_root / row["user_id"] / row["time"]
        screenshots = episode / "Screenshots"
        screenshots.mkdir(parents=True)
        (episode / "action.jsonl").write_text("{}\n", encoding="utf-8")
        (episode / "survey_result.json").write_text(json.dumps(row), encoding="utf-8")
        (screenshots / "frame.jpg").write_bytes(b"image")
        (screenshots / "frame.xml").write_text("<hierarchy />", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
