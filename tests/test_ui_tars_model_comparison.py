from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = PROJECT_ROOT / "ui_tars_proactive" / "compare_models.py"
    spec = importlib.util.spec_from_file_location("compare_models", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UiTarsModelComparisonTest(unittest.TestCase):
    def test_paired_comparison_reports_regression(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "reference"
            candidate = root / "candidate"
            for level in range(4):
                self.write_level(reference, level, [0.8, 0.6])
                self.write_level(candidate, level, [0.7, 0.5])
            output = root / "output"
            report = module.compare_models(
                reference,
                candidate,
                reference_label="sft",
                candidate_label="listwise",
                output_dir=output,
                bootstrap_samples=100,
                seed=42,
                case_limit=5,
            )
            self.assertAlmostEqual(report["macro_across_levels"]["official_delta"], -0.1)
            self.assertLess(report["macro_across_levels"]["task_bootstrap_ci95_high"], 0)
            self.assertTrue((output / "regression_cases.csv").exists())
            self.assertTrue((output / "paired_model_comparison.md").exists())

    @staticmethod
    def write_level(root: Path, level: int, scores: list[float]) -> None:
        path = root / f"level_{level}" / "metrics" / "proactive_predictions_scored.csv"
        path.parent.mkdir(parents=True)
        fields = [
            "task_id",
            "user_id",
            "original_intent",
            "predicted_intent",
            "official_similarity",
            "edit_similarity",
            "semantic_similarity",
            "time",
            "token",
            "error",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for index, score in enumerate(scores):
                writer.writerow(
                    {
                        "task_id": f"task-{index}",
                        "user_id": str(index),
                        "original_intent": f"intent-{index}",
                        "predicted_intent": f"prediction-{index}",
                        "official_similarity": score,
                        "edit_similarity": score,
                        "semantic_similarity": score,
                        "time": 1.0,
                        "token": 10,
                        "error": "",
                    }
                )


if __name__ == "__main__":
    unittest.main()
