from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class UiTarsProactiveTest(unittest.TestCase):
    def test_summary_collects_base_sft_and_deltas(self) -> None:
        summary = self.load_script(PROJECT_ROOT / "ui_tars_proactive" / "summarize_results.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_metrics(root, "ui_tars_7b_base", 0, 0.3, 0.2, 0.4)
            self.write_metrics(root, "ui_tars_7b_sft", 0, 0.5, 0.4, 0.6)
            rows = summary.collect_results(
                root,
                "strict_holdout",
                ["ui_tars_7b_base", "ui_tars_7b_sft"],
            )
            self.assertEqual(len(rows), 2)
            comparison = summary.compare_sft_vs_base(rows)
            self.assertEqual(len(comparison), 1)
            self.assertEqual(comparison[0]["level"], 0)
            self.assertAlmostEqual(comparison[0]["official_similarity_delta"], 0.2)
            self.assertAlmostEqual(comparison[0]["edit_similarity_delta"], 0.2)
            self.assertAlmostEqual(comparison[0]["semantic_similarity_delta"], 0.2)

    @staticmethod
    def write_metrics(root: Path, model: str, level: int, official: float, edit: float, semantic: float) -> None:
        path = root / model / "strict_holdout" / f"level_{level}" / "metrics" / "benchmark_metrics.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "proactive_suggestion": {
                        f"level_{level}": {
                            "count": 2,
                            "official_similarity": {
                                "mean": official,
                                "median": official,
                                "ci95_low": official - 0.1,
                                "ci95_high": official + 0.1,
                            },
                            "official_similarity_raw": {"mean": official},
                            "edit_similarity": {"mean": edit},
                            "semantic_similarity": {"mean": semantic},
                            "time": {"mean": 1.0, "median": 1.0},
                            "token": {"mean": 10.0, "median": 10.0},
                            "error_rate": 0.0,
                        }
                    }
                }
            ),
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
