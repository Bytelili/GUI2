from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from epipeline.io_utils import manifest_identity_matches, read_json, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize scored execution CSVs with the bundled official evaluator.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-non-paper-eligible", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    evaluator_path = project_root / "evaluation" / "fingertip" / "evaluate_reports.py"
    spec = importlib.util.spec_from_file_location("bundled_fingertip_evaluator", evaluator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load bundled evaluator: {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = read_json(args.manifest)
    if not manifest_identity_matches(manifest):
        raise ValueError("Experiment manifest identity is missing or changed")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_metrics = {}
    for index, entry in enumerate(manifest["runs"]):
        run_id = str(entry["id"])
        run_dir = Path(str(entry["run_dir"]))
        score_path = run_dir / "score_report.json"
        scored_path = run_dir / "execution_results_scored.csv"
        if not score_path.exists() or not scored_path.exists():
            raise FileNotFoundError(f"Missing scored execution artifacts for run: {run_id}")
        score = read_json(score_path)
        if score.get("scored_csv_sha256") != sha256_file(scored_path):
            raise ValueError(f"Scored execution CSV changed after scoring: {run_id}")
        if not args.allow_non_paper_eligible and not score.get("paper_eligible"):
            raise ValueError(f"Official evaluation refuses non-paper-eligible run: {run_id}")
        run_output = output / run_id
        run_output.mkdir(parents=True, exist_ok=True)
        _data, metrics = module.evaluate_execution(
            [str(scored_path)],
            args.bootstrap_samples,
            np.random.default_rng(args.seed + index),
            run_output,
        )
        run_metrics[run_id] = {
            "scored_csv": str(scored_path),
            "paper_eligible": bool(score.get("paper_eligible")),
            "metrics": metrics,
        }
    report = {
        "status": "completed",
        "evaluator_path": str(evaluator_path),
        "runs": run_metrics,
    }
    write_json(output / "official_execution_metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("BUNDLED OFFICIAL EXECUTION EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
