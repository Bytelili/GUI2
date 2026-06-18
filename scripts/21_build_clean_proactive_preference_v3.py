from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_preference_cleaner import (  # noqa: E402
    CleanConfig,
    clean_preference_split,
    run_quality_gate_on_clean_artifacts,
    update_dataset_info_with_v3,
)
from papo.proactive_quality_gate import read_json_array, write_json, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build clean v3 proactive preference data with source-aware filtering and quality gate."
    )
    parser.add_argument("--data-dir", default="LLaMA-Factory/data/papo")
    parser.add_argument("--output-dir", default="LLaMA-Factory/data/papo")
    parser.add_argument("--report-dir", default="reports/proactive/clean_preference_v3")
    parser.add_argument("--raw-root", default="/home/dumike/zyy/GUI/data/raw/fingertip20k")
    parser.add_argument("--oracle-weight", type=float, default=0.80)
    parser.add_argument("--min-oracle-margin", type=float, default=0.10)
    parser.add_argument("--max-negatives-per-group", type=int, default=3)
    parser.add_argument("--max-answer-frequency", type=int, default=100)
    parser.add_argument("--allow-history-negatives", action="store_true")
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    config = CleanConfig(
        oracle_weight=args.oracle_weight,
        min_oracle_margin=args.min_oracle_margin,
        max_negatives_per_group=args.max_negatives_per_group,
        max_same_answer_frequency=args.max_answer_frequency,
        allow_history_negatives=args.allow_history_negatives,
    )

    all_reports: dict[str, Any] = {
        "status": "running",
        "config": config.__dict__,
        "splits": {},
        "quality_gate": {},
        "outputs": {},
    }
    any_failed = False

    for split in ("train", "eval"):
        input_path = data_dir / f"papo_proactive_{split}_listwise.json"
        print(f"===== Build clean v3 {split}: {input_path} =====", flush=True)
        if not input_path.exists():
            print(f"SKIP: missing {input_path}", flush=True)
            continue

        rows = read_json_array(input_path)
        artifacts = clean_preference_split(rows, split=split, config=config)
        gate_report = run_quality_gate_on_clean_artifacts(
            artifacts,
            split=split,
            config=config,
            image_roots=[Path(args.raw_root), output_dir, PROJECT_ROOT],
        )

        listwise_path = output_dir / f"papo_proactive_{split}_listwise_v3.json"
        dpo_path = output_dir / f"papo_proactive_{split}_dpo_v3.json"
        rejected_path = report_dir / f"{split}_rejected_rows.jsonl"
        split_report_path = report_dir / f"{split}_clean_report.json"
        gate_report_path = report_dir / f"{split}_quality_gate_report.json"

        write_json(listwise_path, artifacts.listwise_rows)
        write_json(dpo_path, artifacts.dpo_rows)
        write_jsonl(rejected_path, artifacts.rejected_rows)
        write_json(split_report_path, artifacts.report)
        write_json(gate_report_path, gate_report)

        print(
            f"{split}: input_rows={len(rows)} listwise_v3={len(artifacts.listwise_rows)} "
            f"dpo_v3={len(artifacts.dpo_rows)} rejected={len(artifacts.rejected_rows)} "
            f"gate={gate_report['status']}",
            flush=True,
        )
        if gate_report["status"] == "failed":
            any_failed = True
            print("Blocking reasons:", flush=True)
            for reason in gate_report["blocking_reasons"]:
                print(f"  - {reason}", flush=True)

        all_reports["splits"][split] = artifacts.report
        all_reports["quality_gate"][split] = {
            "status": gate_report["status"],
            "blocking_reasons": gate_report["blocking_reasons"],
            "warning_reasons": gate_report["warning_reasons"],
            "issue_counts": gate_report["issue_counts"],
        }
        all_reports["outputs"][split] = {
            "listwise": str(listwise_path),
            "dpo": str(dpo_path),
            "rejected": str(rejected_path),
            "clean_report": str(split_report_path),
            "quality_gate_report": str(gate_report_path),
        }

    update_dataset_info_with_v3(output_dir / "dataset_info.json")
    all_reports["outputs"]["dataset_info"] = str(output_dir / "dataset_info.json")
    all_reports["status"] = "failed" if any_failed else "passed"

    manifest_path = report_dir / "clean_v3_manifest.json"
    write_json(manifest_path, all_reports)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"QUALITY STATUS: {all_reports['status'].upper()}", flush=True)

    if any_failed and not args.allow_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
