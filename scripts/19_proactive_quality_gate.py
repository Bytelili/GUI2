from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_quality_gate import (  # noqa: E402
    ProactiveQualityGate,
    print_decision,
    read_json_array,
    write_issues_csv,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run blocking quality gate for proactive preference datasets."
    )
    parser.add_argument("--data-dir", default="LLaMA-Factory/data/papo")
    parser.add_argument("--output-dir", default="reports/proactive/quality_gate")
    parser.add_argument("--raw-root", default="/home/dumike/zyy/GUI/data/raw/fingertip20k")
    parser.add_argument("--asset-root", default="LLaMA-Factory/data/papo")
    parser.add_argument("--min-oracle-margin", type=float, default=0.10)
    parser.add_argument("--max-answer-frequency", type=int, default=100)
    parser.add_argument("--max-non-oracle-mass", type=float, default=0.35)
    parser.add_argument("--leak-weight-threshold", type=float, default=0.05)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Write reports but return exit code 0 even when the quality gate fails.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_roots = [
        Path(args.raw_root),
        Path(args.asset_root),
        PROJECT_ROOT,
    ]
    gate = ProactiveQualityGate(
        image_roots=image_roots,
        min_oracle_margin=args.min_oracle_margin,
        max_answer_frequency=args.max_answer_frequency,
        max_non_oracle_mass=args.max_non_oracle_mass,
        leak_weight_threshold=args.leak_weight_threshold,
        progress_every=args.progress_every,
        fail_fast=args.fail_fast,
    )

    summaries = []
    inputs = {
        "train_listwise": data_dir / "papo_proactive_train_listwise.json",
        "eval_listwise": data_dir / "papo_proactive_eval_listwise.json",
        "train_dpo": data_dir / "papo_proactive_train_dpo.json",
        "eval_dpo": data_dir / "papo_proactive_eval_dpo.json",
    }

    print("===== Proactive preference quality gate =====", flush=True)
    print("Inputs:", flush=True)
    for name, path in inputs.items():
        print(f"  - {name}: {path} exists={path.exists()}", flush=True)

    for name in ("train_listwise", "eval_listwise"):
        path = inputs[name]
        if not path.exists():
            continue
        rows = read_json_array(path)
        print(f"===== Audit {name}: rows={len(rows)} =====", flush=True)
        summaries.append(gate.audit_listwise(rows, name=name))

    for name in ("train_dpo", "eval_dpo"):
        path = inputs[name]
        if not path.exists():
            continue
        rows = read_json_array(path)
        print(f"===== Audit {name}: rows={len(rows)} =====", flush=True)
        summaries.append(gate.audit_dpo(rows, name=name))

    decision = gate.decide(summaries)
    report = {
        "status": decision.status,
        "blocking_reasons": decision.blocking_reasons,
        "warning_reasons": decision.warning_reasons,
        "summaries": summaries,
        "issue_counts": {
            "by_severity": _count(issue.severity for issue in gate.issues),
            "by_category": _count(issue.category for issue in gate.issues),
        },
        "thresholds": {
            "min_oracle_margin": args.min_oracle_margin,
            "max_answer_frequency": args.max_answer_frequency,
            "max_non_oracle_mass": args.max_non_oracle_mass,
            "leak_weight_threshold": args.leak_weight_threshold,
        },
        "inputs": {name: str(path) for name, path in inputs.items()},
    }

    report_path = output_dir / "quality_report.json"
    issues_path = output_dir / "quality_issues.csv"
    manifest_path = output_dir / "blocked_manifest.json"
    summary_path = output_dir / "quality_summary.txt"

    write_json(report_path, report)
    write_issues_csv(issues_path, gate.issues)
    write_json(
        manifest_path,
        {
            "status": decision.status,
            "blocked": decision.status == "failed",
            "blocking_reasons": decision.blocking_reasons,
            "inputs": {name: str(path) for name, path in inputs.items()},
        },
    )
    with summary_path.open("w", encoding="utf-8") as f:
        print_decision(decision, file=f)
        f.write("\n")
        f.write(json.dumps(report["issue_counts"], ensure_ascii=False, indent=2))
        f.write("\n")

    print_decision(decision)
    print(f"Report: {report_path}")
    print(f"Issues: {issues_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")

    if decision.status == "failed" and not args.allow_fail:
        raise SystemExit(1)


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


if __name__ == "__main__":
    main()
