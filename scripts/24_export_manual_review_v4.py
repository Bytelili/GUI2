from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_listwise_v4 import V4ValidationError, sha256_file, write_json  # noqa: E402
from papo.proactive_manual_review import export_manual_review  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export human-sized PAPO v4 candidate and group review CSV files.")
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "eval"), required=True)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--regression-cases", type=Path)
    parser.add_argument("--seed", type=int, default=20260621)
    args = parser.parse_args()
    groups = json.loads(args.groups.read_text(encoding="utf-8", errors="strict"))
    observed_splits = {
        str((group.get("metadata") or {}).get("partition") or "")
        for group in groups
        if isinstance(group, dict)
    }
    if observed_splits != {args.split}:
        raise V4ValidationError(
            f"Review export split mismatch: expected={args.split!r}, observed={sorted(observed_splits)!r}"
        )
    output_dir = args.workspace / "manual_review" / args.split
    report = export_manual_review(
        groups,
        output_dir / f"manual_candidate_review_{args.split}.csv",
        output_dir / f"manual_group_review_{args.split}.csv",
        sample_size=args.sample_size,
        regression_cases=args.regression_cases,
        seed=args.seed,
    )
    manifest = {
        "schema_version": "papo_listwise_v4_manual_review_export",
        "split": args.split,
        "source_groups": str(args.groups.resolve()),
        "source_groups_sha256": sha256_file(args.groups),
        **report,
        "candidate_csv_sha256": sha256_file(report["candidate_csv"]),
        "group_csv_sha256": sha256_file(report["group_csv"]),
    }
    manifest_path = output_dir / f"manual_review_export_{args.split}.manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({**manifest, "manifest": str(manifest_path.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
