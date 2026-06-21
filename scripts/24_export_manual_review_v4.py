from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_manual_review import export_manual_review  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export human-sized PAPO v4 candidate and group review CSV files.")
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--regression-cases", type=Path)
    parser.add_argument("--seed", type=int, default=20260621)
    args = parser.parse_args()
    groups = json.loads(args.groups.read_text(encoding="utf-8", errors="strict"))
    report = export_manual_review(
        groups,
        args.workspace / "manual_review" / "manual_candidate_review.csv",
        args.workspace / "manual_review" / "manual_group_review.csv",
        sample_size=args.sample_size,
        regression_cases=args.regression_cases,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
