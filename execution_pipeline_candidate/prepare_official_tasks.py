from __future__ import annotations

import argparse
import json

from epipeline.official_tasks import prepare_official_execution_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare strict official-test personalized-execution tasks.")
    parser.add_argument("--project-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--target-split",
        default="",
        choices=["", "test_execution.csv", "sampled_test_execution.csv"],
    )
    args = parser.parse_args()
    report = prepare_official_execution_tasks(
        args.project_config,
        args.output,
        limit=args.limit,
        target_split=args.target_split,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("OFFICIAL EXECUTION TASK PREPARATION PASSED")


if __name__ == "__main__":
    main()
