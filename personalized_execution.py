from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.io import write_jsonl  # noqa: E402
from papo.tasks import build_personalized_execution_tasks  # noqa: E402


def main() -> None:
    official = PROJECT_ROOT / "data/official/fingertip20k"
    parser = argparse.ArgumentParser(description="Build FingerTip-style personalized execution tasks.")
    parser.add_argument("--test", default=str(official / "test_execution.csv"))
    parser.add_argument("--catalog", default=str(official / "total.csv"))
    parser.add_argument("--profiles", default=str(official / "user_profile.csv"))
    parser.add_argument("--raw_root", default=str(PROJECT_ROOT / "data/raw/fingertip20k"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data/papo_tasks/personalized_execution.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include_missing", action="store_true")
    args = parser.parse_args()

    tasks = build_personalized_execution_tasks(
        args.test,
        args.catalog,
        args.profiles,
        args.raw_root,
        limit=args.limit,
        require_complete=not args.include_missing,
    )
    write_jsonl(args.out, tasks)
    print(f"tasks: {len(tasks)}")
    print(f"with_episode_assets: {sum(bool(task['metadata']['episode_path']) for task in tasks)}")
    print(f"with_same_user_reference: {sum(task['input']['same_user_action_reference'] is not None for task in tasks)}")
    print(f"with_cross_user_reference: {sum(task['input']['cross_user_action_reference'] is not None for task in tasks)}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
