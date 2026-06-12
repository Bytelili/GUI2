from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.io import write_jsonl  # noqa: E402
from papo.tasks import build_proactive_suggestion_tasks  # noqa: E402


def main() -> None:
    official = PROJECT_ROOT / "data/official/fingertip20k"
    parser = argparse.ArgumentParser(description="Build FingerTip-style proactive suggestion tasks.")
    parser.add_argument("--test", default=str(official / "test_suggestion.csv"))
    parser.add_argument("--catalog", default=str(official / "total.csv"))
    parser.add_argument("--profiles", default=str(official / "user_profile.csv"))
    parser.add_argument("--raw_root", default=str(PROJECT_ROOT / "data/raw/fingertip20k"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data/papo_tasks/proactive_suggestion.jsonl"))
    parser.add_argument("--screenshot_level", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument("--history_limit", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include_missing", action="store_true")
    args = parser.parse_args()

    tasks = build_proactive_suggestion_tasks(
        args.test,
        args.catalog,
        args.profiles,
        args.raw_root,
        screenshot_level=args.screenshot_level,
        history_limit=args.history_limit,
        limit=args.limit,
        require_complete=not args.include_missing,
    )
    write_jsonl(args.out, tasks)
    print(f"tasks: {len(tasks)}")
    print(f"with_episode_assets: {sum(bool(task['metadata']['episode_path']) for task in tasks)}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
