from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.raw_builder import build_from_raw, write_json, write_jsonl  # noqa: E402
from papo.official_data import read_csv_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", default=str(PROJECT_ROOT / "data/raw/fingertip20k"))
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "data/papo_raw"))
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--max_episodes_per_user", type=int, default=0)
    parser.add_argument(
        "--catalog",
        action="append",
        default=[],
        help="Optional official CSV used to select episodes. Repeat to merge splits.",
    )
    parser.add_argument("--include_incomplete", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    selected = None
    if args.catalog:
        selected = set()
        for catalog in args.catalog:
            selected.update(
                (str(row.get("user_id") or ""), str(row.get("time") or ""))
                for row in read_csv_rows(catalog)
            )
    episodes, steps, audit = build_from_raw(
        Path(args.raw_root),
        max_episodes=args.max_episodes,
        require_complete=not args.include_incomplete,
        selected_episodes=selected,
        max_episodes_per_user=args.max_episodes_per_user,
    )
    write_jsonl(out_dir / "papo_episodes.jsonl", episodes)
    write_jsonl(out_dir / "papo_steps.jsonl", steps)
    write_json(out_dir / "papo_raw_audit.json", audit)

    print(f"episodes: {len(episodes)}")
    print(f"steps: {len(steps)}")
    print(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()
