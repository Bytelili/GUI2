from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.io import read_jsonl, write_jsonl  # noqa: E402
from papo.propagation import add_advantages, candidate_rewards  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", default=str(PROJECT_ROOT / "data/papo_raw/papo_trees.jsonl"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data/papo_raw/papo_advantages.jsonl"))
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    trees = read_jsonl(args.trees)
    reward_rows = []
    for tree in trees:
        reward_rows.extend(candidate_rewards(tree))
    advantage_rows = add_advantages(reward_rows, alpha=args.alpha)
    write_jsonl(args.out, advantage_rows)
    print(f"advantage rows: {len(advantage_rows)}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
