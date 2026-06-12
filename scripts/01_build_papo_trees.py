from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.io import read_jsonl, write_jsonl  # noqa: E402
from papo.official_data import read_csv_rows  # noqa: E402
from papo.tree_builder import build_depth1_tree, build_offline_counterfactual_tree, build_tree_context  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default=str(PROJECT_ROOT / "data/papo_raw/papo_steps.jsonl"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data/papo_raw/papo_trees.jsonl"))
    parser.add_argument("--mode", choices=["depth1", "offline"], default="offline")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--history_top_k", type=int, default=5)
    parser.add_argument("--same_user_k", type=int, default=2)
    parser.add_argument("--cross_user_k", type=int, default=2)
    parser.add_argument("--max_candidates", type=int, default=6)
    parser.add_argument("--user_threshold", type=float, default=0.6)
    parser.add_argument("--root_catalog", default="", help="Optional official CSV selecting root episodes.")
    parser.add_argument("--root_only", action="store_true", help="Build only the first step of each selected episode.")
    args = parser.parse_args()

    steps = read_jsonl(args.steps)
    selected = steps
    if args.root_catalog:
        selected_episodes = {
            f"{row.get('user_id', '')}__{row.get('time', '')}"
            for row in read_csv_rows(args.root_catalog)
        }
        selected = [step for step in selected if str(step.get("episode_id") or "") in selected_episodes]
    if args.root_only:
        selected = [step for step in selected if int(step.get("step_index", 0) or 0) == 0]
    if args.limit > 0:
        selected = selected[: args.limit]
    context = build_tree_context(steps) if args.mode == "offline" else None
    trees = []
    for step in selected:
        if args.mode == "depth1":
            tree = build_depth1_tree(
                step,
                steps,
                history_top_k=args.history_top_k,
                same_user_k=args.same_user_k,
                cross_user_k=args.cross_user_k,
                max_candidates=args.max_candidates,
            ).to_dict()
        else:
            tree = build_offline_counterfactual_tree(
                step,
                steps,
                max_depth=args.max_depth,
                same_user_k=args.same_user_k,
                cross_user_k=args.cross_user_k,
                max_candidates=args.max_candidates,
                user_threshold=args.user_threshold,
                context=context,
            )
        trees.append(tree)
    write_jsonl(args.out, trees)
    print(f"trees: {len(trees)}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
