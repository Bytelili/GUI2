from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.io import read_jsonl, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", default=str(PROJECT_ROOT / "data/papo_raw/papo_trees.jsonl"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data/papo_raw/papo_tree_summary.json"))
    args = parser.parse_args()

    trees = read_jsonl(args.trees)
    node_counts = []
    leaf_counts = []
    user_reward_leaves = 0
    task_reward_leaves = 0
    source_counts: Counter[str] = Counter()
    depth_counts: Counter[int] = Counter()
    candidate_counts = []
    user_scores = []
    target_scores = []
    order_scores = []
    habit_scores = []
    avoid_scores = []

    for tree in trees:
        nodes = tree.get("nodes", [])
        leaves = tree.get("leaves", [])
        node_counts.append(len(nodes))
        leaf_counts.append(len(leaves))
        for node in nodes:
            depth_counts[int(node.get("depth", 0) or 0)] += 1
            candidates = node.get("candidates", [])
            candidate_counts.append(len(candidates))
            for cand in candidates:
                for src in str(cand.get("source", "")).split("+"):
                    if src:
                        source_counts[src] += 1
        for leaf in leaves:
            if float(leaf.get("r_user", 0.0) or 0.0) > 0:
                user_reward_leaves += 1
            if float(leaf.get("r_task", 0.0) or 0.0) > 0:
                task_reward_leaves += 1
            if "user_score" in leaf:
                user_scores.append(float(leaf.get("user_score", 0.0) or 0.0))
            comp = leaf.get("user_score_components")
            if isinstance(comp, dict):
                target_scores.append(float(comp.get("target", 0.0) or 0.0))
                order_scores.append(float(comp.get("order", 0.0) or 0.0))
                habit_scores.append(float(comp.get("habit", 0.0) or 0.0))
                avoid_scores.append(float(comp.get("avoid", 0.0) or 0.0))

    def avg(values: list[float | int]) -> float:
        return sum(float(v) for v in values) / max(len(values), 1)

    summary: dict[str, Any] = {
        "trees_path": str(Path(args.trees).resolve()),
        "num_trees": len(trees),
        "avg_nodes_per_tree": avg(node_counts),
        "avg_leaves_per_tree": avg(leaf_counts),
        "avg_candidates_per_node": avg(candidate_counts),
        "num_leaves": sum(leaf_counts),
        "task_reward_leaf_rate": task_reward_leaves / max(sum(leaf_counts), 1),
        "user_reward_leaf_rate": user_reward_leaves / max(sum(leaf_counts), 1),
        "avg_user_score": avg(user_scores),
        "avg_target_score": avg(target_scores),
        "avg_order_score": avg(order_scores),
        "avg_habit_score": avg(habit_scores),
        "avg_avoid_score": avg(avoid_scores),
        "source_counts": source_counts.most_common(),
        "depth_counts": sorted(depth_counts.items()),
        "warnings": [],
    }
    if summary["avg_leaves_per_tree"] < 2:
        summary["warnings"].append("Trees are narrow; add more same/cross-user candidates or loosen matching.")
    if summary["user_reward_leaf_rate"] < 0.05:
        summary["warnings"].append("Few leaves receive user reward; verifier may be too strict or candidates too weak.")

    write_json(args.out, summary)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
