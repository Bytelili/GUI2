from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate config-driven PAPO objective artifacts.")
    parser.add_argument("--work_dir", default="data/papo_config_run")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    root = Path(args.work_dir)
    action_values = _read_jsonl(root / "papo_action_values.jsonl")
    listwise = _read_jsonl(root / "papo_listwise_targets.jsonl")
    pairs = _read_jsonl(root / "papo_dpo_pairs.jsonl")
    scored_trees = _read_jsonl(root / "papo_scored_trees.jsonl")

    evidence_values: list[float] = []
    for tree_index, tree in enumerate(scored_trees):
        transform = str(tree.get("metadata", {}).get("personalization_evidence_transform") or "")
        for leaf_index, leaf in enumerate(tree.get("leaves", [])):
            evidence = float(leaf.get("r_pref", 0.0))
            if not math.isfinite(evidence):
                raise ValueError(f"scored_trees[{tree_index}].leaves[{leaf_index}] has non-finite r_pref")
            if transform == "tanh_log_ratio" and not -1.0 <= evidence <= 1.0:
                raise ValueError(f"scored_trees[{tree_index}].leaves[{leaf_index}] has r_pref outside [-1, 1]")
            evidence_values.append(evidence)

    required = {"q_user", "q_task", "q_pref", "q_user_conservative", "a_delta", "coverage", "uncertainty"}
    for index, row in enumerate(action_values):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"action_values[{index}] missing fields: {sorted(missing)}")
        if not 0.0 <= float(row["coverage"]) <= 1.0:
            raise ValueError(f"action_values[{index}] coverage is outside [0, 1]")
        if not all(math.isfinite(float(row[field])) for field in required):
            raise ValueError(f"action_values[{index}] contains a non-finite value")

    for index, row in enumerate(listwise):
        candidates = row.get("candidates", [])
        probabilities = [float(candidate["target_policy_probability"]) for candidate in candidates]
        priors = [float(candidate["base_policy_probability"]) for candidate in candidates]
        if not candidates:
            raise ValueError(f"listwise[{index}] has no candidates")
        if abs(sum(probabilities) - 1.0) > args.tolerance:
            raise ValueError(f"listwise[{index}] target policy does not sum to one")
        if abs(sum(priors) - 1.0) > args.tolerance:
            raise ValueError(f"listwise[{index}] base policy does not sum to one")

    for index, pair in enumerate(pairs):
        if pair.get("positive_action") == pair.get("negative_action"):
            raise ValueError(f"pairs[{index}] contains identical actions")
        if float(pair.get("advantage_gap", 0.0)) <= 0.0:
            raise ValueError(f"pairs[{index}] has a non-positive advantage gap")
        target_probability = float(pair.get("target_preference_probability", 0.0))
        if not 0.5 < target_probability <= 1.0:
            raise ValueError(f"pairs[{index}] has an invalid target preference probability")

    print(f"action values: {len(action_values)}")
    print(f"listwise states: {len(listwise)}")
    print(f"DPO pairs: {len(pairs)}")
    if evidence_values:
        negative = sum(value < 0.0 for value in evidence_values)
        positive = sum(value > 0.0 for value in evidence_values)
        neutral = len(evidence_values) - negative - positive
        print(
            "personalization evidence: "
            f"min={min(evidence_values):.4f}, max={max(evidence_values):.4f}, "
            f"negative={negative}, neutral={neutral}, positive={positive}"
        )
    print("PAPO artifact validation passed")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
