from __future__ import annotations

import argparse

from pathlib import Path

from tn_dpo_gui.data.dataset import load_step_examples, load_trajectory_records
from tn_dpo_gui.pair_builder.pair_builder import TNDPOPairBuilder
from tn_dpo_gui.retrieval.user_history_index import UserHistoryIndex
from tn_dpo_gui.utils.config import load_config
from tn_dpo_gui.utils.io import write_json, write_jsonl

from . import PROJECT_ROOT, apply_main_project_layout, resolve_config_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Task-Nullspace DPO preference pairs.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "build_pairs.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_main_project_layout(
        config,
        {
            "input": {
                "steps_path": "steps_path",
                "trajectories_path": "trajectories_path",
                "history_index_path": "user_index_path",
            },
            "output": {
                "pairs_path": "pairs_path",
                "summary_path": "pair_summary_path",
            },
        },
    )
    config = resolve_config_paths(
        config,
        {
            "input": ["steps_path", "trajectories_path", "history_index_path"],
            "output": ["pairs_path", "summary_path"],
        },
    )
    examples = load_step_examples(config["input"]["steps_path"])
    trajectories = load_trajectory_records(config["input"]["trajectories_path"])

    history_index_path = config["input"].get("history_index_path")
    if history_index_path and Path(history_index_path).exists():
        history_index = UserHistoryIndex.load(history_index_path)
    else:
        history_index = UserHistoryIndex.build(trajectories)
        if history_index_path:
            history_index.save(history_index_path)

    builder_config = dict(config.get("pair_builder", {}))
    builder_config["encoder"] = config.get("encoder", {})
    pairs = TNDPOPairBuilder(trajectories, history_index=history_index, config=builder_config).build_pairs(examples)
    write_jsonl(config["output"]["pairs_path"], [pair.to_dict() for pair in pairs])

    summary = {
        "num_examples": len(examples),
        "num_pairs": len(pairs),
        "avg_weight": sum(pair.weight for pair in pairs) / max(len(pairs), 1),
        "avg_gate_capacity": sum(pair.gate_capacity for pair in pairs) / max(len(pairs), 1),
        "pair_counts_by_split": {
            split: sum(1 for pair in pairs if pair.split == split) for split in sorted({pair.split for pair in pairs})
        },
    }
    if config.get("_main_project_layout"):
        summary["main_project_layout"] = config["_main_project_layout"]
    write_json(config["output"]["summary_path"], summary)
    print(summary)


if __name__ == "__main__":
    main()
