from __future__ import annotations

import argparse

from tn_dpo_gui.evaluation.eval_gate import evaluate_gate
from tn_dpo_gui.evaluation.eval_offline import evaluate_ranker
from tn_dpo_gui.evaluation.eval_projection import evaluate_projection
from tn_dpo_gui.utils.config import load_config
from tn_dpo_gui.utils.io import write_json

from . import PROJECT_ROOT, apply_main_project_layout, resolve_config_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline evaluation for TN-DPO artifacts.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "eval.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_main_project_layout(
        config,
        {
            "data": {"pairs_path": "pairs_path"},
            "checkpoints": {
                "ranker_path": "ranker_path",
                "gate_path": "gate_path",
            },
            "output": {"report_path": "eval_report_path"},
        },
    )
    config = resolve_config_paths(
        config,
        {
            "data": ["pairs_path"],
            "checkpoints": ["ranker_path", "gate_path"],
            "output": ["report_path"],
        },
    )

    report = {
        "ranker": evaluate_ranker(
            config["data"]["pairs_path"],
            config["checkpoints"]["ranker_path"],
            batch_size=int(config.get("evaluation", {}).get("batch_size", 32)),
        ),
        "gate": evaluate_gate(
            config["data"]["pairs_path"],
            config["checkpoints"]["gate_path"],
            batch_size=int(config.get("evaluation", {}).get("batch_size", 32)),
        ),
        "projection": evaluate_projection(config["data"]["pairs_path"]),
    }
    write_json(config["output"]["report_path"], report)
    print(report)


if __name__ == "__main__":
    main()
