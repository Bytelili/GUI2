from __future__ import annotations

import argparse
from pathlib import Path

from tn_dpo_gui.evaluation.eval_gate import evaluate_gate
from tn_dpo_gui.evaluation.eval_offline import evaluate_ranker
from tn_dpo_gui.evaluation.eval_projection import evaluate_projection
from tn_dpo_gui.utils.config import export_config, load_config
from tn_dpo_gui.utils.io import write_json
from tn_dpo_gui.utils.provenance import runtime_provenance

from . import PROJECT_ROOT, apply_main_project_layout, override_main_project_root_config, resolve_config_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline evaluation for TN-DPO artifacts.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "eval.yaml"))
    parser.add_argument("--root-config", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    config = override_main_project_root_config(config, args.root_config)
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
    allowed_splits = {str(split).lower() for split in config.get("data", {}).get("splits", ["eval"])}

    report = {
        "ranker": evaluate_ranker(
            config["data"]["pairs_path"],
            config["checkpoints"]["ranker_path"],
            batch_size=int(config.get("evaluation", {}).get("batch_size", 32)),
            allowed_splits=allowed_splits,
        ),
        "gate": evaluate_gate(
            config["data"]["pairs_path"],
            config["checkpoints"]["gate_path"],
            batch_size=int(config.get("evaluation", {}).get("batch_size", 32)),
            allowed_splits=allowed_splits,
        ),
        "projection": evaluate_projection(config["data"]["pairs_path"], allowed_splits=allowed_splits),
    }
    if config.get("_config_path"):
        report["config_path"] = config["_config_path"]
    report["provenance"] = runtime_provenance(PROJECT_ROOT)
    write_json(config["output"]["report_path"], report)
    write_json(Path(config["output"]["report_path"]).with_name("resolved_config.json"), export_config(config))
    print(report)


if __name__ == "__main__":
    main()
