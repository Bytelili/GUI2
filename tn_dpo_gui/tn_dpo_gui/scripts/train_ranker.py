from __future__ import annotations

import argparse

from tn_dpo_gui.training.train_ranker import train_ranker
from tn_dpo_gui.utils.config import load_config

from . import PROJECT_ROOT, apply_main_project_layout, override_main_project_root_config, resolve_config_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the TN-DPO action ranker.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "train_ranker.yaml"))
    parser.add_argument("--root-config", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    config = override_main_project_root_config(config, args.root_config)
    config = apply_main_project_layout(
        config,
        {
            "data": {"pairs_path": "pairs_path"},
            "output": {"dir": "ranker_dir"},
        },
    )
    if config.get("_main_project_layout") and not config.get("training", {}).get("base_model_path"):
        config.setdefault("training", {})["base_model_path"] = config["_main_project_layout"]["model_name_or_path"]
    config = resolve_config_paths(config, {"data": ["pairs_path"], "output": ["dir"]})
    metrics = train_ranker(config)
    print(metrics)


if __name__ == "__main__":
    main()
