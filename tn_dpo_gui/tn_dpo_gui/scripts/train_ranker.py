from __future__ import annotations

import argparse

from tn_dpo_gui.training.train_ranker import train_ranker
from tn_dpo_gui.utils.config import load_config

from . import PROJECT_ROOT, resolve_config_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the TN-DPO action ranker.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "train_ranker.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    config = resolve_config_paths(config, {"data": ["pairs_path"], "output": ["dir"]})
    metrics = train_ranker(config)
    print(metrics)


if __name__ == "__main__":
    main()
