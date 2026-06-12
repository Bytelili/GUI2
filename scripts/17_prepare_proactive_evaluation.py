from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.config import config_path, load_config  # noqa: E402
from papo.proactive_evaluation import prepare_proactive_evaluation_tasks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare audited Proactive official-test tasks.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--screenshot-level", type=int, choices=[0, 1, 2, 3], default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.out_dir) if args.out_dir else config_path(config, "paths.task_dir")
    report = prepare_proactive_evaluation_tasks(
        official_root=config_path(config, "paths.official_root"),
        protocol_dir=config_path(config, "paths.protocol_dir"),
        raw_root=config_path(config, "paths.raw_root"),
        output_dir=output_dir,
        screenshot_level=(
            int(args.screenshot_level)
            if args.screenshot_level is not None
            else int(config["data"]["suggestion_screenshot_level"])
        ),
        history_limit=int(config["data"]["suggestion_history_limit"]),
        require_complete=bool(config["data"]["require_complete"]),
        test_split=str(config["data"]["protocol"]["proactive_test_split"]),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("PROACTIVE EVALUATION TASK PREPARATION PASSED")


if __name__ == "__main__":
    main()
