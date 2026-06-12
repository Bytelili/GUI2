from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.config import config_path, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PAPO paths configured in config.yaml.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--create_output_dirs", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    required_dirs = [
        "paths.raw_root",
        "paths.official_root",
        "paths.llamafactory_dir",
    ]
    output_dirs = [
        "paths.work_dir",
        "paths.task_dir",
        "paths.llamafactory_data_dir",
        "paths.checkpoint_root",
        "paths.logging_root",
    ]
    missing: list[str] = []

    for key in required_dirs:
        path = config_path(config, key)
        print(f"{key}: {path}")
        if not path.is_dir():
            missing.append(f"{key}: {path}")

    official_root = config_path(config, "paths.official_root")
    for name in ["total.csv", "train_set.csv", "test_execution.csv", "test_suggestion.csv", "user_profile.csv"]:
        path = official_root / name
        print(f"official_file: {path}")
        if not path.is_file():
            missing.append(f"official_file: {path}")

    model_value = str(config["paths"]["qwen_model_path"])
    model_path = Path(model_value)
    if model_path.is_absolute() or model_value.startswith("."):
        resolved_model = model_path if model_path.is_absolute() else Path(config["_project_root"]) / model_path
        print(f"paths.qwen_model_path: {resolved_model}")
        if not resolved_model.is_dir():
            missing.append(f"paths.qwen_model_path: {resolved_model}")
    else:
        print(f"paths.qwen_model_path: {model_value} (remote model ID)")

    for key in output_dirs:
        path = config_path(config, key)
        if args.create_output_dirs:
            path.mkdir(parents=True, exist_ok=True)
        print(f"{key}: {path}")

    if missing:
        raise FileNotFoundError("Missing configured paths:\n" + "\n".join(missing))
    print("Config path validation passed")


if __name__ == "__main__":
    main()
