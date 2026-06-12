from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.config import config_path, load_config  # noqa: E402
from papo.data_protocol import build_formal_protocol  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict train/eval protocol files from official FingerTip splits.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    protocol = config["data"]["protocol"]
    manifest = build_formal_protocol(
        config_path(config, "paths.official_root"),
        config_path(config, "paths.protocol_dir"),
        source_train_split=str(protocol["source_train_split"]),
        proactive_test_split=str(protocol["proactive_test_split"]),
        execution_test_split=str(protocol["execution_test_split"]),
        validation_fraction=float(protocol["validation_fraction"]),
        min_validation_per_user=int(protocol["min_validation_per_user"]),
        protocol_id=str(protocol["protocol_id"]),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("STRICT DATA PROTOCOL BUILD PASSED")


if __name__ == "__main__":
    main()
