from __future__ import annotations

import argparse
import json

from epipeline.prepare import prepare_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an isolated personalized-execution experiment matrix.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    manifest = prepare_experiment(args.config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("EXECUTION EXPERIMENT PREPARATION PASSED")


if __name__ == "__main__":
    main()
