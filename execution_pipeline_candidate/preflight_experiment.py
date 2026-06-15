from __future__ import annotations

import argparse
import json

from epipeline.io_utils import read_json
from epipeline.preflight import preflight_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight an isolated personalized-execution experiment.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--skip-device-connection", action="store_true")
    args = parser.parse_args()
    report = preflight_manifest(
        read_json(args.manifest),
        check_device_connection=not args.skip_device_connection,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("EXECUTION EXPERIMENT PREFLIGHT FAILED")
    print("EXECUTION EXPERIMENT PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
