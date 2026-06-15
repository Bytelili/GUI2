from __future__ import annotations

import argparse
import json

from epipeline.audit import audit_manifest
from epipeline.io_utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit an isolated personalized-execution experiment.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--require-paper-eligible", action="store_true")
    args = parser.parse_args()
    report = audit_manifest(read_json(args.manifest), require_paper_eligible=args.require_paper_eligible)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("EXECUTION EXPERIMENT AUDIT FAILED")
    print("EXECUTION EXPERIMENT AUDIT PASSED")


if __name__ == "__main__":
    main()
