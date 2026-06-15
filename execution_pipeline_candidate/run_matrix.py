from __future__ import annotations

import argparse
import json

from epipeline.io_utils import read_json
from epipeline.runner import run_entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a resumable isolated personalized-execution matrix.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    selected = set(args.run_id)
    reports = []
    for entry in manifest["runs"]:
        if selected and str(entry["id"]) not in selected:
            continue
        reports.append(run_entry(manifest, entry, limit=args.limit))
    print(json.dumps({"status": "completed", "runs": reports}, ensure_ascii=False, indent=2))
    if any(report["status"] != "completed" for report in reports):
        raise RuntimeError("At least one execution run has retryable task failures")
    print("EXECUTION MATRIX RUN COMPLETED")


if __name__ == "__main__":
    main()
