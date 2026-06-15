from __future__ import annotations

import argparse
import json

from epipeline.analysis import analyze_manifest
from epipeline.io_utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired personalized-execution experiment results.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    report = analyze_manifest(read_json(args.manifest), bootstrap_samples=args.bootstrap_samples)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("EXECUTION EXPERIMENT ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
