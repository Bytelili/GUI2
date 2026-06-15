from __future__ import annotations

import argparse
import json

from epipeline.io_utils import read_json
from epipeline.scoring import score_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Score isolated personalized-execution results.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--annotations", default="")
    args = parser.parse_args()
    report = score_manifest(read_json(args.manifest), args.annotations or None)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("EXECUTION RESULT SCORING COMPLETED")


if __name__ == "__main__":
    main()
