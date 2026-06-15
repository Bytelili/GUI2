from __future__ import annotations

import argparse
import json

from epipeline.runtime_templates import build_runtime_templates


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reviewable ADB hook and success-rule templates.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = build_runtime_templates(args.tasks, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("RUNTIME TEMPLATE BUILD FAILED")
    print("RUNTIME TEMPLATE BUILD PASSED; REVIEW ALL GENERATED TEMPLATES")


if __name__ == "__main__":
    main()
