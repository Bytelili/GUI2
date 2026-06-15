from __future__ import annotations

import argparse
import json

from epipeline.official_reference import audit_official_reference


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the local project against the FingerTip-20K official source.")
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--project-official-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    report = audit_official_reference(
        args.reference_root,
        args.project_official_root,
        args.output,
        source_only=args.source_only,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("OFFICIAL FINGERTIP REFERENCE AUDIT FAILED")
    print("OFFICIAL FINGERTIP REFERENCE AUDIT PASSED")


if __name__ == "__main__":
    main()
