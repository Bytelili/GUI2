from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_listwise_v4 import V4ValidationError, write_json  # noqa: E402
from papo.proactive_manual_review import apply_manual_review  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and apply PAPO v4 manual review annotations.")
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--candidate-review", type=Path, required=True)
    parser.add_argument("--group-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-log", type=Path, required=True)
    args = parser.parse_args()
    try:
        groups = json.loads(args.groups.read_text(encoding="utf-8", errors="strict"))
        output, report = apply_manual_review(groups, args.candidate_review, args.audit_log, args.group_review)
        write_json(args.output, output)
    except (OSError, ValueError, V4ValidationError) as error:
        print(f"MANUAL REVIEW IMPORT FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
