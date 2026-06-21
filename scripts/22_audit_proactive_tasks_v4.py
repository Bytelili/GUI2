from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_listwise_v4 import audit_source_tasks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local strict Proactive task JSONL files without modifying them.")
    parser.add_argument("--train-tasks", type=Path, required=True)
    parser.add_argument("--eval-tasks", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    try:
        manifest = audit_source_tasks(args.train_tasks, args.eval_tasks, args.workspace, args.image_root)
    except Exception as error:
        print(f"SOURCE TASK AUDIT FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"SOURCE TASK AUDIT STATUS: {manifest['status'].upper()}")
    if manifest["hard_error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
