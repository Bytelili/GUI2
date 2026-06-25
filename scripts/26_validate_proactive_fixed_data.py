from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_fixed_export import (  # noqa: E402
    validate_dpo_rows,
    validate_rerank_rows,
    validate_sft_rows,
    validate_weighted_listwise_rows,
)
from papo.proactive_quality_gate import read_jsonl, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate proactive-fixed exported datasets.")
    parser.add_argument("--sft", required=True)
    parser.add_argument("--dpo", required=True)
    parser.add_argument("--rerank", required=True)
    parser.add_argument("--listwise", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sft_rows = read_jsonl(Path(args.sft))
    dpo_rows = read_jsonl(Path(args.dpo))
    rerank_rows = read_jsonl(Path(args.rerank))
    listwise_rows = read_jsonl(Path(args.listwise))

    report = {
        "status": "passed",
        "sft": validate_sft_rows(sft_rows),
        "dpo": validate_dpo_rows(dpo_rows),
        "rerank": validate_rerank_rows(rerank_rows),
        "weighted_listwise": validate_weighted_listwise_rows(listwise_rows),
    }
    if not all(section.get("passed") for key, section in report.items() if isinstance(section, dict) and "passed" in section):
        report["status"] = "failed"
    write_json(Path(args.out), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
