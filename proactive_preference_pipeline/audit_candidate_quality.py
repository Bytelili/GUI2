from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from ppipeline.io_utils import read_jsonl, write_json, write_jsonl  # noqa: E402
from ppipeline.quality import audit_candidate_quality, build_quality_review_sample  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run the Proactive candidate quality audit.")
    parser.add_argument(
        "--train-scored",
        default="data/proactive_preference/proactive_train_candidates_scored.jsonl",
    )
    parser.add_argument(
        "--eval-scored",
        default="data/proactive_preference/proactive_eval_candidates_scored.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/proactive_preference/candidate_quality_report.rerun.json",
    )
    parser.add_argument(
        "--flags",
        default="data/proactive_preference/candidate_quality_flags.rerun.jsonl",
    )
    parser.add_argument(
        "--review",
        default="data/proactive_preference/candidate_quality_review_sample.rerun.jsonl",
    )
    parser.add_argument("--expect-model-candidates", action="store_true")
    args = parser.parse_args()
    train_rows = read_jsonl(_resolve(args.train_scored))
    eval_rows = read_jsonl(_resolve(args.eval_scored))
    if not train_rows or not eval_rows:
        raise ValueError("Both scored train and eval candidate files must be non-empty")
    report, flags = audit_candidate_quality(
        train_rows,
        eval_rows,
        model_candidates_expected={
            "train": args.expect_model_candidates,
            "eval": args.expect_model_candidates,
        },
    )
    write_json(_resolve(args.output), report)
    write_jsonl(_resolve(args.flags), flags)
    write_jsonl(_resolve(args.review), build_quality_review_sample(train_rows, eval_rows))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Flagged candidates: {len(flags)}")
    if report["status"] == "failed":
        raise ValueError("Candidate quality hard gate failed")
    print("PROACTIVE CANDIDATE QUALITY AUDIT PASSED")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
