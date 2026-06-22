from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_listwise_v4 import (  # noqa: E402
    build_retrieval_candidate_pools,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


def _summary(rows: list[dict]) -> dict:
    candidate_counts: Counter[str] = Counter()
    covered_tasks: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    eligibility_counts: Counter[str] = Counter()
    for row in rows:
        for source, candidates in row["candidates"].items():
            candidate_counts[source] += len(candidates)
            covered_tasks[source] += bool(candidates)
            eligibility_counts.update(str(item.get("eligibility") or "unknown") for item in candidates)
        exclusion_counts.update(row.get("exclusion_counts") or {})
    return {
        "task_count": len(rows),
        "candidate_counts": dict(candidate_counts),
        "covered_tasks": dict(covered_tasks),
        "coverage": {
            source: count / max(len(rows), 1)
            for source, count in covered_tasks.items()
        },
        "exclusion_counts": dict(exclusion_counts),
        "eligibility_counts": dict(eligibility_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build causal same-user and cross-user PAPO v4 retrieval pools.")
    parser.add_argument("--train-tasks", type=Path, required=True)
    parser.add_argument("--eval-tasks", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--max-per-type", type=int, default=2)
    parser.add_argument("--pseudo-negative-similarity", type=float, default=0.55)
    parser.add_argument("--min-same-user-similarity", type=float, default=0.20)
    parser.add_argument("--max-global-text-frequency", type=int)
    args = parser.parse_args()
    train_tasks, eval_tasks = read_jsonl(args.train_tasks), read_jsonl(args.eval_tasks)
    train_rows = build_retrieval_candidate_pools(
        train_tasks,
        train_tasks,
        split="train",
        max_per_type=args.max_per_type,
        pseudo_negative_similarity=args.pseudo_negative_similarity,
        min_same_user_similarity=args.min_same_user_similarity,
        max_global_text_frequency=args.max_global_text_frequency,
    )
    eval_rows = build_retrieval_candidate_pools(
        eval_tasks,
        train_tasks,
        split="eval",
        max_per_type=args.max_per_type,
        pseudo_negative_similarity=args.pseudo_negative_similarity,
        min_same_user_similarity=args.min_same_user_similarity,
        max_global_text_frequency=args.max_global_text_frequency,
    )
    output_dir = args.workspace / "intermediate"
    train_output = output_dir / "retrieval_candidate_pool_train_v4.jsonl"
    eval_output = output_dir / "retrieval_candidate_pool_eval_v4.jsonl"
    write_jsonl(train_output, train_rows)
    write_jsonl(eval_output, eval_rows)
    report = {
        "schema_version": "papo_retrieval_candidate_pool_v4",
        "reference_policy": {
            "train": "strict train tasks strictly earlier than each target",
            "eval": "strict train tasks only, strictly earlier than each eval target",
            "verbatim_prompt_history_copy": "excluded",
            "cross_user_positive_listwise_probability": 0.0,
            "same_user_similar_intent_min_similarity": args.min_same_user_similarity,
            "same_user_context_different_intent_probability": 0.0,
            "global_text_frequency_cap": args.max_global_text_frequency or "max(10, ceil(task_count * 0.005))",
        },
        "inputs": {
            "train_sha256": sha256_file(args.train_tasks),
            "eval_sha256": sha256_file(args.eval_tasks),
        },
        "train": _summary(train_rows),
        "eval": _summary(eval_rows),
        "outputs": {
            "train": str(train_output.resolve()),
            "eval": str(eval_output.resolve()),
            "train_sha256": sha256_file(train_output),
            "eval_sha256": sha256_file(eval_output),
        },
    }
    report_path = args.workspace / "reports" / "retrieval_candidate_pool_v4_report.json"
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
