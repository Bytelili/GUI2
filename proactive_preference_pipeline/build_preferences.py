from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from ppipeline.audit import audit_preference_sets  # noqa: E402
from ppipeline.candidates import build_candidate_sets, model_candidate_map  # noqa: E402
from ppipeline.export import export_preference_datasets, preference_dataset_info  # noqa: E402
from ppipeline.io_utils import read_jsonl, sha256_file, write_json, write_jsonl  # noqa: E402
from ppipeline.quality import (  # noqa: E402
    QualityThresholds,
    audit_candidate_quality,
    build_quality_review_sample,
    drop_invalid_oracle_targets,
    invalid_oracle_targets,
)
from ppipeline.rewards import RewardWeights, score_candidate_sets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build audited Proactive Listwise and PAPO-DPO datasets.")
    parser.add_argument("--train-tasks", default="data/papo_tasks/proactive_train_config.jsonl")
    parser.add_argument("--eval-tasks", default="data/papo_tasks/proactive_eval_config.jsonl")
    parser.add_argument("--protocol-history", default="data/papo_protocol/proactive_history.csv")
    parser.add_argument("--train-model-candidates", default="")
    parser.add_argument("--eval-model-candidates", default="")
    parser.add_argument("--work-dir", default="data/proactive_preference")
    parser.add_argument("--dataset-dir", default="LLaMA-Factory/data/papo")
    parser.add_argument("--raw-root", default="/home/dumike/zyy/GUI/data/raw/fingertip20k")
    parser.add_argument("--asset-prefix", default="RawDataset")
    parser.add_argument("--max-same-user", type=int, default=2)
    parser.add_argument("--max-cross-user", type=int, default=3)
    parser.add_argument("--max-model", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--pair-margin", type=float, default=0.05)
    parser.add_argument("--max-pairs-per-task", type=int, default=2)
    parser.add_argument("--task-weight", type=float, default=0.55)
    parser.add_argument("--user-weight", type=float, default=0.20)
    parser.add_argument("--context-weight", type=float, default=0.15)
    parser.add_argument("--specificity-weight", type=float, default=0.10)
    parser.add_argument("--pseudo-negative-threshold", type=float, default=0.92)
    parser.add_argument("--easy-negative-task-threshold", type=float, default=0.20)
    parser.add_argument("--easy-negative-user-threshold", type=float, default=0.20)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    parser.add_argument("--max-invalid-negative-rate", type=float, default=0.01)
    parser.add_argument("--max-tasks-without-usable-negative-rate", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    train_path = _resolve(args.train_tasks)
    eval_path = _resolve(args.eval_tasks)
    protocol_history_path = _resolve(args.protocol_history)
    work_dir = _resolve(args.work_dir)
    dataset_dir = _resolve(args.dataset_dir)
    train_tasks = read_jsonl(train_path)
    eval_tasks = read_jsonl(eval_path)
    if args.limit > 0:
        train_tasks = train_tasks[: args.limit]
        eval_tasks = eval_tasks[: args.limit]
    if not train_tasks or not eval_tasks:
        raise ValueError("Both strict Proactive train and eval task files must be non-empty")

    train_model = _load_model_candidates(args.train_model_candidates)
    eval_model = _load_model_candidates(args.eval_model_candidates)
    build_args = {
        "reference_tasks": train_tasks,
        "max_same_user": args.max_same_user,
        "max_cross_user": args.max_cross_user,
        "max_model": args.max_model,
    }
    train_sets = build_candidate_sets(
        train_tasks,
        partition="train",
        model_candidates=train_model,
        **build_args,
    )
    eval_sets = build_candidate_sets(
        eval_tasks,
        partition="eval",
        model_candidates=eval_model,
        **build_args,
    )
    weights = RewardWeights(
        task=args.task_weight,
        user=args.user_weight,
        context=args.context_weight,
        specificity=args.specificity_weight,
    )
    score_args = {
        "weights": weights,
        "temperature": args.temperature,
        "pair_margin": args.pair_margin,
        "max_pairs_per_task": args.max_pairs_per_task,
    }
    train_scored = score_candidate_sets(train_sets, **score_args)
    eval_scored = score_candidate_sets(eval_sets, **score_args)
    quality_thresholds = QualityThresholds(
        pseudo_negative_task_match=args.pseudo_negative_threshold,
        easy_negative_task_match=args.easy_negative_task_threshold,
        easy_negative_same_user_similarity=args.easy_negative_user_threshold,
        near_duplicate_similarity=args.near_duplicate_threshold,
        max_invalid_negative_rate=args.max_invalid_negative_rate,
        max_tasks_without_usable_negative_rate=args.max_tasks_without_usable_negative_rate,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    train_scored, excluded_train_oracles = drop_invalid_oracle_targets(
        train_scored,
        thresholds=quality_thresholds,
    )
    eval_invalid_oracles = invalid_oracle_targets(eval_scored, thresholds=quality_thresholds)
    write_jsonl(work_dir / "candidate_quality_excluded_targets.jsonl", excluded_train_oracles)
    write_jsonl(work_dir / "candidate_quality_eval_invalid_oracles.jsonl", eval_invalid_oracles)
    quality_audit, quality_flags = audit_candidate_quality(
        train_scored,
        eval_scored,
        thresholds=quality_thresholds,
        model_candidates_expected={
            "train": bool(train_model),
            "eval": bool(eval_model),
        },
    )
    quality_audit["excluded_train_invalid_oracle_targets"] = {
        "count": len(excluded_train_oracles),
        "policy": "excluded_from_preference_training_only",
        "path": str(work_dir / "candidate_quality_excluded_targets.jsonl"),
    }
    quality_audit["eval_invalid_oracle_targets"] = {
        "count": len(eval_invalid_oracles),
        "policy": "hard_fail_if_nonzero",
        "path": str(work_dir / "candidate_quality_eval_invalid_oracles.jsonl"),
    }
    if eval_invalid_oracles:
        quality_audit["hard_failures"].append(
            f"eval: invalid_oracle_targets={len(eval_invalid_oracles)}"
        )
        quality_audit["status"] = "failed"
    write_json(work_dir / "candidate_quality_report.json", quality_audit)
    write_jsonl(work_dir / "candidate_quality_flags.jsonl", quality_flags)
    write_jsonl(
        work_dir / "candidate_quality_review_sample.jsonl",
        build_quality_review_sample(train_scored, eval_scored),
    )
    if quality_audit["status"] == "failed":
        print(json.dumps(quality_audit, ensure_ascii=False, indent=2))
        raise ValueError("Candidate quality hard gate failed; inspect candidate_quality_report.json")
    train_reference_ids = _read_protocol_episode_ids(protocol_history_path)
    audit = audit_preference_sets(
        train_scored,
        eval_scored,
        train_reference_ids=train_reference_ids,
    )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(work_dir / "proactive_train_candidates_scored.jsonl", train_scored)
    write_jsonl(work_dir / "proactive_eval_candidates_scored.jsonl", eval_scored)
    train_listwise, train_dpo = export_preference_datasets(
        train_scored,
        raw_root=args.raw_root,
        asset_prefix=args.asset_prefix,
    )
    eval_listwise, eval_dpo = export_preference_datasets(
        eval_scored,
        raw_root=args.raw_root,
        asset_prefix=args.asset_prefix,
    )
    datasets = {
        "papo_proactive_train_listwise.json": train_listwise,
        "papo_proactive_eval_listwise.json": eval_listwise,
        "papo_proactive_train_dpo.json": train_dpo,
        "papo_proactive_eval_dpo.json": eval_dpo,
    }
    for filename, rows in datasets.items():
        write_json(dataset_dir / filename, rows)
    _merge_dataset_info(dataset_dir)

    manifest = {
        **audit,
        "method": "proactive_personalized_preference_v1",
        "candidate_quality": quality_audit,
        "limitations": {
            "abstention_training": "not_enabled_without_reliable negative-trigger labels",
            "reward_model": "deterministic decomposed proxy; report each component and ablate weights",
        },
        "inputs": {
            "train_tasks": _file_record(train_path),
            "eval_tasks": _file_record(eval_path),
            "protocol_history": _file_record(protocol_history_path),
            "train_model_candidates": _optional_file_record(args.train_model_candidates),
            "eval_model_candidates": _optional_file_record(args.eval_model_candidates),
        },
        "reward_weights": {
            "task": args.task_weight,
            "user": args.user_weight,
            "context": args.context_weight,
            "specificity": args.specificity_weight,
        },
        "temperature": args.temperature,
        "pair_margin": args.pair_margin,
        "datasets": {
            filename.removesuffix(".json"): {
                "path": str(dataset_dir / filename),
                "rows": len(rows),
                "sha256": sha256_file(dataset_dir / filename),
            }
            for filename, rows in datasets.items()
        },
    }
    write_json(work_dir / "preference_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("PROACTIVE PREFERENCE DATA BUILD PASSED")


def _merge_dataset_info(dataset_dir: Path) -> None:
    path = dataset_dir / "dataset_info.json"
    info = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    info.update(preference_dataset_info())
    write_json(path, info)


def _load_model_candidates(value: str) -> dict[str, list[str]]:
    if not value:
        return {}
    return model_candidate_map(read_jsonl(_resolve(value)))


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _file_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _optional_file_record(value: str) -> dict[str, str] | None:
    return _file_record(_resolve(value)) if value else None


def _read_protocol_episode_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return {
            f"{row.get('user_id', '')}__{row.get('time', '')}"
            for row in csv.DictReader(file)
        }


if __name__ == "__main__":
    main()
