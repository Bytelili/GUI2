from __future__ import annotations

import argparse
import json
import sys
import tarfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_listwise_v4 import (  # noqa: E402
    PROTOCOL_ID,
    V4ValidationError,
    build_groups,
    dataset_info_v4,
    read_jsonl,
    retrieval_pool_map,
    sha256_file,
    sha256_json,
    utc_timestamp,
    write_json,
)
from papo.proactive_quality_gate_v4 import audit_v4_groups  # noqa: E402


def validate_pool(tasks: list[dict[str, Any]], rows: list[dict[str, Any]], split: str) -> None:
    task_ids = [str(item.get("task_id") or "") for item in tasks]
    pool_ids = [str(item.get("task_id") or "") for item in rows]
    if len(task_ids) != len(set(task_ids)) or len(pool_ids) != len(set(pool_ids)):
        raise V4ValidationError(f"{split} task or retrieval pool contains duplicate task IDs")
    if task_ids != pool_ids:
        raise V4ValidationError(f"{split} retrieval pool does not exactly align with source task order")
    for row in rows:
        if row.get("split") != split or row.get("reference_partition") != "train":
            raise V4ValidationError(f"{split} retrieval policy mismatch: {row.get('task_id')}")
        candidates = row.get("candidates")
        if not isinstance(candidates, dict):
            raise V4ValidationError(f"{split} candidate map is malformed: {row.get('task_id')}")


def has_same_user_candidate(row: dict[str, Any]) -> bool:
    candidates = row.get("candidates") or {}
    return bool(
        candidates.get("same_user_similar_intent")
        or candidates.get("same_user_similar_context_different_intent")
    )


def selection_report(
    split: str,
    tasks: list[dict[str, Any]],
    pools: list[dict[str, Any]],
    selected_pools: list[dict[str, Any]],
    groups: list[dict[str, Any]],
) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    eligibility_counts: Counter[str] = Counter()
    excluded_reasons: Counter[str] = Counter()
    selected_source_counts: Counter[str] = Counter()
    group_sizes: Counter[int] = Counter()
    selected_ids = {str(item.get("task_id") or "") for item in selected_pools}
    excluded_ids: list[str] = []
    for row in pools:
        for source, values in (row.get("candidates") or {}).items():
            source_counts[source] += len(values or [])
            eligibility_counts.update(str(item.get("eligibility") or "unknown") for item in (values or []))
        if str(row.get("task_id") or "") not in selected_ids:
            excluded_ids.append(str(row.get("task_id") or ""))
            for reason, count in (row.get("exclusion_counts") or {}).items():
                excluded_reasons[reason] += int(count)
    for group in groups:
        group_sizes[len(group["candidates"])] += 1
        selected_source_counts.update(str(item.get("source") or "unknown") for item in group["candidates"])
        illegal = [item for item in group["candidates"] if str(item.get("source") or "").startswith("cross_user")]
        if illegal:
            raise V4ValidationError(f"Cross-user candidate entered retrieval-only group: {group.get('task_id')}")
    return {
        "split": split,
        "source_task_count": len(tasks),
        "source_pool_count": len(pools),
        "selected_group_count": len(groups),
        "excluded_task_count": len(excluded_ids),
        "excluded_task_ids": excluded_ids,
        "selection_policy": "all tasks with at least one retained same-user historical candidate",
        "source_candidate_counts": dict(source_counts),
        "source_eligibility_counts": dict(eligibility_counts),
        "selected_group_size_distribution": {str(key): value for key, value in sorted(group_sizes.items())},
        "selected_candidate_source_counts": dict(selected_source_counts),
        "excluded_reason_counts": dict(excluded_reasons),
        "cross_user_candidates_in_group": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-scale history-retrieval-only grouped Listwise-v4 data.")
    parser.add_argument("--train-tasks", type=Path, required=True)
    parser.add_argument("--eval-tasks", type=Path, required=True)
    parser.add_argument("--train-pool", type=Path, required=True)
    parser.add_argument("--eval-pool", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, action="append", default=[])
    parser.add_argument("--timestamp")
    args = parser.parse_args()

    source_manifest_path = args.workspace / "manifests" / "source_task_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8", errors="strict"))
    for split, path in (("train", args.train_tasks), ("eval", args.eval_tasks)):
        if source_manifest.get(split, {}).get("sha256") != sha256_file(path):
            raise SystemExit(f"RETRIEVAL-ONLY BUILD FAILED: {split} task SHA256 differs from source manifest")

    train_tasks, eval_tasks = read_jsonl(args.train_tasks), read_jsonl(args.eval_tasks)
    train_pools, eval_pools = read_jsonl(args.train_pool), read_jsonl(args.eval_pool)
    validate_pool(train_tasks, train_pools, "train")
    validate_pool(eval_tasks, eval_pools, "eval")

    train_pairs = [(task, pool) for task, pool in zip(train_tasks, train_pools) if has_same_user_candidate(pool)]
    eval_pairs = [(task, pool) for task, pool in zip(eval_tasks, eval_pools) if has_same_user_candidate(pool)]
    selected_train_tasks, selected_train_pools = map(list, zip(*train_pairs)) if train_pairs else ([], [])
    selected_eval_tasks, selected_eval_pools = map(list, zip(*eval_pairs)) if eval_pairs else ([], [])
    train_groups = build_groups(
        selected_train_tasks,
        split="train",
        model_candidates=None,
        retrieval_candidates=retrieval_pool_map(selected_train_pools),
        synthetic_smoke=False,
        retrieval_only=True,
    )
    eval_groups = build_groups(
        selected_eval_tasks,
        split="eval",
        model_candidates=None,
        retrieval_candidates=retrieval_pool_map(selected_eval_pools),
        synthetic_smoke=False,
        retrieval_only=True,
    )
    selection = {
        "schema_version": "papo_retrieval_only_selection_v4",
        "train": selection_report("train", train_tasks, train_pools, selected_train_pools, train_groups),
        "eval": selection_report("eval", eval_tasks, eval_pools, selected_eval_pools, eval_groups),
        "claim_boundary": "history-retrieval-only engineering data; no model-generated candidates; not full-v4",
    }
    quality, issues = audit_v4_groups(
        train_groups,
        eval_groups,
        image_roots=args.image_root,
        allow_unavailable_images=True,
        source_manifest=source_manifest,
        min_manual_review_fraction=0.0,
    )
    if quality.get("status") == "failed":
        raise SystemExit(f"RETRIEVAL-ONLY BUILD FAILED: quality gate blocked release: {quality.get('issue_counts')}")

    stamp = args.timestamp or utc_timestamp()
    release_dir = args.workspace / "releases" / "retrieval_only_v4" / stamp
    if release_dir.exists():
        raise SystemExit(f"RETRIEVAL-ONLY BUILD FAILED: refusing to overwrite {release_dir}")
    release_dir.mkdir(parents=True)
    files = {
        "papo_proactive_train_listwise_v4.json": train_groups,
        "papo_proactive_eval_listwise_v4.json": eval_groups,
        "dataset_info_v4.json": dataset_info_v4(),
        "listwise_v4_quality_report.json": quality,
        "retrieval_only_selection_report.json": selection,
        "retrieval_only_quality_issues.json": [asdict(issue) for issue in issues],
    }
    for filename, payload in files.items():
        write_json(release_dir / filename, payload)
    manifest = {
        "schema_version": "papo_listwise_v4_manifest",
        "created_at": stamp,
        "release_kind": "retrieval_only_v4",
        "release_status": "retrieval_only_not_for_formal_training",
        "protocol_id": PROTOCOL_ID,
        "source_task_manifest_sha256": sha256_json(source_manifest),
        "source_tasks": {
            "train_sha256": sha256_file(args.train_tasks),
            "eval_sha256": sha256_file(args.eval_tasks),
            "train_rows": len(train_tasks),
            "eval_rows": len(eval_tasks),
        },
        "retrieval_candidate_provenance": {
            "train_sha256": sha256_file(args.train_pool),
            "eval_sha256": sha256_file(args.eval_pool),
            "train_rows": len(train_pools),
            "eval_rows": len(eval_pools),
        },
        "candidate_provenance": None,
        "group_counts": {"train": len(train_groups), "eval": len(eval_groups)},
        "excluded_task_counts": {
            "train": len(train_tasks) - len(train_groups),
            "eval": len(eval_tasks) - len(eval_groups),
        },
        "dataset_hashes": {
            name: sha256_file(release_dir / name)
            for name in (
                "papo_proactive_train_listwise_v4.json",
                "papo_proactive_eval_listwise_v4.json",
                "dataset_info_v4.json",
            )
        },
        "quality_report_sha256": sha256_file(release_dir / "listwise_v4_quality_report.json"),
        "selection_report_sha256": sha256_file(release_dir / "retrieval_only_selection_report.json"),
        "quality_status": quality.get("status"),
        "formal_full_v4_complete": False,
        "training_claim_allowed": False,
        "claim_boundary": "retrieval-only engineering experiment; not full-v4",
    }
    write_json(release_dir / "listwise_v4_manifest.json", manifest)
    artifact_names = [*files, "listwise_v4_manifest.json"]
    sums = release_dir / "SHA256SUMS.txt"
    sums.write_text(
        "".join(f"{sha256_file(release_dir / name)}  {name}\n" for name in artifact_names), encoding="utf-8"
    )
    archive = release_dir.parent / f"PAPO_Listwise_v4_retrieval_only_{stamp}.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for name in [*artifact_names, "SHA256SUMS.txt"]:
            handle.add(release_dir / name, arcname=name)
    (archive.parent / f"{archive.name}.sha256").write_text(
        f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8"
    )
    print(json.dumps({"release_dir": str(release_dir), "archive": str(archive), "manifest": manifest}, ensure_ascii=False, indent=2))
    print("RETRIEVAL-ONLY LISTWISE-V4 RELEASE BUILT; NOT FULL-V4")


if __name__ == "__main__":
    main()
