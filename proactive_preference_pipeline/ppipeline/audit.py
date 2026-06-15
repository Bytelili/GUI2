from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any


def audit_preference_sets(
    train_sets: list[dict[str, Any]],
    eval_sets: list[dict[str, Any]],
    *,
    train_reference_ids: set[str],
) -> dict[str, Any]:
    train_ids = {_episode_id(row) for row in train_sets}
    eval_ids = {_episode_id(row) for row in eval_sets}
    if "" in train_ids or "" in eval_ids:
        raise ValueError("Preference sets contain an empty target episode ID")
    if train_ids & eval_ids:
        raise ValueError(f"Preference train/eval target overlap: {len(train_ids & eval_ids)}")

    reports = {
        "train": _audit_partition(train_sets, "train", train_reference_ids),
        "eval": _audit_partition(eval_sets, "eval", train_reference_ids),
    }
    return {
        "status": "passed",
        "train_targets": len(train_ids),
        "eval_targets": len(eval_ids),
        "train_eval_target_overlap": 0,
        "partitions": reports,
    }


def _audit_partition(
    rows: list[dict[str, Any]],
    partition: str,
    train_reference_ids: set[str],
) -> dict[str, Any]:
    candidate_counts: list[int] = []
    pair_counts: list[int] = []
    source_counts: Counter[str] = Counter()
    top_negative_sources: Counter[str] = Counter()
    reward_totals: list[float] = []
    oracle_margins: list[float] = []
    oracle_top1 = 0
    oracle_user_scores: list[float] = []
    cross_user_scores: list[float] = []
    source_leaks: set[str] = set()
    temporal_violations = 0
    missing_pairs = 0
    for row in rows:
        if row.get("partition") != partition:
            raise ValueError(f"Preference row partition mismatch: expected={partition}")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        target_id = str(metadata.get("papo_episode_id") or "")
        target_time = _episode_time(target_id)
        candidate_ids: set[str] = set()
        for candidate in row.get("candidates", []):
            candidate_id = str(candidate.get("candidate_id") or "")
            if not candidate_id or candidate_id in candidate_ids:
                raise ValueError(f"Duplicate or empty candidate ID in {row.get('task_id')}")
            candidate_ids.add(candidate_id)
            source = str(candidate.get("source") or "")
            source_counts[source] += 1
            reward_totals.append(float(candidate.get("reward", {}).get("total", 0.0) or 0.0))
            if source == "oracle_target":
                oracle_user_scores.append(float(candidate.get("reward", {}).get("user_preference", 0.0) or 0.0))
            if source == "cross_user_hard":
                cross_user_scores.append(float(candidate.get("reward", {}).get("user_preference", 0.0) or 0.0))
            source_id = str(candidate.get("source_episode_id") or "")
            if source not in {"oracle_target", "sft_sample"} and source_id not in train_reference_ids:
                source_leaks.add(source_id)
            source_time = _episode_time(source_id)
            if source not in {"oracle_target", "sft_sample"} and source_time and target_time:
                temporal_violations += source_time >= target_time
        candidate_counts.append(len(candidate_ids))
        pair_counts.append(len(row.get("pairs", [])))
        missing_pairs += not bool(row.get("pairs"))
        ranked = sorted(
            row.get("candidates", []),
            key=lambda item: float(item.get("reward", {}).get("total", 0.0) or 0.0),
            reverse=True,
        )
        oracle = next((item for item in ranked if item.get("source") == "oracle_target"), None)
        negatives = [item for item in ranked if item.get("source") != "oracle_target"]
        if oracle is None:
            raise ValueError(f"Missing oracle target in {row.get('task_id')}")
        oracle_top1 += bool(ranked and ranked[0].get("source") == "oracle_target")
        if negatives:
            top_negative_sources[str(negatives[0].get("source") or "")] += 1
            oracle_margins.append(
                float(oracle.get("reward", {}).get("total", 0.0) or 0.0)
                - float(negatives[0].get("reward", {}).get("total", 0.0) or 0.0)
            )
    if source_leaks or temporal_violations:
        raise ValueError(
            f"{partition} candidate provenance failed: "
            f"outside_train_references={len(source_leaks)}, temporal_violations={temporal_violations}"
        )
    return {
        "targets": len(rows),
        "candidates": sum(candidate_counts),
        "pairs": sum(pair_counts),
        "targets_without_pairs": missing_pairs,
        "mean_candidates_per_target": mean(candidate_counts) if candidate_counts else 0.0,
        "mean_pairs_per_target": mean(pair_counts) if pair_counts else 0.0,
        "candidate_sources": dict(sorted(source_counts.items())),
        "top_hard_negative_sources": dict(sorted(top_negative_sources.items())),
        "reward_total_mean": mean(reward_totals) if reward_totals else 0.0,
        "oracle_top1_rate": oracle_top1 / len(rows) if rows else 0.0,
        "oracle_reward_margin_mean": mean(oracle_margins) if oracle_margins else 0.0,
        "oracle_user_preference_mean": mean(oracle_user_scores) if oracle_user_scores else 0.0,
        "cross_user_preference_mean": mean(cross_user_scores) if cross_user_scores else 0.0,
        "outside_train_reference_episodes": 0,
        "temporal_violations": 0,
    }


def _episode_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("papo_episode_id") or "")


def _episode_time(episode_id: str) -> str:
    return episode_id.split("__", 1)[1] if "__" in episode_id else ""
