from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .proactive_listwise_v4 import V4ValidationError, normalize_text, sha256_json, write_jsonl


CANDIDATE_FIELDS = [
    "split",
    "task_id",
    "group_id",
    "candidate_id",
    "candidate_source",
    "candidate_text",
    "oracle_text",
    "prompt_text",
    "image_paths_json",
    "candidate_eligibility",
    "source_task_id",
    "source_user_id",
    "source_time",
    "retrieval_semantic_similarity",
    "retrieval_scenario_match",
    "retrieval_hour_similarity",
    "reward_task",
    "reward_user",
    "reward_context",
    "reward_specificity",
    "reward_total",
    "current_rank",
    "current_target_probability",
    "decision",
    "corrected_rank",
    "corrected_probability",
    "reason",
    "reviewer",
    "reviewed_at",
]
GROUP_FIELDS = [
    "split",
    "task_id",
    "group_id",
    "user_id",
    "intent_class",
    "candidate_count",
    "oracle_margin",
    "priority_source",
    "prompt_text",
    "image_paths_json",
    "history_recurrence_exact",
    "history_recurrence_substring",
    "decision",
    "reason",
    "reviewer",
    "reviewed_at",
]
DECISIONS = {
    "keep",
    "drop_unrelated",
    "drop_popular_bias",
    "drop_history_copy",
    "drop_cross_user",
    "regenerate",
    "manual_replace",
}
DROP_DECISIONS = {
    "drop_unrelated",
    "drop_popular_bias",
    "drop_history_copy",
    "drop_cross_user",
}


def _regression_ids(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return {str(row.get("task_id") or "") for row in csv.DictReader(handle)}


def _sample_groups(
    groups: list[dict[str, Any]],
    sample_size: int,
    regression_ids: set[str],
    seed: int,
) -> list[dict[str, Any]]:
    if sample_size <= 0 or sample_size >= len(groups):
        return groups
    selected = [group for group in groups if str(group.get("task_id") or "") in regression_ids]
    selected_ids = {str(group.get("group_id") or "") for group in selected}
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        if str(group.get("group_id") or "") in selected_ids:
            continue
        metadata = group.get("metadata") if isinstance(group.get("metadata"), dict) else {}
        buckets[(str(metadata.get("partition")), str(metadata.get("user_id")), str(metadata.get("intent_class")))].append(group)
    rng = random.Random(seed)
    keys = sorted(buckets)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    while len(selected) < sample_size and keys:
        next_keys = []
        for key in keys:
            if buckets[key] and len(selected) < sample_size:
                selected.append(buckets[key].pop())
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    return selected[:sample_size]


def export_manual_review(
    groups: list[dict[str, Any]],
    candidate_csv: str | Path,
    group_csv: str | Path,
    *,
    sample_size: int = 0,
    regression_cases: str | Path | None = None,
    seed: int = 20260621,
) -> dict[str, Any]:
    regression_ids = _regression_ids(regression_cases)
    sampled = _sample_groups(groups, sample_size, regression_ids, seed)
    candidate_path, group_path = Path(candidate_csv), Path(group_csv)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    group_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for group in sampled:
        metadata = group.get("metadata") if isinstance(group.get("metadata"), dict) else {}
        split = str(metadata.get("partition") or "")
        candidates = group["candidates"]
        oracle = candidates[int(group["oracle_index"])]
        prompt_text = "\n".join(
            str(message.get("content") or "")
            for message in group.get("messages", [])
            if isinstance(message, dict)
        )
        image_paths_json = json.dumps(group.get("images") or [], ensure_ascii=False)
        recurrence = metadata.get("target_history_recurrence") or {}
        probabilities = [float(value) for value in group["target_distribution"]]
        margin = probabilities[int(group["oracle_index"])] - max(
            value for index, value in enumerate(probabilities) if index != int(group["oracle_index"])
        )
        priority = "regression_case" if str(group.get("task_id")) in regression_ids else "stratified_random"
        group_rows.append(
            {
                "split": split,
                "task_id": group["task_id"],
                "group_id": group["group_id"],
                "user_id": metadata.get("user_id", ""),
                "intent_class": metadata.get("intent_class", ""),
                "candidate_count": len(candidates),
                "oracle_margin": margin,
                "priority_source": priority,
                "prompt_text": prompt_text,
                "image_paths_json": image_paths_json,
                "history_recurrence_exact": recurrence.get("normalized_exact", ""),
                "history_recurrence_substring": recurrence.get("substring_overlap", ""),
                "decision": "",
                "reason": "",
                "reviewer": "",
                "reviewed_at": "",
            }
        )
        for candidate in candidates:
            reward = candidate.get("reward") if isinstance(candidate.get("reward"), dict) else {}
            candidate_metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
            retrieval = candidate_metadata.get("retrieval") if isinstance(candidate_metadata.get("retrieval"), dict) else {}
            candidate_rows.append(
                {
                    "split": split,
                    "task_id": group["task_id"],
                    "group_id": group["group_id"],
                    "candidate_id": candidate["candidate_id"],
                    "candidate_source": candidate.get("source", ""),
                    "candidate_text": candidate.get("text", ""),
                    "oracle_text": oracle.get("text", ""),
                    "prompt_text": prompt_text,
                    "image_paths_json": image_paths_json,
                    "candidate_eligibility": candidate_metadata.get("eligibility", ""),
                    "source_task_id": candidate_metadata.get("source_task_id", ""),
                    "source_user_id": candidate_metadata.get("source_user_id", ""),
                    "source_time": candidate_metadata.get("source_time", ""),
                    "retrieval_semantic_similarity": retrieval.get("semantic_similarity", ""),
                    "retrieval_scenario_match": retrieval.get("scenario_match", ""),
                    "retrieval_hour_similarity": retrieval.get("hour_similarity", ""),
                    "reward_task": reward.get("R_task", ""),
                    "reward_user": reward.get("R_user", ""),
                    "reward_context": reward.get("R_context", ""),
                    "reward_specificity": reward.get("R_specificity", ""),
                    "reward_total": reward.get("total", ""),
                    "current_rank": candidate_metadata.get("rank", ""),
                    "current_target_probability": candidate_metadata.get("target_probability", ""),
                    "decision": "",
                    "corrected_rank": "",
                    "corrected_probability": "",
                    "reason": "",
                    "reviewer": "",
                    "reviewed_at": "",
                }
            )
    with candidate_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerows(candidate_rows)
    with group_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GROUP_FIELDS)
        writer.writeheader()
        writer.writerows(group_rows)
    return {
        "sampled_groups": len(sampled),
        "candidate_rows": len(candidate_rows),
        "regression_groups": sum(str(group.get("task_id")) in regression_ids for group in sampled),
        "candidate_csv": str(candidate_path.resolve()),
        "group_csv": str(group_path.resolve()),
        "policy": "regression cases first, then deterministic user/class/split stratified sample",
    }


def apply_manual_review(
    groups: list[dict[str, Any]],
    candidate_csv: str | Path,
    audit_log: str | Path,
    group_csv: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_candidate: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for group in groups:
        for candidate in group.get("candidates", []):
            candidate_id = str(candidate.get("candidate_id") or "")
            if candidate_id in by_candidate:
                raise V4ValidationError(f"Duplicate candidate_id in dataset: {candidate_id}")
            by_candidate[candidate_id] = (group, candidate)

    annotations: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    with Path(candidate_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            candidate_id = str(row.get("candidate_id") or "")
            decision = str(row.get("decision") or "").strip()
            if not decision:
                continue
            if candidate_id in annotations:
                errors.append(f"line {line_number}: duplicate annotation for {candidate_id}")
            elif candidate_id not in by_candidate:
                errors.append(f"line {line_number}: unknown candidate_id {candidate_id}")
            elif decision not in DECISIONS:
                errors.append(f"line {line_number}: unknown decision {decision}")
            else:
                group, candidate = by_candidate[candidate_id]
                actual_split = str(group.get("metadata", {}).get("partition") or "")
                if row.get("split") != actual_split or row.get("group_id") != group.get("group_id"):
                    errors.append(f"line {line_number}: split/group mismatch for {candidate_id}")
                if actual_split == "eval" and candidate.get("source") == "oracle_target" and decision != "keep":
                    errors.append(f"line {line_number}: eval oracle target is immutable")
                if actual_split == "eval" and decision == "manual_replace":
                    errors.append(f"line {line_number}: manual replacement is forbidden in eval")
                annotations[candidate_id] = row
    if errors:
        raise V4ValidationError("Manual review import failed:\n- " + "\n- ".join(errors))

    group_annotations: dict[str, dict[str, str]] = {}
    if group_csv is not None:
        known_groups = {str(group.get("group_id") or ""): group for group in groups}
        with Path(group_csv).open("r", encoding="utf-8-sig", newline="") as handle:
            for line_number, row in enumerate(csv.DictReader(handle), start=2):
                group_id = str(row.get("group_id") or "")
                decision = str(row.get("decision") or "").strip()
                if not decision:
                    continue
                if group_id in group_annotations:
                    errors.append(f"group line {line_number}: duplicate annotation for {group_id}")
                elif group_id not in known_groups:
                    errors.append(f"group line {line_number}: unknown group_id {group_id}")
                elif decision not in {"keep", "regenerate"}:
                    errors.append(f"group line {line_number}: decision must be keep or regenerate")
                else:
                    group = known_groups[group_id]
                    actual_split = str(group.get("metadata", {}).get("partition") or "")
                    if row.get("split") != actual_split:
                        errors.append(f"group line {line_number}: split mismatch for {group_id}")
                    if actual_split == "eval" and decision == "regenerate":
                        errors.append(f"group line {line_number}: eval group cannot be regenerated")
                    group_annotations[group_id] = row
        if errors:
            raise V4ValidationError("Manual review import failed:\n- " + "\n- ".join(errors))

    audit_rows: list[dict[str, Any]] = []
    output: list[dict[str, Any]] = []
    for group in groups:
        cloned = {**group, "candidates": [{**item, "metadata": dict(item.get("metadata") or {})} for item in group["candidates"]]}
        group_annotation = group_annotations.get(str(cloned.get("group_id") or ""))
        if group_annotation:
            cloned["metadata"] = {
                **cloned["metadata"],
                "group_reviewed": True,
                "group_review_decision": group_annotation["decision"],
                "group_review_reason": group_annotation.get("reason", ""),
                "group_reviewer": group_annotation.get("reviewer", ""),
                "group_reviewed_at": group_annotation.get("reviewed_at", "") or datetime.now(timezone.utc).isoformat(),
                "requires_regeneration": group_annotation["decision"] == "regenerate",
            }
            audit_rows.append(
                {
                    "task_id": cloned["task_id"],
                    "group_id": cloned["group_id"],
                    "candidate_id": "",
                    "decision": group_annotation["decision"],
                    "before": {"group_reviewed": False},
                    "after": {"group_reviewed": True},
                    "reviewer": group_annotation.get("reviewer", ""),
                    "reviewed_at": group_annotation.get("reviewed_at", ""),
                }
            )
        kept: list[dict[str, Any]] = []
        corrected_probability = False
        for candidate in cloned["candidates"]:
            original_candidate_id = str(candidate["candidate_id"])
            annotation = annotations.get(str(candidate["candidate_id"]))
            if annotation is None:
                kept.append(candidate)
                continue
            decision = annotation["decision"]
            before = {"text": candidate.get("text"), "probability": candidate["metadata"].get("target_probability")}
            if decision in DROP_DECISIONS:
                if candidate.get("source") == "oracle_target":
                    raise V4ValidationError(f"Oracle candidate cannot be dropped: {candidate['candidate_id']}")
            elif decision == "regenerate":
                if str(cloned.get("metadata", {}).get("partition")) == "eval":
                    raise V4ValidationError("Eval groups cannot be regenerated through manual review.")
                cloned["metadata"] = {**cloned["metadata"], "requires_regeneration": True}
                kept.append(candidate)
            else:
                if decision == "manual_replace":
                    replacement = str(annotation.get("candidate_text") or "").strip()
                    if not replacement or normalize_text(replacement) == normalize_text(before["text"]):
                        raise V4ValidationError(f"manual_replace requires edited candidate_text: {candidate['candidate_id']}")
                    candidate["text"] = replacement
                    candidate["candidate_id"] = sha256_json([cloned["task_id"], "manual_replace", normalize_text(replacement)])[:24]
                value = str(annotation.get("corrected_probability") or "").strip()
                if value:
                    probability = float(value)
                    if not math.isfinite(probability) or probability < 0.0:
                        raise V4ValidationError(f"Invalid corrected probability: {candidate['candidate_id']}")
                    candidate["metadata"]["target_probability"] = probability
                    corrected_probability = True
                candidate["metadata"].update(
                    {
                        "reviewed": True,
                        "review_decision": decision,
                        "review_reason": annotation.get("reason", ""),
                        "reviewer": annotation.get("reviewer", ""),
                        "reviewed_at": annotation.get("reviewed_at", "") or datetime.now(timezone.utc).isoformat(),
                        "review_original_candidate_id": original_candidate_id,
                    }
                )
                kept.append(candidate)
            audit_rows.append(
                {
                    "task_id": cloned["task_id"],
                    "group_id": cloned["group_id"],
                    "candidate_id": candidate.get("candidate_id"),
                    "decision": decision,
                    "before": before,
                    "after": {"text": candidate.get("text"), "probability": candidate["metadata"].get("target_probability")},
                    "reviewer": annotation.get("reviewer", ""),
                    "reviewed_at": annotation.get("reviewed_at", ""),
                }
            )
        if len(kept) < 2:
            raise V4ValidationError(f"Manual review left fewer than two candidates: {cloned['group_id']}")
        probabilities = [float(item["metadata"]["target_probability"]) for item in kept]
        total = sum(probabilities)
        if corrected_probability and not math.isclose(total, 1.0, abs_tol=1e-6):
            raise V4ValidationError(f"Corrected probabilities do not sum to one: {cloned['group_id']} ({total})")
        if not corrected_probability:
            probabilities = [value / total for value in probabilities]
        oracle_positions = [index for index, item in enumerate(kept) if item.get("source") == "oracle_target"]
        if len(oracle_positions) != 1:
            raise V4ValidationError(f"Review changed oracle count: {cloned['group_id']}")
        oracle_index = oracle_positions[0]
        if probabilities[oracle_index] + 1e-8 < max(probabilities):
            raise V4ValidationError(f"Oracle is not highest after review: {cloned['group_id']}")
        ranks = {index: rank for rank, index in enumerate(sorted(range(len(kept)), key=probabilities.__getitem__, reverse=True), start=1)}
        for index, candidate in enumerate(kept):
            candidate["metadata"]["rank"] = ranks[index]
            candidate["metadata"]["target_probability"] = probabilities[index]
            original_id = str(candidate["metadata"].pop("review_original_candidate_id", candidate["candidate_id"]))
            corrected_rank = str((annotations.get(original_id) or {}).get("corrected_rank") or "").strip()
            if corrected_rank and int(corrected_rank) != ranks[index]:
                raise V4ValidationError(
                    f"corrected_rank conflicts with corrected_probability for {original_id}: "
                    f"expected={corrected_rank}, derived={ranks[index]}"
                )
        cloned["candidates"] = kept
        cloned["target_distribution"] = probabilities
        cloned["oracle_index"] = oracle_index
        output.append(cloned)
    write_jsonl(audit_log, audit_rows)
    candidate_audit_count = sum(bool(row.get("candidate_id")) for row in audit_rows)
    return output, {
        "status": "passed",
        "candidate_annotations_applied": candidate_audit_count,
        "group_annotations_applied": len(group_annotations),
        "annotations_applied": len(audit_rows),
        "audit_log": str(Path(audit_log).resolve()),
    }
