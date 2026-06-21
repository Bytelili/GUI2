from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .proactive_listwise_v4 import FORBIDDEN_TARGET_SPLITS, PROTOCOL_ID, normalize_text, write_json


@dataclass(frozen=True)
class V4Issue:
    severity: str
    category: str
    split: str
    group_id: str
    candidate_id: str
    detail: str


def _resolve_image(value: str, roots: list[Path]) -> bool:
    original = Path(value)
    if original.is_file():
        return True
    parts = list(Path(value.replace("\\", "/")).parts)
    suffix = parts[parts.index("fingertip20k") + 1 :] if "fingertip20k" in parts else [original.name]
    return any(root.joinpath(*suffix).is_file() for root in roots)


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None, "mean": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "p50": ordered[int((len(ordered) - 1) * 0.50)],
        "p95": ordered[int((len(ordered) - 1) * 0.95)],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def audit_v4_groups(
    train_groups: list[dict[str, Any]],
    eval_groups: list[dict[str, Any]],
    *,
    image_roots: Iterable[str | Path] = (),
    allow_unavailable_images: bool = False,
    source_manifest: dict[str, Any] | None = None,
    min_oracle_margin: float = 0.15,
    max_answer_fraction: float = 0.05,
    min_manual_review_fraction: float = 0.0,
) -> tuple[dict[str, Any], list[V4Issue]]:
    issues: list[V4Issue] = []
    roots = [Path(value) for value in image_roots]
    source_counts: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    group_sizes: list[float] = []
    margins: list[float] = []
    reviewed = 0
    candidate_total = 0
    unavailable_image_count = 0
    split_oracle_hashes: dict[str, set[str]] = {"train": set(), "eval": set()}

    def add(severity: str, category: str, split: str, group_id: str, detail: str, candidate_id: str = "") -> None:
        issue = V4Issue(severity, category, split, group_id, candidate_id, detail)
        issues.append(issue)
        print(f"[{severity.upper()}] {category} | {split} | {group_id} | {detail}", flush=True)

    if source_manifest and source_manifest.get("hard_error_count"):
        add("block", "source_manifest_failed", "all", "source", "source task manifest contains hard errors")

    all_group_ids: set[str] = set()
    all_task_ids: dict[str, set[str]] = {"train": set(), "eval": set()}
    for split, groups in (("train", train_groups), ("eval", eval_groups)):
        for group_index, group in enumerate(groups):
            group_id = str(group.get("group_id") or "")
            task_id = str(group.get("task_id") or "")
            metadata = group.get("metadata") if isinstance(group.get("metadata"), dict) else {}
            if not group_id or group_id in all_group_ids:
                add("block", "duplicate_or_missing_group_id", split, group_id or f"index:{group_index}", "group_id must be unique")
            all_group_ids.add(group_id)
            if not task_id or task_id in all_task_ids[split]:
                add("block", "duplicate_or_missing_task_id", split, group_id, "task_id must be unique inside split")
            all_task_ids[split].add(task_id)
            if metadata.get("partition") != split:
                add("block", "partition_mismatch", split, group_id, str(metadata.get("partition")))
            if metadata.get("protocol_id") != PROTOCOL_ID:
                add("block", "protocol_mismatch", split, group_id, str(metadata.get("protocol_id")))
            if str(metadata.get("target_split") or "").lower() in FORBIDDEN_TARGET_SPLITS:
                add("block", "forbidden_target_split", split, group_id, str(metadata.get("target_split")))
            if metadata.get("history_policy") != "same_user_strictly_before_target_time":
                add("block", "invalid_history_policy", split, group_id, str(metadata.get("history_policy")))
            oracle_hash = str(metadata.get("target_identity_sha256") or "")
            if not oracle_hash:
                add("block", "missing_target_identity_hash", split, group_id, "target_identity_sha256 is required")
            split_oracle_hashes[split].add(oracle_hash)

            messages = group.get("messages")
            if not isinstance(messages, list) or not messages:
                add("block", "invalid_messages", split, group_id, "messages must be a non-empty prompt-only list")
                prompt = ""
            else:
                if any(message.get("role") == "assistant" for message in messages if isinstance(message, dict)):
                    add("block", "assistant_in_prompt", split, group_id, "formal v4 messages must be prompt-only")
                prompt = "\n".join(str(message.get("content") or "") for message in messages if isinstance(message, dict))

            images = group.get("images")
            if not isinstance(images, list) or not images:
                add("block", "missing_images", split, group_id, "at least one original image path is required")
            else:
                for image in images:
                    if not _resolve_image(str(image), roots):
                        unavailable_image_count += 1

            candidates = group.get("candidates")
            probabilities = group.get("target_distribution")
            oracle_index = group.get("oracle_index")
            if not isinstance(candidates, list) or not 2 <= len(candidates) <= 4:
                add("block", "invalid_group_size", split, group_id, "candidate group size must be 2-4")
                continue
            group_sizes.append(float(len(candidates)))
            if not isinstance(probabilities, list) or len(probabilities) != len(candidates):
                add("block", "target_size_mismatch", split, group_id, "target_distribution size differs from candidates")
                continue
            try:
                probs = [float(value) for value in probabilities]
            except (TypeError, ValueError):
                add("block", "invalid_probability", split, group_id, "probabilities must be numeric")
                continue
            if any(not math.isfinite(value) or value < 0.0 for value in probs) or not math.isclose(sum(probs), 1.0, abs_tol=1e-6):
                add("block", "invalid_probability", split, group_id, "probabilities must be finite, non-negative, sum to one")
                continue

            oracle_positions = [index for index, item in enumerate(candidates) if item.get("source") == "oracle_target"]
            if len(oracle_positions) != 1:
                add("block", "oracle_count", split, group_id, f"expected one oracle, got {len(oracle_positions)}")
                continue
            actual_oracle = oracle_positions[0]
            if not isinstance(oracle_index, int) or oracle_index != actual_oracle:
                add("block", "oracle_index", split, group_id, f"expected {actual_oracle}, got {oracle_index}")
            if probs[actual_oracle] + 1e-8 < max(probs):
                add("block", "oracle_not_top_probability", split, group_id, "oracle probability is not highest")
            best_negative = max(value for index, value in enumerate(probs) if index != actual_oracle)
            margin = probs[actual_oracle] - best_negative
            margins.append(margin)
            if margin < min_oracle_margin:
                add("block", "weak_oracle_margin", split, group_id, f"margin={margin:.6f} < {min_oracle_margin:.6f}")

            ids: set[str] = set()
            texts: set[str] = set()
            expected_ranks = {index: rank for rank, index in enumerate(sorted(range(len(probs)), key=probs.__getitem__, reverse=True), start=1)}
            reward_totals: list[float] = []
            for index, candidate in enumerate(candidates):
                candidate_total += 1
                candidate_id = str(candidate.get("candidate_id") or "")
                text = str(candidate.get("text") or "").strip()
                source = str(candidate.get("source") or "")
                key = normalize_text(text)
                if not candidate_id or candidate_id in ids:
                    add("block", "duplicate_or_missing_candidate_id", split, group_id, candidate_id or f"index:{index}")
                ids.add(candidate_id)
                if not key or key in texts:
                    add("block", "duplicate_or_empty_candidate", split, group_id, text, candidate_id)
                texts.add(key)
                if any(marker in text.upper() for marker in ("ERROR", "�")):
                    add("block", "invalid_candidate_text", split, group_id, text, candidate_id)
                if key and key in normalize_text(prompt) and source != "oracle_target":
                    add("block", "prompt_history_copy", split, group_id, "candidate appears verbatim in prompt/history", candidate_id)
                if source == "cross_user_hard":
                    add("block", "cross_user_positive_candidate", split, group_id, "cross-user candidate cannot receive positive mass", candidate_id)
                reward = candidate.get("reward")
                required_reward = {"R_task", "R_user", "R_context", "R_specificity", "total"}
                if not isinstance(reward, dict) or not required_reward.issubset(reward):
                    add("block", "missing_reward_components", split, group_id, "reward components are incomplete", candidate_id)
                    reward_totals.append(float("-inf"))
                else:
                    reward_totals.append(float(reward["total"]))
                candidate_metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
                if candidate_metadata.get("rank") != expected_ranks[index]:
                    add("block", "rank_probability_mismatch", split, group_id, f"candidate rank != {expected_ranks[index]}", candidate_id)
                try:
                    metadata_probability = float(candidate_metadata.get("target_probability"))
                except (TypeError, ValueError):
                    metadata_probability = math.nan
                if not math.isclose(metadata_probability, probs[index], abs_tol=1e-8):
                    add("block", "candidate_probability_mismatch", split, group_id, "candidate metadata probability differs", candidate_id)
                if candidate_metadata.get("reviewed") is True:
                    reviewed += 1
                source_counts[source] += 1
                answer_counts[key] += 1
            if not any(bool((item.get("metadata") or {}).get("reviewed")) for item in candidates):
                reward_ranks = {
                    index: rank
                    for rank, index in enumerate(
                        sorted(range(len(reward_totals)), key=reward_totals.__getitem__, reverse=True), start=1
                    )
                }
                for index, candidate in enumerate(candidates):
                    if candidate.get("metadata", {}).get("rank") != reward_ranks[index]:
                        add(
                            "block",
                            "reward_rank_mismatch",
                            split,
                            group_id,
                            f"reward-derived rank={reward_ranks[index]}",
                            str(candidate.get("candidate_id") or ""),
                        )

    overlap = all_task_ids["train"] & all_task_ids["eval"]
    if overlap:
        add("block", "train_eval_task_overlap", "all", "dataset", f"overlap_count={len(overlap)}")
    target_overlap = split_oracle_hashes["train"] & split_oracle_hashes["eval"]
    if target_overlap:
        add("block", "train_eval_target_overlap", "all", "dataset", f"overlap_count={len(target_overlap)}")
    if unavailable_image_count:
        severity = "warn" if allow_unavailable_images else "block"
        add(
            severity,
            "unavailable_image",
            "all",
            "dataset",
            f"count={unavailable_image_count}; original paths retained; see source unavailable-image report",
        )

    for answer, count in answer_counts.items():
        fraction = count / max(candidate_total, 1)
        if answer and count >= 20 and fraction > max_answer_fraction:
            add("block", "popular_answer_bias", "all", "dataset", f"answer={answer[:80]!r}, fraction={fraction:.6f}")
    manual_fraction = reviewed / max(candidate_total, 1)
    if manual_fraction + 1e-12 < min_manual_review_fraction:
        add("block", "manual_review_coverage", "all", "dataset", f"coverage={manual_fraction:.6f}")

    block_count = sum(issue.severity == "block" for issue in issues)
    warning_count = sum(issue.severity == "warn" for issue in issues)
    report = {
        "schema_version": "papo_listwise_v4_quality_report",
        "status": "failed" if block_count else "passed_with_warnings" if warning_count else "passed",
        "group_counts": {"train": len(train_groups), "eval": len(eval_groups)},
        "candidate_count": candidate_total,
        "unavailable_image_count": unavailable_image_count,
        "block_count": block_count,
        "warning_count": warning_count,
        "issue_counts": dict(Counter(issue.category for issue in issues)),
        "source_counts": dict(source_counts),
        "group_size": _quantiles(group_sizes),
        "oracle_margin": _quantiles(margins),
        "manual_review_fraction": manual_fraction,
        "image_policy": "unavailable paths retained" if allow_unavailable_images else "all images required locally",
        "train_eval_task_overlap_count": len(overlap),
        "train_eval_target_overlap_count": len(target_overlap),
        "top_answers": [{"text": text, "count": count} for text, count in answer_counts.most_common(20)],
    }
    return report, issues


def write_quality_outputs(report: dict[str, Any], issues: list[V4Issue], report_dir: str | Path) -> dict[str, str]:
    root = Path(report_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "listwise_v4_quality_report.json"
    csv_path = root / "listwise_v4_quality_issues.csv"
    summary_path = root / "listwise_v4_quality_summary.txt"
    write_json(json_path, report)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(V4Issue("", "", "", "", "", "")).keys()))
        writer.writeheader()
        writer.writerows(asdict(issue) for issue in issues)
    summary_path.write_text(
        f"QUALITY STATUS: {report['status'].upper()}\n"
        f"groups: train={report['group_counts']['train']} eval={report['group_counts']['eval']}\n"
        f"candidates={report['candidate_count']} blocks={report['block_count']} warnings={report['warning_count']}\n",
        encoding="utf-8",
    )
    return {"json": str(json_path), "csv": str(csv_path), "summary": str(summary_path)}


def verify_training_dataset_binding(training_config: dict[str, Any], release_dir: str | Path) -> list[str]:
    root = Path(release_dir)
    manifest = json.loads((root / "listwise_v4_manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    if training_config.get("use_papo_group_listwise") is not True:
        errors.append("training config does not enable use_papo_group_listwise")
    if training_config.get("use_papo_listwise"):
        errors.append("legacy use_papo_listwise must be disabled for v4")
    expected = manifest.get("dataset_hashes", {})
    configured = training_config.get("papo_dataset_hashes")
    if configured != expected:
        errors.append("training config dataset hashes do not match release manifest")
    return errors
