from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from statistics import mean
from typing import Any

from .candidates import normalize_text
from .rewards import build_pairs


@dataclass(frozen=True)
class QualityThresholds:
    min_candidate_chars: int = 4
    repeated_character_rate: float = 0.60
    pseudo_negative_task_match: float = 0.92
    easy_negative_task_match: float = 0.20
    easy_negative_same_user_similarity: float = 0.20
    near_duplicate_similarity: float = 0.92
    max_invalid_negative_rate: float = 0.01
    max_tasks_without_usable_negative_rate: float = 0.05
    warning_pseudo_negative_rate: float = 0.25
    warning_easy_negative_rate: float = 0.60
    warning_min_valid_hard_negative_rate: float = 0.20
    warning_min_model_candidate_coverage: float = 0.90

    def validate(self) -> None:
        if self.min_candidate_chars < 1:
            raise ValueError("min_candidate_chars must be positive")
        for name, value in asdict(self).items():
            if name == "min_candidate_chars":
                continue
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"Quality threshold must be in [0, 1]: {name}={value}")


def audit_candidate_quality(
    train_sets: list[dict[str, Any]],
    eval_sets: list[dict[str, Any]],
    *,
    thresholds: QualityThresholds | None = None,
    model_candidates_expected: dict[str, bool] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Annotate candidates, remove unsafe DPO pairs, and report proxy quality."""
    thresholds = thresholds or QualityThresholds()
    thresholds.validate()
    model_candidates_expected = model_candidates_expected or {}
    flags: list[dict[str, Any]] = []
    reports: dict[str, dict[str, Any]] = {}
    hard_failures: list[str] = []
    warnings: list[str] = []
    for partition, rows in [("train", train_sets), ("eval", eval_sets)]:
        report, partition_flags = _audit_partition(
            rows,
            partition,
            thresholds,
            model_candidates_expected=bool(model_candidates_expected.get(partition)),
        )
        reports[partition] = report
        flags.extend(partition_flags)
        hard_failures.extend(f"{partition}: {message}" for message in report["hard_failures"])
        warnings.extend(f"{partition}: {message}" for message in report["warnings"])
    status = "failed" if hard_failures else ("warning" if warnings else "passed")
    return (
        {
            "status": status,
            "method": "deterministic_proxy_candidate_quality_v1",
            "interpretation": (
                "Structural failures are hard gates. Semantic tiers are lexical/context proxies "
                "and must be supplemented by sampled human review."
            ),
            "thresholds": asdict(thresholds),
            "hard_failures": hard_failures,
            "warnings": warnings,
            "partitions": reports,
        },
        flags,
    )


def drop_invalid_oracle_targets(
    rows: list[dict[str, Any]],
    *,
    thresholds: QualityThresholds | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    excluded = invalid_oracle_targets(rows, thresholds=thresholds)
    excluded_ids = {str(row.get("task_id") or "") for row in excluded}
    return [row for row in rows if str(row.get("task_id") or "") not in excluded_ids], excluded


def invalid_oracle_targets(
    rows: list[dict[str, Any]],
    *,
    thresholds: QualityThresholds | None = None,
) -> list[dict[str, Any]]:
    thresholds = thresholds or QualityThresholds()
    thresholds.validate()
    invalid: list[dict[str, Any]] = []
    for row in rows:
        oracle = next(
            (candidate for candidate in row.get("candidates", []) if candidate.get("source") == "oracle_target"),
            None,
        )
        if oracle is None:
            invalid.append(_invalid_oracle_record(row, {}, ["missing_oracle_target"]))
            continue
        reasons = _structural_reasons(str(oracle.get("text") or ""), thresholds)
        if reasons:
            invalid.append(_invalid_oracle_record(row, oracle, reasons))
    return invalid


def build_quality_review_sample(
    train_sets: list[dict[str, Any]],
    eval_sets: list[dict[str, Any]],
    *,
    per_bucket: int = 10,
) -> list[dict[str, Any]]:
    if per_bucket < 1:
        raise ValueError("per_bucket must be positive")
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for partition, rows in [("train", train_sets), ("eval", eval_sets)]:
        for row in rows:
            inputs = row.get("input") if isinstance(row.get("input"), dict) else {}
            target = row.get("target") if isinstance(row.get("target"), dict) else {}
            history = list(inputs.get("previous_intents") or [])
            for candidate in row.get("candidates", []):
                source = str(candidate.get("source") or "")
                if source == "oracle_target":
                    continue
                quality = candidate.get("quality") if isinstance(candidate.get("quality"), dict) else {}
                quality_class = str(quality.get("class") or "")
                record = {
                    "partition": partition,
                    "task_id": row.get("task_id", ""),
                    "time": inputs.get("time", ""),
                    "scenario": inputs.get("scenario", ""),
                    "recent_history": history[-3:],
                    "target": target.get("intent", ""),
                    "candidate": candidate.get("text", ""),
                    "candidate_source": source,
                    "candidate_source_episode_id": candidate.get("source_episode_id", ""),
                    "quality": quality,
                    "reward": candidate.get("reward", {}),
                }
                buckets.setdefault((partition, quality_class, source), []).append(record)
    sample: list[dict[str, Any]] = []
    for key in sorted(buckets):
        rows = sorted(
            buckets[key],
            key=lambda row: (str(row.get("task_id") or ""), str(row.get("candidate") or "")),
        )
        sample.extend(rows[:per_bucket])
    return sample


def _audit_partition(
    rows: list[dict[str, Any]],
    partition: str,
    thresholds: QualityThresholds,
    *,
    model_candidates_expected: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    classifications: Counter[str] = Counter()
    source_classes: dict[str, Counter[str]] = {}
    flags: list[dict[str, Any]] = []
    negative_count = 0
    invalid_oracles = 0
    tasks_without_usable = 0
    tasks_with_model_candidates = 0
    near_duplicate_pairs = 0
    model_near_duplicate_pairs = 0
    model_pair_similarities: list[float] = []
    removed_unsafe_dpo_pairs = 0
    recovered_safe_dpo_pairs = 0
    final_dpo_pairs = 0
    for row in rows:
        candidates = list(row.get("candidates") or [])
        candidate_by_id: dict[str, dict[str, Any]] = {}
        usable_negatives = 0
        model_candidates: list[dict[str, Any]] = []
        negatives: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id") or "")
            candidate_by_id[candidate_id] = candidate
            source = str(candidate.get("source") or "")
            structural_reasons = _structural_reasons(str(candidate.get("text") or ""), thresholds)
            if source == "oracle_target":
                quality_class = "oracle"
                reasons = structural_reasons
                invalid_oracles += bool(structural_reasons)
            else:
                negative_count += 1
                negatives.append(candidate)
                quality_class, semantic_reasons = _negative_class(candidate, structural_reasons, thresholds)
                reasons = structural_reasons + semantic_reasons
                classifications[quality_class] += 1
                source_classes.setdefault(source, Counter())[quality_class] += 1
                usable_negatives += quality_class in {"easy_negative", "valid_hard_negative"}
                if quality_class != "valid_hard_negative":
                    flags.append(_flag_record(row, candidate, quality_class, reasons))
                if source == "sft_sample":
                    model_candidates.append(candidate)
            candidate["quality"] = {
                "class": quality_class,
                "reasons": reasons,
                "listwise_eligible": quality_class != "invalid",
                "dpo_rejected_eligible": quality_class in {"easy_negative", "valid_hard_negative"},
            }
        tasks_without_usable += usable_negatives == 0
        tasks_with_model_candidates += bool(model_candidates)
        near_duplicate_pairs += _count_near_duplicate_pairs(negatives, thresholds.near_duplicate_similarity)
        model_pairs = _pair_similarities(model_candidates)
        model_pair_similarities.extend(model_pairs)
        model_near_duplicate_pairs += sum(
            similarity >= thresholds.near_duplicate_similarity for similarity in model_pairs
        )
        original_pairs = list(row.get("pairs", []))
        safe_original_pair_ids: set[str] = set()
        for pair in original_pairs:
            rejected = candidate_by_id.get(str(pair.get("rejected_candidate_id") or ""), {})
            quality = rejected.get("quality") if isinstance(rejected.get("quality"), dict) else {}
            if quality.get("dpo_rejected_eligible"):
                safe_original_pair_ids.add(str(pair.get("rejected_candidate_id") or ""))
            else:
                removed_unsafe_dpo_pairs += 1
        safe_candidates = [
            candidate
            for candidate in candidates
            if candidate.get("source") == "oracle_target"
            or bool((candidate.get("quality") or {}).get("dpo_rejected_eligible"))
        ]
        rebuilt_pairs = build_pairs(
            safe_candidates,
            float(row.get("pair_margin", 0.05) or 0.05),
            int(row.get("max_pairs_per_task", max(2, len(original_pairs))) or 2),
            float(row.get("temperature", 0.2) or 0.2),
        )
        recovered_safe_dpo_pairs += sum(
            str(pair.get("rejected_candidate_id") or "") not in safe_original_pair_ids
            for pair in rebuilt_pairs
        )
        final_dpo_pairs += len(rebuilt_pairs)
        row["pairs"] = rebuilt_pairs

    targets = len(rows)
    invalid_rate = _rate(classifications["invalid"], negative_count)
    no_usable_rate = _rate(tasks_without_usable, targets)
    pseudo_rate = _rate(classifications["pseudo_negative"], negative_count)
    easy_rate = _rate(classifications["easy_negative"], negative_count)
    valid_hard_rate = _rate(classifications["valid_hard_negative"], negative_count)
    model_coverage = _rate(tasks_with_model_candidates, targets)
    hard_failures: list[str] = []
    warnings: list[str] = []
    if invalid_oracles:
        hard_failures.append(f"invalid_oracle_targets={invalid_oracles}")
    if invalid_rate > thresholds.max_invalid_negative_rate:
        hard_failures.append(
            f"invalid_negative_rate={invalid_rate:.4f} exceeds "
            f"{thresholds.max_invalid_negative_rate:.4f}"
        )
    if no_usable_rate > thresholds.max_tasks_without_usable_negative_rate:
        hard_failures.append(
            f"tasks_without_usable_negative_rate={no_usable_rate:.4f} exceeds "
            f"{thresholds.max_tasks_without_usable_negative_rate:.4f}"
        )
    if pseudo_rate > thresholds.warning_pseudo_negative_rate:
        warnings.append(
            f"pseudo_negative_rate={pseudo_rate:.4f} exceeds "
            f"{thresholds.warning_pseudo_negative_rate:.4f}"
        )
    if easy_rate > thresholds.warning_easy_negative_rate:
        warnings.append(
            f"easy_negative_rate={easy_rate:.4f} exceeds {thresholds.warning_easy_negative_rate:.4f}"
        )
    if negative_count and valid_hard_rate < thresholds.warning_min_valid_hard_negative_rate:
        warnings.append(
            f"valid_hard_negative_rate={valid_hard_rate:.4f} is below "
            f"{thresholds.warning_min_valid_hard_negative_rate:.4f}"
        )
    if model_candidates_expected and model_coverage < thresholds.warning_min_model_candidate_coverage:
        warnings.append(
            f"model_candidate_coverage={model_coverage:.4f} is below "
            f"{thresholds.warning_min_model_candidate_coverage:.4f}"
        )
    return (
        {
            "targets": targets,
            "negative_candidates": negative_count,
            "classification_counts": dict(sorted(classifications.items())),
            "classification_rates": {
                name: _rate(classifications[name], negative_count)
                for name in ["invalid", "pseudo_negative", "easy_negative", "valid_hard_negative"]
            },
            "source_classification_counts": {
                source: dict(sorted(counts.items()))
                for source, counts in sorted(source_classes.items())
            },
            "invalid_oracle_targets": invalid_oracles,
            "tasks_without_usable_negative": tasks_without_usable,
            "tasks_without_usable_negative_rate": no_usable_rate,
            "near_duplicate_negative_pairs": near_duplicate_pairs,
            "model_candidates_expected": model_candidates_expected,
            "tasks_with_model_candidates": tasks_with_model_candidates,
            "model_candidate_coverage": model_coverage,
            "model_candidate_pair_similarity_mean": (
                mean(model_pair_similarities) if model_pair_similarities else 0.0
            ),
            "model_candidate_near_duplicate_pairs": model_near_duplicate_pairs,
            "removed_unsafe_dpo_pairs": removed_unsafe_dpo_pairs,
            "recovered_safe_dpo_pairs": recovered_safe_dpo_pairs,
            "final_dpo_pairs": final_dpo_pairs,
            "hard_failures": hard_failures,
            "warnings": warnings,
        },
        flags,
    )


def _negative_class(
    candidate: dict[str, Any],
    structural_reasons: list[str],
    thresholds: QualityThresholds,
) -> tuple[str, list[str]]:
    if structural_reasons:
        return "invalid", []
    reward = candidate.get("reward") if isinstance(candidate.get("reward"), dict) else {}
    task_match = float(reward.get("task_match", 0.0) or 0.0)
    same_user = float(reward.get("same_user_similarity", 0.0) or 0.0)
    if task_match >= thresholds.pseudo_negative_task_match:
        return "pseudo_negative", ["near_target_lexical_match"]
    if (
        task_match <= thresholds.easy_negative_task_match
        and same_user <= thresholds.easy_negative_same_user_similarity
    ):
        return "easy_negative", ["low_target_and_user_history_match"]
    return "valid_hard_negative", []


def _structural_reasons(text: str, thresholds: QualityThresholds) -> list[str]:
    normalized = normalize_text(text)
    reasons: list[str] = []
    if len(normalized) < thresholds.min_candidate_chars:
        reasons.append("too_short")
    if "\ufffd" in text:
        reasons.append("replacement_character")
    if any(ord(character) < 32 and character not in "\t\n\r" for character in text):
        reasons.append("control_character")
    if len(normalized) >= 8:
        most_common = Counter(normalized).most_common(1)[0][1]
        if most_common / len(normalized) >= thresholds.repeated_character_rate:
            reasons.append("repeated_character_output")
    return reasons


def _flag_record(
    row: dict[str, Any],
    candidate: dict[str, Any],
    quality_class: str,
    reasons: list[str],
) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return {
        "partition": row.get("partition", ""),
        "task_id": row.get("task_id", ""),
        "target_episode_id": metadata.get("papo_episode_id", ""),
        "candidate_id": candidate.get("candidate_id", ""),
        "candidate_source": candidate.get("source", ""),
        "candidate_source_episode_id": candidate.get("source_episode_id", ""),
        "quality_class": quality_class,
        "reasons": reasons,
        "text": candidate.get("text", ""),
        "reward": candidate.get("reward", {}),
    }


def _invalid_oracle_record(
    row: dict[str, Any],
    oracle: dict[str, Any],
    reasons: list[str],
) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    target = row.get("target") if isinstance(row.get("target"), dict) else {}
    inputs = row.get("input") if isinstance(row.get("input"), dict) else {}
    return {
        "partition": row.get("partition", ""),
        "task_id": row.get("task_id", ""),
        "target_episode_id": metadata.get("papo_episode_id", ""),
        "time": inputs.get("time", ""),
        "scenario": inputs.get("scenario", ""),
        "target": target.get("intent", ""),
        "oracle_candidate_id": oracle.get("candidate_id", ""),
        "oracle_text": oracle.get("text", ""),
        "reasons": reasons,
    }


def _count_near_duplicate_pairs(candidates: list[dict[str, Any]], threshold: float) -> int:
    return sum(similarity >= threshold for similarity in _pair_similarities(candidates))


def _pair_similarities(candidates: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for index, left in enumerate(candidates):
        left_text = normalize_text(left.get("text"))
        for right in candidates[index + 1 :]:
            right_text = normalize_text(right.get("text"))
            if left_text and right_text:
                values.append(SequenceMatcher(None, left_text, right_text).ratio())
    return values


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
