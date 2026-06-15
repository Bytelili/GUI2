from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any

from .candidates import normalize_text


@dataclass(frozen=True)
class RewardWeights:
    task: float = 0.55
    user: float = 0.20
    context: float = 0.15
    specificity: float = 0.10

    def validate(self) -> None:
        values = asdict(self)
        if any(value < 0.0 for value in values.values()):
            raise ValueError(f"Reward weights must be non-negative: {values}")
        if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-8):
            raise ValueError(f"Reward weights must sum to one: {values}")


def score_candidate_sets(
    candidate_sets: list[dict[str, Any]],
    *,
    weights: RewardWeights,
    temperature: float = 0.2,
    pair_margin: float = 0.05,
    max_pairs_per_task: int = 2,
) -> list[dict[str, Any]]:
    weights.validate()
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    scored_sets: list[dict[str, Any]] = []
    for row in candidate_sets:
        inputs = row["input"]
        target_text = str(row["target"].get("intent") or "")
        same_history = [
            str(item.get("intent") or "")
            for item in inputs.get("previous_intents", [])
            if str(item.get("intent") or "")
        ]
        cross_texts = [
            str(item.get("text") or "")
            for item in row["candidates"]
            if item.get("source") == "cross_user_hard"
        ]
        candidates: list[dict[str, Any]] = []
        for candidate in row["candidates"]:
            text = str(candidate.get("text") or "")
            task_score = text_similarity(text, target_text)
            same_score = max((text_similarity(text, item) for item in same_history), default=0.0)
            cross_score = max((text_similarity(text, item) for item in cross_texts), default=0.0)
            user_score = (same_score - cross_score + 1.0) / 2.0
            context_score = _context_score(inputs, candidate)
            specificity_score = _specificity(text)
            total = (
                weights.task * task_score
                + weights.user * user_score
                + weights.context * context_score
                + weights.specificity * specificity_score
            )
            candidates.append(
                {
                    **candidate,
                    "reward": {
                        "task_match": task_score,
                        "same_user_similarity": same_score,
                        "cross_user_similarity": cross_score,
                        "user_preference": user_score,
                        "context_match": context_score,
                        "specificity": specificity_score,
                        "total": total,
                    },
                }
            )
        probabilities = _softmax([candidate["reward"]["total"] / temperature for candidate in candidates])
        for candidate, probability in zip(candidates, probabilities):
            candidate["target_policy_probability"] = probability
        candidates.sort(key=lambda item: (item["reward"]["total"], item["source"] == "oracle_target"), reverse=True)
        pairs = _build_pairs(candidates, pair_margin, max_pairs_per_task, temperature)
        scored_sets.append(
            {
                **row,
                "reward_weights": asdict(weights),
                "temperature": temperature,
                "pair_margin": pair_margin,
                "candidates": candidates,
                "pairs": pairs,
            }
        )
    return scored_sets


def text_similarity(left: str, right: str) -> float:
    left_key = normalize_text(left)
    right_key = normalize_text(right)
    if not left_key or not right_key:
        return 0.0
    return SequenceMatcher(None, left_key, right_key).ratio()


def _build_pairs(
    candidates: list[dict[str, Any]],
    margin: float,
    limit: int,
    temperature: float,
) -> list[dict[str, Any]]:
    oracle = next((item for item in candidates if item.get("source") == "oracle_target"), None)
    if oracle is None:
        raise ValueError("Every preference set must contain an oracle target")
    negatives = [item for item in candidates if item is not oracle]
    negatives.sort(key=lambda item: item["reward"]["total"], reverse=True)
    pairs: list[dict[str, Any]] = []
    for rejected in negatives:
        gap = float(oracle["reward"]["total"]) - float(rejected["reward"]["total"])
        if gap <= margin:
            continue
        pairs.append(
            {
                "chosen_candidate_id": oracle["candidate_id"],
                "rejected_candidate_id": rejected["candidate_id"],
                "chosen": oracle["text"],
                "rejected": rejected["text"],
                "chosen_source": oracle["source"],
                "rejected_source": rejected["source"],
                "reward_gap": gap,
                "weight": min(5.0, max(0.1, gap / max(margin, 1e-8))),
                "target_preference_probability": _sigmoid(gap / temperature),
            }
        )
        if len(pairs) >= limit:
            break
    return pairs


def _context_score(inputs: dict[str, Any], candidate: dict[str, Any]) -> float:
    source = str(candidate.get("source") or "")
    if source in {"oracle_target", "sft_sample"}:
        return 1.0
    scenario_match = float(
        bool(inputs.get("scenario"))
        and str(inputs.get("scenario")) == str(candidate.get("source_scenario") or "")
    )
    target_hour = _hour(str(inputs.get("time") or ""))
    source_hour = _hour(str(candidate.get("source_time") or ""))
    temporal = 0.5
    if target_hour is not None and source_hour is not None:
        distance = min(abs(target_hour - source_hour), 24 - abs(target_hour - source_hour))
        temporal = 1.0 - distance / 12.0
    return 0.5 * scenario_match + 0.5 * temporal


def _specificity(text: str) -> float:
    length = len(normalize_text(text))
    if length < 4:
        return length / 4.0
    if length <= 48:
        return 1.0
    return max(0.0, 1.0 - (length - 48) / 96.0)


def _hour(timestamp: str) -> int | None:
    try:
        return int(timestamp.split("_", 1)[1][:2])
    except (IndexError, TypeError, ValueError):
        return None


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)
