from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.papo.official_data import (
    EpisodeRef,
    episode_assets,
    intent_similarity,
    official_age_group,
)


@dataclass(frozen=True)
class RetrievedEpisode:
    episode: EpisodeRef
    episode_dir: Path
    intent_similarity: float
    actions: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            **self.episode.to_dict(),
            "episode_path": str(self.episode_dir),
            "intent_similarity": self.intent_similarity,
            "actions": self.actions,
        }


def retrieve(
    target: EpisodeRef,
    catalog: Iterable[EpisodeRef],
    raw_index: dict[tuple[str, str], Path],
    mode: str,
    top_k: int = 1,
    seed: int = 42,
) -> list[RetrievedEpisode]:
    candidates: list[EpisodeRef] = []
    for candidate in catalog:
        if candidate.episode_id == target.episode_id:
            continue
        if (candidate.user_id, candidate.time) not in raw_index:
            continue
        if not _eligible(target, candidate, mode):
            continue
        candidates.append(candidate)

    rng = random.Random(f"{seed}:{target.episode_id}:{mode}")
    if mode.startswith("random_"):
        rng.shuffle(candidates)
        selected = candidates[:top_k]
    else:
        candidates.sort(
            key=lambda item: (intent_similarity(target.intent, item.intent), item.time),
            reverse=True,
        )
        selected = candidates[:top_k]

    return [
        RetrievedEpisode(
            episode=item,
            episode_dir=raw_index[(item.user_id, item.time)],
            intent_similarity=intent_similarity(target.intent, item.intent),
            actions=list(episode_assets(raw_index[(item.user_id, item.time)])["actions"]),
        )
        for item in selected
    ]


def retrieve_all_modes(
    target: EpisodeRef,
    catalog: Iterable[EpisodeRef],
    raw_index: dict[tuple[str, str], Path],
    modes: list[str],
    top_k: int = 1,
    seed: int = 42,
) -> dict[str, list[RetrievedEpisode]]:
    """Retrieve all baseline variants while computing each intent similarity once."""
    scored: list[tuple[EpisodeRef, float]] = []
    for candidate in catalog:
        if candidate.episode_id == target.episode_id:
            continue
        if (candidate.user_id, candidate.time) not in raw_index:
            continue
        scored.append((candidate, intent_similarity(target.intent, candidate.intent)))

    output: dict[str, list[RetrievedEpisode]] = {}
    for mode in modes:
        candidates = [(item, score) for item, score in scored if _eligible(target, item, mode)]
        rng = random.Random(f"{seed}:{target.episode_id}:{mode}")
        if mode.startswith("random_"):
            rng.shuffle(candidates)
        else:
            candidates.sort(key=lambda item: (item[1], item[0].time), reverse=True)
        selected = candidates[:top_k]
        output[mode] = [
            RetrievedEpisode(
                episode=item,
                episode_dir=raw_index[(item.user_id, item.time)],
                intent_similarity=score,
                actions=list(episode_assets(raw_index[(item.user_id, item.time)])["actions"]),
            )
            for item, score in selected
        ]
    return output


def _eligible(target: EpisodeRef, candidate: EpisodeRef, mode: str) -> bool:
    same_user = candidate.user_id == target.user_id
    strict_past = candidate.time < target.time
    different_type = official_age_group(candidate.user_id) != official_age_group(target.user_id)
    if mode == "same_user_top1":
        return same_user and strict_past
    if mode == "same_user_no_same_intent":
        return same_user and strict_past and candidate.intent != target.intent
    if mode == "random_same_user":
        return same_user and strict_past
    if mode == "cross_user_top1":
        return not same_user and different_type
    if mode == "cross_user_strict_past":
        return not same_user and different_type and strict_past
    if mode == "random_cross_user":
        return not same_user and different_type
    raise ValueError(f"Unknown retrieval mode: {mode}")
