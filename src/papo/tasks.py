from __future__ import annotations

from pathlib import Path
from typing import Any

from .official_data import (
    EpisodeRef,
    best_reference,
    best_references,
    complete_raw_index,
    episode_assets,
    episode_ref,
    index_catalog,
    load_profiles,
    official_age_group,
    previous_episodes,
    read_csv_rows,
)


def _profile(user_id: str, profiles: dict[str, dict[str, str]]) -> dict[str, str]:
    return dict(profiles.get(user_id, {}))


def build_proactive_suggestion_tasks(
    test_path: str | Path,
    catalog_path: str | Path,
    profiles_path: str | Path,
    raw_root: str | Path,
    screenshot_level: int = 0,
    history_limit: int = 20,
    limit: int = 0,
    require_complete: bool = True,
    provenance: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if screenshot_level not in {0, 1, 2, 3}:
        raise ValueError("screenshot_level must be one of 0, 1, 2, or 3")

    catalog = index_catalog(read_csv_rows(catalog_path))
    profiles = load_profiles(profiles_path)
    raw_index = complete_raw_index(raw_root)
    tasks: list[dict[str, Any]] = []
    test_rows = read_csv_rows(test_path)
    for row in test_rows:
        target = episode_ref(row)
        if require_complete and (target.user_id, target.time) not in raw_index:
            continue
        assets = episode_assets(raw_index.get((target.user_id, target.time)))
        history = previous_episodes(target, catalog, limit=history_limit)
        tasks.append(
            {
                "task_id": f"suggestion__{target.episode_id}",
                "task_type": "proactive_suggestion",
                "input": {
                    "user_id": target.user_id,
                    "time": target.time,
                    "scenario": target.scenario,
                    "user_profile": _profile(target.user_id, profiles),
                    "previous_intents": [item.to_dict() for item in history],
                    "initial_screenshots": assets["screenshots"][:screenshot_level],
                },
                "target": {
                    "intent": target.intent,
                    "app": target.app,
                    "intent_class": target.intent_class,
                },
                "metadata": {
                    "episode_path": assets["episode_path"],
                    "papo_episode_id": target.episode_id,
                    "screenshot_level": screenshot_level,
                    "history_policy": "same_user_strictly_before_target_time",
                    "target_is_hidden_from_input": True,
                    "history_episode_ids": [item.episode_id for item in history],
                    **(provenance or {}),
                },
            }
        )
        if limit > 0 and len(tasks) >= limit:
            break
    return tasks


def build_personalized_execution_tasks(
    test_path: str | Path,
    catalog_path: str | Path,
    profiles_path: str | Path,
    raw_root: str | Path,
    limit: int = 0,
    require_complete: bool = True,
    same_user_top_k: int = 1,
    cross_user_top_k: int = 1,
    intent_similarity_threshold: float = 0.0,
    exclude_same_intent: bool = False,
    provenance: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    catalog_rows = [episode_ref(row) for row in read_csv_rows(catalog_path)]
    catalog = index_catalog(read_csv_rows(catalog_path))
    profiles = load_profiles(profiles_path)
    raw_index = complete_raw_index(raw_root)
    tasks: list[dict[str, Any]] = []

    test_rows = read_csv_rows(test_path)
    for row in test_rows:
        target = episode_ref(row)
        if require_complete and (target.user_id, target.time) not in raw_index:
            continue
        target_assets = episode_assets(raw_index.get((target.user_id, target.time)))

        same_candidates = previous_episodes(target, catalog, limit=0)
        same_refs = best_references(
            target,
            same_candidates,
            raw_root,
            same_user_top_k,
            raw_index,
            similarity_threshold=intent_similarity_threshold,
            exclude_same_intent=exclude_same_intent,
        )
        target_age_group = official_age_group(target.user_id)
        cross_candidates = [
            item for item in catalog_rows
            if item.user_id != target.user_id
            and item.time < target.time
            and official_age_group(item.user_id) != target_age_group
        ]
        cross_refs = best_references(
            target,
            cross_candidates,
            raw_root,
            cross_user_top_k,
            raw_index,
            similarity_threshold=intent_similarity_threshold,
            exclude_same_intent=False,
        )

        tasks.append(
            {
                "task_id": f"execution__{target.episode_id}",
                "task_type": "personalized_execution",
                "input": {
                    "user_id": target.user_id,
                    "time": target.time,
                    "scenario": target.scenario,
                    "app": target.app,
                    "instruction": target.intent,
                    "user_profile": _profile(target.user_id, profiles),
                    "initial_screenshot": (target_assets["screenshots"] or [""])[0],
                    "initial_xml": (target_assets["xml_files"] or [""])[0],
                    "same_user_action_reference": _reference_payload(same_refs[0]) if same_refs else None,
                    "cross_user_action_reference": _reference_payload(cross_refs[0]) if cross_refs else None,
                    "same_user_action_references": [_reference_payload(ref) for ref in same_refs],
                    "cross_user_action_references": [_reference_payload(ref) for ref in cross_refs],
                    "target_age_group": target_age_group,
                },
                "target": {
                    "actions": target_assets["actions"],
                    "intent_class": target.intent_class,
                },
                "metadata": {
                    "episode_path": target_assets["episode_path"],
                    "papo_episode_id": target.episode_id,
                    "papo_root_step_id": f"{target.episode_id}__0000",
                    "papo_tree_id": f"papo_tree__{target.episode_id}__0000",
                    "reference_policy": "most_similar_intent_strictly_before_target_time",
                    "same_user_top_k": same_user_top_k,
                    "cross_user_top_k": cross_user_top_k,
                    "intent_similarity_threshold": intent_similarity_threshold,
                    "exclude_same_intent": exclude_same_intent,
                    "same_user_reference_is_personalization_context": True,
                    "cross_user_reference_is_different_age_group_counterfactual": True,
                    "target_actions_are_evaluation_only": True,
                    "same_user_reference_episode_ids": [ref[0].episode_id for ref in same_refs],
                    "cross_user_reference_episode_ids": [ref[0].episode_id for ref in cross_refs],
                    **(provenance or {}),
                },
            }
        )
        if limit > 0 and len(tasks) >= limit:
            break
    return tasks


def _reference_payload(reference: tuple[EpisodeRef, Path, float] | None) -> dict[str, Any] | None:
    if reference is None:
        return None
    ref, episode_dir, similarity = reference
    assets = episode_assets(episode_dir)
    return {
        **ref.to_dict(),
        "intent_similarity": similarity,
        "actions": assets["actions"],
        "episode_path": assets["episode_path"],
    }
