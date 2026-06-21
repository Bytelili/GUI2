from __future__ import annotations

from typing import Any


def flatten_papo_group_features(
    features: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int], list[float], list[bool]]:
    r"""Expand complete PAPO groups while retaining candidate-to-group metadata."""
    flat_features: list[dict[str, Any]] = []
    group_indices: list[int] = []
    target_probabilities: list[float] = []
    oracle_mask: list[bool] = []
    for group_index, feature in enumerate(features):
        candidate_ids = feature["input_ids"]
        candidate_labels = feature["labels"]
        candidate_masks = feature["attention_mask"]
        probabilities = feature["papo_group_target"]
        oracle_index = int(feature["papo_group_oracle_index"])
        size = len(candidate_ids)
        if size < 2 or len(candidate_labels) != size or len(candidate_masks) != size:
            raise ValueError("Malformed PAPO group token features.")
        if len(probabilities) != size or oracle_index < 0 or oracle_index >= size:
            raise ValueError("Malformed PAPO group target metadata.")
        common = {
            key: value
            for key, value in feature.items()
            if key
            not in {
                "input_ids",
                "labels",
                "attention_mask",
                "listwise_weight",
                "papo_group_target",
                "papo_group_oracle_index",
                "papo_group_id",
            }
        }
        for candidate_index in range(size):
            flat_features.append(
                {
                    **common,
                    "input_ids": candidate_ids[candidate_index],
                    "labels": candidate_labels[candidate_index],
                    "attention_mask": candidate_masks[candidate_index],
                }
            )
            group_indices.append(group_index)
            target_probabilities.append(float(probabilities[candidate_index]))
            oracle_mask.append(candidate_index == oracle_index)
    return flat_features, group_indices, target_probabilities, oracle_mask
