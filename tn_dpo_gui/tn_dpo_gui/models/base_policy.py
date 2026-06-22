from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter

from tn_dpo_gui.data.action_schema import Action


class BasePolicy(ABC):
    @abstractmethod
    def action_log_probs(self, example, candidates: list[Action], history_records=None) -> dict[str, float]:
        raise NotImplementedError

    def suggest_actions(self, example) -> list[Action]:
        return []


class UniformBasePolicy(BasePolicy):
    def action_log_probs(self, example, candidates: list[Action], history_records=None) -> dict[str, float]:
        if not candidates:
            return {}
        log_prob = -math.log(len(candidates))
        return {candidate.normalized_key(): log_prob for candidate in candidates}


class LoggedFrequencyPolicy(BasePolicy):
    def action_log_probs(self, example, candidates: list[Action], history_records=None) -> dict[str, float]:
        history_records = history_records or []
        counts = Counter(action.normalized_key() for record in history_records for action in record.actions)
        total = sum(counts.values()) + len(candidates)
        return {
            candidate.normalized_key(): math.log((counts[candidate.normalized_key()] + 1.0) / max(total, 1.0))
            for candidate in candidates
        }
