from __future__ import annotations

from tn_dpo_gui.data.schema import TrajectoryRecord
from tn_dpo_gui.data.split import HISTORICAL_SPLITS, assert_no_test_splits
from tn_dpo_gui.encoders.text_encoder import SimpleTextEncoder
from tn_dpo_gui.utils.math_utils import cosine_similarity

from .user_history_index import UserHistoryIndex


class UserHistoryRetriever:
    def __init__(self, index: UserHistoryIndex, text_encoder: SimpleTextEncoder | None = None) -> None:
        self.index = index
        self.text_encoder = text_encoder or SimpleTextEncoder()

    def retrieve(
        self,
        user_id: str,
        query_instruction: str | None = None,
        limit: int = 5,
        allowed_splits: tuple[str, ...] = ("train", "history"),
        exclude_trajectory_ids: set[str] | None = None,
    ) -> list[TrajectoryRecord]:
        assert_no_test_splits(allowed_splits)
        allowed = {split.lower() for split in allowed_splits} or HISTORICAL_SPLITS
        excluded = {trajectory_id for trajectory_id in (exclude_trajectory_ids or set()) if trajectory_id}
        candidates = [
            record
            for record in self.index.get(user_id)
            if record.split.lower() in allowed and record.trajectory_id not in excluded
        ]
        if not query_instruction or not candidates:
            return candidates[:limit]

        query_vector = self.text_encoder.encode_text(query_instruction)
        scored = []
        for record in candidates:
            score = cosine_similarity(query_vector, self.text_encoder.encode_text(record.instruction))
            scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:limit]]
