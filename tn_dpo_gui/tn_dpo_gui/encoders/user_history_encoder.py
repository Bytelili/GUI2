from __future__ import annotations

import numpy as np

from tn_dpo_gui.data.schema import TrajectoryRecord
from tn_dpo_gui.utils.math_utils import cosine_similarity, l2_normalize, softmax

from .text_encoder import SimpleTextEncoder
from .trajectory_encoder import TrajectoryEncoder


class UserHistoryEncoder:
    def __init__(self, text_encoder: SimpleTextEncoder, trajectory_encoder: TrajectoryEncoder) -> None:
        self.text_encoder = text_encoder
        self.trajectory_encoder = trajectory_encoder

    @property
    def output_dim(self) -> int:
        return self.text_encoder.output_dim

    def zero_vector(self) -> np.ndarray:
        return np.zeros(self.output_dim, dtype=np.float32)

    def summarize_history(self, history: list[TrajectoryRecord], limit: int = 5) -> str:
        lines = []
        for record in history[:limit]:
            action_blob = " -> ".join(action.to_text() for action in record.actions[:4]) or "empty"
            lines.append(f"{record.task_id}: {record.instruction} [{action_blob}]")
        return "\n".join(lines) or "no_user_history"

    def encode_user_history(self, history: list[TrajectoryRecord], query_instruction: str | None = None) -> np.ndarray:
        if not history:
            return self.zero_vector()
        trajectory_vectors = np.asarray([self.trajectory_encoder.encode_record(record) for record in history], dtype=np.float32)
        if query_instruction:
            query_vector = self.text_encoder.encode_text(query_instruction)
            instruction_vectors = self.text_encoder.encode_texts([record.instruction for record in history])
            similarities = [cosine_similarity(query_vector, vector) for vector in instruction_vectors]
            weights = softmax(similarities)
        else:
            weights = np.full(len(history), 1.0 / len(history), dtype=np.float32)
        pooled = np.sum(trajectory_vectors * weights[:, None], axis=0)
        return l2_normalize(pooled.astype(np.float32))
