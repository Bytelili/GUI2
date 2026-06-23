from __future__ import annotations

from dataclasses import replace

from tn_dpo_gui.data.action_normalizer import normalize_action
from tn_dpo_gui.data.schema import GUIStepExample, TrajectoryContinuation, TrajectoryRecord
from tn_dpo_gui.data.split import assert_no_test_splits
from tn_dpo_gui.encoders.text_encoder import SimpleTextEncoder
from tn_dpo_gui.utils.math_utils import cosine_similarity


class ContinuationRetriever:
    def __init__(
        self,
        records: list[TrajectoryRecord],
        text_encoder: SimpleTextEncoder | None = None,
        allowed_splits: tuple[str, ...] = ("train", "history"),
        fallback_current_future: bool = True,
    ) -> None:
        assert_no_test_splits(allowed_splits)
        self.records = [record for record in records if record.split.lower() in {split.lower() for split in allowed_splits}]
        self.text_encoder = text_encoder or SimpleTextEncoder()
        self.fallback_current_future = fallback_current_future

    def _build_from_record(self, record: TrajectoryRecord, action_key: str, retrieval_score: float) -> list[TrajectoryContinuation]:
        continuations: list[TrajectoryContinuation] = []
        for index, record_action in enumerate(record.actions):
            if record_action.normalized_key() != action_key:
                continue
            continuations.append(
                TrajectoryContinuation(
                    source_example_id=record.trajectory_id,
                    source_action=record_action,
                    instruction=record.instruction,
                    actions=list(record.actions[index + 1 :]),
                    task_success=record.task_success,
                    progress=record.progress,
                    goal_state=record.goal_state,
                    invalid_count=record.invalid_count,
                    risk_score=record.risk_score,
                    retrieval_score=retrieval_score,
                )
            )
        return continuations

    def retrieve(self, example: GUIStepExample, action, limit: int = 4) -> list[TrajectoryContinuation]:
        normalized_action = normalize_action(action)
        action_key = normalized_action.normalized_key()
        query_vector = self.text_encoder.encode_text(example.instruction)
        candidates: list[TrajectoryContinuation] = []
        source_trajectory_id = example.source_trajectory_id or ""

        for record in self.records:
            if source_trajectory_id and record.trajectory_id == source_trajectory_id:
                continue
            instruction_score = cosine_similarity(query_vector, self.text_encoder.encode_text(record.instruction))
            same_task_bonus = 1.0 if record.task_id == example.task_id else 0.0
            exact_action_bonus = 1.0 if any(act.normalized_key() == action_key for act in record.actions) else 0.0
            retrieval_score = instruction_score + 0.5 * same_task_bonus + 0.5 * exact_action_bonus
            candidates.extend(self._build_from_record(record, action_key, retrieval_score))
            if candidates and len(candidates) >= limit * 2:
                continue

        if not candidates:
            for record in self.records:
                if source_trajectory_id and record.trajectory_id == source_trajectory_id:
                    continue
                if record.task_id != example.task_id:
                    continue
                instruction_score = cosine_similarity(query_vector, self.text_encoder.encode_text(record.instruction))
                if instruction_score <= 0.0:
                    continue
                candidates.append(
                    TrajectoryContinuation(
                        source_example_id=record.trajectory_id,
                        source_action=normalized_action,
                        instruction=record.instruction,
                        actions=list(record.actions),
                        task_success=record.task_success,
                        progress=record.progress,
                        goal_state=record.goal_state,
                        invalid_count=record.invalid_count,
                        risk_score=record.risk_score,
                        retrieval_score=0.05 + instruction_score * 0.2,
                    )
                )
                if len(candidates) >= limit:
                    break

        if (
            self.fallback_current_future
            and example.split.lower() in {"train", "history"}
            and normalized_action.normalized_key() == example.current_action.normalized_key()
        ):
            candidates.append(
                TrajectoryContinuation(
                    source_example_id=example.example_id,
                    source_action=replace(example.current_action),
                    instruction=example.instruction,
                    actions=list(example.future_trajectory),
                    task_success=example.task_success,
                    progress=example.progress,
                    goal_state=example.goal_state,
                    invalid_count=example.invalid_count,
                    risk_score=example.risk_score,
                    retrieval_score=2.0,
                )
            )

        candidates.sort(key=lambda continuation: continuation.retrieval_score, reverse=True)
        return candidates[:limit]
