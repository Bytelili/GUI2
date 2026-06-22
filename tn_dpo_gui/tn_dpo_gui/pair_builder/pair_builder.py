from __future__ import annotations

import math
from itertools import combinations

from tn_dpo_gui.candidates.candidate_generator import CandidateGenerator
from tn_dpo_gui.counterfactual.continuation import select_min_distance_pair
from tn_dpo_gui.counterfactual.task_equivalence import task_equivalence_distance
from tn_dpo_gui.data.schema import GUIStepExample, TrajectoryRecord
from tn_dpo_gui.encoders.state_encoder import StateEncoder
from tn_dpo_gui.encoders.text_encoder import SimpleTextEncoder
from tn_dpo_gui.encoders.trajectory_encoder import TrajectoryEncoder
from tn_dpo_gui.encoders.user_history_encoder import UserHistoryEncoder
from tn_dpo_gui.models.base_policy import BasePolicy, UniformBasePolicy
from tn_dpo_gui.retrieval.continuation_retriever import ContinuationRetriever
from tn_dpo_gui.retrieval.user_history_index import UserHistoryIndex
from tn_dpo_gui.retrieval.user_history_retriever import UserHistoryRetriever
from tn_dpo_gui.scoring.nullspace_projection import project_preference_to_task_nullspace
from tn_dpo_gui.scoring.preference_reward import preference_margin, preference_score
from tn_dpo_gui.scoring.task_reward import continuation_task_reward
from tn_dpo_gui.scoring.uncertainty import pair_uncertainty, pair_weight
from tn_dpo_gui.training.losses import capacity_slack
from tn_dpo_gui.utils.logging import get_logger

from .pair_schema import TNDPOPair


class TNDPOPairBuilder:
    def __init__(
        self,
        trajectory_records: list[TrajectoryRecord],
        history_index: UserHistoryIndex | None = None,
        text_encoder: SimpleTextEncoder | None = None,
        candidate_generator: CandidateGenerator | None = None,
        base_policy: BasePolicy | None = None,
        config: dict | None = None,
    ) -> None:
        self.config = config or {}
        self.text_encoder = text_encoder or SimpleTextEncoder(**self.config.get("encoder", {}))
        self.state_encoder = StateEncoder(self.text_encoder)
        self.trajectory_encoder = TrajectoryEncoder(self.text_encoder)
        self.user_history_encoder = UserHistoryEncoder(self.text_encoder, self.trajectory_encoder)
        self.history_index = history_index or UserHistoryIndex.build(trajectory_records)
        self.history_retriever = UserHistoryRetriever(self.history_index, self.text_encoder)
        self.continuation_retriever = ContinuationRetriever(
            trajectory_records,
            self.text_encoder,
            allowed_splits=tuple(self.config.get("allowed_history_splits", ("train", "history"))),
            fallback_current_future=bool(self.config.get("fallback_current_future", True)),
        )
        self.candidate_generator = candidate_generator or CandidateGenerator(max_candidates=int(self.config.get("max_candidates", 8)))
        self.base_policy = base_policy or UniformBasePolicy()
        self.logger = get_logger("tn_dpo_gui.pair_builder")

    def _trajectory_vector(self, example: GUIStepExample, action, continuation):
        return self.trajectory_encoder.encode_action_continuation(
            instruction=continuation.instruction or example.instruction,
            action=action,
            continuation_actions=continuation.actions,
            goal_state=continuation.goal_state or example.goal_state,
        )

    def build_pairs(self, examples: list[GUIStepExample]) -> list[TNDPOPair]:
        built_pairs: list[TNDPOPair] = []
        history_limit = int(self.config.get("history_limit", 5))
        continuation_limit = int(self.config.get("continuation_limit", 4))
        gamma = float(self.config.get("gamma", 0.75))
        lambda_u = float(self.config.get("lambda_u", 0.5))
        tau_omega = float(self.config.get("tau_omega", 0.25))
        min_null_margin = float(self.config.get("min_null_margin", 0.02))

        for example in examples:
            history = self.history_retriever.retrieve(example.user_id, example.instruction, limit=history_limit)
            history_text = self.user_history_encoder.summarize_history(history, limit=history_limit)
            user_vector = self.user_history_encoder.encode_user_history(history, example.instruction)
            user_context_text = history_text
            state_text = self.state_encoder.compose_state_text(example.instruction, example.ui_tree, example.action_history)
            candidates = self.candidate_generator.generate(example, history_records=history, base_policy=self.base_policy)
            if len(candidates) < 2:
                self.logger.warning("skip %s because only %s candidate(s) were available", example.example_id, len(candidates))
                continue

            reference_logps = self.base_policy.action_log_probs(example, candidates, history_records=history)
            candidate_continuations = {
                candidate.normalized_key(): self.continuation_retriever.retrieve(example, candidate, limit=continuation_limit)
                for candidate in candidates
            }

            raw_pairs = []
            for left_action, right_action in combinations(candidates, 2):
                left_candidates = candidate_continuations[left_action.normalized_key()]
                right_candidates = candidate_continuations[right_action.normalized_key()]
                if not left_candidates or not right_candidates:
                    continue
                left_continuation, right_continuation, task_distance = select_min_distance_pair(
                    left_candidates,
                    right_candidates,
                    task_equivalence_distance,
                )
                if left_continuation is None or right_continuation is None:
                    continue

                left_task_reward = continuation_task_reward(left_continuation)
                right_task_reward = continuation_task_reward(right_continuation)
                task_margin = left_task_reward - right_task_reward
                left_pref = preference_score(user_vector, self._trajectory_vector(example, left_action, left_continuation))
                right_pref = preference_score(user_vector, self._trajectory_vector(example, right_action, right_continuation))
                pref_margin = preference_margin(left_pref, right_pref)
                init_weight = math.exp(-gamma * task_distance)
                uncertainty = pair_uncertainty(left_candidates, right_candidates)
                raw_pairs.append(
                    {
                        "left_action": left_action,
                        "right_action": right_action,
                        "left_continuation": left_continuation,
                        "right_continuation": right_continuation,
                        "task_margin": task_margin,
                        "pref_margin": pref_margin,
                        "task_distance": task_distance,
                        "uncertainty": uncertainty,
                        "init_weight": init_weight,
                    }
                )

            if not raw_pairs:
                self.logger.warning("skip %s because no valid continuation pair could be built", example.example_id)
                continue

            rho, null_margins = project_preference_to_task_nullspace(
                [item["task_margin"] for item in raw_pairs],
                [item["pref_margin"] for item in raw_pairs],
                weights=[item["init_weight"] for item in raw_pairs],
            )

            state_pairs: list[TNDPOPair] = []
            capacity_terms: list[float] = []
            for raw_pair, null_margin in zip(raw_pairs, null_margins.tolist()):
                if abs(null_margin) < min_null_margin:
                    continue
                omega = pair_weight(null_margin, raw_pair["uncertainty"], lambda_u=lambda_u, tau_omega=tau_omega)
                final_weight = raw_pair["init_weight"] * omega
                if final_weight <= 0.0:
                    continue

                sign = 1.0 if null_margin >= 0.0 else -1.0
                chosen_action = raw_pair["left_action"] if sign > 0 else raw_pair["right_action"]
                rejected_action = raw_pair["right_action"] if sign > 0 else raw_pair["left_action"]
                oriented_task_margin = sign * raw_pair["task_margin"]
                oriented_pref_margin = sign * raw_pair["pref_margin"]
                oriented_null_margin = sign * float(null_margin)

                pair = TNDPOPair(
                    pair_id=f"{example.example_id}::{chosen_action.normalized_key()}::{rejected_action.normalized_key()}",
                    example_id=example.example_id,
                    user_id=example.user_id,
                    task_id=example.task_id,
                    state_id=example.state_id,
                    instruction=example.instruction,
                    split=example.split,
                    state_text=state_text,
                    user_context_text=user_context_text,
                    history_text=history_text,
                    chosen_action=chosen_action,
                    rejected_action=rejected_action,
                    chosen_action_text=chosen_action.to_text(),
                    rejected_action_text=rejected_action.to_text(),
                    chosen_logp_ref=reference_logps.get(chosen_action.normalized_key(), -math.log(len(candidates))),
                    rejected_logp_ref=reference_logps.get(rejected_action.normalized_key(), -math.log(len(candidates))),
                    task_margin=oriented_task_margin,
                    preference_margin=oriented_pref_margin,
                    null_margin=oriented_null_margin,
                    projection_rho=float(rho),
                    task_distance=float(raw_pair["task_distance"]),
                    uncertainty=float(raw_pair["uncertainty"]),
                    omega=float(omega),
                    init_weight=float(raw_pair["init_weight"]),
                    weight=float(final_weight),
                    candidate_count=len(candidates),
                    metadata={
                        "left_action": raw_pair["left_action"].to_dict(),
                        "right_action": raw_pair["right_action"].to_dict(),
                        "left_source": raw_pair["left_continuation"].source_example_id,
                        "right_source": raw_pair["right_continuation"].source_example_id,
                        "num_candidates": len(candidates),
                        "history_context": history_text,
                        "base_policy": self.base_policy.__class__.__name__,
                    },
                )
                capacity_terms.append(capacity_slack(pair.null_margin, pair.uncertainty, lambda_u=lambda_u))
                state_pairs.append(pair)

            if not state_pairs:
                self.logger.warning("skip %s because all candidate pairs were filtered by null margin threshold", example.example_id)
                continue

            gate_capacity = sum(capacity_terms) / len(capacity_terms)
            for pair in state_pairs:
                pair.gate_capacity = gate_capacity
            built_pairs.extend(state_pairs)
        return built_pairs
