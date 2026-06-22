from __future__ import annotations

from tn_dpo_gui.data.action_normalizer import action_texts
from tn_dpo_gui.data.schema import GUIStepExample

from .text_encoder import SimpleTextEncoder


class StateEncoder:
    def __init__(self, text_encoder: SimpleTextEncoder) -> None:
        self.text_encoder = text_encoder

    def compose_state_text(
        self,
        instruction: str,
        ui_tree: str | None = None,
        action_history: list | None = None,
    ) -> str:
        history_text = " | ".join(action_texts(action_history or [])) or "none"
        ui_text = (ui_tree or "unknown_ui").strip()
        return f"instruction: {instruction}\nui: {ui_text}\naction_history: {history_text}"

    def encode_example(self, example: GUIStepExample):
        text = self.compose_state_text(example.instruction, example.ui_tree, example.action_history)
        return self.text_encoder.encode_text(text)
