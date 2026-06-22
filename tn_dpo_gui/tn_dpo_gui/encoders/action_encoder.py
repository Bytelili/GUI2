from __future__ import annotations

from tn_dpo_gui.data.action_normalizer import normalize_action

from .text_encoder import SimpleTextEncoder


class ActionEncoder:
    def __init__(self, text_encoder: SimpleTextEncoder) -> None:
        self.text_encoder = text_encoder

    def encode(self, action) -> object:
        return self.text_encoder.encode_text(normalize_action(action).to_text())
