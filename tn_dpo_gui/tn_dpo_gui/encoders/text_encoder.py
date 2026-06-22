from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

from tn_dpo_gui.utils.math_utils import l2_normalize

from .base import BaseEncoder

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


class SimpleTextEncoder(BaseEncoder):
    """Deterministic text encoder with an optional sentence-transformer backend."""

    def __init__(
        self,
        dim: int = 256,
        backend: str = "hashing",
        sentence_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.dim = int(dim)
        self.backend = backend
        self.sentence_model_name = sentence_model_name
        self._sentence_model = None

    @property
    def output_dim(self) -> int:
        return self.dim

    def _load_sentence_model(self) -> None:
        if self._sentence_model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._sentence_model = SentenceTransformer(self.sentence_model_name)

    def _tokenize(self, text: str) -> list[str]:
        tokens = TOKEN_PATTERN.findall((text or "").lower())
        if not tokens:
            tokens = [chunk for chunk in (text or "").lower().split() if chunk]
        return tokens

    def _hash_encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row_index, text in enumerate(texts):
            tokens = self._tokenize(text)
            if not tokens:
                continue
            for token in tokens:
                digest = hashlib.md5(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row_index, index] += sign
            matrix[row_index] = l2_normalize(matrix[row_index])
        return matrix

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if self.backend == "sentence_transformers":
            try:
                self._load_sentence_model()
                encoded = self._sentence_model.encode(texts, normalize_embeddings=True)
                return np.asarray(encoded, dtype=np.float32)
            except Exception:
                return self._hash_encode(texts)
        return self._hash_encode(texts)

    def get_config(self) -> dict[str, str | int]:
        return {"dim": self.dim, "backend": self.backend, "sentence_model_name": self.sentence_model_name}
