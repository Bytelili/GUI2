from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class BaseEncoder(ABC):
    @property
    @abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]
