from __future__ import annotations

from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


class TextEncoder:
    def __init__(self, model_name: str, normalize: bool = True):
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")
