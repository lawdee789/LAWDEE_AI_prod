from __future__ import annotations

import threading
from typing import Iterable, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Wrapper around SentenceTransformer with thread-safe encoding."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model_lock = threading.Lock()
        self._model = SentenceTransformer(model_name, trust_remote_code=True)

    def encode(self, texts: Sequence[str] | Iterable[str]) -> np.ndarray:
        cleaned_texts = [text.strip() for text in texts]
        with self._model_lock:
            return self._model.encode(
                cleaned_texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
