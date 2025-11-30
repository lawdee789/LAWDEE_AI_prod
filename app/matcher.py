"""Sentence Transformer powered semantic matcher."""

from __future__ import annotations

import asyncio
from typing import List, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from . import schemas


class SemanticMatcher:
    """Thin wrapper over SentenceTransformer with async helpers."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._model_lock = asyncio.Lock()

    async def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            async with self._model_lock:
                if self._model is None:
                    loop = asyncio.get_running_loop()
                    self._model = await loop.run_in_executor(
                        None, lambda: SentenceTransformer(self.model_name)
                    )
        return self._model

    async def warm_up(self) -> None:
        """Preload the transformer weights."""

        await self._ensure_model()

    async def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        model = await self._ensure_model()
        loop = asyncio.get_running_loop()
        vectors = await loop.run_in_executor(
            None, lambda: model.encode(list(texts), normalize_embeddings=True)
        )
        return np.asarray(vectors, dtype=np.float32)

    async def embed_lawyer(self, profile: schemas.LawyerProfile) -> List[float]:
        """Return normalized embedding suitable for persistence."""

        vector = await self._encode_texts([profile.as_corpus()])
        return vector[0].tolist()

    async def embed_case(self, case: schemas.CasePayload) -> np.ndarray:
        """Return normalized embedding for the incoming case."""

        vector = await self._encode_texts([case.as_corpus()])
        return vector[0]

    def rank(
        self,
        case_vector: np.ndarray,
        records: Sequence[schemas.LawyerRecord],
        top_k: int,
    ) -> List[Tuple[schemas.LawyerRecord, float]]:
        """Compute cosine similarity and return ordered matches."""

        if not records:
            return []

        matrix = np.stack(
            [np.asarray(record.embedding, dtype=np.float32) for record in records]
        )
        similarities = matrix @ case_vector.astype(np.float32)
        limit = min(top_k, len(records))
        top_indices = np.argsort(similarities)[::-1][:limit]
        return [
            (records[index], float(similarities[index])) for index in top_indices.tolist()
        ]

