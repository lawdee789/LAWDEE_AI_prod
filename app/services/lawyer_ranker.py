from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


class LawyerRanker:
    """Ranks lawyers against a case description using sentence transformers."""

    def __init__(self, model_name: str, lawyer_data_path: str, default_top_k: int = 5) -> None:
        self.model_name = model_name
        self.lawyer_data_path = Path(lawyer_data_path)
        self.default_top_k = default_top_k

        self._model: SentenceTransformer | None = None
        self._lawyers: List[Dict[str, Any]] = []
        self._embeddings: np.ndarray | None = None
        self._lock = threading.Lock()

        self._warm_up()

    def _warm_up(self) -> None:
        with self._lock:
            self._model = self._model or self._load_model()
            self._lawyers = self._load_lawyers()
            documents = [self._lawyer_to_document(meta) for meta in self._lawyers]
            self._embeddings = self._model.encode(
                documents, convert_to_numpy=True, normalize_embeddings=True
            )

    def _load_model(self) -> SentenceTransformer:
        try:
            return SentenceTransformer(self.model_name)
        except Exception as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError(f"Failed to load model '{self.model_name}': {exc}") from exc

    def _load_lawyers(self) -> List[Dict[str, Any]]:
        if not self.lawyer_data_path.exists():
            raise FileNotFoundError(
                f"Lawyer dataset not found at '{self.lawyer_data_path}'. "
                "Did you run the setup instructions?"
            )
        with self.lawyer_data_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list) or not data:
            raise ValueError("Lawyer dataset must be a non-empty list")
        return data

    @staticmethod
    def _lawyer_to_document(meta: Dict[str, Any]) -> str:
        specialties = meta.get("specialties") or []
        tags = meta.get("tags") or []
        sections = [
            meta.get("name", ""),
            meta.get("summary", ""),
            meta.get("description", ""),
            ", ".join(meta.get("languages", [])),
            ", ".join(specialties),
            ", ".join(meta.get("awards", [])),
            ", ".join(tags),
        ]
        return "\n".join(filter(None, sections))

    def rank(self, description: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        if not description or not description.strip():
            raise ValueError("Case description must not be empty")

        k = top_k or self.default_top_k
        if k <= 0:
            raise ValueError("top_k must be a positive integer")

        if self._embeddings is None or self._model is None:
            self._warm_up()

        query_vector = self._model.encode(
            description, convert_to_numpy=True, normalize_embeddings=True
        )
        embeddings = self._embeddings
        if embeddings is None:
            raise RuntimeError("Embeddings are not initialized")
        similarities = embeddings @ query_vector

        top_indices = np.argsort(similarities)[::-1][: min(k, len(similarities))]
        recommendations: List[Dict[str, Any]] = []
        for idx in top_indices:
            lawyer_payload = dict(self._lawyers[int(idx)])
            lawyer_payload["score"] = round(float(similarities[int(idx)]), 4)
            recommendations.append(lawyer_payload)
        return recommendations

    def reload(self) -> None:
        """Reload lawyer metadata and refresh cached embeddings."""
        self._warm_up()
