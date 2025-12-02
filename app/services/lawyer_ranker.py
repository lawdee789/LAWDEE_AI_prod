from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import requests
from sentence_transformers import SentenceTransformer


class LawyerRanker:
    """Ranks lawyers against a case description using sentence transformers."""

    def __init__(
        self,
        model_name: str,
        lawyer_data_source: str,
        default_top_k: int = 5,
        data_fetch_timeout: float = 10.0,
    ) -> None:
        self.model_name = model_name
        self.lawyer_data_source = lawyer_data_source
        self.default_top_k = default_top_k
        self.data_fetch_timeout = data_fetch_timeout

        self._model: SentenceTransformer | None = None
        self._lawyers: List[Dict[str, Any]] = []
        self._embeddings: np.ndarray | None = None
        self._lock = threading.Lock()

        self._warm_up()

    def _warm_up(self) -> None:
        with self._lock:
            self._model = self._model or self._load_model()
            if self.lawyer_data_source:
                self._lawyers = self._load_lawyers()
                self._embeddings = self._embed_lawyers(self._lawyers)
            else:
                self._lawyers = []
                self._embeddings = None

    def _load_model(self) -> SentenceTransformer:
        try:
            return SentenceTransformer(self.model_name)
        except Exception as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError(f"Failed to load model '{self.model_name}': {exc}") from exc

    def _load_lawyers(self) -> List[Dict[str, Any]]:
        payload = self._load_dataset_payload()

        dataset: Any = None
        if isinstance(payload, dict):
            for key in ("data", "results", "items", "lawyers"):
                maybe = payload.get(key)
                if isinstance(maybe, list):
                    dataset = maybe
                    break
        elif isinstance(payload, list):
            dataset = payload

        if not isinstance(dataset, list) or not dataset:
            raise ValueError("Lawyer dataset must be a non-empty list of objects")
        return dataset

    @staticmethod
    def _is_remote_source(source: str) -> bool:
        normalized = source.strip().lower()
        return normalized.startswith(("http://", "https://"))

    def _load_dataset_payload(self) -> Any:
        source = self.lawyer_data_source
        if not source:
            raise ValueError("LAWYER_DATA_URL/LAWYER_DATA_PATH is not configured")
        if self._is_remote_source(source):
            return self._load_from_url(source)
        return self._load_from_file(Path(source))

    def _load_from_file(self, path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(
                f"Lawyer dataset not found at '{path}'. Did you run the setup instructions?"
            )
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_from_url(self, url: str) -> Any:
        try:
            response = requests.get(url, timeout=self.data_fetch_timeout)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network call
            raise RuntimeError(f"Failed to download lawyer dataset: {exc}") from exc
        try:
            return response.json()
        except ValueError as exc:
            raise ValueError(f"Lawyer dataset at '{url}' is not valid JSON") from exc

    def _embed_lawyers(self, lawyers: List[Dict[str, Any]]) -> np.ndarray:
        if self._model is None:
            self._warm_up()
        model = self._model
        if model is None:
            raise RuntimeError("SentenceTransformer model is not initialized")
        documents = [self._lawyer_to_document(meta) for meta in lawyers]
        if not documents:
            raise ValueError("No lawyer text documents available for embedding")
        return model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)

    @staticmethod
    def _stringify(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return "true" if value else "false"
        text = str(value).strip()
        return text or None

    @classmethod
    def _collect_text_chunks(
        cls, value: Any, fields: Sequence[str] | None = None
    ) -> List[str]:
        if value is None:
            return []
        items: Iterable[Any]
        if isinstance(value, list):
            items = value
        else:
            items = [value]

        chunks: List[str] = []
        for item in items:
            if isinstance(item, dict):
                keys = fields or item.keys()
                for key in keys:
                    chunk = cls._stringify(item.get(key))
                    if chunk:
                        chunks.append(chunk)
            else:
                chunk = cls._stringify(item)
                if chunk:
                    chunks.append(chunk)
        return chunks

    @classmethod
    def _lawyer_to_document(cls, meta: Dict[str, Any]) -> str:
        user = meta.get("user") or {}
        sections: List[str] = []

        def add(value: Any) -> None:
            text = cls._stringify(value)
            if text:
                sections.append(text)

        add(meta.get("lawyer_id"))
        add(meta.get("name") or user.get("name"))
        add(meta.get("slogan"))
        add(meta.get("summary"))
        add(meta.get("description"))
        add(meta.get("note"))
        add(meta.get("lawfirm_name") or meta.get("law_firm"))
        add(meta.get("education"))
        add(meta.get("account_status"))
        add(f"Service scope: {meta.get('service')}" if meta.get("service") else None)

        numeric_fields = {
            "cases_closed_count": "cases closed",
            "avg_rating": "average rating",
            "consult_min_price": "consult min price",
            "consult_max_price": "consult max price",
            "document_delivery_min_price": "document delivery min price",
            "document_delivery_max_price": "document delivery max price",
        }
        for field, label in numeric_fields.items():
            value = meta.get(field)
            if value is not None:
                add(f"{label}: {value}")

        add(
            f"has lawyer license: {meta.get('has_lawyer_license')}"
            if meta.get("has_lawyer_license") is not None
            else None
        )
        add(
            f"is verified by court: {meta.get('is_verified_by_court')}"
            if meta.get("is_verified_by_court") is not None
            else None
        )

        sections.extend(cls._collect_text_chunks(meta.get("specialties")))
        sections.extend(
            cls._collect_text_chunks(meta.get("specializations"), ("specialization",))
        )
        sections.extend(cls._collect_text_chunks(meta.get("languages")))
        sections.extend(cls._collect_text_chunks(meta.get("tags")))
        sections.extend(cls._collect_text_chunks(meta.get("awards")))
        sections.extend(
            cls._collect_text_chunks(meta.get("achievements"), ("title", "description"))
        )
        sections.extend(
            cls._collect_text_chunks(
                meta.get("verification_docs"),
                ("issuer", "doc_number", "docs", "title"),
            )
        )
        sections.extend(
            cls._collect_text_chunks(
                meta.get("cases"), ("title", "category", "status", "service")
            )
        )
        sections.extend(
            cls._collect_text_chunks(
                user,
                (
                    "email",
                    "tel",
                    "role",
                    "note",
                    "street",
                    "district",
                    "city",
                    "province",
                    "zipcode",
                ),
            )
        )

        return "\n".join(sections)

    def rank(
        self,
        description: str,
        top_k: int | None = None,
        lawyers: List[Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        if not description or not description.strip():
            raise ValueError("Case description must not be empty")

        k = top_k or self.default_top_k
        if k <= 0:
            raise ValueError("top_k must be a positive integer")

        if self._model is None:
            self._warm_up()

        dataset = lawyers or self._lawyers
        if not dataset:
            raise ValueError(
                "No lawyer candidates available. Provide `lawyers` in the request payload "
                "or configure LAWYER_DATA_URL/LAWYER_DATA_PATH."
            )

        if lawyers is not None:
            embeddings = self._embed_lawyers(dataset)
        else:
            embeddings = self._embeddings
            if embeddings is None:
                embeddings = self._embed_lawyers(dataset)
                self._embeddings = embeddings

        query_vector = self._model.encode(
            description, convert_to_numpy=True, normalize_embeddings=True
        )

        similarities = embeddings @ query_vector

        top_indices = np.argsort(similarities)[::-1][: min(k, len(similarities))]
        recommendations: List[Dict[str, Any]] = []
        for idx in top_indices:
            lawyer_payload = dict(dataset[int(idx)])
            lawyer_payload["score"] = round(float(similarities[int(idx)]), 4)
            recommendations.append(lawyer_payload)
        return recommendations

    def reload(self) -> None:
        """Reload lawyer metadata and refresh cached embeddings."""
        self._warm_up()
