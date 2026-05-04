from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

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
        rating_score_weight: float = 0.10,
        max_rating: float = 5.0,
    ) -> None:
        self.model_name = model_name
        self.lawyer_data_source = lawyer_data_source
        self.default_top_k = default_top_k
        self.data_fetch_timeout = data_fetch_timeout
        self.rating_score_weight = max(0.0, rating_score_weight)
        self.max_rating = max(0.1, max_rating)

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

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Text must not be empty")
        if self._model is None:
            self._warm_up()
        model = self._model
        if model is None:
            raise RuntimeError("SentenceTransformer model is not initialized")
        vector = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return [float(item) for item in vector.tolist()]

    def embed_lawyer(self, lawyer: Dict[str, Any]) -> List[float]:
        document = self._lawyer_to_document(lawyer)
        if not document.strip():
            raise ValueError("Lawyer profile has no text available for embedding")
        return self.embed_text(document)

    @classmethod
    def _embedding_from_lawyers(
        cls, lawyers: List[Dict[str, Any]]
    ) -> np.ndarray | None:
        vectors: List[List[float]] = []
        expected_size: int | None = None
        for lawyer in lawyers:
            raw_vector = lawyer.get("lawyer_embedding") or lawyer.get("embedding")
            if not isinstance(raw_vector, list) or not raw_vector:
                return None
            try:
                vector = [float(item) for item in raw_vector]
            except (TypeError, ValueError):
                return None
            if expected_size is None:
                expected_size = len(vector)
            elif len(vector) != expected_size:
                return None
            vectors.append(vector)

        if not vectors:
            return None
        return np.array(vectors, dtype=float)

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
    def _normalize_filter_token(cls, value: Any) -> str | None:
        text = cls._stringify(value)
        if not text:
            return None
        normalized = re.sub(r"[\s\-]+", "_", text.strip().lower())
        normalized = re.sub(r"_+", "_", normalized)
        return normalized.strip("_") or None

    @classmethod
    def _filter_tokens_from_value(
        cls, value: Any, fields: Sequence[str] | None = None
    ) -> Set[str]:
        tokens: Set[str] = set()
        for chunk in cls._collect_text_chunks(value, fields):
            token = cls._normalize_filter_token(chunk)
            if token:
                tokens.add(token)
        return tokens

    @classmethod
    def _lawyer_filter_tokens(cls, meta: Dict[str, Any]) -> Set[str]:
        token_fields = (
            "ประเภทของงาน",
            "หัวข้อบริการ",
            "ประเภทงานที่เชี่ยวชาญ",
            "category",
            "type",
            "work_type",
            "workType",
            "job_type",
            "jobType",
            "service",
            "service_topic",
            "serviceTopic",
            "slogan",
            "summary",
        )
        tokens: Set[str] = set()
        for field in token_fields:
            tokens.update(cls._filter_tokens_from_value(meta.get(field)))

        tokens.update(cls._filter_tokens_from_value(meta.get("specialties")))
        tokens.update(
            cls._filter_tokens_from_value(meta.get("specializations"), ("specialization",))
        )
        tokens.update(cls._filter_tokens_from_value(meta.get("expertise")))
        tokens.update(cls._filter_tokens_from_value(meta.get("expertise_types")))
        tokens.update(cls._filter_tokens_from_value(meta.get("ประเภทงานที่เชี่ยวชาญ")))
        tokens.update(cls._filter_tokens_from_value(meta.get("work_types")))
        tokens.update(cls._filter_tokens_from_value(meta.get("job_types")))
        tokens.update(cls._filter_tokens_from_value(meta.get("services")))
        tokens.update(cls._filter_tokens_from_value(meta.get("service_topics")))
        tokens.update(cls._filter_tokens_from_value(meta.get("tags")))
        return tokens

    @classmethod
    def _lawyer_matches_filter(
        cls, meta: Dict[str, Any], filter_groups: Sequence[Set[str]]
    ) -> bool:
        if not filter_groups:
            return True
        lawyer_tokens = cls._lawyer_filter_tokens(meta)
        for filter_criteria in filter_groups:
            for criterion in filter_criteria:
                for lawyer_token in lawyer_tokens:
                    if criterion == lawyer_token:
                        return True
                    if criterion in lawyer_token or lawyer_token in criterion:
                        return True
        return False

    @classmethod
    def _normalize_filter_groups(cls, filter_criteria: Any) -> List[Set[str]]:
        if not filter_criteria:
            return []

        raw_groups: Iterable[Any]
        if isinstance(filter_criteria, Mapping):
            raw_groups = filter_criteria.values()
        else:
            raw_groups = [filter_criteria]

        groups: List[Set[str]] = []
        for raw_group in raw_groups:
            values = raw_group if isinstance(raw_group, (list, tuple, set)) else [raw_group]
            group = {
                token
                for token in (cls._normalize_filter_token(item) for item in values)
                if token
            }
            if group:
                groups.append(group)
        return groups

    @classmethod
    def _lawyer_to_document(cls, meta: Dict[str, Any]) -> str:
        sections: List[str] = []

        def add(value: Any) -> None:
            text = cls._stringify(value)
            if text:
                sections.append(text)

        add(meta.get("slogan"))
        add(meta.get("summary"))
        add(meta.get("description"))
        sections.extend(cls._collect_text_chunks(meta.get("ประเภทงานที่เชี่ยวชาญ")))

        return "\n".join(sections)

    @staticmethod
    def _numeric_value(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _rating_score(self, meta: Dict[str, Any]) -> float:
        for field in ("avg_rating", "rating", "review_rating", "score_rating"):
            rating = self._numeric_value(meta.get(field))
            if rating is not None:
                normalized = rating / self.max_rating
                return float(np.clip(normalized, 0.0, 1.0))
        return 0.0

    def rank(
        self,
        description: str,
        top_k: int | None = None,
        lawyers: List[Dict[str, Any]] | None = None,
        filter_criteria: Any = None,
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

        filter_groups = self._normalize_filter_groups(filter_criteria)
        if filter_groups:
            dataset = [
                lawyer
                for lawyer in dataset
                if self._lawyer_matches_filter(lawyer, filter_groups)
            ]
            if not dataset:
                return []

        stored_embeddings = self._embedding_from_lawyers(dataset)
        if stored_embeddings is not None:
            embeddings = stored_embeddings
        elif lawyers is not None or filter_groups:
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
        rating_scores = np.array([self._rating_score(lawyer) for lawyer in dataset])
        final_scores = similarities + (self.rating_score_weight * rating_scores)

        top_indices = np.argsort(final_scores)[::-1][: min(k, len(final_scores))]
        recommendations: List[Dict[str, Any]] = []
        for idx in top_indices:
            lawyer_payload = dict(dataset[int(idx)])
            lawyer_payload["similarity_score"] = round(float(similarities[int(idx)]), 4)
            lawyer_payload["rating_score"] = round(float(rating_scores[int(idx)]), 4)
            lawyer_payload["score"] = round(float(final_scores[int(idx)]), 4)
            recommendations.append(lawyer_payload)
        return recommendations

    def reload(self) -> None:
        """Reload lawyer metadata and refresh cached embeddings."""
        self._warm_up()
