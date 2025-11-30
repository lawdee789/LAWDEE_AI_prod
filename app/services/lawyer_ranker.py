from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List

import numpy as np

from ..repository.lawyer_repository import LawyerRepository
from .embedding_service import EmbeddingService


@dataclass(frozen=True)
class RankedLawyer:
    lawyer_id: str
    name: str
    summary: str
    civil_case_specialization: List[str]
    criminal_case_specialization: List[str]
    description: str
    score: float
    metadata: Dict[str, Any]


class LawyerRanker:
    """Compute semantic similarity between case descriptions and lawyers."""

    def __init__(
        self,
        repository: LawyerRepository,
        embedder: EmbeddingService,
        min_score: float = 0.0,
    ) -> None:
        self.repository = repository
        self.embedder = embedder
        self.min_score = min_score

        self._lock = Lock()
        self._lawyer_profiles: List[Dict[str, Any]] = []
        self._lawyer_embeddings: np.ndarray | None = None

        self._build_index()

    def _build_index(self) -> None:
        lawyers = self.repository.get_all()
        if not lawyers:
            self._lawyer_embeddings = np.empty((0, 0))
            self._lawyer_profiles = []
            return

        texts = [self._profile_to_text(lawyer) for lawyer in lawyers]
        embeddings = self.embedder.encode(texts)

        with self._lock:
            self._lawyer_profiles = lawyers
            self._lawyer_embeddings = embeddings

    def refresh(self) -> None:
        """Reload lawyers from the repository and rebuild the embedding index."""

        self.repository.reload()
        self._build_index()

    def recommend(
        self, case_description: str, top_k: int, min_score: float | None = None
    ) -> List[RankedLawyer]:
        if self._lawyer_embeddings is None or not len(self._lawyer_profiles):
            return []

        cleaned_description = case_description.strip()
        limit = max(1, min(top_k, len(self._lawyer_profiles)))
        query_embedding = self.embedder.encode([cleaned_description])[0]

        embeddings = self._lawyer_embeddings
        scores = embeddings @ query_embedding
        ranked_indices = np.argsort(scores)[::-1][:limit]
        effective_min_score = self.min_score if min_score is None else min_score

        recommendations: List[RankedLawyer] = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score < effective_min_score:
                continue
            lawyer = self._lawyer_profiles[idx]
            recommendations.append(
                RankedLawyer(
                    lawyer_id=lawyer["lawyer_id"],
                    name=lawyer["name"],
                    summary=lawyer.get("summary", ""),
                    civil_case_specialization=lawyer.get("civil_case_specialization", []),
                    criminal_case_specialization=lawyer.get(
                        "criminal_case_specialization", []
                    ),
                    description=lawyer.get("description", ""),
                    score=score,
                    metadata={
                        "lawfirm_name": lawyer.get("lawfirm_name"),
                        "consult_min_price": lawyer.get("consult_min_price"),
                        "consult_max_price": lawyer.get("consult_max_price"),
                        "document_delivery_min_price": lawyer.get(
                            "document_delivery_min_price"
                        ),
                        "document_delivery_max_price": lawyer.get(
                            "document_delivery_max_price"
                        ),
                        "years_experience": lawyer.get("years_experience"),
                        "languages": lawyer.get("languages", []),
                    },
                )
            )

        return recommendations

    @staticmethod
    def _profile_to_text(lawyer: Dict[str, Any]) -> str:
        parts = [lawyer.get("summary", ""), lawyer.get("description", "")]
        civil = ", ".join(lawyer.get("civil_case_specialization", []))
        criminal = ", ".join(lawyer.get("criminal_case_specialization", []))
        lawfirm = lawyer.get("lawfirm_name", "")
        extras = [
            f"civil: {civil}" if civil else "",
            f"criminal: {criminal}" if criminal else "",
            lawfirm,
        ]
        parts.extend(extras)
        return " \n".join(filter(None, parts))
