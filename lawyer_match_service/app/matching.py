from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import numpy as np

from .config import Settings
from .encoder import TextEncoder
from .models import CaseCategory, CasePayload, LawyerMatch, MatchResponse, ScoreBreakdown
from .repository import LawyerRepository


class MatchingEngine:
    """
    Keeps lawyer embeddings in-memory and exposes a cosine-similarity retrieval API.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        encoder: TextEncoder,
        repository: LawyerRepository,
    ):
        self.settings = settings
        self.encoder = encoder
        self.repository = repository
        self._lawyers = []
        self._embeddings = np.empty((0, 0), dtype="float32")
        self._lock = asyncio.Lock()

    async def refresh(self) -> None:
        """Reloads lawyers from repository and recomputes embeddings."""
        async with self._lock:
            lawyers = await self.repository.load()
            if not lawyers:
                self._lawyers = []
                self._embeddings = np.empty((0, 0), dtype="float32")
                return
            texts = [lawyer.as_text_blob() for lawyer in lawyers]
            embeddings = self.encoder.encode(texts)
            self._lawyers = lawyers
            self._embeddings = embeddings

    @property
    def lawyer_count(self) -> int:
        return len(self._lawyers)

    def match(self, case: CasePayload, top_k: int | None = None) -> MatchResponse:
        if not self._lawyers:
            raise RuntimeError("No lawyers loaded. Call refresh() before matching.")

        query_vec = self.encoder.encode([case.as_text_blob()])[0]
        similarities = self._embeddings @ query_vec
        top_k = top_k or self.settings.default_top_k

        ranked_indices = np.argsort(similarities)[::-1][: min(top_k * 2, len(similarities))]
        matches: list[LawyerMatch] = []

        for idx in ranked_indices:
            lawyer = self._lawyers[idx]
            sim = float(similarities[idx])
            if sim < self.settings.min_score_threshold:
                continue
            breakdown = self._build_breakdown(case, lawyer, sim)
            final_score = breakdown.similarity + breakdown.specialization_bonus + breakdown.service_bonus
            matches.append(
                LawyerMatch(
                    lawyer=lawyer,
                    score=round(final_score, 4),
                    breakdown=breakdown,
                )
            )
            if len(matches) >= top_k:
                break

        return MatchResponse(
            case_id=case.case_id,
            generated_at=datetime.now(timezone.utc),
            lawyers=matches,
        )

    def _build_breakdown(self, case: CasePayload, lawyer, similarity: float) -> ScoreBreakdown:
        category_value = (
            case.category.value if isinstance(case.category, CaseCategory) else case.category
        )
        specialization_bonus = (
            self.settings.specialization_weight
            if lawyer.matches_category(str(category_value))
            else 0.0
        )
        service_bonus = (
            self.settings.service_weight
            if case.service and case.service.lower() in [s.lower() for s in lawyer.services]
            else 0.0
        )
        return ScoreBreakdown(
            similarity=round(similarity, 4),
            specialization_bonus=round(specialization_bonus, 4),
            service_bonus=round(service_bonus, 4),
        )
