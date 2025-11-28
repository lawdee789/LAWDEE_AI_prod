from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import Settings
from .repository import FileLawyerRepository
from .schemas import CasePayload, LawyerPayload, MatchRequest, MatchResponse, MatchCandidate


class LawyerMatchEngine:
    def __init__(self, settings: Settings, repository: FileLawyerRepository) -> None:
        self.settings = settings
        self.repository = repository
        self._lawyers: List[LawyerPayload] = []
        self._id_to_index: Dict[str, int] = {}
        self._embeddings: Optional[np.ndarray] = None
        self._model: Optional[SentenceTransformer] = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.settings.model_name)
        return self._model

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._get_model()
        return model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=self.settings.normalize_embeddings,
            show_progress_bar=False,
        )

    def _build_embeddings(self) -> None:
        if not self._lawyers:
            self._embeddings = np.empty((0, 0))
            return
        corpus = [lawyer.compound_text() for lawyer in self._lawyers]
        self._embeddings = self._encode(corpus)
        self._id_to_index = {lawyer.lawyer_id: idx for idx, lawyer in enumerate(self._lawyers)}

    def load_initial_state(self) -> None:
        self._lawyers = self.repository.get_all()
        self._build_embeddings()

    def upsert_lawyers(self, lawyers: Iterable[LawyerPayload]) -> List[LawyerPayload]:
        self._lawyers = self.repository.upsert(lawyers)
        self._build_embeddings()
        return self._lawyers

    @property
    def lawyer_count(self) -> int:
        return len(self._lawyers)

    def _select_indices(self, allowed_ids: Optional[Sequence[str]]) -> np.ndarray:
        if not allowed_ids:
            return np.arange(len(self._lawyers))
        valid_indices = [self._id_to_index[idx] for idx in allowed_ids if idx in self._id_to_index]
        return np.array(valid_indices, dtype=int)

    def _match_scores(self, case_vector: np.ndarray, indices: np.ndarray) -> np.ndarray:
        if self._embeddings is None or self._embeddings.size == 0:
            return np.array([])
        vectors = self._embeddings[indices]
        scores = vectors @ case_vector
        return scores

    def rank(self, request: MatchRequest) -> MatchResponse:
        if self._embeddings is None:
            raise RuntimeError("Engine not initialized. Call load_initial_state first.")

        case_vector = self._encode([request.case.compound_text()])[0]
        subset_indices = self._select_indices(request.allowed_lawyer_ids)
        if subset_indices.size == 0:
            return MatchResponse(
                case_id=request.case.case_id,
                top_k=request.top_k or self.settings.top_k,
                count=0,
                results=[],
            )

        scores = self._match_scores(case_vector, subset_indices)
        top_k = min(request.top_k or self.settings.top_k, subset_indices.size)
        if top_k == 0:
            return MatchResponse(
                case_id=request.case.case_id,
                top_k=top_k,
                count=0,
                results=[],
            )

        best_indices = np.argpartition(-scores, kth=top_k - 1)[:top_k]
        ordered = best_indices[np.argsort(-scores[best_indices])]

        results: List[MatchCandidate] = []
        for offset in ordered:
            lawyer_idx = subset_indices[offset]
            lawyer = self._lawyers[lawyer_idx]
            score = float(scores[offset])
            debug = None
            if request.include_vectors:
                debug = {
                    "lawyer_vector": self._embeddings[lawyer_idx].tolist(),
                    "case_vector": case_vector.tolist(),
                }
            results.append(
                MatchCandidate(
                    lawyer_id=lawyer.lawyer_id,
                    score=round(score, 4),
                    lawyer=lawyer,
                    debug=debug,
                )
            )

        return MatchResponse(
            case_id=request.case.case_id,
            top_k=top_k,
            count=len(results),
            results=results,
        )
