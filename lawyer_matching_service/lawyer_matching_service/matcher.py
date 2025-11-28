"""Core similarity logic for ranking lawyers against a case."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import Settings, get_settings
from .schemas import CasePayload, LawyerProfile, RankedLawyer


class LawyerMatcher:
    """Computes similarity scores between a case and lawyer profiles."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._vectorizer = TfidfVectorizer(
            analyzer=self._settings.vectorizer_analyzer,
            ngram_range=(self._settings.vectorizer_ngram_min, self._settings.vectorizer_ngram_max),
            lowercase=True,
            strip_accents=None,
        )

    def rank(self, case: CasePayload, lawyers: Sequence[LawyerProfile], top_k: int) -> List[RankedLawyer]:
        if not lawyers:
            return []

        case_document = self._compose_case_document(case)
        lawyer_documents = [self._compose_lawyer_document(lawyer) for lawyer in lawyers]

        tfidf_matrix = self._vectorizer.fit_transform([*lawyer_documents, case_document])
        lawyer_vectors = tfidf_matrix[:-1]
        case_vector = tfidf_matrix[-1]

        if case_vector.nnz == 0:
            similarities = np.zeros(len(lawyers))
        else:
            similarities = cosine_similarity(lawyer_vectors, case_vector).flatten()

        results: List[RankedLawyer] = []
        for idx, (lawyer, similarity) in enumerate(zip(lawyers, similarities)):
            bonus, bonus_reasons = self._metadata_bonus(case, lawyer)
            raw_score = float(similarity + bonus)
            bounded_score = max(0.0, min(1.0, raw_score))
            reasons = [f"Text similarity score {similarity:.3f}", *bonus_reasons]
            results.append(
                RankedLawyer(
                    lawyer_id=lawyer.lawyer_id,
                    score=round(bounded_score, 4),
                    reasons=[reason for reason in reasons if reason],
                    lawyer=lawyer,
                )
            )

        results.sort(key=lambda match: match.score, reverse=True)
        return results[: min(top_k, len(results))]

    @staticmethod
    def _compose_case_document(case: CasePayload) -> str:
        parts = [case.title, case.description, case.category or "", case.note or ""]
        document = " \n ".join(part.strip() for part in parts if part)
        return document or case.case_id

    @staticmethod
    def _compose_lawyer_document(lawyer: LawyerProfile) -> str:
        specializations = [*lawyer.civil_case_specialization, *lawyer.criminal_case_specialization]
        meta_flags = []
        if lawyer.has_lawyer_license:
            meta_flags.append("licensed attorney")
        if lawyer.is_verified_by_court:
            meta_flags.append("court verified")

        parts = [
            lawyer.name or "",
            lawyer.summary,
            lawyer.description,
            lawyer.lawfirm_name or "",
            " ".join(specializations),
            " ".join(meta_flags),
        ]
        document = " \n ".join(part.strip() for part in parts if part)
        return document or lawyer.lawyer_id

    def _metadata_bonus(self, case: CasePayload, lawyer: LawyerProfile) -> Tuple[float, List[str]]:
        bonus = 0.0
        reasons: List[str] = []

        category = (case.category or "").strip().lower()
        normalized_specs = {spec.strip().lower() for spec in self._collect_specializations(lawyer)}

        if category:
            if category in normalized_specs:
                bonus += 0.07
                reasons.append(f"Specialized in {category} cases")
            elif category in {"criminal", "civil"}:
                target_specs = (
                    lawyer.criminal_case_specialization
                    if category == "criminal"
                    else lawyer.civil_case_specialization
                )
                if target_specs:
                    bonus += 0.05
                    reasons.append(f"Has {category} specialization experience")

        if lawyer.has_lawyer_license:
            bonus += 0.02
            reasons.append("Active law license")

        if lawyer.is_verified_by_court:
            bonus += 0.03
            reasons.append("Verified by court")

        if lawyer.avg_rating and lawyer.review_count:
            if lawyer.avg_rating >= 4.5 and lawyer.review_count >= 10:
                bonus += 0.01
                reasons.append("High client rating")

        return bonus, reasons

    @staticmethod
    def _collect_specializations(lawyer: LawyerProfile) -> Sequence[str]:
        return [*lawyer.civil_case_specialization, *lawyer.criminal_case_specialization]
