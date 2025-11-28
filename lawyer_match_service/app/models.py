from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CaseCategory(str, Enum):
    CIVIL = "CIVIL"
    CRIMINAL = "CRIMINAL"
    LABOR = "LABOR"
    FAMILY = "FAMILY"
    TAX = "TAX"
    COMMERCIAL = "COMMERCIAL"
    OTHER = "OTHER"


class CasePayload(BaseModel):
    case_id: str | None = None
    client_id: str | None = None
    title: str
    description: str
    category: CaseCategory | str = CaseCategory.OTHER
    note: str | None = None
    service: str | None = None
    metadata: dict[str, Any] | None = Field(default_factory=dict)

    def as_text_blob(self) -> str:
        """Concatenate the case details into a single text block."""
        parts: list[str] = []
        for chunk in (
            self.title,
            str(self.category.value if isinstance(self.category, CaseCategory) else self.category),
            self.description,
            self.service or "",
            self.note or "",
        ):
            if chunk:
                parts.append(chunk.strip())
        if self.metadata:
            meta_text = " ".join(
                f"{key}: {value}" for key, value in self.metadata.items() if value
            )
            if meta_text:
                parts.append(meta_text)
        return " \n ".join(parts)


class LawyerProfile(BaseModel):
    lawyer_id: str
    name: str
    slogan: str | None = None
    summary: str = ""
    description: str | None = None
    lawfirm_name: str | None = None
    consult_min_price: float | None = None
    consult_max_price: float | None = None
    document_delivery_min_price: float | None = None
    document_delivery_max_price: float | None = None
    civil_case_specialization: list[str] = Field(default_factory=list)
    criminal_case_specialization: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    has_lawyer_license: bool | None = None
    is_verified_by_court: bool | None = None
    languages: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    notable_cases: list[str] = Field(default_factory=list)
    avg_rating: float | None = None
    review_count: int | None = None
    updated_at: datetime | None = None
    raw_payload: dict[str, Any] | None = None

    def as_text_blob(self) -> str:
        """Flatten the lawyer profile into text for embedding."""
        parts = [
            self.name,
            self.slogan or "",
            self.summary,
            self.description or "",
            " ".join(self.civil_case_specialization),
            " ".join(self.criminal_case_specialization),
            " ".join(self.services),
            " ".join(self.education),
            " ".join(self.notable_cases),
        ]
        return " \n ".join(chunk for chunk in parts if chunk)

    def matches_category(self, category: str) -> bool:
        normalized = category.lower()
        if normalized == "criminal":
            return bool(self.criminal_case_specialization)
        if normalized == "civil":
            return bool(self.civil_case_specialization)
        return bool(self.civil_case_specialization or self.criminal_case_specialization)


class MatchRequest(BaseModel):
    case: CasePayload
    top_k: int | None = Field(default=None, ge=1, le=50)
    allow_inactive: bool = False


class ScoreBreakdown(BaseModel):
    similarity: float
    specialization_bonus: float
    service_bonus: float


class LawyerMatch(BaseModel):
    lawyer: LawyerProfile
    score: float
    breakdown: ScoreBreakdown


class MatchResponse(BaseModel):
    case_id: str | None = None
    generated_at: datetime
    lawyers: list[LawyerMatch]
