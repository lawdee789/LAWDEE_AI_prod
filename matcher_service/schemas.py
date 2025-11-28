from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LawyerPayload(BaseModel):
    lawyer_id: str = Field(..., description="Primary identifier from the Express backend")
    name: Optional[str] = None
    summary: Optional[str] = None
    slogan: Optional[str] = None
    description: Optional[str] = None
    civil_case_specialization: List[str] = Field(default_factory=list)
    criminal_case_specialization: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    consult_min_price: Optional[float] = None
    consult_max_price: Optional[float] = None
    document_delivery_min_price: Optional[float] = None
    document_delivery_max_price: Optional[float] = None
    tags: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)

    def compound_text(self) -> str:
        """Concatenate important lawyer attributes for embedding."""

        blocks = [
            self.name or "",
            self.summary or "",
            self.slogan or "",
            self.description or "",
            " ".join(self.civil_case_specialization),
            " ".join(self.criminal_case_specialization),
            " ".join(self.languages),
            " ".join(self.tags),
        ]
        return " \n ".join(filter(None, (block.strip() for block in blocks)))


class CasePayload(BaseModel):
    case_id: Optional[str] = None
    title: str
    category: Optional[str] = None
    description: str
    note: Optional[str] = None
    service: Optional[str] = None

    def compound_text(self) -> str:
        blocks = [self.title, self.category or "", self.description, self.note or "", self.service or ""]
        return " \n ".join(filter(None, (block.strip() for block in blocks)))


class MatchRequest(BaseModel):
    case: CasePayload
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    allowed_lawyer_ids: Optional[List[str]] = Field(
        default=None,
        description="Restrict search to these lawyer ids if provided",
    )
    include_vectors: bool = Field(
        default=False,
        description="Return raw embedding vectors for debugging only.",
    )


class MatchCandidate(BaseModel):
    lawyer_id: str
    score: float
    lawyer: LawyerPayload
    debug: Optional[Dict[str, Any]] = None


class MatchResponse(BaseModel):
    case_id: Optional[str] = None
    top_k: int
    count: int
    results: List[MatchCandidate]


class BulkUpsertRequest(BaseModel):
    lawyers: List[LawyerPayload]
