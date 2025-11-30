"""Data transfer objects for the service."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class LawyerProfile(BaseModel):
    """Normalized lawyer profile coming from the existing platform."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    lawyer_id: str = Field(..., alias="user_id", description="Primary identifier")
    name: str = Field(..., description="Display name taken from the user record")
    slogan: str = Field(..., description="Short elevator pitch")
    summary: str = Field(..., description="Concise summary shown on cards")
    lawfirm_name: str = Field(..., description="Law firm or chamber name")
    consult_min_price: float = Field(..., ge=0)
    consult_max_price: float = Field(..., ge=0)
    document_delivery_min_price: Optional[float] = Field(default=None, ge=0)
    document_delivery_max_price: Optional[float] = Field(default=None, ge=0)
    education: Optional[str] = None
    description: str = Field(..., description="Long-form profile text")
    civil_case_specialization: List[str] = Field(default_factory=list)
    criminal_case_specialization: List[str] = Field(default_factory=list)
    has_lawyer_license: bool = Field(default=False)
    is_verified_by_court: bool = Field(default=False)
    verification_docs: List[str] = Field(default_factory=list)
    avg_rating: float = Field(default=0, ge=0, le=5)
    review_count: int = Field(default=0, ge=0)
    office_location: Optional[str] = Field(
        default=None, description="Free-form location string for ranking context"
    )
    phone_number: Optional[str] = None
    languages: List[str] = Field(default_factory=list)
    years_of_experience: Optional[int] = Field(default=None, ge=0, le=80)

    def as_corpus(self) -> str:
        """Combine relevant profile fields into a single searchable string."""

        parts = [
            self.name,
            self.slogan,
            self.summary,
            self.description,
            f"Firm: {self.lawfirm_name}",
            f"Education: {self.education}" if self.education else "",
            f"Office: {self.office_location}" if self.office_location else "",
            "Civil: " + ", ".join(self.civil_case_specialization)
            if self.civil_case_specialization
            else "",
            "Criminal: " + ", ".join(self.criminal_case_specialization)
            if self.criminal_case_specialization
            else "",
            "Languages: " + ", ".join(self.languages) if self.languages else "",
            f"Experience: {self.years_of_experience} years"
            if self.years_of_experience
            else "",
        ]
        return "\n".join(p.strip() for p in parts if p and p.strip())


class LawyerRecord(LawyerProfile):
    """Lawyer profile plus the vector embedding and timestamps."""

    embedding: List[float]
    created_at: datetime
    updated_at: datetime


class LawyerResponse(LawyerProfile):
    """Serializable lawyer representation returned to API clients."""

    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, record: LawyerRecord) -> "LawyerResponse":
        return cls(**record.model_dump(exclude={"embedding"}))


class CasePayload(BaseModel):
    """Incoming case payload from the Node.js backend."""

    case_id: str
    title: str
    description: str
    category: Optional[str] = None
    note: Optional[str] = None
    status: Optional[str] = None
    service: Optional[str] = None
    client_id: Optional[str] = None

    def as_corpus(self) -> str:
        parts = [
            self.title,
            self.description,
            f"Category: {self.category}" if self.category else "",
            f"Note: {self.note}" if self.note else "",
            f"Service: {self.service}" if self.service else "",
            f"Status: {self.status}" if self.status else "",
        ]
        return "\n".join(p.strip() for p in parts if p and p.strip())


class MatchRequest(BaseModel):
    """Request body for semantic search."""

    case: CasePayload
    top_k: int = Field(default=5, ge=1, le=20)


class MatchResult(BaseModel):
    """Single ranked lawyer result."""

    score: float = Field(..., ge=-1, le=1)
    lawyer: LawyerResponse


class MatchResponse(BaseModel):
    """Response body for `/match-lawyers`."""

    case_id: str
    top_k: int
    results: List[MatchResult]
    computed_at: datetime


class LawyerListResponse(BaseModel):
    """Response wrapper for `/lawyers`."""

    count: int
    data: List[LawyerResponse]


class HealthResponse(BaseModel):
    """Simple shape for health probes."""

    status: str
    model_name: str
    total_lawyers: int

