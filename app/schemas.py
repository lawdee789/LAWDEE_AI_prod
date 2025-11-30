from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class RecommendationRequest(BaseModel):
    description: str = Field(..., min_length=10)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @validator("description")
    def description_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Description must contain non-whitespace characters")
        return value


class RecommendationResponseLawyer(BaseModel):
    lawyer_id: str
    name: str
    summary: str
    civil_case_specialization: List[str]
    criminal_case_specialization: List[str]
    description: str
    score: float
    metadata: Dict[str, Any]


class RecommendationResponse(BaseModel):
    success: bool = True
    count: int
    data: List[RecommendationResponseLawyer]
    metadata: Dict[str, Any]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
