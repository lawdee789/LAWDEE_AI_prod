"""Pydantic schemas used across the service."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class LawyerProfile(BaseModel):
    """Normalized representation of a lawyer profile."""

    lawyer_id: str = Field(..., description="Primary identifier for a lawyer record.")
    name: Optional[str] = None
    summary: str = ""
    description: str = ""
    lawfirm_name: Optional[str] = None
    consult_min_price: Optional[float] = None
    consult_max_price: Optional[float] = None
    document_delivery_min_price: Optional[float] = None
    document_delivery_max_price: Optional[float] = None
    civil_case_specialization: List[str] = Field(default_factory=list)
    criminal_case_specialization: List[str] = Field(default_factory=list)
    has_lawyer_license: Optional[bool] = None
    is_verified_by_court: Optional[bool] = None
    avg_rating: Optional[float] = None
    review_count: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        populate_by_name = True
        extra = "allow"

    @staticmethod
    def _extract_specializations(value: Any) -> List[str]:
        if isinstance(value, list):
            normalized: List[str] = []
            for item in value:
                if isinstance(item, str):
                    normalized.append(item)
                elif isinstance(item, dict):
                    specialization = item.get("specialization") or item.get("name")
                    if specialization:
                        normalized.append(str(specialization))
            return normalized
        return []

    @staticmethod
    def _extract_rate(value: Any, key: str) -> Optional[float]:
        if isinstance(value, dict):
            raw = value.get(key)
            if raw is None:
                return None
            try:
                return float(raw)
            except (TypeError, ValueError):
                return None
        return None

    @model_validator(mode="before")
    @classmethod
    def harmonize_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        if "lawyer_id" not in data:
            for candidate_key in ("_id", "id", "user_id"):
                if candidate_key in data:
                    data["lawyer_id"] = str(data[candidate_key])
                    break

        if "consultationRate" in data:
            rate = data["consultationRate"]
            data.setdefault("consult_min_price", cls._extract_rate(rate, "min"))
            data.setdefault("consult_max_price", cls._extract_rate(rate, "max"))

        if "documentDeliveryRate" in data:
            rate = data["documentDeliveryRate"]
            data.setdefault("document_delivery_min_price", cls._extract_rate(rate, "min"))
            data.setdefault("document_delivery_max_price", cls._extract_rate(rate, "max"))

        if "civilCase_specialized" in data and "civil_case_specialization" not in data:
            data["civil_case_specialization"] = cls._extract_specializations(data["civilCase_specialized"])

        if "criminalCase_specialized" in data and "criminal_case_specialization" not in data:
            data["criminal_case_specialization"] = cls._extract_specializations(data["criminalCase_specialized"])

        return data

    @field_validator("civil_case_specialization", "criminal_case_specialization", mode="before")
    @classmethod
    def ensure_list(cls, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if value is None:
            return []
        return [str(value)]


class CasePayload(BaseModel):
    case_id: str = Field(..., description="Primary case identifier from the Express backend.")
    title: str
    description: str
    category: Optional[str] = None
    note: Optional[str] = None


class MatchRequest(BaseModel):
    case: CasePayload
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    lawyers: Optional[List[LawyerProfile]] = None


class RankedLawyer(BaseModel):
    lawyer_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    reasons: List[str]
    lawyer: LawyerProfile


class MatchResponse(BaseModel):
    success: bool
    count: int
    case_id: str
    lawyers: List[RankedLawyer]
    metadata: Dict[str, Any]
