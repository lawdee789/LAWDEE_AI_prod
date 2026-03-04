from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .config import AppConfig
from .services.lawyer_ranker import LawyerRanker

router = APIRouter()


def _get_ranker(request: Request) -> LawyerRanker:
    ranker = request.app.state.lawyer_ranker
    if ranker is None:
        raise RuntimeError("LawyerRanker has not been initialized")
    return ranker


def _get_config(request: Request) -> AppConfig:
    return request.app.state.config


class RecommendationRequest(BaseModel):
    description: str = Field(default="")
    top_k: Optional[int] = None
    case: Optional[Dict[str, Any]] = None
    case_id: Optional[str] = None
    lawyers: Optional[List[Dict[str, Any]]] = None


@router.get("/health")
def health_check(cfg: AppConfig = Depends(_get_config)):
    return {
        "success": True,
        "message": "Lawyer recommender online",
        "model": cfg.model_name if cfg else None,
    }


def _sanitize_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip()
    return text or None


def _collect_from_nested(
    payload: Any, fields: Iterable[str], fallback: str | None = None
) -> List[str]:
    if payload is None:
        return []
    items = payload if isinstance(payload, list) else [payload]
    results: List[str] = []
    for item in items:
        if isinstance(item, dict):
            for field in fields:
                text = _sanitize_text(item.get(field))
                if text:
                    results.append(text)
        else:
            text = _sanitize_text(item)
            if text:
                results.append(text)
    if not results and fallback:
        text = _sanitize_text(fallback)
        if text:
            results.append(text)
    return results


def _case_to_prompt(case_payload: Dict[str, Any], explicit_case_id: str | None = None) -> str:
    sections: List[str] = []

    def add(label: str, value: Any) -> None:
        text = _sanitize_text(value)
        if text:
            sections.append(f"{label}: {text}")

    add("Case ID", case_payload.get("case_id") or explicit_case_id)
    add("Title", case_payload.get("title"))
    add("Category", case_payload.get("category"))
    add("Status", case_payload.get("status"))
    add("Service", case_payload.get("service"))
    add("Note", case_payload.get("note"))
    add("Case description", case_payload.get("description"))

    legal_case = case_payload.get("legal_case") or {}
    if isinstance(legal_case, dict) and legal_case:
        add("Verdict date", legal_case.get("verdict_date"))
        add("Subpoena date", legal_case.get("subpoena_date"))
        add("Is served", legal_case.get("is_served"))

    client = case_payload.get("client") or {}
    if isinstance(client, dict):
        add("Client name", client.get("name"))
        add("Client phone", client.get("tel"))

    chosen_lawyer = case_payload.get("chosen_lawyer")
    if isinstance(chosen_lawyer, dict):
        chosen_user = chosen_lawyer.get("user")
        if isinstance(chosen_user, dict):
            add("Chosen lawyer name", chosen_user.get("name"))
            add("Chosen lawyer phone", chosen_user.get("tel"))
        add("Chosen lawyer specialization", chosen_lawyer.get("slogan"))

    offered_lawyers = case_payload.get("offered_lawyers") or []
    offered_names: List[str] = []
    for candidate in offered_lawyers:
        lawyer = candidate.get("lawyer") if isinstance(candidate, dict) else {}
        user = lawyer.get("user") if isinstance(lawyer, dict) else {}
        name = (
            user.get("name")
            if isinstance(user, dict)
            else lawyer.get("name") if isinstance(lawyer, dict) else None
        )
        if name:
            offered_names.append(str(name))
    if offered_names:
        sections.append(f"Invited lawyers: {', '.join(offered_names)}")

    file_names = _collect_from_nested(case_payload.get("files"), ("file",))
    if file_names:
        sections.append(f"Attached files: {', '.join(file_names)}")

    timeline_titles = _collect_from_nested(case_payload.get("timelines"), ("title",))
    if timeline_titles:
        sections.append(f"Timeline entries: {', '.join(timeline_titles)}")
    else:
        add("Timeline count", len(case_payload.get("timelines") or []))

    add("Appointment count", len(case_payload.get("appointments") or []))
    return "\n".join(sections)


@router.post("/recommendations")
def recommendations(data: RecommendationRequest, ranker: LawyerRanker = Depends(_get_ranker), cfg: AppConfig = Depends(_get_config)):
    description = data.description
    top_k = data.top_k
    case_payload = data.case
    case_id = data.case_id
    lawyers_payload = data.lawyers

    if lawyers_payload is not None and not isinstance(lawyers_payload, list):
        raise HTTPException(
            status_code=400,
            detail="`lawyers` must be an array of lawyer objects",
        )

    if isinstance(case_payload, dict):
        case_prompt = _case_to_prompt(case_payload, explicit_case_id=case_id)
        if description:
            description = f"{description.strip()}\n\n{case_prompt}"
        else:
            description = case_prompt
    elif case_id:
        case_line = f"Case ID: {case_id}"
        description = f"{description.strip()}\n\n{case_line}" if description else case_line

    if not description or not description.strip():
        raise HTTPException(
            status_code=400,
            detail="`description` or `case` data is required in the request body",
        )

    if lawyers_payload is not None and len(lawyers_payload) == 0:
        raise HTTPException(
            status_code=400,
            detail="`lawyers` cannot be an empty array",
        )

    try:
        normalized_top_k = int(top_k) if top_k is not None else cfg.default_top_k
        if normalized_top_k <= 0:
            raise ValueError
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="`top_k` must be a positive integer",
        )

    try:
        recommendations = ranker.rank(
            description=description, top_k=normalized_top_k, lawyers=lawyers_payload
        )
    except ValueError as exc:  # Raised by the ranker for invalid inputs
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as e:  # pragma: no cover - we don't expect to reach this
        print(f"Unexpected error while ranking lawyers: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to compute recommendations",
        )

    response_payload = {
        "success": True,
        "case_id": case_id,
        "count": len(recommendations),
        "total": len(recommendations),
        "data": recommendations,
    }
    return response_payload
