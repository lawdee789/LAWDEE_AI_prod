from __future__ import annotations

from typing import Any, Dict, Iterable, List

from flask import Blueprint, current_app, jsonify, request

from .config import AppConfig
from .services.lawyer_ranker import LawyerRanker

api_bp = Blueprint("api", __name__)


def _get_ranker() -> LawyerRanker:
    ranker = current_app.extensions.get("lawyer_ranker")
    if ranker is None:
        raise RuntimeError("LawyerRanker has not been initialized")
    return ranker


@api_bp.route("/health", methods=["GET"])
def health_check():
    cfg: AppConfig = current_app.config.get("APP_CONFIG")
    return jsonify(
        {
            "success": True,
            "message": "Lawyer recommender online",
            "model": cfg.model_name if cfg else None,
        }
    )


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


@api_bp.route("/recommendations", methods=["POST"])
def recommendations():
    payload = request.get_json(silent=True) or {}
    description = payload.get("description", "")
    top_k = payload.get("top_k")
    case_payload = payload.get("case")
    case_id = payload.get("case_id")
    lawyers_payload = payload.get("lawyers")

    if lawyers_payload is not None and not isinstance(lawyers_payload, list):
        return (
            jsonify(
                {
                    "success": False,
                    "message": "`lawyers` must be an array of lawyer objects",
                }
            ),
            400,
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
        return (
            jsonify(
                {
                    "success": False,
                    "message": "`description` or `case` data is required in the request body",
                }
            ),
            400,
        )

    if lawyers_payload is not None and len(lawyers_payload) == 0:
        return (
            jsonify(
                {
                    "success": False,
                    "message": "`lawyers` cannot be an empty array",
                }
            ),
            400,
        )

    cfg: AppConfig = current_app.config.get("APP_CONFIG")

    try:
        normalized_top_k = int(top_k) if top_k is not None else cfg.default_top_k
        if normalized_top_k <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return (
            jsonify(
                {
                    "success": False,
                    "message": "`top_k` must be a positive integer",
                }
            ),
            400,
        )

    ranker = _get_ranker()

    try:
        recommendations = ranker.rank(
            description=description, top_k=normalized_top_k, lawyers=lawyers_payload
        )
    except ValueError as exc:  # Raised by the ranker for invalid inputs
        return jsonify({"success": False, "message": str(exc)}), 400
    except Exception:  # pragma: no cover - we don't expect to reach this
        current_app.logger.exception("Unexpected error while ranking lawyers")
        return (
            jsonify(
                {
                    "success": False,
                    "message": "Failed to compute recommendations",
                }
            ),
            500,
        )

    response_payload = {
        "success": True,
        "case_id": case_id,
        "count": len(recommendations),
        "total": len(recommendations),
        "data": recommendations,
    }
    return jsonify(response_payload)
