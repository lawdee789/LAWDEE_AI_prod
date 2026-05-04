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
            "rating_score_weight": cfg.rating_score_weight if cfg else None,
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


def _case_filter_criteria(payload: Dict[str, Any], case_payload: Any) -> Dict[str, List[str]]:
    work_type_fields = (
        "ประเภทของงาน",
        "category",
        "type",
        "work_type",
        "workType",
        "job_type",
        "jobType",
    )
    service_fields = (
        "หัวข้อบริการ",
        "service",
        "service_topic",
        "serviceTopic",
        "service_title",
        "serviceTitle",
    )

    def collect(fields: Iterable[str]) -> List[str]:
        criteria: List[str] = []
        for field in fields:
            text = _sanitize_text(payload.get(field))
            if text:
                criteria.append(text)
        if isinstance(case_payload, dict):
            for field in fields:
                text = _sanitize_text(case_payload.get(field))
                if text:
                    criteria.append(text)
        return criteria

    return {
        "work_type": collect(work_type_fields),
        "service": collect(service_fields),
    }


def _first_text(payloads: Iterable[Dict[str, Any]], fields: Iterable[str]) -> str | None:
    for payload in payloads:
        for field in fields:
            text = _sanitize_text(payload.get(field))
            if text:
                return text
    return None


def _case_to_prompt(case_payload: Dict[str, Any], root_payload: Dict[str, Any]) -> str:
    sections: List[str] = []

    def add(label: str, value: Any) -> None:
        text = _sanitize_text(value)
        if text:
            sections.append(f"{label}: {text}")

    payloads = [case_payload, root_payload]
    add(
        "ประเภทของงาน",
        _first_text(payloads, ("ประเภทของงาน", "category", "type", "work_type", "workType")),
    )
    add(
        "หัวข้อบริการ",
        _first_text(payloads, ("หัวข้อบริการ", "service", "service_topic", "serviceTopic")),
    )
    add("หัวข้องาน", _first_text(payloads, ("หัวข้องาน", "title")))
    add("รายละเอียดงาน", _first_text(payloads, ("รายละเอียดงาน", "description")))
    add("หมายเหตุเพิ่มเติม", _first_text(payloads, ("หมายเหตุเพิ่มเติม", "note")))
    return "\n".join(sections)


@api_bp.route("/embeddings", methods=["POST"])
def embeddings():
    payload = request.get_json(silent=True) or {}
    ranker = _get_ranker()

    try:
        if isinstance(payload.get("lawyer"), dict):
            vector = ranker.embed_lawyer(payload["lawyer"])
        else:
            text = _sanitize_text(payload.get("text"))
            if not text:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": "`text` or `lawyer` is required in the request body",
                        }
                    ),
                    400,
                )
            vector = ranker.embed_text(text)
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400
    except Exception:  # pragma: no cover - runtime/model failure
        current_app.logger.exception("Unexpected error while creating embedding")
        return (
            jsonify(
                {
                    "success": False,
                    "message": "Failed to compute embedding",
                }
            ),
            500,
        )

    cfg: AppConfig = current_app.config.get("APP_CONFIG")
    return jsonify(
        {
            "success": True,
            "model": cfg.model_name if cfg else None,
            "dimension": len(vector),
            "embedding": vector,
        }
    )


@api_bp.route("/recommendations", methods=["POST"])
def recommendations():
    payload = request.get_json(silent=True) or {}
    description = ""
    top_k = payload.get("top_k")
    case_payload = payload.get("case")
    case_id = payload.get("case_id")
    lawyers_payload = payload.get("lawyers")
    filter_criteria = _case_filter_criteria(payload, case_payload)

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
        description = _case_to_prompt(case_payload, root_payload=payload)
    else:
        description = _case_to_prompt({}, root_payload=payload)

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
            description=description,
            top_k=normalized_top_k,
            lawyers=lawyers_payload,
            filter_criteria=filter_criteria,
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
