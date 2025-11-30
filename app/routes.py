from __future__ import annotations

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


@api_bp.route("/recommendations", methods=["POST"])
def recommendations():
    payload = request.get_json(silent=True) or {}
    description = payload.get("description", "")
    top_k = payload.get("top_k")

    if not description or not description.strip():
        return (
            jsonify(
                {
                    "success": False,
                    "message": "`description` is required in the request body",
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
        recommendations = ranker.rank(description=description, top_k=normalized_top_k)
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

    return jsonify({"success": True, "count": len(recommendations), "data": recommendations})
