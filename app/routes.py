from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

from flask import Blueprint, Flask, jsonify, request
from pydantic import ValidationError

from .config import Settings
from .schemas import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendationResponseLawyer,
)
from .services.lawyer_ranker import LawyerRanker


def register_routes(app: Flask, ranker: LawyerRanker, settings: Settings) -> None:
    blueprint = Blueprint("recommendations", __name__)

    @blueprint.get("/health")
    def health() -> tuple[dict, int]:
        payload = {
            "success": True,
            "model_name": settings.model_name,
            "lawyers_indexed": len(ranker.repository.get_all()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return jsonify(payload), 200

    @blueprint.post("/recommendations")
    def recommendations() -> tuple[dict, int]:
        try:
            payload = RecommendationRequest(**(request.get_json(force=True) or {}))
        except ValidationError as exc:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Invalid request body",
                        "errors": exc.errors(),
                    }
                ),
                400,
            )

        top_k = payload.top_k or settings.default_top_k
        min_score = payload.min_score

        ranked_lawyers = ranker.recommend(
            case_description=payload.description,
            top_k=top_k,
            min_score=min_score,
        )

        lawyer_models = [
            RecommendationResponseLawyer(**asdict(lawyer)) for lawyer in ranked_lawyers
        ]

        response = RecommendationResponse(
            count=len(lawyer_models),
            data=lawyer_models,
            metadata={
                "model_name": settings.model_name,
                "top_k": top_k,
                "min_score": min_score or ranker.min_score,
            },
        )

        response_dict = response.dict()
        response_dict["timestamp"] = response_dict["timestamp"].isoformat()
        return jsonify(response_dict), 200

    app.register_blueprint(blueprint)
