from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class AppConfig:
    """Container for runtime configuration loaded from environment variables."""

    model_name: str
    lawyer_data_source: str
    default_top_k: int
    rating_score_weight: float = 0.10
    max_rating: float = 5.0
    cors_origins: List[str] = field(default_factory=list)
    debug: bool = False
    port: int = 8000

    @staticmethod
    def _to_bool(value: str | None, default: bool = False) -> bool:
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def from_env(cls) -> "AppConfig":
        model_name = os.getenv(
            "SENTENCE_MODEL",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        )
        lawyer_data_source = os.getenv(
            "LAWYER_DATA_URL", os.getenv("LAWYER_DATA_PATH", "http://localhost:5050/api/v1/lawyers")
        )
        default_top_k = int(os.getenv("DEFAULT_TOP_K", "5"))
        rating_score_weight = float(os.getenv("RATING_SCORE_WEIGHT", "0.10"))
        max_rating = float(os.getenv("MAX_RATING", "5.0"))
        cors_raw = os.getenv("CORS_ORIGINS", "*")
        cors_origins = [origin.strip() for origin in cors_raw.split(",") if origin.strip()]
        if not cors_origins:
            cors_origins = ["*"]
        debug = cls._to_bool(os.getenv("FLASK_DEBUG")) or cls._to_bool(
            os.getenv("DEBUG")
        )
        port = int(os.getenv("PORT", os.getenv("FLASK_RUN_PORT", "8000")))
        return cls(
            model_name=model_name,
            lawyer_data_source=lawyer_data_source,
            default_top_k=default_top_k,
            rating_score_weight=rating_score_weight,
            max_rating=max_rating,
            cors_origins=cors_origins,
            debug=debug,
            port=port,
        )
