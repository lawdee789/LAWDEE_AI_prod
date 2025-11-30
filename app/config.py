"""Runtime configuration helpers."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Application runtime settings."""

    app_name: str = Field(default="Lawyer Matching Service")
    model_name: str = Field(
        default=os.getenv(
            "SENTENCE_MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        )
    )
    data_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data")
        )
    )
    default_top_k: int = Field(default=5, ge=1, le=20)
    max_top_k: int = Field(default=20, ge=1, le=50)


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""

    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings

