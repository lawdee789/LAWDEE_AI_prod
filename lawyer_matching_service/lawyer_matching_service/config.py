"""Application configuration settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings

_PACKAGE_ROOT = Path(__file__).resolve().parent
_DEFAULT_DATASET = _PACKAGE_ROOT / "data" / "sample_lawyers.json"


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    app_name: str = "Lawyer Matching Service"
    default_top_k: int = 5
    lawyer_source_url: Optional[AnyHttpUrl] = None
    lawyer_source_token: Optional[str] = None
    request_timeout_seconds: float = 5.0
    fallback_dataset_path: str = Field(
        default=str(_DEFAULT_DATASET),
        description="Absolute path to the JSON file used when remote lawyer data is unavailable.",
    )
    vectorizer_analyzer: str = Field(
        default="char_wb",
        description="Analyzer type passed to scikit-learn's TfidfVectorizer.",
    )
    vectorizer_ngram_min: int = 3
    vectorizer_ngram_max: int = 5
    enable_debug_logging: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""

    return Settings()
