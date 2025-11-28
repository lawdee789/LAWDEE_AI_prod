from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Runtime configuration for the lawyer matching microservice.
    Values can be overridden through standard environment variables.
    """

    embedding_model_name: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        description="Sentence Transformer model used to embed cases and lawyers.",
    )
    lawyer_repository_path: Path = Field(
        default=Path(__file__).resolve().parents[1] / "data" / "lawyers_sample.json",
        description="Local JSON file used to seed the in-memory lawyer store.",
    )
    lawyer_repository_url: str | None = Field(
        default=None,
        description="Optional HTTP endpoint in the Express backend that exposes lawyer profiles.",
    )
    default_top_k: int = Field(default=5, ge=1, le=50)
    min_score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    specialization_weight: float = Field(
        default=0.15,
        description="Extra score awarded when the lawyer specialization matches the case category.",
    )
    service_weight: float = Field(
        default=0.05,
        description="Bonus when a lawyer lists the same service name as the incoming case.",
    )
    refresh_interval_seconds: int = Field(default=600, ge=30)
    similarity_metric: Literal["cosine"] = "cosine"
    allow_cross_origin: bool = Field(default=True)

    class Config:
        env_file = ".env"
        env_prefix = "LAWYER_MATCH_"
