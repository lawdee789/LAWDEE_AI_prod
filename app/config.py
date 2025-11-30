from functools import lru_cache
from pathlib import Path
from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """Centralized application configuration loaded from environment variables."""

    model_name: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        env="MODEL_NAME",
    )
    data_path: str = Field(default="data/lawyers.sample.json", env="LAWYERS_DATA_PATH")
    default_top_k: int = Field(default=5, env="DEFAULT_TOP_K")
    min_similarity_score: float = Field(default=0.2, env="MIN_SIMILARITY_SCORE")
    flask_host: str = Field(default="0.0.0.0", env="FLASK_HOST")
    flask_port: int = Field(default=5050, env="FLASK_PORT")
    flask_debug: bool = Field(default=False, env="FLASK_DEBUG")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator("data_path", pre=True)
    def _resolve_data_path(cls, value: str) -> str:
        return str(Path(value))

    @validator("min_similarity_score")
    def _validate_similarity_score(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("MIN_SIMILARITY_SCORE must be within [0, 1]")
        return value


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()
