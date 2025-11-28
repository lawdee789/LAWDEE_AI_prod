from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


@dataclass(slots=True)
class Settings:
    """Container for runtime configuration values."""

    model_name: str = os.getenv(
        "SENTENCE_TRANSFORMER_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    data_path: Path = Path(
        os.getenv("LAWYER_DATA_PATH", Path("data/lawyers_seed.json").as_posix())
    )
    top_k: int = int(os.getenv("MATCHER_DEFAULT_TOP_K", "5"))
    similarity_metric: str = os.getenv("MATCHER_SIMILARITY", "cosine")
    normalize_embeddings: bool = bool(int(os.getenv("MATCHER_NORMALIZE", "1")))

    def ensure_data_path(self) -> None:
        self.data_path.parent.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
