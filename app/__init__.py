from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import AppConfig
from .routes import router
from .services.lawyer_ranker import LawyerRanker


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Application factory."""

    cfg = config or AppConfig.from_env()
    app = FastAPI(title="Lawyer Recommender API")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize lawyer ranker and store in app state
    app.state.config = cfg
    app.state.lawyer_ranker = LawyerRanker(
        model_name=cfg.model_name,
        lawyer_data_source=cfg.lawyer_data_source,
        default_top_k=cfg.default_top_k,
    )

    app.include_router(router)

    return app


__all__ = ["create_app"]
