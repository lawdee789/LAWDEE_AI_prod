from __future__ import annotations

from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings
from .encoder import TextEncoder
from .matching import MatchingEngine
from .models import MatchRequest, MatchResponse
from .repository import FileLawyerRepository, HttpLawyerRepository, LawyerRepository


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    repository = _build_repository(settings)
    encoder = TextEncoder(settings.embedding_model_name)
    engine = MatchingEngine(settings=settings, encoder=encoder, repository=repository)

    app = FastAPI(
        title="Case-Lawyer Matching Service",
        version="1.0.0",
        description="Embeddings-based retrieval service that returns the best lawyers for a case payload.",
    )

    if settings.allow_cross_origin:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.on_event("startup")
    async def startup_event() -> None:
        await engine.refresh()

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "lawyers_loaded": engine.lawyer_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.post("/match-lawyers", response_model=MatchResponse)
    async def match_lawyers(payload: MatchRequest, engine_dep: MatchingEngine = Depends(lambda: engine)) -> MatchResponse:
        try:
            return engine_dep.match(payload.case, payload.top_k)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

    @app.post("/refresh-lawyers")
    async def refresh_lawyers(engine_dep: MatchingEngine = Depends(lambda: engine)) -> dict:
        await engine_dep.refresh()
        return {"success": True, "lawyers_loaded": engine_dep.lawyer_count}

    return app


def _build_repository(settings: Settings) -> LawyerRepository:
    if settings.lawyer_repository_url:
        return HttpLawyerRepository(settings.lawyer_repository_url)
    return FileLawyerRepository(settings.lawyer_repository_path)
