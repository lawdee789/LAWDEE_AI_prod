from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from matcher_service.config import get_settings
from matcher_service.engine import LawyerMatchEngine
from matcher_service.repository import FileLawyerRepository
from matcher_service.schemas import BulkUpsertRequest, MatchRequest, MatchResponse

settings = get_settings()
repository = FileLawyerRepository(settings.data_path)
engine = LawyerMatchEngine(settings, repository)

app = FastAPI(
    title="Lawyer Semantic Match Service",
    version="0.1.0",
    description="SentenceTransformer-based semantic search for lawyer selection",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    settings.ensure_data_path()
    engine.load_initial_state()


@app.get("/healthz")
def health_check() -> dict:
    return {
        "status": "ok",
        "model": settings.model_name,
        "lawyers_indexed": engine.lawyer_count,
    }


@app.post("/api/v1/match", response_model=MatchResponse)
async def match_lawyers(request: MatchRequest) -> MatchResponse:
    try:
        return await run_in_threadpool(engine.rank, request)
    except RuntimeError as exc:  # pragma: no cover - defensive guard
        raise HTTPException(503, detail=str(exc)) from exc


@app.post("/api/v1/lawyers")
async def bulk_upsert(request: BulkUpsertRequest) -> dict:
    lawyers = await run_in_threadpool(engine.upsert_lawyers, request.lawyers)
    return {"success": True, "count": len(lawyers)}
