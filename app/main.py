"""FastAPI entry point for the semantic lawyer matcher."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .matcher import SemanticMatcher
from .storage import LawyerStore
from . import schemas

settings = get_settings()
store = LawyerStore(settings.data_dir)
matcher = SemanticMatcher(settings.model_name)

app = FastAPI(title=settings.app_name, version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def warm_up_model() -> None:
    await matcher.warm_up()


@app.get("/healthz", response_model=schemas.HealthResponse)
def healthz() -> schemas.HealthResponse:
    return schemas.HealthResponse(
        status="ok", model_name=settings.model_name, total_lawyers=store.count()
    )


@app.get("/v1/lawyers", response_model=schemas.LawyerListResponse)
def list_lawyers() -> schemas.LawyerListResponse:
    data = [schemas.LawyerResponse.from_record(record) for record in store.list()]
    return schemas.LawyerListResponse(count=len(data), data=data)


@app.post("/v1/lawyers", response_model=schemas.LawyerResponse, status_code=201)
async def upsert_lawyer(payload: schemas.LawyerProfile) -> schemas.LawyerResponse:
    embedding = await matcher.embed_lawyer(payload)
    record = store.upsert(payload, embedding)
    return schemas.LawyerResponse.from_record(record)


@app.post("/v1/match-lawyers", response_model=schemas.MatchResponse)
async def match_lawyers(payload: schemas.MatchRequest) -> schemas.MatchResponse:
    records = store.list()
    if not records:
        raise HTTPException(
            status_code=409, detail="No lawyers have been indexed yet. Add lawyers first."
        )

    requested_top_k = min(payload.top_k, settings.max_top_k)
    case_vector = await matcher.embed_case(payload.case)
    ranked = matcher.rank(case_vector, records, requested_top_k)
    results = [
        schemas.MatchResult(
            score=score,
            lawyer=schemas.LawyerResponse.from_record(record),
        )
        for record, score in ranked
    ]
    return schemas.MatchResponse(
        case_id=payload.case.case_id,
        top_k=len(results),
        results=results,
        computed_at=datetime.now(timezone.utc),
    )


def get_app() -> FastAPI:
    """Compatibility helper for external ASGI servers."""

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )

