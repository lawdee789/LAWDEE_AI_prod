"""FastAPI application exposing the lawyer matching service."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .config import get_settings
from .logger import get_logger
from .matcher import LawyerMatcher
from .providers import LawyerProvider
from .schemas import MatchRequest, MatchResponse

settings = get_settings()
logger = get_logger(__name__)

app = FastAPI(title=settings.app_name, version="0.1.0")
provider = LawyerProvider(settings)
matcher = LawyerMatcher(settings)


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    """Lightweight health endpoint for Kubernetes / uptime checks."""

    return {"status": "ok", "service": settings.app_name}


@app.post("/match-lawyers", response_model=MatchResponse)
async def match_lawyers(request: MatchRequest) -> MatchResponse:
    """Return the best lawyer matches for a case payload."""

    top_k = request.top_k or settings.default_top_k

    if request.lawyers and len(request.lawyers) > 0:
        lawyer_pool = request.lawyers
        lawyer_source = "payload"
    else:
        lawyer_pool = await provider.fetch()
        lawyer_source = "provider"

    if not lawyer_pool:
        raise HTTPException(status_code=404, detail="No lawyers available for matching.")

    matches = matcher.rank(case=request.case, lawyers=lawyer_pool, top_k=top_k)

    logger.info(
        "Computed %s matches for case %s (requested top_k=%s, pool=%s)",
        len(matches),
        request.case.case_id,
        top_k,
        len(lawyer_pool),
    )

    return MatchResponse(
        success=True,
        count=len(matches),
        case_id=request.case.case_id,
        lawyers=matches,
        metadata={
            "top_k_requested": top_k,
            "lawyer_source": lawyer_source,
            "total_candidates": len(lawyer_pool),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("lawyer_matching_service.app:app", host="0.0.0.0", port=8000, reload=False)
