import asyncio
import logging

from fastapi import APIRouter, Request

from vectorist.api.schemas import HealthResponse

router = APIRouter()
logger = logging.getLogger(__name__)

_CHECK_TIMEOUT_SECONDS = 2.0


async def _check(name: str, component) -> str:
    try:
        await asyncio.wait_for(component.ping(), timeout=_CHECK_TIMEOUT_SECONDS)
    except Exception:
        logger.exception("health check failed for %s", name)
        return "error"
    return "ok"


@router.get("/v1/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    cache = await _check("cache", request.app.state.cache)
    vector_store = await _check("vector_store", request.app.state.store)
    status = "ok" if cache == "ok" and vector_store == "ok" else "degraded"
    return HealthResponse(
        status=status,
        cache=cache,
        vector_store=vector_store,
        embedder="ok",
    )
