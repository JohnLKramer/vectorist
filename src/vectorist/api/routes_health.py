from fastapi import APIRouter

from vectorist.api.schemas import HealthResponse

router = APIRouter()


@router.get("/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        cache="ok",
        vector_store="ok",
        embedder="ok",
    )
