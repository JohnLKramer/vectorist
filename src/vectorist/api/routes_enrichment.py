from fastapi import APIRouter, Request

from vectorist.api.schemas import EnrichmentCountsResponse

router = APIRouter()


@router.get("/v1/enrichment", response_model=EnrichmentCountsResponse)
async def enrichment_counts(request: Request) -> EnrichmentCountsResponse:
    counts = await request.app.state.cache.job_counts()
    return EnrichmentCountsResponse(
        pending=counts["PENDING"],
        running=counts["RUNNING"],
        done=counts["DONE"],
        failed=counts["FAILED"],
    )
