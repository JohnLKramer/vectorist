import asyncio

from fastapi import APIRouter, Request

from vectorist.api.schemas import SearchHit, SearchRequest, SearchResponse
from vectorist.storage.models import SearchResult

router = APIRouter()


def _hit(result: SearchResult) -> SearchHit:
    return SearchHit(
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        source_uri=result.source_uri,
        content_text=result.content_text,
        score=result.score,
        breadcrumbs=list(result.breadcrumbs),
        metadata=dict(result.metadata),
    )


@router.post("/v1/search", response_model=SearchResponse)
async def search(body: SearchRequest, request: Request) -> SearchResponse:
    query_vector = await asyncio.to_thread(
        request.app.state.embedder.embed_query, body.query
    )
    hits = await request.app.state.store.search(
        query_vector,
        limit=body.limit,
        filters=body.filters,
    )
    return SearchResponse(results=[_hit(hit) for hit in hits])
