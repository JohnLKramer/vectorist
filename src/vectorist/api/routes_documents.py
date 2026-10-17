from fastapi import APIRouter, HTTPException, Query, Request, Response

from vectorist.api.schemas import (
    DocumentListResponse,
    DocumentResponse,
    DocumentSummaryResponse,
)

router = APIRouter()


@router.get("/v1/documents", response_model=DocumentListResponse)
async def list_documents(
    request: Request,
    source_prefix: str = Query(min_length=1),
    limit: int = Query(500, ge=1, le=1000),
    cursor: str | None = None,
) -> DocumentListResponse:
    rows = await request.app.state.cache.list_documents_by_prefix(
        source_prefix, limit + 1, cursor
    )
    page = rows[:limit]
    return DocumentListResponse(
        documents=[
            DocumentSummaryResponse(
                id=row.id,
                source_uri=row.source_uri,
                content_hash=row.content_hash,
                status=row.status.value,
            )
            for row in page
        ],
        next_cursor=page[-1].source_uri if len(rows) > limit else None,
    )


@router.get("/v1/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str, request: Request) -> DocumentResponse:
    document = await request.app.state.cache.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    return DocumentResponse(
        id=document.id,
        source_uri=document.source_uri,
        content_hash=document.content_hash,
        status=document.status.value,
        error_message=document.error_message,
        raw_content=document.raw_content,
    )


@router.delete("/v1/documents/{document_id}", status_code=204)
async def delete_document(document_id: str, request: Request) -> Response:
    document = await request.app.state.cache.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    await request.app.state.store.delete_chunks_by_document_id(document_id)
    await request.app.state.cache.delete_document(document_id)
    return Response(status_code=204)
