from fastapi import APIRouter, HTTPException, Request, Response

from vectorist.api.schemas import DocumentResponse

router = APIRouter()


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
