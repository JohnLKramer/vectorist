from fastapi import APIRouter, HTTPException, Request

from vectorist.api.schemas import AssetResponse
from vectorist.ir.models import EnrichmentStatus, MediaNode

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


@router.put("/v1/documents/{document_id}/assets", response_model=AssetResponse)
async def upload_asset(document_id: str, ref: str, request: Request) -> AssetResponse:
    state = request.app.state
    enricher = state.enricher
    if enricher is None:
        raise HTTPException(
            status_code=409,
            detail="image enrichment is disabled (VECTORIST_ENRICHER is none)",
        )
    cache = state.cache
    if await cache.get_document(document_id) is None:
        raise HTTPException(status_code=404, detail="document not found")

    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported image content type {content_type!r}; "
            f"allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    data = await request.body()
    if len(data) > state.max_asset_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"asset is {len(data)} bytes; limit is {state.max_asset_bytes}",
        )

    snapshot = await cache.get_ir_snapshot(document_id)
    has_ref = snapshot is not None and any(
        isinstance(node, MediaNode)
        and node.enrichment_status != EnrichmentStatus.SKIPPED
        and node.metadata.get("ref") == ref
        for node in snapshot.nodes.values()
    )
    if not has_ref:
        raise HTTPException(
            status_code=404,
            detail=f"document has no enrichable image with ref {ref!r}",
        )

    asset_hash = await cache.put_asset(data, content_type)
    await cache.link_asset(document_id, ref, asset_hash)
    if asset_hash in await cache.get_enrichments([asset_hash], enricher.provider_id):
        result = await state.ingest.reindex(document_id)
        if result is None:
            raise HTTPException(status_code=404, detail="document not found")
        if result.status == "failed":
            raise HTTPException(
                status_code=500,
                detail=f"re-index of document {document_id} failed: "
                f"{result.error['message']}",
            )
        if result.status == "skipped":
            # The document is not INDEXED (its last ingest failed); the next
            # ingest applies the stored description.
            return AssetResponse(asset_hash=asset_hash, status="queued")
        return AssetResponse(asset_hash=asset_hash, status="done")
    await cache.enqueue_job(asset_hash)
    return AssetResponse(asset_hash=asset_hash, status="queued")
