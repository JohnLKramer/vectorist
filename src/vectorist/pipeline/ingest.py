from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from uuid import uuid4

from vectorist.adapters.base import AdapterProtocol
from vectorist.adapters.html import HtmlAdapter
from vectorist.adapters.markdown import MarkdownAdapter
from vectorist.cache.models import DocumentRecord, DocumentStatus
from vectorist.cache.protocol import CacheProtocol
from vectorist.chunking.chunker import Chunk, Chunker
from vectorist.embeddings.protocol import EmbeddingProviderProtocol
from vectorist.enrichment.apply import apply_enrichments, has_legacy_media, media_ref
from vectorist.hashing import sha256_text
from vectorist.ir.models import EnrichmentStatus, IRDocument, MediaNode
from vectorist.storage.models import ChunkRecord
from vectorist.storage.protocol import VectorStoreProtocol

logger = logging.getLogger(__name__)

@dataclass
class IngestResult:
    source_uri: str
    document_id: str | None
    status: str  # accepted | skipped | failed | indexed
    error: dict[str, str] | None = None
    pending_assets: list[str] = field(default_factory=list)


class IngestService:
    def __init__(
        self,
        cache: CacheProtocol,
        vector_store: VectorStoreProtocol,
        embedder: EmbeddingProviderProtocol,
        adapter: AdapterProtocol | None = None,
        adapters: dict[str, AdapterProtocol] | None = None,
        chunker: Chunker | None = None,
        enricher_id: str | None = None,
    ) -> None:
        self._cache = cache
        self._vector_store = vector_store
        self._embedder = embedder
        self._adapters: dict[str, AdapterProtocol] = {
            "text/markdown": MarkdownAdapter() if adapter is None else adapter,
            "text/html": HtmlAdapter(),
        }
        if adapters is not None:
            self._adapters.update(adapters)
        self._chunker = Chunker(max_tokens=512) if chunker is None else chunker
        self._enricher_id = enricher_id
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, document_id: str) -> asyncio.Lock:
        return self._locks.setdefault(document_id, asyncio.Lock())

    async def ingest_document(
        self,
        source_uri: str,
        content: str,
        content_type: str = "text/markdown",
    ) -> IngestResult:
        adapter = self._adapters.get(content_type)
        if adapter is None:
            return IngestResult(
                source_uri=source_uri,
                document_id=None,
                status="failed",
                error={
                    "code": "PARSE_ERROR",
                    "message": f"unsupported content type: {content_type}",
                },
            )

        content_hash = sha256_text(content)
        existing = await self._cache.get_document_by_uri(source_uri)
        if (
            existing is not None
            and existing.content_hash == content_hash
            and existing.status == DocumentStatus.INDEXED
            and not await self._snapshot_predates_refs(existing.id)
        ):
            return IngestResult(
                source_uri=source_uri,
                document_id=existing.id,
                status="skipped",
                pending_assets=await self.pending_refs(existing.id),
            )

        document_id = existing.id if existing is not None else str(uuid4())
        async with self._lock_for(document_id):
            await self._cache.upsert_document(
                DocumentRecord(
                    id=document_id,
                    source_uri=source_uri,
                    content_hash=content_hash,
                    raw_content=content,
                    status=DocumentStatus.PENDING,
                    error_message=None,
                )
            )
            try:
                ir_document = adapter.parse(source_uri, content)
                await self._cache.save_ir_snapshot(document_id, ir_document)
                await self._cache.update_status(document_id, DocumentStatus.PARSED)
                await self._index(document_id, source_uri, ir_document)
            except Exception as exc:
                return await self._failed(source_uri, document_id, exc)

        return IngestResult(
            source_uri=source_uri,
            document_id=document_id,
            status="indexed",
            pending_assets=await self.pending_refs(document_id),
        )

    async def reindex(self, document_id: str) -> IngestResult | None:
        """Re-chunk a stored document from its IR snapshot with current image text.

        Returns ``None`` when the document or its IR snapshot is missing. A
        document that is not ``INDEXED`` is left alone and the result is
        ``skipped``: its snapshot may be older than its stored content, and the
        next ingest of the document applies the stored image text.
        """
        async with self._lock_for(document_id):
            document = await self._cache.get_document(document_id)
            if document is None:
                return None
            if document.status != DocumentStatus.INDEXED:
                logger.info(
                    "not re-indexing document %s: status is %s, not INDEXED",
                    document_id,
                    document.status.value,
                )
                return IngestResult(
                    source_uri=document.source_uri,
                    document_id=document_id,
                    status="skipped",
                )
            ir_document = await self._cache.get_ir_snapshot(document_id)
            if ir_document is None:
                return None
            try:
                await self._index(document_id, document.source_uri, ir_document)
            except Exception as exc:
                return await self._failed(document.source_uri, document_id, exc)

        return IngestResult(
            source_uri=document.source_uri,
            document_id=document_id,
            status="indexed",
            pending_assets=await self.pending_refs(document_id),
        )

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document's chunks and cache row under its lock.

        Returns ``False`` when the document does not exist. Holding the lock
        keeps an in-flight re-index from writing chunks back after the delete.
        """
        async with self._lock_for(document_id):
            if await self._cache.get_document(document_id) is None:
                return False
            await self._vector_store.delete_chunks_by_document_id(document_id)
            await self._cache.delete_document(document_id)
        return True

    async def pending_refs(self, document_id: str) -> list[str]:
        """Unique refs, in document order, of image nodes the client should upload.

        These are pending image nodes with no linked asset, and failed image
        nodes even when linked, so a ``sync`` or ``crawl`` re-run retries them.
        """
        if self._enricher_id is None:
            return []
        ir_document = await self._cache.get_ir_snapshot(document_id)
        if ir_document is None:
            return []
        links = await self._cache.get_document_assets(document_id)
        refs: list[str] = []
        for node in ir_document.nodes.values():
            if not isinstance(node, MediaNode):
                continue
            if node.enrichment_status == EnrichmentStatus.PENDING:
                ref = media_ref(ir_document, node)
                wanted = ref not in links
            elif node.enrichment_status == EnrichmentStatus.FAILED:
                ref = media_ref(ir_document, node)
                wanted = True
            else:
                continue
            if wanted and ref not in refs:
                refs.append(ref)
        return refs

    async def _snapshot_predates_refs(self, document_id: str) -> bool:
        """True when enrichment is on and the snapshot has image nodes without refs.

        Such a legacy snapshot is re-parsed instead of skipped so it gains refs.
        """
        if self._enricher_id is None:
            return False
        ir_document = await self._cache.get_ir_snapshot(document_id)
        return ir_document is not None and has_legacy_media(ir_document)

    async def _failed(
        self, source_uri: str, document_id: str, exc: Exception
    ) -> IngestResult:
        await self._cache.update_status(
            document_id, DocumentStatus.FAILED, error_message=str(exc)
        )
        return IngestResult(
            source_uri=source_uri,
            document_id=document_id,
            status="failed",
            error={"code": "INGEST_FAILED", "message": str(exc)},
        )

    async def _index(
        self, document_id: str, source_uri: str, ir_document: IRDocument
    ) -> None:
        if self._enricher_id is not None:
            links = await self._cache.get_document_assets(document_id)
            hashes = sorted(set(links.values()))
            texts = await self._cache.get_enrichments(hashes, self._enricher_id)
            statuses = await self._cache.get_job_statuses(hashes)
            ir_document = apply_enrichments(ir_document, links, texts, statuses)
            await self._cache.save_ir_snapshot(document_id, ir_document)

        chunks = self._chunker.chunk_document(ir_document)
        existing_hashes = await self._vector_store.get_chunk_hashes(document_id)
        changed: list[tuple[str, str, Chunk]] = []
        new_ids: set[str] = set()
        for chunk in chunks:
            chunk_id = f"{document_id}:{chunk.chunk_index}"
            chunk_hash = sha256_text(chunk.content_text + document_id)
            new_ids.add(chunk_id)
            if existing_hashes.get(chunk_id) != chunk_hash:
                changed.append((chunk_id, chunk_hash, chunk))

        removed_ids = [
            chunk_id for chunk_id in existing_hashes if chunk_id not in new_ids
        ]

        embeddings: list[list[float]] = []
        if changed:
            embeddings = await asyncio.to_thread(
                self._embedder.embed_documents,
                [chunk.content_text for _, _, chunk in changed],
            )

        if removed_ids:
            await self._vector_store.delete_chunks_by_ids(removed_ids)

        if changed:
            records: list[ChunkRecord] = []
            for (chunk_id, chunk_hash, chunk), embedding in zip(
                changed, embeddings, strict=True
            ):
                metadata = dict(chunk.metadata)
                metadata["source_uri"] = source_uri
                metadata["breadcrumbs"] = list(chunk.breadcrumbs)
                metadata["line_range"] = list(chunk.line_range)
                records.append(
                    ChunkRecord(
                        id=chunk_id,
                        document_id=document_id,
                        chunk_index=chunk.chunk_index,
                        chunk_hash=chunk_hash,
                        content_text=chunk.content_text,
                        embedding=embedding,
                        node_types=list(chunk.node_types),
                        metadata=metadata,
                    )
                )
            await self._vector_store.upsert_chunks(records)

        await self._cache.update_status(
            document_id, DocumentStatus.INDEXED, error_message=None
        )
