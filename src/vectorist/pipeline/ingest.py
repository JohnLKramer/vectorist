from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from vectorist.adapters.base import AdapterProtocol
from vectorist.adapters.markdown import MarkdownAdapter
from vectorist.cache.models import DocumentRecord, DocumentStatus
from vectorist.cache.protocol import CacheProtocol
from vectorist.chunking.chunker import Chunk, Chunker
from vectorist.embeddings.protocol import EmbeddingProviderProtocol
from vectorist.hashing import sha256_text
from vectorist.storage.models import ChunkRecord
from vectorist.storage.protocol import VectorStoreProtocol


@dataclass
class IngestResult:
    source_uri: str
    document_id: str | None
    status: str  # accepted | skipped | failed | indexed
    error: dict[str, str] | None = None


class IngestService:
    def __init__(
        self,
        cache: CacheProtocol,
        vector_store: VectorStoreProtocol,
        embedder: EmbeddingProviderProtocol,
        adapter: AdapterProtocol | None = None,
        chunker: Chunker | None = None,
    ) -> None:
        self._cache = cache
        self._vector_store = vector_store
        self._embedder = embedder
        self._adapter = MarkdownAdapter() if adapter is None else adapter
        self._chunker = Chunker(max_tokens=512) if chunker is None else chunker

    async def ingest_document(
        self,
        source_uri: str,
        content: str,
        content_type: str = "text/markdown",
    ) -> IngestResult:
        if content_type != "text/markdown":
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
        ):
            return IngestResult(
                source_uri=source_uri,
                document_id=existing.id,
                status="skipped",
            )

        document_id = existing.id if existing is not None else str(uuid4())
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
            ir_document = self._adapter.parse(source_uri, content)
            await self._cache.save_ir_snapshot(document_id, ir_document)
            await self._cache.update_status(document_id, DocumentStatus.PARSED)

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
                embeddings = self._embedder.embed_documents(
                    [chunk.content_text for _, _, chunk in changed]
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
        except Exception as exc:
            await self._cache.update_status(
                document_id, DocumentStatus.FAILED, error_message=str(exc)
            )
            return IngestResult(
                source_uri=source_uri,
                document_id=document_id,
                status="failed",
                error={"code": "INGEST_FAILED", "message": str(exc)},
            )

        return IngestResult(
            source_uri=source_uri,
            document_id=document_id,
            status="indexed",
        )
