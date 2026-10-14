from __future__ import annotations

from typing import Any, Protocol

from vectorist.storage.models import ChunkRecord, SearchResult


class VectorStoreProtocol(Protocol):
    async def initialize(self) -> None: ...

    async def upsert_chunks(self, chunks: list[ChunkRecord]) -> None: ...

    async def delete_chunks_by_document_id(self, document_id: str) -> None: ...

    async def get_chunk_hashes(self, document_id: str) -> dict[str, str]: ...

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]: ...
