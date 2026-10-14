from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import asyncpg

from vectorist.migrations.runner import _parse_changesets
from vectorist.storage.models import ChunkRecord, SearchResult

_POSTGRES_MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[3] / "migrations" / "postgres"
)

_SCHEMA_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS _schema_migrations (
    id TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL
)
"""


def _metadata_for_storage(chunk: ChunkRecord) -> dict[str, Any]:
    stored = dict(chunk.metadata)
    stored["node_types"] = list(chunk.node_types)
    return stored


def _embedding_param(embedding: list[float]) -> str:
    return "[" + ",".join(f"{x:g}" for x in embedding) + "]"


async def _execute_sql_script(conn: asyncpg.Connection, script: str) -> None:
    statements = [part.strip() for part in script.split(";") if part.strip()]
    for statement in statements:
        await conn.execute(statement)


async def _apply_postgres_migrations(
    conn: asyncpg.Connection, migrations_dir: Path
) -> list[str]:
    await conn.execute(_SCHEMA_MIGRATIONS_DDL)
    rows = await conn.fetch("SELECT id FROM _schema_migrations")
    already = {row["id"] for row in rows}

    applied: list[str] = []
    now = datetime.now(timezone.utc)

    for path in sorted(migrations_dir.glob("*.sql")):
        sql_text = path.read_text(encoding="utf-8")
        for changeset_id, body in _parse_changesets(sql_text):
            if changeset_id in already:
                continue
            if body:
                await _execute_sql_script(conn, body)
            await conn.execute(
                "INSERT INTO _schema_migrations (id, applied_at) VALUES ($1, $2)",
                changeset_id,
                now,
            )
            already.add(changeset_id)
            applied.append(changeset_id)

    return applied


def _search_result_from_row(row: asyncpg.Record, distance: float) -> SearchResult:
    meta = row["metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    source_uri = str(meta.get("source_uri", ""))
    breadcrumbs = list(meta.get("breadcrumbs", []))
    result_metadata = {
        key: value
        for key, value in meta.items()
        if key not in ("source_uri", "breadcrumbs")
    }
    score = 1.0 - float(distance)
    return SearchResult(
        chunk_id=row["id"],
        document_id=row["document_id"],
        source_uri=source_uri,
        content_text=row["content_text"],
        score=score,
        breadcrumbs=breadcrumbs,
        metadata=result_metadata,
    )


class PgVectorStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(self._dsn)
        async with self._pool.acquire() as conn:
            await _apply_postgres_migrations(conn, _POSTGRES_MIGRATIONS_DIR)

    def _pool_required(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PgVectorStore.initialize() must be called first")
        return self._pool

    async def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        pool = self._pool_required()
        async with pool.acquire() as conn:
            for chunk in chunks:
                metadata_json = json.dumps(
                    _metadata_for_storage(chunk), sort_keys=True
                )
                embedding = _embedding_param(chunk.embedding)
                await conn.execute(
                    """
                    INSERT INTO vector_chunks (
                        id, document_id, chunk_index, chunk_hash,
                        content_text, embedding, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6::vector, $7::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        chunk_index = EXCLUDED.chunk_index,
                        chunk_hash = EXCLUDED.chunk_hash,
                        content_text = EXCLUDED.content_text,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                    """,
                    chunk.id,
                    chunk.document_id,
                    chunk.chunk_index,
                    chunk.chunk_hash,
                    chunk.content_text,
                    embedding,
                    metadata_json,
                )

    async def delete_chunks_by_document_id(self, document_id: str) -> None:
        pool = self._pool_required()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM vector_chunks WHERE document_id = $1",
                document_id,
            )

    async def get_chunk_hashes(self, document_id: str) -> dict[str, str]:
        pool = self._pool_required()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, chunk_hash
                FROM vector_chunks
                WHERE document_id = $1
                """,
                document_id,
            )
        return {row["id"]: row["chunk_hash"] for row in rows}

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        pool = self._pool_required()
        query_embedding = _embedding_param(query_vector)
        async with pool.acquire() as conn:
            if filters and "source_uri_prefix" in filters:
                prefix_param = f"{str(filters['source_uri_prefix'])}%"
                count_row = await conn.fetchrow(
                    """
                    SELECT COUNT(*) AS n
                    FROM vector_chunks
                    WHERE metadata->>'source_uri' LIKE $1
                    """,
                    prefix_param,
                )
                if count_row is None or int(count_row["n"]) == 0:
                    return []
                rows = await conn.fetch(
                    """
                    SELECT
                        id,
                        document_id,
                        content_text,
                        metadata,
                        embedding <=> $1::vector AS distance
                    FROM vector_chunks
                    WHERE metadata->>'source_uri' LIKE $2
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                    """,
                    query_embedding,
                    prefix_param,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT
                        id,
                        document_id,
                        content_text,
                        metadata,
                        embedding <=> $1::vector AS distance
                    FROM vector_chunks
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                    """,
                    query_embedding,
                    limit,
                )
        return [_search_result_from_row(row, row["distance"]) for row in rows]
