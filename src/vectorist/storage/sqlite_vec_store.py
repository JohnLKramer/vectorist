from __future__ import annotations

import asyncio
import json
import queue
import sqlite3
import threading
from pathlib import Path
from typing import Any

import sqlite_vec

from vectorist.migrations.runner import apply_migrations
from vectorist.storage.models import ChunkRecord, SearchResult

_SQLITE_MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[3] / "migrations" / "sqlite"
)
_VEC_DIMENSIONS = 384
_VEC_TABLE = "chunk_vectors"
_VEC_TABLE_DDL = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS {_VEC_TABLE} USING vec0(
    chunk_id TEXT PRIMARY KEY,
    embedding float[{_VEC_DIMENSIONS}]
)
"""


def _open_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    apply_migrations(conn, _SQLITE_MIGRATIONS_DIR)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(_VEC_TABLE_DDL)
    conn.commit()
    return conn


def _metadata_for_storage(chunk: ChunkRecord) -> dict[str, Any]:
    stored = dict(chunk.metadata)
    stored["node_types"] = list(chunk.node_types)
    return stored


def _upsert_chunks_sync(conn: sqlite3.Connection, chunks: list[ChunkRecord]) -> None:
    for chunk in chunks:
        metadata_json = json.dumps(_metadata_for_storage(chunk), sort_keys=True)
        conn.execute(
            """
            INSERT INTO chunk_metadata (
                chunk_id, document_id, chunk_index, chunk_hash, content_text, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                document_id = excluded.document_id,
                chunk_index = excluded.chunk_index,
                chunk_hash = excluded.chunk_hash,
                content_text = excluded.content_text,
                metadata_json = excluded.metadata_json
            """,
            (
                chunk.id,
                chunk.document_id,
                chunk.chunk_index,
                chunk.chunk_hash,
                chunk.content_text,
                metadata_json,
            ),
        )
        conn.execute(f"DELETE FROM {_VEC_TABLE} WHERE chunk_id = ?", (chunk.id,))
        conn.execute(
            f"INSERT INTO {_VEC_TABLE} (chunk_id, embedding) VALUES (?, ?)",
            (chunk.id, sqlite_vec.serialize_float32(chunk.embedding)),
        )
    conn.commit()


def _delete_chunks_by_document_id_sync(conn: sqlite3.Connection, document_id: str) -> None:
    rows = conn.execute(
        "SELECT chunk_id FROM chunk_metadata WHERE document_id = ?",
        (document_id,),
    ).fetchall()
    chunk_ids = [row[0] for row in rows]
    for chunk_id in chunk_ids:
        conn.execute(f"DELETE FROM {_VEC_TABLE} WHERE chunk_id = ?", (chunk_id,))
    conn.execute("DELETE FROM chunk_metadata WHERE document_id = ?", (document_id,))
    conn.commit()


def _get_chunk_hashes_sync(conn: sqlite3.Connection, document_id: str) -> dict[str, str]:
    rows = conn.execute(
        "SELECT chunk_id, chunk_hash FROM chunk_metadata WHERE document_id = ?",
        (document_id,),
    ).fetchall()
    return {row[0]: row[1] for row in rows}


def _search_result_from_row(row: sqlite3.Row, distance: float) -> SearchResult:
    meta = json.loads(row["metadata_json"])
    source_uri = str(meta.get("source_uri", ""))
    breadcrumbs = list(meta.get("breadcrumbs", []))
    result_metadata = {
        key: value
        for key, value in meta.items()
        if key not in ("source_uri", "breadcrumbs")
    }
    score = 1.0 - float(distance)
    return SearchResult(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        source_uri=source_uri,
        content_text=row["content_text"],
        score=score,
        breadcrumbs=breadcrumbs,
        metadata=result_metadata,
    )


def _search_sync(
    conn: sqlite3.Connection,
    query_vector: list[float],
    limit: int,
    filters: dict[str, Any] | None,
) -> list[SearchResult]:
    knn_k = limit
    params: list[Any] = [sqlite_vec.serialize_float32(query_vector)]
    prefix_clause = ""
    prefix_param: str | None = None
    if filters and "source_uri_prefix" in filters:
        prefix = str(filters["source_uri_prefix"])
        prefix_clause = "AND json_extract(m.metadata_json, '$.source_uri') LIKE ?"
        prefix_param = f"{prefix}%"
        knn_k = max(limit * 10, 50)

    params.append(knn_k)
    if prefix_param is not None:
        params.append(prefix_param)
    query = f"""
        SELECT
            m.chunk_id,
            m.document_id,
            m.content_text,
            m.metadata_json,
            {_VEC_TABLE}.distance AS distance
        FROM {_VEC_TABLE}
        INNER JOIN chunk_metadata m ON m.chunk_id = {_VEC_TABLE}.chunk_id
        WHERE {_VEC_TABLE}.embedding MATCH ?
          AND k = ?
          {prefix_clause}
        ORDER BY distance
        LIMIT ?
    """
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    return [_search_result_from_row(row, row["distance"]) for row in rows]


class SqliteVecStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None

    async def initialize(self) -> None:
        if self._thread is not None:
            return

        loop = asyncio.get_running_loop()
        init_event = threading.Event()
        init_error: list[BaseException | None] = [None]
        work_queue: queue.Queue = queue.Queue()
        db_path = self._db_path

        def worker() -> None:
            try:
                conn = _open_connection(db_path)
            except BaseException as exc:
                init_error[0] = exc
                return
            finally:
                init_event.set()
            while True:
                job = work_queue.get()
                if job is None:
                    conn.close()
                    break
                future, fn, args, kwargs = job
                try:
                    result = fn(conn, *args, **kwargs)
                except BaseException as exc:
                    loop.call_soon_threadsafe(future.set_exception, exc)
                else:
                    loop.call_soon_threadsafe(future.set_result, result)

        thread = threading.Thread(target=worker, name="SqliteVecStore", daemon=True)
        thread.start()
        await asyncio.to_thread(init_event.wait)
        if init_error[0] is not None:
            raise init_error[0]
        self._queue = work_queue
        self._thread = thread

    async def _run(self, fn, /, *args, **kwargs):
        if self._queue is None:
            raise RuntimeError("SqliteVecStore.initialize() must be called first")
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._queue.put((future, fn, args, kwargs))
        return await future

    async def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        await self._run(_upsert_chunks_sync, chunks)

    async def delete_chunks_by_document_id(self, document_id: str) -> None:
        await self._run(_delete_chunks_by_document_id_sync, document_id)

    async def get_chunk_hashes(self, document_id: str) -> dict[str, str]:
        return await self._run(_get_chunk_hashes_sync, document_id)

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        return await self._run(_search_sync, query_vector, limit, filters)
