from __future__ import annotations

import asyncio
import queue
import sqlite3
import threading
from pathlib import Path

from vectorist.cache.models import DocumentRecord, DocumentStatus, DocumentSummary
from vectorist.hashing import sha256_text
from vectorist.ir.models import IRDocument
from vectorist.ir.serialize import ir_from_json, ir_to_json
from vectorist.migrations.runner import apply_migrations, packaged_migrations_dir

_SQLITE_MIGRATIONS_DIR = packaged_migrations_dir("sqlite")


class DocumentNotFoundError(LookupError):
    def __init__(self, document_id: str) -> None:
        super().__init__(f"document not found: {document_id}")
        self.document_id = document_id


def _open_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    apply_migrations(conn, _SQLITE_MIGRATIONS_DIR)
    return conn


class SqliteCache:
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
                    try:
                        conn.rollback()
                    finally:
                        loop.call_soon_threadsafe(future.set_exception, exc)
                else:
                    loop.call_soon_threadsafe(future.set_result, result)

        thread = threading.Thread(target=worker, name="SqliteCache", daemon=True)
        thread.start()
        await asyncio.to_thread(init_event.wait)
        if init_error[0] is not None:
            raise init_error[0]
        self._queue = work_queue
        self._thread = thread

    async def _run(self, fn, /, *args, **kwargs):
        if self._queue is None:
            raise RuntimeError("SqliteCache.initialize() must be called first")
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._queue.put((future, fn, args, kwargs))
        return await future

    async def ping(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("SqliteCache worker thread is not running")
        await self._run(_ping_sync)

    async def get_document_by_uri(self, source_uri: str) -> DocumentRecord | None:
        return await self._run(_get_document_by_uri_sync, source_uri)

    async def get_document(self, document_id: str) -> DocumentRecord | None:
        return await self._run(_get_document_sync, document_id)

    async def delete_document(self, document_id: str) -> None:
        await self._run(_delete_document_sync, document_id)

    async def list_documents_by_prefix(
        self, prefix: str, limit: int, after_uri: str | None = None
    ) -> list[DocumentSummary]:
        return await self._run(_list_documents_by_prefix_sync, prefix, limit, after_uri)

    async def upsert_document(self, doc: DocumentRecord) -> None:
        await self._run(_upsert_document_sync, doc)

    async def save_ir_snapshot(self, document_id: str, ir_tree: IRDocument) -> None:
        await self._run(_save_ir_snapshot_sync, document_id, ir_tree)

    async def get_ir_snapshot(self, document_id: str) -> IRDocument | None:
        return await self._run(_get_ir_snapshot_sync, document_id)

    async def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        error_message: str | None = None,
    ) -> None:
        await self._run(
            _update_status_sync, document_id, status, error_message
        )


def _ping_sync(conn: sqlite3.Connection) -> None:
    conn.execute("SELECT 1 FROM documents LIMIT 1").fetchone()


def _get_document_by_uri_sync(
    conn: sqlite3.Connection, source_uri: str
) -> DocumentRecord | None:
    row = conn.execute(
        """
        SELECT id, source_uri, content_hash, raw_content, status, error_message
        FROM documents
        WHERE source_uri = ?
        """,
        (source_uri,),
    ).fetchone()
    if row is None:
        return None
    return _row_to_document(row)


def _get_document_sync(
    conn: sqlite3.Connection, document_id: str
) -> DocumentRecord | None:
    row = conn.execute(
        """
        SELECT id, source_uri, content_hash, raw_content, status, error_message
        FROM documents
        WHERE id = ?
        """,
        (document_id,),
    ).fetchone()
    if row is None:
        return None
    return _row_to_document(row)


def _delete_document_sync(conn: sqlite3.Connection, document_id: str) -> None:
    conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    conn.commit()


def _list_documents_by_prefix_sync(
    conn: sqlite3.Connection, prefix: str, limit: int, after_uri: str | None
) -> list[DocumentSummary]:
    rows = conn.execute(
        """
        SELECT id, source_uri, content_hash, status
        FROM documents
        WHERE substr(source_uri, 1, length(?1)) = ?1
          AND (?2 IS NULL OR source_uri > ?2)
        ORDER BY source_uri
        LIMIT ?3
        """,
        (prefix, after_uri, limit),
    ).fetchall()
    return [
        DocumentSummary(
            id=row["id"],
            source_uri=row["source_uri"],
            content_hash=row["content_hash"],
            status=DocumentStatus(row["status"]),
        )
        for row in rows
    ]


def _upsert_document_sync(conn: sqlite3.Connection, doc: DocumentRecord) -> None:
    conn.execute(
        """
        INSERT INTO documents (
            id, source_uri, content_hash, raw_content, status, error_message
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            source_uri = excluded.source_uri,
            content_hash = excluded.content_hash,
            raw_content = excluded.raw_content,
            status = excluded.status,
            error_message = excluded.error_message,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            doc.id,
            doc.source_uri,
            doc.content_hash,
            doc.raw_content,
            doc.status.value,
            doc.error_message,
        ),
    )
    conn.commit()


def _save_ir_snapshot_sync(
    conn: sqlite3.Connection, document_id: str, ir_tree: IRDocument
) -> None:
    ir_json = ir_to_json(ir_tree)
    node_count = len(ir_tree.nodes)
    snapshot_hash = sha256_text(ir_json)
    conn.execute(
        """
        INSERT INTO ir_snapshots (document_id, ir_tree_json, node_count, snapshot_hash)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(document_id) DO UPDATE SET
            ir_tree_json = excluded.ir_tree_json,
            node_count = excluded.node_count,
            snapshot_hash = excluded.snapshot_hash
        """,
        (document_id, ir_json, node_count, snapshot_hash),
    )
    conn.commit()


def _get_ir_snapshot_sync(
    conn: sqlite3.Connection, document_id: str
) -> IRDocument | None:
    row = conn.execute(
        "SELECT ir_tree_json FROM ir_snapshots WHERE document_id = ?",
        (document_id,),
    ).fetchone()
    if row is None:
        return None
    return ir_from_json(row["ir_tree_json"])


def _update_status_sync(
    conn: sqlite3.Connection,
    document_id: str,
    status: DocumentStatus,
    error_message: str | None,
) -> None:
    cursor = conn.execute(
        """
        UPDATE documents
        SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (status.value, error_message, document_id),
    )
    if cursor.rowcount == 0:
        conn.rollback()
        raise DocumentNotFoundError(document_id)
    conn.commit()


def _row_to_document(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        id=row["id"],
        source_uri=row["source_uri"],
        content_hash=row["content_hash"],
        raw_content=row["raw_content"],
        status=DocumentStatus(row["status"]),
        error_message=row["error_message"],
    )
