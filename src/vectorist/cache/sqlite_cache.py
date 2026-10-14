from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from vectorist.cache.models import DocumentRecord, DocumentStatus
from vectorist.hashing import sha256_text
from vectorist.ir.models import IRDocument
from vectorist.ir.serialize import ir_from_json, ir_to_json
from vectorist.migrations.runner import apply_migrations

_SQLITE_MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[3] / "migrations" / "sqlite"
)


class SqliteCache:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        apply_migrations(conn, _SQLITE_MIGRATIONS_DIR)
        self._conn = conn

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SqliteCache.initialize() must be called first")
        return self._conn

    async def get_document_by_uri(self, source_uri: str) -> DocumentRecord | None:
        return await asyncio.to_thread(self._get_document_by_uri_sync, source_uri)

    def _get_document_by_uri_sync(self, source_uri: str) -> DocumentRecord | None:
        row = self._connection().execute(
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

    async def upsert_document(self, doc: DocumentRecord) -> None:
        await asyncio.to_thread(self._upsert_document_sync, doc)

    def _upsert_document_sync(self, doc: DocumentRecord) -> None:
        self._connection().execute(
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
        self._connection().commit()

    async def save_ir_snapshot(self, document_id: str, ir_tree: IRDocument) -> None:
        await asyncio.to_thread(self._save_ir_snapshot_sync, document_id, ir_tree)

    def _save_ir_snapshot_sync(self, document_id: str, ir_tree: IRDocument) -> None:
        ir_json = ir_to_json(ir_tree)
        node_count = len(ir_tree.nodes)
        snapshot_hash = sha256_text(ir_json)
        self._connection().execute(
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
        self._connection().commit()

    async def get_ir_snapshot(self, document_id: str) -> IRDocument | None:
        return await asyncio.to_thread(self._get_ir_snapshot_sync, document_id)

    def _get_ir_snapshot_sync(self, document_id: str) -> IRDocument | None:
        row = self._connection().execute(
            "SELECT ir_tree_json FROM ir_snapshots WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if row is None:
            return None
        return ir_from_json(row["ir_tree_json"])

    async def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        error_message: str | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._update_status_sync, document_id, status, error_message
        )

    def _update_status_sync(
        self,
        document_id: str,
        status: DocumentStatus,
        error_message: str | None,
    ) -> None:
        self._connection().execute(
            """
            UPDATE documents
            SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (status.value, error_message, document_id),
        )
        self._connection().commit()


def _row_to_document(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        id=row["id"],
        source_uri=row["source_uri"],
        content_hash=row["content_hash"],
        raw_content=row["raw_content"],
        status=DocumentStatus(row["status"]),
        error_message=row["error_message"],
    )
