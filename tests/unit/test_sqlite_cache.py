import asyncio
import sqlite3
from unittest.mock import patch

import pytest
from vectorist.cache.models import DocumentRecord, DocumentStatus
from vectorist.cache import sqlite_cache as sqlite_cache_module
from vectorist.cache.sqlite_cache import DocumentNotFoundError, SqliteCache
from vectorist.ir.models import IRDocument, ParagraphNode


def _sample_paragraph() -> ParagraphNode:
    return ParagraphNode(
        node_id="p1",
        parent_id=None,
        children_ids=[],
        breadcrumbs=["a.md"],
        line_range=(1, 1),
        raw_content="Hello",
        metadata={},
    )


@pytest.mark.asyncio
async def test_initialize_raises_when_db_path_invalid(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    cache = SqliteCache(blocker / "cache.db")
    with pytest.raises(OSError):
        await asyncio.wait_for(cache.initialize(), timeout=1.0)


@pytest.mark.asyncio
async def test_initialize_raises_when_open_connection_fails(tmp_path):
    cache = SqliteCache(tmp_path / "cache.db")

    def boom(_db_path):
        raise RuntimeError("open failed")

    with patch.object(sqlite_cache_module, "_open_connection", boom):
        with pytest.raises(RuntimeError, match="open failed"):
            await asyncio.wait_for(cache.initialize(), timeout=1.0)


@pytest.mark.asyncio
async def test_upsert_and_get_document(tmp_path):
    cache = SqliteCache(tmp_path / "cache.db")
    await cache.initialize()
    doc = DocumentRecord(
        id="d1",
        source_uri="a.md",
        content_hash="abc",
        raw_content="# A",
        status=DocumentStatus.PENDING,
    )
    await cache.upsert_document(doc)
    got = await cache.get_document_by_uri("a.md")
    assert got is not None
    assert got.content_hash == "abc"
    assert got.raw_content == "# A"


@pytest.mark.asyncio
async def test_ir_snapshot_round_trip(tmp_path):
    cache = SqliteCache(tmp_path / "cache.db")
    await cache.initialize()
    doc = DocumentRecord(
        id="d1",
        source_uri="a.md",
        content_hash="abc",
        raw_content="# A",
        status=DocumentStatus.PENDING,
    )
    await cache.upsert_document(doc)
    para = _sample_paragraph()
    ir = IRDocument(source_uri="a.md", root_id="p1", nodes={"p1": para})
    await cache.save_ir_snapshot("d1", ir)
    restored = await cache.get_ir_snapshot("d1")
    assert restored is not None
    assert restored.source_uri == "a.md"
    assert restored.nodes["p1"].raw_content == "Hello"


@pytest.mark.asyncio
async def test_update_status(tmp_path):
    cache = SqliteCache(tmp_path / "cache.db")
    await cache.initialize()
    doc = DocumentRecord(
        id="d1",
        source_uri="a.md",
        content_hash="abc",
        raw_content="# A",
        status=DocumentStatus.PENDING,
    )
    await cache.upsert_document(doc)
    await cache.update_status("d1", DocumentStatus.FAILED, error_message="parse error")
    got = await cache.get_document_by_uri("a.md")
    assert got is not None
    assert got.status == DocumentStatus.FAILED
    assert got.error_message == "parse error"


@pytest.mark.asyncio
async def test_foreign_keys_enabled(tmp_path):
    cache = SqliteCache(tmp_path / "cache.db")
    await cache.initialize()
    fk = await cache._run(
        lambda conn: conn.execute("PRAGMA foreign_keys").fetchone()[0]
    )
    assert fk == 1


@pytest.mark.asyncio
async def test_save_ir_snapshot_rejects_unknown_document(tmp_path):
    cache = SqliteCache(tmp_path / "cache.db")
    await cache.initialize()
    ir = IRDocument(
        source_uri="ghost.md",
        root_id="p1",
        nodes={"p1": _sample_paragraph()},
    )
    with pytest.raises(sqlite3.IntegrityError):
        await cache.save_ir_snapshot("missing-id", ir)


@pytest.mark.asyncio
async def test_update_status_raises_for_missing_document(tmp_path):
    cache = SqliteCache(tmp_path / "cache.db")
    await cache.initialize()
    with pytest.raises(DocumentNotFoundError):
        await cache.update_status("missing-id", DocumentStatus.FAILED)


@pytest.mark.asyncio
async def test_overlapping_operations(tmp_path):
    cache = SqliteCache(tmp_path / "cache.db")
    await cache.initialize()
    docs = [
        DocumentRecord(
            id=f"d{i}",
            source_uri=f"{i}.md",
            content_hash=f"h{i}",
            raw_content=f"# {i}",
            status=DocumentStatus.PENDING,
        )
        for i in range(8)
    ]
    for doc in docs:
        await cache.upsert_document(doc)
    await asyncio.gather(
        *[cache.get_document_by_uri(f"{i}.md") for i in range(8)],
        *[
            cache.update_status(f"d{i}", DocumentStatus.PARSED)
            for i in range(8)
        ],
        cache.upsert_document(
            DocumentRecord(
                id="d0",
                source_uri="0.md",
                content_hash="h0-updated",
                raw_content="# 0 revised",
                status=DocumentStatus.INDEXED,
            )
        ),
    )
    got = await cache.get_document_by_uri("0.md")
    assert got is not None
    assert got.status in (DocumentStatus.PARSED, DocumentStatus.INDEXED)
    assert got.content_hash == "h0-updated"
    for i in range(1, 8):
        row = await cache.get_document_by_uri(f"{i}.md")
        assert row is not None
        assert row.status == DocumentStatus.PARSED
        assert row.content_hash == f"h{i}"
