import pytest
from vectorist.cache.models import DocumentRecord, DocumentStatus
from vectorist.cache.sqlite_cache import SqliteCache
from vectorist.ir.models import IRDocument, ParagraphNode


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
    para = ParagraphNode(
        node_id="p1",
        parent_id=None,
        children_ids=[],
        breadcrumbs=["a.md"],
        line_range=(1, 1),
        raw_content="Hello",
        metadata={},
    )
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
