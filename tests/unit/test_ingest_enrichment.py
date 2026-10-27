import asyncio

from vectorist.cache.models import DocumentStatus
from vectorist.cache.sqlite_cache import SqliteCache
from vectorist.embeddings.fake import FakeEmbeddingProvider
from vectorist.ir.models import MediaNode
from vectorist.pipeline.ingest import IngestService
from vectorist.storage.sqlite_vec_store import SqliteVecStore

_URI = "acme/docs/images.md"
_CONTENT = "# T\n\n![a](x.png)\n\n![b](x.png)\n\n![c](y.png)"


class CountingEmbedder:
    def __init__(self) -> None:
        self._inner = FakeEmbeddingProvider()
        self.document_calls = 0

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)


async def _open(tmp_path, enricher_id="fake"):
    cache = SqliteCache(tmp_path / "cache.db")
    store = SqliteVecStore(tmp_path / "vectors.db")
    await cache.initialize()
    await store.initialize()
    embedder = CountingEmbedder()
    service = IngestService(cache, store, embedder, enricher_id=enricher_id)
    return cache, store, embedder, service


async def _chunk_texts(store, embedder, document_id: str) -> list[str]:
    hits = await store.search(embedder.embed_query("image"), limit=100)
    return [hit.content_text for hit in hits if hit.document_id == document_id]


async def test_ingest_lists_pending_refs_once(tmp_path):
    _, _, _, service = await _open(tmp_path)

    result = await service.ingest_document(_URI, _CONTENT)

    assert result.status == "indexed"
    assert result.pending_assets == ["x.png", "y.png"]


async def test_pending_assets_empty_when_disabled(tmp_path):
    _, _, _, service = await _open(tmp_path, enricher_id=None)

    result = await service.ingest_document(_URI, _CONTENT)

    assert result.status == "indexed"
    assert result.pending_assets == []


async def test_skipped_ingest_still_lists_unlinked_refs(tmp_path):
    cache, _, _, service = await _open(tmp_path)
    first = await service.ingest_document(_URI, _CONTENT)

    second = await service.ingest_document(_URI, _CONTENT)

    assert second.status == "skipped"
    assert second.pending_assets == ["x.png", "y.png"]
    asset_hash = await cache.put_asset(b"x-bytes", "image/png")
    await cache.link_asset(first.document_id, "x.png", asset_hash)
    assert await service.pending_refs(first.document_id) == ["y.png"]


async def test_reindex_applies_text_and_updates_chunks(tmp_path):
    cache, store, embedder, service = await _open(tmp_path)
    first = await service.ingest_document(_URI, _CONTENT)
    asset_hash = await cache.put_asset(b"x-bytes", "image/png")
    await cache.link_asset(first.document_id, "x.png", asset_hash)
    await cache.put_enrichment(asset_hash, "fake", "a red circle")

    reindexed = await service.reindex(first.document_id)

    assert reindexed is not None
    assert reindexed.status == "indexed"
    texts = await _chunk_texts(store, embedder, first.document_id)
    assert any("a red circle" in text for text in texts)
    calls_before = embedder.document_calls
    again = await service.reindex(first.document_id)
    assert again is not None
    assert again.status == "indexed"
    assert embedder.document_calls == calls_before


async def test_reingest_keeps_image_text(tmp_path):
    cache, store, embedder, service = await _open(tmp_path)
    first = await service.ingest_document(_URI, _CONTENT)
    x_hash = await cache.put_asset(b"x-bytes", "image/png")
    y_hash = await cache.put_asset(b"y-bytes", "image/png")
    await cache.link_asset(first.document_id, "x.png", x_hash)
    await cache.link_asset(first.document_id, "y.png", y_hash)
    await cache.put_enrichment(x_hash, "fake", "a red circle")
    await service.reindex(first.document_id)

    edited = await service.ingest_document(_URI, _CONTENT + "\n\nA new paragraph.")

    assert edited.status == "indexed"
    assert edited.pending_assets == []
    texts = await _chunk_texts(store, embedder, first.document_id)
    assert any("a red circle" in text for text in texts)
    assert any("A new paragraph." in text for text in texts)


async def test_reindex_missing_document_returns_none(tmp_path):
    _, _, _, service = await _open(tmp_path)

    assert await service.reindex("no-such-document") is None


async def test_unchanged_legacy_snapshot_without_refs_is_reparsed(tmp_path):
    cache, _, _, service = await _open(tmp_path)
    content = "# T\n\n![a](x.png)"
    first = await service.ingest_document(_URI, content)
    snapshot = await cache.get_ir_snapshot(first.document_id)
    legacy = snapshot.model_copy(deep=True)
    for node in legacy.nodes.values():
        if isinstance(node, MediaNode):
            node.metadata.pop("ref")
    await cache.save_ir_snapshot(first.document_id, legacy)

    second = await service.ingest_document(_URI, content)

    assert second.status == "indexed"
    assert second.pending_assets == ["x.png"]


async def test_reindex_legacy_snapshot_fails_with_explicit_message(tmp_path):
    cache, _, _, service = await _open(tmp_path)
    first = await service.ingest_document(_URI, "# T\n\n![a](x.png)")
    legacy = (await cache.get_ir_snapshot(first.document_id)).model_copy(deep=True)
    for node in legacy.nodes.values():
        if isinstance(node, MediaNode):
            node.metadata.pop("ref")
    await cache.save_ir_snapshot(first.document_id, legacy)

    result = await service.reindex(first.document_id)

    assert result.status == "failed"
    assert "predates image refs" in result.error["message"]


async def test_failed_linked_ref_is_pending_again(tmp_path):
    cache, _, _, service = await _open(tmp_path)
    first = await service.ingest_document(_URI, _CONTENT)
    x_hash = await cache.put_asset(b"x-bytes", "image/png")
    y_hash = await cache.put_asset(b"y-bytes", "image/png")
    await cache.link_asset(first.document_id, "x.png", x_hash)
    await cache.link_asset(first.document_id, "y.png", y_hash)
    for asset_hash in (x_hash, y_hash):
        await cache.enqueue_job(asset_hash)
    assert await cache.claim_job() == x_hash
    await cache.fail_job(x_hash, "boom", max_attempts=1)
    await service.reindex(first.document_id)

    again = await service.ingest_document(_URI, _CONTENT)

    assert again.status == "skipped"
    assert again.pending_assets == ["x.png"]


class GatedEmbedder(CountingEmbedder):
    """Blocks ``embed_documents`` on an asyncio.Event once ``armed`` is set."""

    def __init__(self, loop) -> None:
        super().__init__()
        self._loop = loop
        self.armed = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.armed:
            self._loop.call_soon_threadsafe(self.entered.set)
            asyncio.run_coroutine_threadsafe(self.release.wait(), self._loop).result(
                timeout=10
            )
        return super().embed_documents(texts)


async def test_delete_document_waits_for_inflight_reindex(tmp_path):
    cache = SqliteCache(tmp_path / "cache.db")
    store = SqliteVecStore(tmp_path / "vectors.db")
    await cache.initialize()
    await store.initialize()
    embedder = GatedEmbedder(asyncio.get_running_loop())
    service = IngestService(cache, store, embedder, enricher_id="fake")
    first = await service.ingest_document(_URI, _CONTENT)
    document_id = first.document_id
    asset_hash = await cache.put_asset(b"x-bytes", "image/png")
    await cache.link_asset(document_id, "x.png", asset_hash)
    await cache.put_enrichment(asset_hash, "fake", "a red circle")
    embedder.armed = True

    reindex_task = asyncio.create_task(service.reindex(document_id))
    await embedder.entered.wait()
    delete_task = asyncio.create_task(service.delete_document(document_id))
    for _ in range(20):
        await asyncio.sleep(0)
    assert not delete_task.done()
    embedder.release.set()
    await reindex_task

    assert await delete_task is True
    assert await store.get_chunk_hashes(document_id) == {}
    assert await cache.get_document(document_id) is None


async def test_delete_document_missing_returns_false(tmp_path):
    _, _, _, service = await _open(tmp_path)

    assert await service.delete_document("no-such-document") is False


_HTML = (
    "<html><body><h1>Pooling</h1><p>connection pooling notes</p>"
    '<img src="x.png" alt="a"></body></html>'
)
_EMPTY_HTML = "<html><body></body></html>"


async def test_reindex_of_failed_document_is_skipped(tmp_path):
    cache, store, embedder, service = await _open(tmp_path)
    first = await service.ingest_document(_URI, _HTML, "text/html")
    assert first.status == "indexed"
    document_id = first.document_id
    texts_before = sorted(await _chunk_texts(store, embedder, document_id))
    failed = await service.ingest_document(_URI, _EMPTY_HTML, "text/html")
    assert failed.status == "failed"
    document = await cache.get_document(document_id)
    assert document.status == DocumentStatus.FAILED
    assert document.raw_content == _EMPTY_HTML
    assert await cache.get_ir_snapshot(document_id) is not None
    asset_hash = await cache.put_asset(b"x-bytes", "image/png")
    await cache.link_asset(document_id, "x.png", asset_hash)
    await cache.put_enrichment(asset_hash, "fake", "a red circle")

    result = await service.reindex(document_id)

    assert result is not None
    assert result.status == "skipped"
    assert result.document_id == document_id
    assert (await cache.get_document(document_id)).status == DocumentStatus.FAILED
    assert sorted(await _chunk_texts(store, embedder, document_id)) == texts_before
