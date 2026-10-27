from __future__ import annotations

import asyncio
import sqlite3
import time

import pytest

from vectorist.cache.sqlite_cache import SqliteCache
from vectorist.embeddings.fake import FakeEmbeddingProvider
from vectorist.enrichment.worker import EnrichmentWorker
from vectorist.ir.models import EnrichmentStatus, MediaNode
from vectorist.pipeline.ingest import IngestResult, IngestService
from vectorist.storage.sqlite_vec_store import SqliteVecStore

_PROVIDER = "stub"
_CONTENT = "# T\n\n![d](a.png)"


class StubEnricher:
    def __init__(self, result: str | Exception = "a red circle") -> None:
        self._result = result
        self.calls: list[tuple[bytes, str, str]] = []

    @property
    def provider_id(self) -> str:
        return _PROVIDER

    def describe(self, image: bytes, content_type: str, alt_text: str) -> str:
        self.calls.append((image, content_type, alt_text))
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


async def _open(tmp_path, enricher, max_attempts=3):
    cache = SqliteCache(tmp_path / "cache.db")
    store = SqliteVecStore(tmp_path / "vectors.db")
    await cache.initialize()
    await store.initialize()
    embedder = FakeEmbeddingProvider()
    service = IngestService(cache, store, embedder, enricher_id=_PROVIDER)
    worker = EnrichmentWorker(
        cache, service, enricher, poll_seconds=0.01, max_attempts=max_attempts
    )
    return cache, store, embedder, service, worker


async def _ingest_with_asset(cache, service, uri="acme/docs/a.md"):
    result = await service.ingest_document(uri, _CONTENT)
    asset_hash = await cache.put_asset(b"a-bytes", "image/png")
    await cache.link_asset(result.document_id, "a.png", asset_hash)
    await cache.enqueue_job(asset_hash)
    return result.document_id, asset_hash


async def _chunk_texts(store, embedder, document_id: str) -> list[str]:
    hits = await store.search(embedder.embed_query("image"), limit=100)
    return [hit.content_text for hit in hits if hit.document_id == document_id]


def _job_error(tmp_path, asset_hash: str) -> str | None:
    conn = sqlite3.connect(tmp_path / "cache.db")
    try:
        row = conn.execute(
            "SELECT error FROM enrichment_jobs WHERE asset_hash = ?", (asset_hash,)
        ).fetchone()
    finally:
        conn.close()
    return row[0]


async def _media_status(cache, document_id: str) -> EnrichmentStatus:
    snapshot = await cache.get_ir_snapshot(document_id)
    statuses = [
        node.enrichment_status
        for node in snapshot.nodes.values()
        if isinstance(node, MediaNode)
    ]
    assert len(statuses) == 1
    return statuses[0]


async def test_run_once_enriches_and_reindexes(tmp_path):
    cache, store, embedder, service, worker = await _open(tmp_path, StubEnricher())
    document_id, asset_hash = await _ingest_with_asset(cache, service)

    assert await worker.run_once() is True

    assert await cache.get_job_statuses([asset_hash]) == {asset_hash: "DONE"}
    assert await cache.get_enrichments([asset_hash], _PROVIDER) == {
        asset_hash: "a red circle"
    }
    texts = await _chunk_texts(store, embedder, document_id)
    assert any("a red circle" in text for text in texts)
    assert await _media_status(cache, document_id) == EnrichmentStatus.DONE
    assert await worker.run_once() is False


async def test_alt_text_reaches_provider(tmp_path):
    enricher = StubEnricher()
    cache, _, _, service, worker = await _open(tmp_path, enricher)
    await _ingest_with_asset(cache, service)

    await worker.run_once()

    assert enricher.calls == [(b"a-bytes", "image/png", "d")]


async def test_shared_image_enriches_once(tmp_path):
    enricher = StubEnricher()
    cache, store, embedder, service, worker = await _open(tmp_path, enricher)
    first_id, asset_hash = await _ingest_with_asset(cache, service, "acme/docs/a.md")
    second_id, second_hash = await _ingest_with_asset(
        cache, service, "acme/docs/b.md"
    )
    assert second_hash == asset_hash

    assert await worker.run_once() is True
    assert await worker.run_once() is False

    assert len(enricher.calls) == 1
    for document_id in (first_id, second_id):
        texts = await _chunk_texts(store, embedder, document_id)
        assert any("a red circle" in text for text in texts)


async def test_provider_error_retries_then_fails(tmp_path):
    enricher = StubEnricher(RuntimeError("boom"))
    cache, _, _, service, worker = await _open(tmp_path, enricher, max_attempts=2)
    document_id, asset_hash = await _ingest_with_asset(cache, service)

    assert await worker.run_once() is True
    assert await cache.get_job_statuses([asset_hash]) == {asset_hash: "PENDING"}
    assert await _media_status(cache, document_id) == EnrichmentStatus.PENDING
    assert await worker.run_once() is True

    assert len(enricher.calls) == 2
    assert await cache.get_job_statuses([asset_hash]) == {asset_hash: "FAILED"}
    assert _job_error(tmp_path, asset_hash) == "boom"
    assert await _media_status(cache, document_id) == EnrichmentStatus.FAILED
    assert await worker.run_once() is False


async def test_empty_text_counts_as_failure(tmp_path):
    enricher = StubEnricher("  ")
    cache, _, _, service, worker = await _open(tmp_path, enricher, max_attempts=2)
    document_id, asset_hash = await _ingest_with_asset(cache, service)

    await worker.run_once()
    await worker.run_once()

    assert await cache.get_job_statuses([asset_hash]) == {asset_hash: "FAILED"}
    assert await cache.get_enrichments([asset_hash], _PROVIDER) == {}
    assert await _media_status(cache, document_id) == EnrichmentStatus.FAILED


async def test_document_deleted_during_describe_is_dropped(tmp_path):
    cache = SqliteCache(tmp_path / "cache.db")
    store = SqliteVecStore(tmp_path / "vectors.db")
    await cache.initialize()
    await store.initialize()
    embedder = FakeEmbeddingProvider()
    service = IngestService(cache, store, embedder, enricher_id=_PROVIDER)
    document_id, asset_hash = await _ingest_with_asset(cache, service)
    loop = asyncio.get_running_loop()

    class DeletingEnricher(StubEnricher):
        def describe(self, image: bytes, content_type: str, alt_text: str) -> str:
            asyncio.run_coroutine_threadsafe(
                self._delete(), loop
            ).result(timeout=5)
            return super().describe(image, content_type, alt_text)

        async def _delete(self) -> None:
            await cache.delete_document(document_id)
            await store.delete_chunks_by_document_id(document_id)

    enricher = DeletingEnricher()
    worker = EnrichmentWorker(
        cache, service, enricher, poll_seconds=0.01, max_attempts=3
    )

    assert await worker.run_once() is True

    assert len(enricher.calls) == 1
    assert await _chunk_texts(store, embedder, document_id) == []
    assert await cache.get_enrichments([asset_hash], _PROVIDER) == {}
    assert await cache.get_document(document_id) is None


async def test_recover_requeues_running_jobs(tmp_path):
    enricher = StubEnricher()
    cache, _, _, service, worker = await _open(tmp_path, enricher)
    _, asset_hash = await _ingest_with_asset(cache, service)
    assert await cache.claim_job() == asset_hash
    assert await worker.run_once() is False

    await worker.recover()

    assert await worker.run_once() is True
    assert await cache.get_job_statuses([asset_hash]) == {asset_hash: "DONE"}


async def test_job_stays_running_until_reindex_finishes(tmp_path):
    cache, store, embedder, service, worker = await _open(tmp_path, StubEnricher())
    document_id, asset_hash = await _ingest_with_asset(cache, service)
    real_reindex = service.reindex
    statuses_during_reindex: list[str] = []

    async def recording_reindex(doc_id: str):
        statuses_during_reindex.append(
            (await cache.get_job_statuses([asset_hash]))[asset_hash]
        )
        return await real_reindex(doc_id)

    service.reindex = recording_reindex

    assert await worker.run_once() is True

    assert statuses_during_reindex == ["RUNNING"]
    assert await cache.get_job_statuses([asset_hash]) == {asset_hash: "DONE"}
    texts = await _chunk_texts(store, embedder, document_id)
    assert any("a red circle" in text for text in texts)


async def test_crash_mid_reindex_recovers_without_describing_again(tmp_path):
    enricher = StubEnricher()
    cache, store, embedder, service, worker = await _open(tmp_path, enricher)
    document_id, asset_hash = await _ingest_with_asset(cache, service)
    real_reindex = service.reindex

    async def crashing_reindex(doc_id: str):
        raise RuntimeError("process died mid re-index")

    service.reindex = crashing_reindex
    with pytest.raises(RuntimeError, match="process died"):
        await worker.run_once()

    assert await cache.get_job_statuses([asset_hash]) == {asset_hash: "RUNNING"}
    service.reindex = real_reindex

    await worker.recover()
    assert await worker.run_once() is True

    assert len(enricher.calls) == 1
    assert await cache.get_job_statuses([asset_hash]) == {asset_hash: "DONE"}
    texts = await _chunk_texts(store, embedder, document_id)
    assert any("a red circle" in text for text in texts)
    assert await _media_status(cache, document_id) == EnrichmentStatus.DONE


async def test_cancel_mid_reindex_leaves_job_running(tmp_path):
    cache, _, _, service, worker = await _open(tmp_path, StubEnricher())
    _, asset_hash = await _ingest_with_asset(cache, service)

    async def cancelled_reindex(doc_id: str):
        raise asyncio.CancelledError

    service.reindex = cancelled_reindex
    with pytest.raises(asyncio.CancelledError):
        await worker.run_once()

    assert await cache.get_job_statuses([asset_hash]) == {asset_hash: "RUNNING"}


async def test_document_linked_during_describe_is_reindexed(tmp_path):
    enricher = StubEnricher()
    cache, store, embedder, service, worker = await _open(tmp_path, enricher)
    first_id, asset_hash = await _ingest_with_asset(cache, service, "acme/docs/a.md")
    second = await service.ingest_document("acme/docs/b.md", _CONTENT)
    real_put = cache.put_enrichment

    async def linking_put(asset: str, provider_id: str, text: str) -> None:
        await real_put(asset, provider_id, text)
        await cache.link_asset(second.document_id, "a.png", asset_hash)

    cache.put_enrichment = linking_put

    assert await worker.run_once() is True

    for document_id in (first_id, second.document_id):
        texts = await _chunk_texts(store, embedder, document_id)
        assert any("a red circle" in text for text in texts)


async def test_start_and_stop_process_jobs(tmp_path):
    cache, store, embedder, service, worker = await _open(tmp_path, StubEnricher())
    worker.start()
    try:
        document_id, asset_hash = await _ingest_with_asset(cache, service)

        for _ in range(500):
            if await cache.get_job_statuses([asset_hash]) == {asset_hash: "DONE"}:
                break
            await asyncio.sleep(0.01)
        assert await cache.get_job_statuses([asset_hash]) == {asset_hash: "DONE"}
        texts = await _chunk_texts(store, embedder, document_id)
        assert any("a red circle" in text for text in texts)
    finally:
        await worker.stop()


async def test_unexpected_error_ending_polling_task_is_logged(tmp_path, caplog):
    cache, _, _, service, worker = await _open(tmp_path, StubEnricher())

    async def broken_claim() -> str | None:
        raise RuntimeError("cache is gone")

    cache.claim_job = broken_claim
    def logged() -> bool:
        return any(
            r.levelname == "ERROR"
            and r.exc_info is not None
            and "cache is gone" in str(r.exc_info[1])
            for r in caplog.records
        )

    worker.start()
    for _ in range(500):
        if logged():
            break
        await asyncio.sleep(0.01)

    assert logged()
    await worker.stop()


async def _open_with_delay(tmp_path, enricher, max_attempts, retry_delay_seconds):
    cache = SqliteCache(tmp_path / "cache.db")
    store = SqliteVecStore(tmp_path / "vectors.db")
    await cache.initialize()
    await store.initialize()
    service = IngestService(cache, store, FakeEmbeddingProvider(), enricher_id=_PROVIDER)
    worker = EnrichmentWorker(
        cache,
        service,
        enricher,
        poll_seconds=0.01,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
    )
    return cache, service, worker


async def test_retryable_failure_waits_before_returning(tmp_path):
    cache, service, worker = await _open_with_delay(
        tmp_path, StubEnricher(RuntimeError("boom")), 3, 0.05
    )
    _, asset_hash = await _ingest_with_asset(cache, service)

    started = time.monotonic()
    assert await worker.run_once() is True
    elapsed = time.monotonic() - started

    assert await cache.get_job_statuses([asset_hash]) == {asset_hash: "PENDING"}
    assert elapsed >= 0.05


async def test_terminal_failure_does_not_wait(tmp_path, monkeypatch):
    cache, service, worker = await _open_with_delay(
        tmp_path, StubEnricher(RuntimeError("boom")), 1, 30.0
    )
    _, asset_hash = await _ingest_with_asset(cache, service)
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def recording_sleep(delay: float, *args, **kwargs):
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr("vectorist.enrichment.worker.asyncio.sleep", recording_sleep)

    assert await worker.run_once() is True

    assert await cache.get_job_statuses([asset_hash]) == {asset_hash: "FAILED"}
    assert sleeps == []


async def test_retry_delay_defaults_to_poll_seconds(tmp_path, monkeypatch):
    cache = SqliteCache(tmp_path / "cache.db")
    store = SqliteVecStore(tmp_path / "vectors.db")
    await cache.initialize()
    await store.initialize()
    service = IngestService(cache, store, FakeEmbeddingProvider(), enricher_id=_PROVIDER)
    worker = EnrichmentWorker(
        cache, service, StubEnricher(RuntimeError("boom")), poll_seconds=0.02, max_attempts=3
    )
    await _ingest_with_asset(cache, service)
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def recording_sleep(delay: float, *args, **kwargs):
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr("vectorist.enrichment.worker.asyncio.sleep", recording_sleep)

    assert await worker.run_once() is True

    assert sleeps == [0.02]


async def test_skipped_reindex_counts_as_success(tmp_path):
    cache, _, _, service, worker = await _open(tmp_path, StubEnricher())
    document_id, asset_hash = await _ingest_with_asset(cache, service)

    async def skipped_reindex(doc_id: str):
        return IngestResult(source_uri="acme/docs/a.md", document_id=doc_id, status="skipped")

    service.reindex = skipped_reindex

    assert await worker.run_once() is True

    assert await cache.get_job_statuses([asset_hash]) == {asset_hash: "DONE"}
