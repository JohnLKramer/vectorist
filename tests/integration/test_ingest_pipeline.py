import threading

from vectorist.adapters.markdown import MarkdownAdapter
from vectorist.cache.models import DocumentStatus
from vectorist.cache.sqlite_cache import SqliteCache
from vectorist.chunking.chunker import Chunker
from vectorist.embeddings.fake import FakeEmbeddingProvider
from vectorist.hashing import sha256_text
from vectorist.pipeline.ingest import IngestService
from vectorist.storage.sqlite_vec_store import SqliteVecStore

_TWO_SECTIONS = """# Guide

## Stable

The stable paragraph stays exactly as written.

## Mutable

The original mutable paragraph discusses widgets.
"""


class RecordingEmbedder:
    def __init__(self) -> None:
        self._inner = FakeEmbeddingProvider()
        self.document_calls: list[list[str]] = []
        self.document_call_threads: list[int] = []

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        self.document_call_threads.append(threading.get_ident())
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)


class FlakyEmbedder:
    def __init__(self) -> None:
        self._inner = FakeEmbeddingProvider()
        self.failures_left = 1

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.failures_left:
            self.failures_left -= 1
            raise RuntimeError("embed boom")
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)


class WrongDimensionsOnceEmbedder:
    def __init__(self) -> None:
        self._inner = FakeEmbeddingProvider()
        self.failures_left = 1

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.failures_left:
            self.failures_left -= 1
            return [[1.0, 0.0, 0.0] for _ in texts]
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)


async def _open_service(tmp_path, embedder=None):
    cache = SqliteCache(tmp_path / "cache.db")
    store = SqliteVecStore(tmp_path / "vectors.db")
    await cache.initialize()
    await store.initialize()
    if embedder is None:
        embedder = FakeEmbeddingProvider()
    service = IngestService(cache, store, embedder)
    return cache, store, embedder, service


async def test_ingest_pooling_twice_skips_second_and_search_returns_chunk(
    tmp_path, fixtures_dir
):
    cache, store, embedder, service = await _open_service(tmp_path)
    content = (fixtures_dir / "pooling.md").read_text(encoding="utf-8")
    source_uri = "acme/docs/pooling.md"

    first = await service.ingest_document(source_uri, content)
    assert first.status == "indexed"
    assert first.document_id is not None
    assert first.error is None
    assert first.source_uri == source_uri

    stored = await cache.get_document_by_uri(source_uri)
    assert stored is not None
    assert stored.id == first.document_id
    assert stored.status == DocumentStatus.INDEXED
    assert stored.content_hash == sha256_text(content)
    assert await cache.get_ir_snapshot(first.document_id) is not None

    second = await service.ingest_document(source_uri, content)
    assert second.status == "skipped"
    assert second.document_id == first.document_id
    assert second.error is None

    hits = await store.search(embedder.embed_query("connection pooling"), limit=5)
    matched = [hit for hit in hits if hit.document_id == first.document_id]
    assert matched
    assert matched[0].source_uri == source_uri
    assert matched[0].breadcrumbs == [
        "pooling.md",
        "Connection Pooling",
        "Overview",
    ]
    assert "Connection pooling reuses database connections" in matched[0].content_text
    assert matched[0].metadata["line_range"] == [5, 17]
    assert matched[0].metadata["language"] == "python"
    assert "CODE_BLOCK" in matched[0].metadata["node_types"]


async def test_changed_document_reuses_id_and_is_not_skipped(tmp_path):
    _cache, _store, _embedder, service = await _open_service(tmp_path)
    source_uri = "guide.md"
    first = await service.ingest_document(source_uri, _TWO_SECTIONS)
    changed = _TWO_SECTIONS.replace("widgets", "gizmos")

    second = await service.ingest_document(source_uri, changed)

    assert first.status == "indexed"
    assert second.status == "indexed"
    assert second.document_id == first.document_id
    assert second.document_id is not None


async def test_text_plain_returns_parse_error_and_writes_nothing(tmp_path):
    cache, store, _embedder, service = await _open_service(tmp_path)
    source_uri = "notes.txt"

    result = await service.ingest_document(
        source_uri,
        "plain text",
        content_type="text/plain",
    )

    assert result.status == "failed"
    assert result.document_id is None
    assert result.source_uri == source_uri
    assert result.error == {
        "code": "PARSE_ERROR",
        "message": "unsupported content type: text/plain",
    }
    assert await cache.get_document_by_uri(source_uri) is None
    assert await store.get_chunk_hashes("unused") == {}


async def test_content_change_embeds_only_the_changed_chunk(tmp_path):
    embedder = RecordingEmbedder()
    _cache, store, _embedder, service = await _open_service(tmp_path, embedder)
    source_uri = "guide.md"

    first = await service.ingest_document(source_uri, _TWO_SECTIONS)
    assert first.document_id is not None
    document_id = first.document_id

    hashes = await store.get_chunk_hashes(document_id)
    assert set(hashes) == {f"{document_id}:0", f"{document_id}:1"}
    hits = await store.search(embedder.embed_query("stable paragraph"), limit=5)
    by_id = {hit.chunk_id: hit for hit in hits if hit.document_id == document_id}
    assert set(by_id) == set(hashes)
    for chunk_id, hit in by_id.items():
        assert hashes[chunk_id] == sha256_text(hit.content_text + document_id)
        assert hit.source_uri == source_uri

    assert len(embedder.document_calls) == 1
    assert len(embedder.document_calls[0]) == 2
    embedder.document_calls.clear()

    changed = _TWO_SECTIONS.replace("widgets", "gizmos")
    second = await service.ingest_document(source_uri, changed)
    assert second.status == "indexed"
    assert second.document_id == document_id

    assert len(embedder.document_calls) == 1
    assert len(embedder.document_calls[0]) == 1
    embedded = embedder.document_calls[0][0]
    assert "gizmos" in embedded
    assert "stable paragraph" not in embedded

    updated = await store.get_chunk_hashes(document_id)
    assert updated[f"{document_id}:0"] == hashes[f"{document_id}:0"]
    assert updated[f"{document_id}:1"] != hashes[f"{document_id}:1"]


async def test_removed_section_deletes_chunk_without_embedding_unchanged(tmp_path):
    embedder = RecordingEmbedder()
    _cache, store, _embedder, service = await _open_service(tmp_path, embedder)
    source_uri = "guide.md"
    first = await service.ingest_document(source_uri, _TWO_SECTIONS)
    assert first.document_id is not None
    document_id = first.document_id
    embedder.document_calls.clear()

    shorter = """# Guide

## Stable

The stable paragraph stays exactly as written.
"""
    second = await service.ingest_document(source_uri, shorter)

    assert second.status == "indexed"
    assert second.document_id == document_id
    assert embedder.document_calls == []
    assert await store.get_chunk_hashes(document_id) == {
        f"{document_id}:0": sha256_text(
            Chunker(max_tokens=512)
            .chunk_document(MarkdownAdapter().parse(source_uri, shorter))[0]
            .content_text
            + document_id
        )
    }


async def test_failed_ingest_is_retried_instead_of_skipped(tmp_path, fixtures_dir):
    embedder = FlakyEmbedder()
    cache, store, _embedder, service = await _open_service(tmp_path, embedder)
    content = (fixtures_dir / "pooling.md").read_text(encoding="utf-8")
    source_uri = "acme/docs/pooling.md"

    failed = await service.ingest_document(source_uri, content)
    assert failed.status == "failed"
    assert failed.document_id is not None
    assert failed.error == {"code": "INGEST_FAILED", "message": "embed boom"}

    stored = await cache.get_document_by_uri(source_uri)
    assert stored is not None
    assert stored.id == failed.document_id
    assert stored.status == DocumentStatus.FAILED
    assert stored.error_message == "embed boom"

    retried = await service.ingest_document(source_uri, content)
    assert retried.status == "indexed"
    assert retried.document_id == failed.document_id
    assert retried.error is None
    assert (await cache.get_document_by_uri(source_uri)).status == DocumentStatus.INDEXED
    hits = await store.search(embedder.embed_query("connection pooling"), limit=5)
    assert any(hit.document_id == failed.document_id for hit in hits)


async def test_same_hash_parsed_document_is_ingested_again(tmp_path, fixtures_dir):
    cache, _store, _embedder, service = await _open_service(tmp_path)
    content = (fixtures_dir / "pooling.md").read_text(encoding="utf-8")
    source_uri = "acme/docs/pooling.md"
    first = await service.ingest_document(source_uri, content)
    assert first.document_id is not None

    await cache.update_status(first.document_id, DocumentStatus.PARSED)
    again = await service.ingest_document(source_uri, content)

    assert again.status == "indexed"
    assert again.document_id == first.document_id
    stored = await cache.get_document_by_uri(source_uri)
    assert stored is not None
    assert stored.status == DocumentStatus.INDEXED


async def test_failed_vector_write_is_rolled_back_so_retry_indexes_chunks(
    tmp_path, fixtures_dir
):
    embedder = WrongDimensionsOnceEmbedder()
    cache, store, _embedder, service = await _open_service(tmp_path, embedder)
    content = (fixtures_dir / "pooling.md").read_text(encoding="utf-8")
    source_uri = "acme/docs/pooling.md"

    failed = await service.ingest_document(source_uri, content)
    assert failed.status == "failed"
    assert failed.document_id is not None
    assert await store.get_chunk_hashes(failed.document_id) == {}

    retried = await service.ingest_document(source_uri, content)
    assert retried.status == "indexed"
    hits = await store.search(embedder.embed_query("connection pooling"), limit=5)
    assert any(hit.document_id == failed.document_id for hit in hits)


async def test_embed_documents_runs_off_the_event_loop_thread(tmp_path):
    embedder = RecordingEmbedder()
    _cache, _store, _embedder, service = await _open_service(tmp_path, embedder)

    result = await service.ingest_document("guide.md", _TWO_SECTIONS)

    assert result.status == "indexed"
    assert embedder.document_call_threads
    assert threading.get_ident() not in embedder.document_call_threads
