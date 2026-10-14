import pytest

from vectorist.embeddings.fake import FakeEmbeddingProvider
from vectorist.storage.models import ChunkRecord
from vectorist.storage.sqlite_vec_store import SqliteVecStore


def _chunk(
    *,
    chunk_id: str,
    document_id: str,
    chunk_index: int,
    text: str,
    source_uri: str,
    embedding: list[float],
    chunk_hash: str = "hash",
) -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        chunk_hash=chunk_hash,
        content_text=text,
        embedding=embedding,
        node_types=["paragraph"],
        metadata={
            "source_uri": source_uri,
            "breadcrumbs": ["Doc", f"Section {chunk_index}"],
        },
    )


@pytest.mark.asyncio
async def test_initialize_upsert_two_chunks_search_finds_nearest(tmp_path):
    provider = FakeEmbeddingProvider()
    store = SqliteVecStore(tmp_path / "vectors.db")
    await store.initialize()

    near_text = "connection pool sizing for postgres"
    far_text = "unrelated topic about gardening tools"

    chunks = [
        _chunk(
            chunk_id="c-near",
            document_id="doc-1",
            chunk_index=0,
            text=near_text,
            source_uri="acme/docs/pool.md",
            embedding=provider.embed_documents([near_text])[0],
            chunk_hash="near-hash",
        ),
        _chunk(
            chunk_id="c-far",
            document_id="doc-1",
            chunk_index=1,
            text=far_text,
            source_uri="acme/docs/pool.md",
            embedding=provider.embed_documents([far_text])[0],
            chunk_hash="far-hash",
        ),
    ]
    await store.upsert_chunks(chunks)

    query_vector = provider.embed_query(near_text)
    results = await store.search(query_vector, limit=2)

    assert len(results) == 2
    assert results[0].chunk_id == "c-near"
    assert results[0].document_id == "doc-1"
    assert results[0].source_uri == "acme/docs/pool.md"
    assert results[0].content_text == near_text
    assert results[0].breadcrumbs == ["Doc", "Section 0"]
    assert results[0].score >= results[1].score

    hashes = await store.get_chunk_hashes("doc-1")
    assert hashes == {"c-near": "near-hash", "c-far": "far-hash"}


@pytest.mark.asyncio
async def test_search_filters_by_source_uri_prefix(tmp_path):
    provider = FakeEmbeddingProvider()
    store = SqliteVecStore(tmp_path / "vectors.db")
    await store.initialize()

    query_text = "shared query text for filtering"
    query_vector = provider.embed_query(query_text)

    await store.upsert_chunks(
        [
            _chunk(
                chunk_id="match",
                document_id="d1",
                chunk_index=0,
                text=query_text,
                source_uri="acme/docs/guide.md",
                embedding=provider.embed_documents([query_text])[0],
            ),
            _chunk(
                chunk_id="other",
                document_id="d2",
                chunk_index=0,
                text=query_text,
                source_uri="other/docs/guide.md",
                embedding=provider.embed_documents([query_text])[0],
            ),
        ]
    )

    results = await store.search(
        query_vector,
        limit=10,
        filters={"source_uri_prefix": "acme/docs/"},
    )
    assert [r.chunk_id for r in results] == ["match"]


@pytest.mark.asyncio
async def test_delete_chunks_by_document_id(tmp_path):
    provider = FakeEmbeddingProvider()
    store = SqliteVecStore(tmp_path / "vectors.db")
    await store.initialize()

    text = "delete me"
    await store.upsert_chunks(
        [
            _chunk(
                chunk_id="c1",
                document_id="doc-del",
                chunk_index=0,
                text=text,
                source_uri="x.md",
                embedding=provider.embed_documents([text])[0],
            )
        ]
    )
    assert await store.get_chunk_hashes("doc-del") == {"c1": "hash"}

    await store.delete_chunks_by_document_id("doc-del")
    assert await store.get_chunk_hashes("doc-del") == {}
    assert await store.search(provider.embed_query(text), limit=5) == []
