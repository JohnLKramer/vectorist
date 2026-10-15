import math
import os

import asyncpg
import pytest

from vectorist.embeddings.fake import FakeEmbeddingProvider
from vectorist.storage.models import ChunkRecord
from vectorist.storage.pgvector_store import PgVectorStore

pytestmark = pytest.mark.skipif(
    not os.getenv("VECTORIST_PG_URL"),
    reason="VECTORIST_PG_URL not set",
)


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


@pytest.fixture
async def store():
    dsn = os.environ["VECTORIST_PG_URL"]
    pg_store = PgVectorStore(dsn)
    await pg_store.initialize()
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("TRUNCATE vector_chunks")
    finally:
        await conn.close()
    try:
        yield pg_store
    finally:
        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute("TRUNCATE vector_chunks")
        finally:
            await conn.close()
        await pg_store.close()


@pytest.mark.asyncio
async def test_initialize_upsert_two_chunks_search_finds_nearest(store):
    provider = FakeEmbeddingProvider()

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
async def test_search_filters_by_source_uri_prefix(store):
    provider = FakeEmbeddingProvider()

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


def _unit_vector(*, index: int, dimensions: int = 384) -> list[float]:
    values = [0.0] * dimensions
    values[index] = 1.0
    return values


def _near_query_vector(i: int, dimensions: int = 384) -> list[float]:
    values = _unit_vector(index=0, dimensions=dimensions)
    values[2 + i % (dimensions - 2)] = 0.01 * (1 + i // (dimensions - 2))
    return values


@pytest.mark.asyncio
async def test_search_prefix_filter_keeps_match_outside_hnsw_candidate_window(store):
    # Disabling sequential scans makes the planner pick the HNSW index, which
    # only returns hnsw.ef_search (default 40) candidates before filtering.
    index_preferring_store = PgVectorStore(
        f"{os.environ['VECTORIST_PG_URL']}?enable_seqscan=off"
    )
    await index_preferring_store.initialize()
    query_vector = _unit_vector(index=0)
    nearer_than_ef_search = 200
    noise_chunks = [
        _chunk(
            chunk_id=f"noise-{i}",
            document_id=f"d-noise-{i}",
            chunk_index=0,
            text=f"noise {i}",
            source_uri=f"other/docs/noise-{i}.md",
            embedding=_near_query_vector(i),
        )
        for i in range(nearer_than_ef_search)
    ]
    match_chunk = _chunk(
        chunk_id="prefix-match",
        document_id="d-match",
        chunk_index=0,
        text="acme match",
        source_uri="acme/docs/target.md",
        embedding=_unit_vector(index=1),
    )
    await store.upsert_chunks([*noise_chunks, match_chunk])

    try:
        results = await index_preferring_store.search(
            query_vector,
            limit=1,
            filters={"source_uri_prefix": "acme/docs/"},
        )
        assert [r.chunk_id for r in results] == ["prefix-match"]
        assert results[0].source_uri == "acme/docs/target.md"
        assert math.isclose(results[0].score, 0.0, abs_tol=1e-6)
    finally:
        await index_preferring_store.close()


@pytest.mark.asyncio
async def test_delete_chunks_by_document_id(store):
    provider = FakeEmbeddingProvider()

    delete_text = "delete me"
    keep_text = "keep me"
    await store.upsert_chunks(
        [
            _chunk(
                chunk_id="c1",
                document_id="doc-del",
                chunk_index=0,
                text=delete_text,
                source_uri="delete.md",
                embedding=provider.embed_documents([delete_text])[0],
                chunk_hash="del-hash",
            ),
            _chunk(
                chunk_id="c2",
                document_id="doc-keep",
                chunk_index=0,
                text=keep_text,
                source_uri="keep.md",
                embedding=provider.embed_documents([keep_text])[0],
                chunk_hash="keep-hash",
            ),
        ]
    )
    assert await store.get_chunk_hashes("doc-del") == {"c1": "del-hash"}
    assert await store.get_chunk_hashes("doc-keep") == {"c2": "keep-hash"}

    await store.delete_chunks_by_document_id("doc-del")
    assert await store.get_chunk_hashes("doc-del") == {}
    assert await store.get_chunk_hashes("doc-keep") == {"c2": "keep-hash"}

    delete_hits = await store.search(provider.embed_query(delete_text), limit=5)
    assert all(r.document_id != "doc-del" for r in delete_hits)
    keep_hits = await store.search(provider.embed_query(keep_text), limit=5)
    assert any(r.chunk_id == "c2" and r.document_id == "doc-keep" for r in keep_hits)
