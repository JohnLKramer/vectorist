import sqlite3

import pytest
from vectorist.cache.models import DocumentRecord, DocumentStatus
from vectorist.cache.sqlite_cache import SqliteCache
from vectorist.hashing import sha256_bytes


@pytest.fixture
async def cache(tmp_path):
    c = SqliteCache(tmp_path / "c.db")
    await c.initialize()
    return c


def _doc(doc_id: str) -> DocumentRecord:
    return DocumentRecord(
        id=doc_id,
        source_uri=f"file:///{doc_id}.md",
        content_hash="h",
        raw_content="x",
        status=DocumentStatus.PENDING,
    )


async def test_put_asset_is_idempotent_and_content_addressed(cache):
    data = b"x" * 10
    first = await cache.put_asset(data, "image/png")
    second = await cache.put_asset(data, "image/png")

    assert first == second == sha256_bytes(data)
    asset = await cache.get_asset(first)
    assert asset is not None
    assert asset.size == 10
    assert asset.data == data
    assert asset.content_type == "image/png"
    assert asset.hash == first


async def test_link_and_lookup(cache):
    await cache.upsert_document(_doc("d1"))
    h = await cache.put_asset(b"img", "image/png")

    await cache.link_asset("d1", "a.png", h)

    assert await cache.get_document_assets("d1") == {"a.png": h}
    assert await cache.document_ids_for_asset(h) == ["d1"]


async def test_enrichment_is_per_provider(cache):
    h = await cache.put_asset(b"img", "image/png")
    await cache.put_enrichment(h, "fake", "t")

    assert await cache.get_enrichments([h], "fake") == {h: "t"}
    assert await cache.get_enrichments([h], "ocr") == {}


async def test_claim_is_atomic_and_fifo(cache):
    await cache.enqueue_job("h1")
    await cache.enqueue_job("h2")

    assert await cache.claim_job() == "h1"
    assert await cache.claim_job() == "h2"
    assert await cache.claim_job() is None
    assert (await cache.job_counts())["RUNNING"] == 2


async def test_job_counts_zero_when_empty(cache):
    assert await cache.job_counts() == {
        "PENDING": 0,
        "RUNNING": 0,
        "DONE": 0,
        "FAILED": 0,
    }


async def test_fail_job_retries_then_fails(cache):
    await cache.enqueue_job("h1")
    await cache.claim_job()

    assert await cache.fail_job("h1", "boom", max_attempts=2) == "PENDING"
    assert await cache.claim_job() == "h1"
    assert await cache.fail_job("h1", "boom", max_attempts=2) == "FAILED"
    assert await cache.get_job_statuses(["h1"]) == {"h1": "FAILED"}


async def test_fail_job_on_missing_job_does_not_raise(cache):
    assert await cache.fail_job("nope", "boom", max_attempts=2) == "FAILED"


async def test_enqueue_resets_failed_job(cache):
    await cache.enqueue_job("h1")
    await cache.claim_job()
    await cache.fail_job("h1", "boom", max_attempts=1)
    assert await cache.get_job_statuses(["h1"]) == {"h1": "FAILED"}

    await cache.enqueue_job("h1")
    assert await cache.get_job_statuses(["h1"]) == {"h1": "PENDING"}
    assert await cache.claim_job() == "h1"

    await cache.enqueue_job("h1")
    assert await cache.get_job_statuses(["h1"]) == {"h1": "RUNNING"}


async def test_enqueue_resets_done_job(cache):
    await cache.enqueue_job("h1")
    await cache.claim_job()
    await cache.complete_job("h1")

    await cache.enqueue_job("h1")

    assert await cache.get_job_statuses(["h1"]) == {"h1": "PENDING"}
    assert await cache.claim_job() == "h1"


async def test_enqueue_reset_clears_attempts_and_error(cache, tmp_path):
    await cache.enqueue_job("h1")
    await cache.claim_job()
    await cache.fail_job("h1", "boom", max_attempts=1)

    await cache.enqueue_job("h1")

    conn = sqlite3.connect(tmp_path / "c.db")
    try:
        row = conn.execute(
            "SELECT status, attempts, error FROM enrichment_jobs WHERE asset_hash = ?",
            ("h1",),
        ).fetchone()
    finally:
        conn.close()
    assert row == ("PENDING", 0, None)


async def test_enqueue_leaves_pending_and_running_jobs_alone(cache):
    await cache.enqueue_job("h1")
    await cache.claim_job()
    await cache.fail_job("h1", "boom", max_attempts=3)
    await cache.enqueue_job("h2")
    await cache.claim_job()
    await cache.claim_job()
    await cache.enqueue_job("h3")

    await cache.enqueue_job("h1")
    await cache.enqueue_job("h2")
    await cache.enqueue_job("h3")

    assert await cache.get_job_statuses(["h1", "h2", "h3"]) == {
        "h1": "RUNNING",
        "h2": "RUNNING",
        "h3": "PENDING",
    }


async def test_requeue_running_jobs(cache):
    await cache.enqueue_job("h1")
    await cache.claim_job()

    assert await cache.requeue_running_jobs() == 1
    assert await cache.get_job_statuses(["h1"]) == {"h1": "PENDING"}


async def test_delete_document_collects_unreferenced_assets(cache):
    await cache.upsert_document(_doc("d1"))
    await cache.upsert_document(_doc("d2"))
    h = await cache.put_asset(b"img", "image/png")
    await cache.link_asset("d1", "a.png", h)
    await cache.link_asset("d2", "b.png", h)
    await cache.put_enrichment(h, "fake", "t")
    await cache.enqueue_job(h)

    await cache.delete_document("d1")

    assert await cache.get_asset(h) is not None
    assert await cache.get_enrichments([h], "fake") == {h: "t"}
    assert await cache.document_ids_for_asset(h) == ["d2"]

    await cache.delete_document("d2")

    assert await cache.get_asset(h) is None
    assert await cache.get_enrichments([h], "fake") == {}
    assert await cache.get_job_statuses([h]) == {}


async def test_delete_document_keeps_unlinked_assets_of_other_documents(cache):
    await cache.upsert_document(_doc("d1"))
    unlinked = await cache.put_asset(b"fresh-upload", "image/png")

    await cache.delete_document("d1")

    assert await cache.get_asset(unlinked) is not None
