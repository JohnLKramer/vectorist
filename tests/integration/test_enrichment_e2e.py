import httpx
import pytest

from enrich_helpers import wait_for_enrichment
from vectorist.api.app import create_app
from vectorist.enrichment.fake import FakeImageEnricher

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64
DESCRIPTION = "fake description of image"


async def _ingest(client, uri, content):
    response = await client.post(
        "/v1/ingest", json={"source_uri": uri, "content": content}
    )
    assert response.status_code == 200
    return response.json()


async def _upload(client, document_id, ref, body=PNG):
    return await client.put(
        f"/v1/documents/{document_id}/assets",
        params={"ref": ref},
        content=body,
        headers={"Content-Type": "image/png"},
    )


async def _search_texts(client, query="image"):
    response = await client.post("/v1/search", json={"query": query, "limit": 10})
    return [hit["content_text"] for hit in response.json()["results"]]


async def _counts(client):
    return (await client.get("/v1/enrichment")).json()


async def test_two_documents_share_one_image(enrich_env, app_client):
    first = await _ingest(app_client, "a.md", "# A\n\nfirst\n\n![x](one.png)")
    second = await _ingest(app_client, "b.md", "# B\n\nsecond\n\n![y](two.png)")
    assert (await _upload(app_client, first["document_id"], "one.png")).status_code == 200
    assert (await _upload(app_client, second["document_id"], "two.png")).status_code == 200
    await wait_for_enrichment(app_client)

    texts = await _search_texts(app_client)
    assert sum(DESCRIPTION in text for text in texts) >= 2
    assert (await _counts(app_client))["done"] == 1


async def test_duplicate_src_in_one_document(enrich_env, app_client):
    doc = await _ingest(
        app_client, "a.md", "# A\n\n![a](x.png)\n\ntext between\n\n![a](x.png)"
    )
    assert doc["pending_assets"] == ["x.png"]
    assert (await _upload(app_client, doc["document_id"], "x.png")).status_code == 200
    await wait_for_enrichment(app_client)

    texts = await _search_texts(app_client)
    assert sum(text.count(DESCRIPTION) for text in texts) == 2


async def test_edit_keeps_image_text_without_second_upload(enrich_env, app_client):
    content = "# A\n\nbody\n\n![a](a.png)"
    doc = await _ingest(app_client, "a.md", content)
    await _upload(app_client, doc["document_id"], "a.png")
    await wait_for_enrichment(app_client)

    edited = await _ingest(app_client, "a.md", content + "\n\nadded paragraph")
    assert edited["pending_assets"] == []
    texts = await _search_texts(app_client)
    assert any(DESCRIPTION in text for text in texts)


async def test_restart_finishes_pending_job(enrich_env, sqlite_env, monkeypatch):
    monkeypatch.setenv("VECTORIST_ENRICH_POLL_SECONDS", "3600")
    first = create_app()
    async with first.router.lifespan_context(first):
        transport = httpx.ASGITransport(app=first)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            doc = await _ingest(client, "a.md", "# A\n\nbody\n\n![a](a.png)")
            assert (await _upload(client, doc["document_id"], "a.png")).status_code == 200
            assert (await _counts(client)) == {
                "pending": 1,
                "running": 0,
                "done": 0,
                "failed": 0,
            }

    monkeypatch.setenv("VECTORIST_ENRICH_POLL_SECONDS", "0.05")
    second = create_app()
    async with second.router.lifespan_context(second):
        transport = httpx.ASGITransport(app=second)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            await wait_for_enrichment(client)
            assert (await _counts(client))["done"] == 1
            texts = await _search_texts(client)
            assert any(DESCRIPTION in text for text in texts)


@pytest.fixture
def failing_describe(monkeypatch):
    monkeypatch.setenv("VECTORIST_ENRICH_MAX_ATTEMPTS", "1")
    original = FakeImageEnricher.describe

    def boom(self, image, content_type, alt_text):
        raise RuntimeError("provider down")

    monkeypatch.setattr(FakeImageEnricher, "describe", boom)
    return lambda: monkeypatch.setattr(FakeImageEnricher, "describe", original)


async def test_failed_enrichment_marks_node_failed(
    enrich_env, failing_describe, app_client
):
    doc = await _ingest(app_client, "a.md", "# A\n\nbody\n\n![a](a.png)")
    await _upload(app_client, doc["document_id"], "a.png")
    await wait_for_enrichment(app_client)
    assert (await _counts(app_client))["failed"] == 1
    assert not any(DESCRIPTION in text for text in await _search_texts(app_client))

    failing_describe()
    response = await _upload(app_client, doc["document_id"], "a.png")
    assert response.status_code == 200
    await wait_for_enrichment(app_client)
    assert any(DESCRIPTION in text for text in await _search_texts(app_client))



async def test_sync_rerun_retries_failed_image(
    enrich_env, failing_describe, app_client
):
    content = "# A\n\nbody\n\n![a](a.png)"
    doc = await _ingest(app_client, "a.md", content)
    await _upload(app_client, doc["document_id"], "a.png")
    await wait_for_enrichment(app_client)
    assert (await _counts(app_client))["failed"] == 1

    failing_describe()
    again = await _ingest(app_client, "a.md", content)
    assert again["status"] in ("skipped", "indexed")
    assert again["pending_assets"] == ["a.png"]
    response = await _upload(app_client, doc["document_id"], "a.png")
    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    await wait_for_enrichment(app_client)

    assert any(DESCRIPTION in text for text in await _search_texts(app_client))
    final = await _ingest(app_client, "a.md", content)
    assert final["pending_assets"] == []


async def test_linked_pending_or_done_image_is_not_listed_again(enrich_env, app_client):
    content = "# A\n\nbody\n\n![a](a.png)"
    doc = await _ingest(app_client, "a.md", content)
    await _upload(app_client, doc["document_id"], "a.png")
    assert (await _ingest(app_client, "a.md", content))["pending_assets"] == []
    await wait_for_enrichment(app_client)
    assert (await _ingest(app_client, "a.md", content))["pending_assets"] == []


async def test_new_provider_describes_image_done_by_old_provider(
    enrich_env, sqlite_env, monkeypatch
):
    first = create_app()
    async with first.router.lifespan_context(first):
        transport = httpx.ASGITransport(app=first)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            doc = await _ingest(client, "a.md", "# A\n\nfirst\n\n![a](a.png)")
            assert (await _upload(client, doc["document_id"], "a.png")).status_code == 200
            await wait_for_enrichment(client)
            assert (await _counts(client))["done"] == 1

    monkeypatch.setattr(FakeImageEnricher, "provider_id", property(lambda self: "fake2"))
    second = create_app()
    async with second.router.lifespan_context(second):
        transport = httpx.ASGITransport(app=second)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            doc = await _ingest(client, "b.md", "# B\n\nsecond\n\n![b](b.png)")
            assert doc["pending_assets"] == ["b.png"]
            response = await _upload(client, doc["document_id"], "b.png")
            assert response.status_code == 200
            assert response.json()["status"] == "queued"
            await wait_for_enrichment(client)

            asset_hash = response.json()["asset_hash"]
            texts = await second.state.cache.get_enrichments([asset_hash], "fake2")
            assert DESCRIPTION in texts[asset_hash]
            hits = await client.post("/v1/search", json={"query": "second", "limit": 10})
            assert any(
                hit["document_id"] == doc["document_id"]
                and DESCRIPTION in hit["content_text"]
                for hit in hits.json()["results"]
            )
