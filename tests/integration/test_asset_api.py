import pytest

from enrich_helpers import wait_for_enrichment
from vectorist.pipeline.ingest import IngestResult, IngestService

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64
CONTENT = "# T\n\npooling text\n\n![a](a.png)"


async def _ingest(client, uri="a.md", content=CONTENT):
    response = await client.post(
        "/v1/ingest", json={"source_uri": uri, "content": content}
    )
    assert response.status_code == 200
    return response.json()


async def _upload(client, document_id, ref="a.png", body=PNG, content_type="image/png"):
    return await client.put(
        f"/v1/documents/{document_id}/assets",
        params={"ref": ref},
        content=body,
        headers={"Content-Type": content_type},
    )


async def _search_texts(client, query):
    response = await client.post("/v1/search", json={"query": query, "limit": 20})
    return [hit["content_text"] for hit in response.json()["results"]]


async def test_ingest_lists_pending_assets(enrich_env, app_client):
    body = await _ingest(app_client)
    assert body["pending_assets"] == ["a.png"]


async def test_upload_enriches_and_makes_text_searchable(enrich_env, app_client):
    doc = await _ingest(app_client)
    response = await _upload(app_client, doc["document_id"])
    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    await wait_for_enrichment(app_client)
    texts = await _search_texts(app_client, "image")
    assert any("fake description of image" in text for text in texts)
    again = await _ingest(app_client)
    assert again["status"] == "skipped"
    assert again["pending_assets"] == []


async def test_text_searchable_while_job_pending(enrich_env, app_client):
    await _ingest(app_client)
    texts = await _search_texts(app_client, "pooling text")
    assert any("pooling text" in text for text in texts)


async def test_reupload_of_enriched_image_is_done(enrich_env, app_client):
    first = await _ingest(app_client)
    await _upload(app_client, first["document_id"])
    await wait_for_enrichment(app_client)
    second = await _ingest(app_client, uri="b.md", content="# B\n\nother\n\n![a](a.png)")
    response = await _upload(app_client, second["document_id"])
    assert response.status_code == 200
    assert response.json()["status"] == "done"
    texts = await _search_texts(app_client, "image")
    assert sum("fake description of image" in text for text in texts) >= 2


async def _ingest_enriched_then_second(client):
    first = await _ingest(client)
    await _upload(client, first["document_id"])
    await wait_for_enrichment(client)
    return await _ingest(client, uri="b.md", content="# B\n\nother\n\n![a](a.png)")


async def test_reupload_with_failed_reindex_is_server_error(
    enrich_env, app_client, monkeypatch
):
    second = await _ingest_enriched_then_second(app_client)

    async def failed_reindex(self, document_id):
        return IngestResult(
            source_uri="b.md",
            document_id=document_id,
            status="failed",
            error={"code": "INGEST_FAILED", "message": "vector store exploded"},
        )

    monkeypatch.setattr(IngestService, "reindex", failed_reindex)
    response = await _upload(app_client, second["document_id"])
    assert response.status_code == 500
    assert "vector store exploded" in response.text


async def test_reupload_with_document_gone_during_reindex_is_not_found(
    enrich_env, app_client, monkeypatch
):
    second = await _ingest_enriched_then_second(app_client)

    async def missing_reindex(self, document_id):
        return None

    monkeypatch.setattr(IngestService, "reindex", missing_reindex)
    response = await _upload(app_client, second["document_id"])
    assert response.status_code == 404
    assert "document not found" in response.text


async def test_reupload_when_reindex_skips_failed_document_is_queued(
    enrich_env, app_client, monkeypatch
):
    second = await _ingest_enriched_then_second(app_client)

    async def skipped_reindex(self, document_id):
        return IngestResult(source_uri="b.md", document_id=document_id, status="skipped")

    monkeypatch.setattr(IngestService, "reindex", skipped_reindex)
    response = await _upload(app_client, second["document_id"])
    assert response.status_code == 200
    assert response.json()["status"] == "queued"


async def test_upload_errors(enrich_env, app_client):
    doc = await _ingest(
        app_client, content="# T\n\n![a](a.png)\n\n![](<>)"
    )
    document_id = doc["document_id"]
    assert (await _upload(app_client, "missing")).status_code == 404
    assert (await _upload(app_client, document_id, ref="zzz.png")).status_code == 404
    assert (await _upload(app_client, document_id, ref="")).status_code == 404
    assert (
        await _upload(app_client, document_id, content_type="image/svg+xml")
    ).status_code == 415


@pytest.fixture
def small_assets(monkeypatch):
    monkeypatch.setenv("VECTORIST_MAX_ASSET_BYTES", "100")


async def test_upload_too_large(enrich_env, small_assets, app_client):
    doc = await _ingest(app_client)
    response = await _upload(app_client, doc["document_id"], body=b"0" * 101)
    assert response.status_code == 413


async def test_upload_without_enricher_is_conflict(app_client):
    doc = await _ingest(app_client)
    assert doc["pending_assets"] == []
    assert (await _upload(app_client, doc["document_id"])).status_code == 409


async def test_enrichment_counts(enrich_env, app_client):
    body = (await app_client.get("/v1/enrichment")).json()
    assert body == {"pending": 0, "running": 0, "done": 0, "failed": 0}


async def test_delete_document_removes_assets(enrich_env, app_client):
    doc = await _ingest(app_client)
    await _upload(app_client, doc["document_id"])
    await wait_for_enrichment(app_client)
    assert (await app_client.delete(f"/v1/documents/{doc['document_id']}")).status_code == 204
    body = (await app_client.get("/v1/enrichment")).json()
    assert body == {"pending": 0, "running": 0, "done": 0, "failed": 0}
