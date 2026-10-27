import pytest

pytestmark = pytest.mark.backends

CONTENT = "# Failing\n\nconnection pooling notes\n"


async def test_failed_upsert_is_recorded_and_retry_indexes(app_client):
    store = app_client._transport.app.state.store
    original = store.upsert_chunks
    calls = 0

    async def fail_once(records):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")
        return await original(records)

    store.upsert_chunks = fail_once
    try:
        body = {"source_uri": "fail/doc.md", "content": CONTENT}
        failed = (await app_client.post("/v1/ingest", json=body)).json()
        assert failed["status"] == "failed"
        assert failed["error"]["code"] == "INGEST_FAILED"

        document = (await app_client.get(f"/v1/documents/{failed['document_id']}")).json()
        assert document["status"] == "FAILED"
        assert "boom" in document["error_message"]
        search = (await app_client.post("/v1/search", json={"query": "pooling", "limit": 5})).json()
        assert search["results"] == []

        retry = (await app_client.post("/v1/ingest", json=body)).json()
    finally:
        del store.upsert_chunks
    assert retry["status"] == "indexed"
    assert retry["document_id"] == failed["document_id"]
    search = (await app_client.post("/v1/search", json={"query": "pooling", "limit": 5})).json()
    assert [hit["source_uri"] for hit in search["results"]] == ["fail/doc.md"]
