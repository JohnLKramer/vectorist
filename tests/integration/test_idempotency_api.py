import httpx

from vectorist.api.app import create_app


def _configure_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VECTORIST_CACHE_PATH", str(tmp_path / "cache.db"))
    monkeypatch.setenv("VECTORIST_VEC_PATH", str(tmp_path / "vec.db"))
    monkeypatch.setenv("VECTORIST_EMBEDDER", "fake")
    monkeypatch.delenv("VECTORIST_VECTOR_BACKEND", raising=False)
    monkeypatch.delenv("VECTORIST_DATABASE_URL", raising=False)
    monkeypatch.delenv("VECTORIST_PG_URL", raising=False)


async def test_second_identical_ingest_is_skipped(tmp_path, monkeypatch):
    _configure_env(monkeypatch, tmp_path)
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    body = {
        "source_uri": "docs/pooling.md",
        "content": "# Pool\n\nThe connection pool is shared.\n",
    }
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            first = await client.post("/v1/ingest", json=body)
            assert first.status_code == 200
            first_payload = first.json()
            assert first_payload["status"] == "indexed"
            document_id = first_payload["document_id"]
            assert document_id

            second = await client.post("/v1/ingest", json=body)
            assert second.status_code == 200
            second_payload = second.json()
            assert second_payload["status"] == "skipped"
            assert second_payload["document_id"] == document_id
