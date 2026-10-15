import json

import httpx
import pytest

from vectorist.api.app import create_app


def _configure_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VECTORIST_CACHE_PATH", str(tmp_path / "cache.db"))
    monkeypatch.setenv("VECTORIST_VEC_PATH", str(tmp_path / "vec.db"))
    monkeypatch.setenv("VECTORIST_EMBEDDER", "fake")
    monkeypatch.delenv("VECTORIST_VECTOR_BACKEND", raising=False)
    monkeypatch.delenv("VECTORIST_DATABASE_URL", raising=False)
    monkeypatch.delenv("VECTORIST_PG_URL", raising=False)


async def test_ingest_stream_and_search(tmp_path, monkeypatch, fixtures_dir):
    _configure_env(monkeypatch, tmp_path)
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            body = (
                '{"source_uri":"repo/pooling.md","content_type":"text/markdown","content":'
                + json.dumps((fixtures_dir / "pooling.md").read_text())
                + "}\n"
            )
            response = await client.post(
                "/v1/ingest/stream",
                content=body,
                headers={"Content-Type": "application/x-ndjson"},
            )
            assert response.status_code == 200
            assert response.json()[0]["status"] in {"indexed", "accepted"}
            search = await client.post(
                "/v1/search", json={"query": "connection pool", "limit": 5}
            )
            assert search.status_code == 200
            assert len(search.json()["results"]) >= 1
            hit = search.json()["results"][0]
            assert hit["source_uri"] == "repo/pooling.md"
            assert "chunk_id" in hit
            assert "document_id" in hit
            assert "content_text" in hit
            assert "score" in hit
            assert "breadcrumbs" in hit
            assert "metadata" in hit


async def test_ingest_stream_skips_blank_lines_and_reports_malformed_lines(
    tmp_path, monkeypatch
):
    _configure_env(monkeypatch, tmp_path)
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    body = (
        "\n"
        "   \n"
        "{not json}\n"
        + json.dumps(
            {
                "source_uri": "notes/ok.md",
                "content": "# Notes\n\nA short note about pooling.\n",
            }
        )
        + "\n"
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            response = await client.post(
                "/v1/ingest/stream",
                content=body,
                headers={"Content-Type": "application/x-ndjson"},
            )
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 2
    assert results[0]["status"] == "failed"
    assert results[0]["document_id"] is None
    assert results[0]["error"]["code"]
    assert results[1]["status"] == "indexed"
    assert results[1]["source_uri"] == "notes/ok.md"


async def test_single_ingest_get_delete_and_health(tmp_path, monkeypatch):
    _configure_env(monkeypatch, tmp_path)
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            created = await client.post(
                "/v1/ingest",
                json={
                    "source_uri": "docs/pooling.md",
                    "content": "# Pool\n\nThe connection pool is shared.\n",
                },
            )
            assert created.status_code == 200
            payload = created.json()
            assert payload["status"] == "indexed"
            document_id = payload["document_id"]
            assert document_id

            fetched = await client.get(f"/v1/documents/{document_id}")
            assert fetched.status_code == 200
            document = fetched.json()
            assert document["id"] == document_id
            assert document["source_uri"] == "docs/pooling.md"
            assert document["content_hash"]
            assert document["status"] == "INDEXED"
            assert document["error_message"] is None
            assert "connection pool" in document["raw_content"]

            missing = await client.get("/v1/documents/missing")
            assert missing.status_code == 404

            deleted = await client.delete(f"/v1/documents/{document_id}")
            assert deleted.status_code == 204
            assert deleted.content == b""

            gone = await client.get(f"/v1/documents/{document_id}")
            assert gone.status_code == 404

            deleted_again = await client.delete(f"/v1/documents/{document_id}")
            assert deleted_again.status_code == 404

            search = await client.post(
                "/v1/search", json={"query": "connection pool", "limit": 5}
            )
            assert search.status_code == 200
            assert search.json()["results"] == []

            health = await client.get("/v1/health")
            assert health.status_code == 200
            assert health.json() == {
                "status": "ok",
                "cache": "ok",
                "vector_store": "ok",
                "embedder": "ok",
            }


async def test_search_applies_source_uri_prefix_and_default_limit(
    tmp_path, monkeypatch
):
    _configure_env(monkeypatch, tmp_path)
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            for source_uri in ("repo/a.md", "other/b.md"):
                response = await client.post(
                    "/v1/ingest",
                    json={
                        "source_uri": source_uri,
                        "content": f"# {source_uri}\n\nconnection pool details\n",
                    },
                )
                assert response.status_code == 200
                assert response.json()["status"] == "indexed"
            filtered = await client.post(
                "/v1/search",
                json={
                    "query": "connection pool",
                    "filters": {"source_uri_prefix": "repo/"},
                },
            )
    assert filtered.status_code == 200
    results = filtered.json()["results"]
    assert results
    assert all(item["source_uri"].startswith("repo/") for item in results)


async def test_unknown_vector_backend_fails_at_startup(tmp_path, monkeypatch):
    _configure_env(monkeypatch, tmp_path)
    monkeypatch.setenv("VECTORIST_VECTOR_BACKEND", "redis")
    app = create_app()
    with pytest.raises(RuntimeError, match="vector backend"):
        async with app.router.lifespan_context(app):
            pass


async def test_pgvector_without_database_url_fails_at_startup(tmp_path, monkeypatch):
    _configure_env(monkeypatch, tmp_path)
    monkeypatch.setenv("VECTORIST_VECTOR_BACKEND", "pgvector")
    app = create_app()
    with pytest.raises(RuntimeError, match="VECTORIST_DATABASE_URL"):
        async with app.router.lifespan_context(app):
            pass
