import json

import httpx

from vectorist.api.app import create_app


def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("VECTORIST_CACHE_PATH", str(tmp_path / "cache.db"))
    monkeypatch.setenv("VECTORIST_VEC_PATH", str(tmp_path / "vec.db"))
    monkeypatch.setenv("VECTORIST_EMBEDDER", "fake")
    for name in ("VECTORIST_VECTOR_BACKEND", "VECTORIST_DATABASE_URL", "VECTORIST_PG_URL"):
        monkeypatch.delenv(name, raising=False)


async def test_html_ingest_and_search(tmp_path, monkeypatch, fixtures_dir):
    _env(monkeypatch, tmp_path)
    html = (fixtures_dir / "html" / "guide.html").read_text()
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            single = await client.post(
                "/v1/ingest",
                json={"source_uri": "https://docs.example/pool", "content": html,
                      "content_type": "text/html"},
            )
            assert single.json()["status"] == "indexed"
            line = json.dumps({"source_uri": "https://docs.example/pool2", "content": html,
                               "content_type": "text/html"}) + "\n"
            stream = await client.post("/v1/ingest/stream", content=line)
            assert stream.json()[0]["status"] == "indexed"

            hits = (await client.post(
                "/v1/search", json={"query": "reuse connections latency", "limit": 5}
            )).json()["results"]
            assert hits
            hit = hits[0]
            assert hit["source_uri"].startswith("https://docs.example/pool")
            assert hit["breadcrumbs"][0] == "Pooling"
            assert len(hit["metadata"]["line_range"]) == 2
            assert "Copyright" not in hit["content_text"]

            bad = await client.post(
                "/v1/ingest",
                json={"source_uri": "x", "content": "x", "content_type": "application/pdf"},
            )
            assert bad.json()["status"] == "failed"
            empty = await client.post(
                "/v1/ingest",
                json={"source_uri": "https://e/x", "content": "<nav>menu</nav>",
                      "content_type": "text/html"},
            )
            assert empty.json()["status"] == "failed"
