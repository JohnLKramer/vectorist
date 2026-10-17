import json

import httpx

from vectorist.api.app import create_app


def _configure_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VECTORIST_CACHE_PATH", str(tmp_path / "cache.db"))
    monkeypatch.setenv("VECTORIST_VEC_PATH", str(tmp_path / "vec.db"))
    monkeypatch.setenv("VECTORIST_EMBEDDER", "fake")
    monkeypatch.delenv("VECTORIST_VECTOR_BACKEND", raising=False)
    monkeypatch.delenv("VECTORIST_DATABASE_URL", raising=False)
    monkeypatch.delenv("VECTORIST_PG_URL", raising=False)


async def _seed(client, uris):
    body = "".join(
        json.dumps({"source_uri": uri, "content": f"# {uri}\n\ntext\n"}) + "\n"
        for uri in uris
    )
    response = await client.post(
        "/v1/ingest/stream",
        content=body,
        headers={"Content-Type": "application/x-ndjson"},
    )
    assert response.status_code == 200


async def test_list_requires_prefix_and_valid_limit(tmp_path, monkeypatch):
    _configure_env(monkeypatch, tmp_path)
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            assert (await client.get("/v1/documents")).status_code == 422
            assert (await client.get("/v1/documents?source_prefix=")).status_code == 422
            for limit in (0, 1001):
                response = await client.get(
                    f"/v1/documents?source_prefix=a&limit={limit}"
                )
                assert response.status_code == 422


async def test_list_pages_through_every_document_once(tmp_path, monkeypatch):
    _configure_env(monkeypatch, tmp_path)
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            await _seed(client, ["r/1.md", "r/2.md", "r/3.md", "other/1.md"])
            seen = []
            cursor = None
            while True:
                url = "/v1/documents?source_prefix=r/&limit=2"
                if cursor:
                    url += f"&cursor={cursor}"
                body = (await client.get(url)).json()
                seen += [d["source_uri"] for d in body["documents"]]
                cursor = body["next_cursor"]
                if cursor is None:
                    break
            assert seen == ["r/1.md", "r/2.md", "r/3.md"]
            empty = (await client.get("/v1/documents?source_prefix=none/")).json()
            assert empty == {"documents": [], "next_cursor": None}
