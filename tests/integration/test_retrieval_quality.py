import httpx
import pytest

from vectorist.api.app import create_app

pytestmark = pytest.mark.embedder

DOCUMENTS = ("pooling.md", "retrieval/logging.md", "retrieval/auth.md", "retrieval/deployment.md")


def _uri(name: str) -> str:
    return f"file:///corpus/{name}"


@pytest.fixture
async def client(sqlite_env, monkeypatch, fixtures_dir):
    monkeypatch.setenv("VECTORIST_EMBEDDER", "fastembed")
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=300) as c:
            for name in DOCUMENTS:
                response = await c.post(
                    "/v1/ingest",
                    json={"source_uri": _uri(name), "content": (fixtures_dir / name).read_text()},
                )
                assert response.status_code == 200
                assert response.json()["status"] != "failed"
            yield c


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("keeping a set of ready database links so each request avoids reconnect delay", "pooling.md"),
        ("machine readable output lines for our monitoring tools", "retrieval/logging.md"),
        ("what happens when a user's access credential times out", "retrieval/auth.md"),
        ("how do we ship a new version gradually and undo a bad one", "retrieval/deployment.md"),
    ],
)
async def test_expected_document_ranks_first(client, query, expected):
    response = await client.post("/v1/search", json={"query": query, "limit": 5})
    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    assert results[0]["source_uri"] == _uri(expected)
