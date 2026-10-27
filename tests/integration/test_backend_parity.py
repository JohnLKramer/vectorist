import json

import pytest

pytestmark = pytest.mark.backends

NDJSON = {"Content-Type": "application/x-ndjson"}


async def _seed(client, uris):
    body = "".join(
        json.dumps({"source_uri": uri, "content": f"# {uri}\n\ntext\n"}) + "\n"
        for uri in uris
    )
    response = await client.post("/v1/ingest/stream", content=body, headers=NDJSON)
    assert response.status_code == 200


async def test_ingest_stream_and_search(app_client, fixtures_dir):
    body = (
        '{"source_uri":"repo/pooling.md","content_type":"text/markdown","content":'
        + json.dumps((fixtures_dir / "pooling.md").read_text())
        + "}\n"
    )
    response = await app_client.post("/v1/ingest/stream", content=body, headers=NDJSON)
    assert response.status_code == 200
    assert response.json()[0]["status"] in {"indexed", "accepted"}
    search = await app_client.post(
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
    assert hit["metadata"]["line_range"] == [5, 17]
    assert hit["metadata"]["language"] == "python"
    assert "CODE_BLOCK" in hit["metadata"]["node_types"]


async def test_ingest_stream_skips_blank_lines_and_reports_malformed_lines(app_client):
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
    response = await app_client.post("/v1/ingest/stream", content=body, headers=NDJSON)
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 2
    assert results[0]["status"] == "failed"
    assert results[0]["document_id"] is None
    assert results[0]["error"]["code"]
    assert results[1]["status"] == "indexed"
    assert results[1]["source_uri"] == "notes/ok.md"


async def test_single_ingest_get_delete_and_health(app_client):
    created = await app_client.post(
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

    fetched = await app_client.get(f"/v1/documents/{document_id}")
    assert fetched.status_code == 200
    document = fetched.json()
    assert document["id"] == document_id
    assert document["source_uri"] == "docs/pooling.md"
    assert document["content_hash"]
    assert document["status"] == "INDEXED"
    assert document["error_message"] is None
    assert "connection pool" in document["raw_content"]

    missing = await app_client.get("/v1/documents/missing")
    assert missing.status_code == 404

    deleted = await app_client.delete(f"/v1/documents/{document_id}")
    assert deleted.status_code == 204
    assert deleted.content == b""

    gone = await app_client.get(f"/v1/documents/{document_id}")
    assert gone.status_code == 404

    deleted_again = await app_client.delete(f"/v1/documents/{document_id}")
    assert deleted_again.status_code == 404

    search = await app_client.post(
        "/v1/search", json={"query": "connection pool", "limit": 5}
    )
    assert search.status_code == 200
    assert search.json()["results"] == []

    health = await app_client.get("/v1/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "cache": "ok",
        "vector_store": "ok",
        "embedder": "ok",
    }


async def test_search_applies_source_uri_prefix_and_default_limit(app_client):
    for source_uri in ("repo/a.md", "other/b.md"):
        response = await app_client.post(
            "/v1/ingest",
            json={
                "source_uri": source_uri,
                "content": f"# {source_uri}\n\nconnection pool details\n",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "indexed"
    filtered = await app_client.post(
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


@pytest.mark.parametrize("limit", [0, -1, 4097])
async def test_search_rejects_out_of_range_limit(app_client, limit):
    response = await app_client.post(
        "/v1/search", json={"query": "connection pool", "limit": limit}
    )
    assert response.status_code == 422


@pytest.mark.parametrize("limit", [1, 4096])
async def test_search_accepts_limit_bounds(app_client, limit):
    await app_client.post(
        "/v1/ingest",
        json={"source_uri": "a.md", "content": "# A\n\nconnection pool\n"},
    )
    response = await app_client.post(
        "/v1/search", json={"query": "connection pool", "limit": limit}
    )
    assert response.status_code == 200
    assert len(response.json()["results"]) == 1


async def test_indented_code_and_html_blocks_are_searchable(app_client):
    content = (
        "# Legacy\n\n"
        "    indented_code = 1\n\n"
        "<div>Raw HTML note</div>\n"
    )
    created = await app_client.post(
        "/v1/ingest", json={"source_uri": "docs/legacy.md", "content": content}
    )
    assert created.json()["status"] == "indexed"
    search = await app_client.post(
        "/v1/search", json={"query": "indented_code", "limit": 10}
    )
    texts = "\n".join(hit["content_text"] for hit in search.json()["results"])
    assert "indented_code = 1" in texts
    assert "Raw HTML note" in texts


async def test_second_identical_ingest_is_skipped(app_client):
    body = {
        "source_uri": "docs/pooling.md",
        "content": "# Pool\n\nThe connection pool is shared.\n",
    }
    first = await app_client.post("/v1/ingest", json=body)
    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["status"] == "indexed"
    document_id = first_payload["document_id"]
    assert document_id

    second = await app_client.post("/v1/ingest", json=body)
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["status"] == "skipped"
    assert second_payload["document_id"] == document_id


async def test_changed_document_replaces_old_chunks(app_client):
    def content(second_section: str) -> str:
        return (
            "# Guide\n\n## Stable\n\nThe stable section never changes.\n\n"
            f"## Volatile\n\n{second_section}\n"
        )

    first = await app_client.post(
        "/v1/ingest",
        json={"source_uri": "docs/guide.md", "content": content("Zebra marker text.")},
    )
    assert first.json()["status"] == "indexed"
    second = await app_client.post(
        "/v1/ingest",
        json={"source_uri": "docs/guide.md", "content": content("Walrus marker text.")},
    )
    assert second.json()["status"] == "indexed"
    assert second.json()["document_id"] == first.json()["document_id"]

    search = await app_client.post("/v1/search", json={"query": "marker text", "limit": 20})
    texts = "\n".join(hit["content_text"] for hit in search.json()["results"])
    assert "Walrus marker text." in texts
    assert "Zebra" not in texts
    assert "stable section never changes" in texts


async def test_list_requires_prefix_and_valid_limit(app_client):
    assert (await app_client.get("/v1/documents")).status_code == 422
    assert (await app_client.get("/v1/documents?source_prefix=")).status_code == 422
    for limit in (0, 1001):
        response = await app_client.get(f"/v1/documents?source_prefix=a&limit={limit}")
        assert response.status_code == 422


async def test_list_pages_through_every_document_once(app_client):
    await _seed(app_client, ["r/1.md", "r/2.md", "r/3.md", "other/1.md"])
    seen = []
    cursor = None
    while True:
        url = "/v1/documents?source_prefix=r/&limit=2"
        if cursor:
            url += f"&cursor={cursor}"
        body = (await app_client.get(url)).json()
        seen += [d["source_uri"] for d in body["documents"]]
        cursor = body["next_cursor"]
        if cursor is None:
            break
    assert seen == ["r/1.md", "r/2.md", "r/3.md"]
    empty = (await app_client.get("/v1/documents?source_prefix=none/")).json()
    assert empty == {"documents": [], "next_cursor": None}
