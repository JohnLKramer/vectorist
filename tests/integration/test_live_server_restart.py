import httpx
import pytest

pytestmark = pytest.mark.live

DOCS = {
    "restart/a.md": "# Alpha\n\nconnection pooling notes\n",
    "restart/b.md": "# Beta\n\nlogging configuration notes\n",
}


def _ingest(server, uri, content):
    response = httpx.post(
        f"{server.base_url}/v1/ingest", json={"source_uri": uri, "content": content}
    )
    assert response.status_code == 200
    return response.json()


def _search_uris(server):
    response = httpx.post(f"{server.base_url}/v1/search", json={"query": "notes", "limit": 10})
    assert response.status_code == 200
    return {hit["source_uri"] for hit in response.json()["results"]}


def _restart(server):
    server.stop()
    server.start()


def test_data_survives_restart(live_server):
    for uri, content in DOCS.items():
        assert _ingest(live_server, uri, content)["status"] == "indexed"
    assert _search_uris(live_server) == set(DOCS)

    _restart(live_server)

    assert _search_uris(live_server) == set(DOCS)
    for uri, content in DOCS.items():
        assert _ingest(live_server, uri, content)["status"] == "skipped"


def test_edit_after_restart_reindexes_only_changed(live_server):
    for uri, content in DOCS.items():
        _ingest(live_server, uri, content)
    _restart(live_server)

    assert _ingest(live_server, "restart/a.md", "# Alpha\n\nrewritten pooling notes\n")["status"] == "indexed"
    assert _ingest(live_server, "restart/b.md", DOCS["restart/b.md"])["status"] == "skipped"


def test_deleted_document_stays_gone_after_restart(live_server):
    ids = {uri: _ingest(live_server, uri, content)["document_id"] for uri, content in DOCS.items()}
    assert httpx.delete(f"{live_server.base_url}/v1/documents/{ids['restart/a.md']}").status_code == 204

    _restart(live_server)

    assert _search_uris(live_server) == {"restart/b.md"}
    assert httpx.get(f"{live_server.base_url}/v1/documents/{ids['restart/a.md']}").status_code == 404
