import pytest

import live


@pytest.fixture
def live_server(backend, tmp_path):
    env = live.server_env(backend, tmp_path)
    dsn = env.get("VECTORIST_DATABASE_URL")
    if dsn:
        live.truncate_vector_chunks(dsn)
    server = live.LiveServer(env)
    try:
        server.start()
        yield server
    finally:
        server.stop()
        if dsn:
            live.truncate_vector_chunks(dsn)


@pytest.fixture
def enrich_env(monkeypatch):
    monkeypatch.setenv("VECTORIST_ENRICHER", "fake")
    monkeypatch.setenv("VECTORIST_ENRICH_POLL_SECONDS", "0.05")
