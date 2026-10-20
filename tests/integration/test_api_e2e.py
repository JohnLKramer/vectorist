import threading

import httpx
import pytest

from vectorist.api.app import create_app


async def test_unknown_vector_backend_fails_at_startup(sqlite_env, monkeypatch):
    monkeypatch.setenv("VECTORIST_VECTOR_BACKEND", "redis")
    app = create_app()
    with pytest.raises(RuntimeError, match="vector backend"):
        async with app.router.lifespan_context(app):
            pass


async def test_pgvector_without_database_url_fails_at_startup(sqlite_env, monkeypatch):
    monkeypatch.setenv("VECTORIST_VECTOR_BACKEND", "pgvector")
    app = create_app()
    with pytest.raises(RuntimeError, match="VECTORIST_DATABASE_URL"):
        async with app.router.lifespan_context(app):
            pass


async def test_search_embeds_query_off_the_event_loop_thread(sqlite_env, monkeypatch):
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    query_threads: list[int] = []
    async with app.router.lifespan_context(app):
        embedder = app.state.embedder
        original = embedder.embed_query

        def recording_embed_query(text: str) -> list[float]:
            query_threads.append(threading.get_ident())
            return original(text)

        monkeypatch.setattr(embedder, "embed_query", recording_embed_query)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            response = await client.post("/v1/search", json={"query": "pool"})
    assert response.status_code == 200
    assert query_threads
    assert threading.get_ident() not in query_threads


async def test_unknown_embedder_fails_at_startup(sqlite_env, monkeypatch):
    monkeypatch.setenv("VECTORIST_EMBEDDER", "Fake")
    import vectorist.embeddings.fastembed_provider as fastembed_provider

    def must_not_construct(*args, **kwargs):
        raise AssertionError("FastEmbedProvider must not be constructed")

    monkeypatch.setattr(fastembed_provider, "FastEmbedProvider", must_not_construct)
    app = create_app()
    with pytest.raises(RuntimeError, match="unsupported embedder: Fake"):
        async with app.router.lifespan_context(app):
            pass


@pytest.mark.parametrize("component", ["cache", "store"])
async def test_health_reports_degraded_when_a_component_check_fails(
    sqlite_env, monkeypatch, component
):
    app = create_app()
    transport = httpx.ASGITransport(app=app)

    async def failing_ping() -> None:
        raise RuntimeError("backend down")

    async with app.router.lifespan_context(app):
        monkeypatch.setattr(getattr(app.state, component), "ping", failing_ping)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            health = await client.get("/v1/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "degraded"
    failed_field = "cache" if component == "cache" else "vector_store"
    healthy_field = "vector_store" if component == "cache" else "cache"
    assert body[failed_field] == "error"
    assert body[healthy_field] == "ok"
    assert body["embedder"] == "ok"
