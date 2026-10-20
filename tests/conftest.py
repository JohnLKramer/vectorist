import os
from pathlib import Path

import asyncpg
import httpx
import pytest

from vectorist.api.app import create_app

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def cli_client_factory():
    """Build a CLI ServerClient factory that calls an in-process ASGI app.

    The CLI client is synchronous and runs in a worker thread; its requests are
    forwarded onto the event loop that owns the app. Call from inside a running loop.
    """
    import asyncio

    import httpx

    from vectorist.cli.client import ServerClient

    def make(async_client: httpx.AsyncClient):
        loop = asyncio.get_running_loop()

        class Bridge(httpx.BaseTransport):
            def handle_request(self, request):
                query = f"?{request.url.query.decode()}" if request.url.query else ""
                future = asyncio.run_coroutine_threadsafe(
                    async_client.request(
                        request.method,
                        request.url.path + query,
                        content=request.read(),
                        headers=dict(request.headers),
                    ),
                    loop,
                )
                response = future.result(timeout=60)
                return httpx.Response(
                    response.status_code, headers=response.headers, content=response.content
                )

        return lambda _config: ServerClient(
            httpx.Client(transport=Bridge(), base_url="http://t")
        )

    return make


def _set_env(monkeypatch, tmp_path, backend: str) -> dict[str, str]:
    pg_url = os.environ.get("VECTORIST_PG_URL")
    monkeypatch.setenv("VECTORIST_CACHE_PATH", str(tmp_path / "cache.db"))
    monkeypatch.setenv("VECTORIST_VEC_PATH", str(tmp_path / "vec.db"))
    monkeypatch.setenv("VECTORIST_EMBEDDER", "fake")
    monkeypatch.delenv("VECTORIST_VECTOR_BACKEND", raising=False)
    monkeypatch.delenv("VECTORIST_DATABASE_URL", raising=False)
    monkeypatch.delenv("VECTORIST_PG_URL", raising=False)
    env = {
        "VECTORIST_CACHE_PATH": str(tmp_path / "cache.db"),
        "VECTORIST_VEC_PATH": str(tmp_path / "vec.db"),
        "VECTORIST_EMBEDDER": "fake",
    }
    if backend == "pgvector":
        env["VECTORIST_VECTOR_BACKEND"] = "pgvector"
        env["VECTORIST_DATABASE_URL"] = pg_url
        monkeypatch.setenv("VECTORIST_VECTOR_BACKEND", "pgvector")
        monkeypatch.setenv("VECTORIST_DATABASE_URL", env["VECTORIST_DATABASE_URL"])
    return env


async def _truncate_vector_chunks(dsn: str) -> None:
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("TRUNCATE vector_chunks")
    finally:
        await conn.close()


@pytest.fixture(params=["sqlite", "pgvector"])
def backend(request) -> str:
    if request.param == "pgvector" and not os.getenv("VECTORIST_PG_URL"):
        pytest.skip("VECTORIST_PG_URL not set")
    return request.param


@pytest.fixture
def server_env(backend, tmp_path, monkeypatch) -> dict[str, str]:
    return _set_env(monkeypatch, tmp_path, backend)


@pytest.fixture
def sqlite_env(tmp_path, monkeypatch) -> dict[str, str]:
    return _set_env(monkeypatch, tmp_path, "sqlite")


@pytest.fixture
async def app_client(backend, server_env):
    app = create_app()
    dsn = server_env.get("VECTORIST_DATABASE_URL")
    async with app.router.lifespan_context(app):
        if dsn:
            await _truncate_vector_chunks(dsn)
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
                yield client
        finally:
            if dsn:
                await _truncate_vector_chunks(dsn)
