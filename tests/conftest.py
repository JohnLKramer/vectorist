from pathlib import Path
import pytest

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
