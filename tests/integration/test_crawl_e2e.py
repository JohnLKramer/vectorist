import asyncio
import functools
import http.server
import os
import threading

import httpx
import pytest

from vectorist.api.app import create_app
from vectorist.cli.__main__ import main

playwright_sync = pytest.importorskip("playwright.sync_api")


def _chromium_available() -> bool:
    with playwright_sync.sync_playwright() as p:
        return os.path.exists(p.chromium.executable_path)


pytestmark = pytest.mark.skipif(
    not _chromium_available(),
    reason="Chromium is not installed: run `playwright install chromium`",
)

INDEX = """<html><body><main><h1>Docs home</h1><p>Welcome to the docs.</p>
<a href="/docs/static.html">static</a> <a href="/docs/js.html">js</a>
<a href="/docs/gone.html">gone</a></main></body></html>"""
STATIC = "<html><body><main><h1>Static</h1><p>Plain pooling guidance.</p></main></body></html>"
GONE = "<html><body><main><h1>Gone</h1><p>Soon removed page.</p></main></body></html>"
JS = """<html><body><main id="m"><p>loading</p></main>
<script>document.getElementById('m').innerHTML =
  '<h1>Dynamic</h1><p>Rendered by javascript only.</p>';</script></body></html>"""


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


async def test_crawl_renders_javascript_and_prunes(tmp_path, monkeypatch, cli_client_factory):
    monkeypatch.setenv("VECTORIST_CACHE_PATH", str(tmp_path / "cache.db"))
    monkeypatch.setenv("VECTORIST_VEC_PATH", str(tmp_path / "vec.db"))
    monkeypatch.setenv("VECTORIST_EMBEDDER", "fake")
    for name in ("VECTORIST_VECTOR_BACKEND", "VECTORIST_DATABASE_URL", "VECTORIST_PG_URL"):
        monkeypatch.delenv(name, raising=False)

    site = tmp_path / "site"
    (site / "docs").mkdir(parents=True)
    for name, body in {"index.html": INDEX, "static.html": STATIC, "js.html": JS, "gone.html": GONE}.items():
        (site / "docs" / name).write_text(body)
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_Quiet, directory=str(site))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    argv = ["crawl", f"{base}/docs/index.html", "--prune", "--delay", "0"]

    app = create_app()
    try:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as aclient:
                factory = cli_client_factory(aclient)

                async def run():
                    return await asyncio.to_thread(main, argv, factory)

                assert await run() == 0
                docs = (await aclient.get(f"/v1/documents?source_prefix={base}/docs")).json()
                assert len(docs["documents"]) == 4
                hits = (await aclient.post(
                    "/v1/search", json={"query": "rendered by javascript", "limit": 5}
                )).json()["results"]
                assert any("javascript" in h["content_text"] for h in hits)

                (site / "docs" / "gone.html").unlink()
                (site / "docs" / "index.html").write_text(INDEX.replace(
                    '<a href="/docs/gone.html">gone</a>', ""))
                assert await run() == 0
                docs = (await aclient.get(f"/v1/documents?source_prefix={base}/docs")).json()
                assert sorted(d["source_uri"].rsplit("/", 1)[-1] for d in docs["documents"]) == [
                    "index.html", "js.html", "static.html"]
    finally:
        server.shutdown()
