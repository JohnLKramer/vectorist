import json
from argparse import Namespace

import httpx
import pytest

from vectorist.cli.client import ServerClient
from vectorist.cli.config import load_crawl_config
from vectorist.cli.crawl.command import run_crawl
from vectorist.cli.crawl.render import RenderedPage
from vectorist.cli.errors import CliError, ConfigError

START = "https://h.example/docs"


class FakeRenderer:
    def __init__(self, pages):
        self.pages = pages

    def render(self, url):
        return self.pages[url]

    def close(self):
        pass


def _html(links=()):
    anchors = "".join(f'<a href="{l}">x</a>' for l in links)
    return f"<body><p>content</p>{anchors}</body>"


def _pages(spec):
    return {u: RenderedPage(u, status, _html(links)) for u, (status, links) in spec.items()}


class FakeServer:
    def __init__(self, existing=(), pending=None):
        self.requests = []
        self.existing = list(existing)
        self.pending = pending or {}
        self.assets = []

    def __call__(self, request):
        self.requests.append(request)
        if request.url.path == "/v1/ingest/stream":
            lines = [json.loads(l) for l in request.content.decode().splitlines()]
            return httpx.Response(200, json=[
                {"source_uri": l["source_uri"], "document_id": "x", "status": "indexed", "error": None,
                 "pending_assets": self.pending.get(l["source_uri"], [])}
                for l in lines])
        if request.method == "PUT" and "/assets" in request.url.path:
            self.assets.append(request)
            return httpx.Response(200, json={"status": "pending"})
        if request.method == "GET":
            return httpx.Response(200, json={"documents": self.existing, "next_cursor": None})
        return httpx.Response(204)

    def client(self):
        return ServerClient(httpx.Client(transport=httpx.MockTransport(self), base_url="http://t"))

    def bodies(self):
        return [json.loads(l) for r in self.requests if r.method == "POST"
                for l in r.content.decode().splitlines()]

    def deletes(self):
        return [r.url.path for r in self.requests if r.method == "DELETE"]


def _fetch():
    return httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))


def _config(**overrides):
    base = dict(url=START, server=None, max_pages=None, delay=0.0, concurrency=None,
                keep_query=False, prune=False, dry_run=False, batch_size=None)
    base.update(overrides)
    return load_crawl_config(Namespace(**base), {}, __import__("pathlib").Path("/nonexistent"))


def _run(config, server, pages):
    return run_crawl(config, server.client(), _fetch(), lambda: FakeRenderer(pages))


def test_sends_html_with_content_type_in_batches():
    pages = _pages({START: (200, ["/docs/a", "/docs/b"]),
                    f"{START}/a": (200, []), f"{START}/b": (200, [])})
    server = FakeServer()
    summary = _run(_config(batch_size=2), server, pages)
    assert summary.ingested == 3
    assert len([r for r in server.requests if r.method == "POST"]) == 2
    assert {b["content_type"] for b in server.bodies()} == {"text/html"}


def test_prune_deletes_unseen_in_scope_documents_only():
    pages = _pages({START: (200, [])})
    existing = [
        {"id": "keep", "source_uri": START},
        {"id": "gone", "source_uri": f"{START}/old"},
        {"id": "other", "source_uri": "https://h.example/docs-old/x"},
    ]
    server = FakeServer(existing)
    summary = _run(_config(prune=True), server, pages)
    assert server.deletes() == ["/v1/documents/gone"]
    assert summary.pruned == [f"{START}/old"]


def test_prune_refused_on_truncation():
    pages = _pages({START: (200, ["/docs/a"]), f"{START}/a": (200, [])})
    server = FakeServer()
    with pytest.raises(CliError, match="max-pages"):
        _run(_config(prune=True, max_pages=1), server, pages)
    assert server.requests == []


def test_prune_refused_on_page_failure():
    pages = _pages({START: (200, ["/docs/a"]), f"{START}/a": (500, [])})
    with pytest.raises(CliError, match="failed"):
        _run(_config(prune=True), FakeServer(), pages)


def test_prune_refused_with_zero_pages():
    pages = _pages({START: (404, [])})
    with pytest.raises(CliError, match="no pages"):
        _run(_config(prune=True), FakeServer(), pages)


def test_dry_run_sends_and_deletes_nothing():
    pages = _pages({START: (200, [])})
    server = FakeServer([{"id": "gone", "source_uri": f"{START}/old"}])
    summary = _run(_config(prune=True, dry_run=True), server, pages)
    assert [r for r in server.requests if r.method != "GET"] == []
    assert summary.sent == 1 and summary.pruned == [f"{START}/old"]


def test_non_http_start_url_is_a_config_error():
    with pytest.raises(ConfigError):
        _run(_config(url="ftp://x/y"), FakeServer(), {})


def test_config_requires_url_and_validates_numbers(tmp_path):
    base = dict(url=None, server=None, max_pages=None, delay=None, concurrency=None,
                keep_query=False, prune=False, dry_run=False, batch_size=None)
    with pytest.raises(ConfigError, match="URL"):
        load_crawl_config(Namespace(**base), {}, tmp_path)
    for bad in (dict(max_pages=0), dict(delay=-1.0), dict(concurrency=0)):
        with pytest.raises(ConfigError):
            load_crawl_config(Namespace(**{**base, "url": START, **bad}), {}, tmp_path)


def test_config_file_crawl_table_and_precedence(tmp_path):
    (tmp_path / ".vectorist.toml").write_text(
        'server = "http://file"\n[crawl]\nurl = "https://h.example/docs"\nmax_pages = 7\ndelay = 1\n')
    base = dict(url=None, server=None, max_pages=None, delay=None, concurrency=None,
                keep_query=False, prune=False, dry_run=False, batch_size=None)
    config = load_crawl_config(Namespace(**base), {}, tmp_path)
    assert (config.url, config.max_pages, config.delay, config.server) == (START, 7, 1.0, "http://file")
    flagged = load_crawl_config(Namespace(**{**base, "max_pages": 3}), {}, tmp_path)
    assert flagged.max_pages == 3


def test_config_unknown_crawl_key_is_fatal(tmp_path):
    (tmp_path / ".vectorist.toml").write_text("[crawl]\nmax_page = 3\n")
    base = dict(url=START, server=None, max_pages=None, delay=None, concurrency=None,
                keep_query=False, prune=False, dry_run=False, batch_size=None)
    with pytest.raises(ConfigError, match="max_page"):
        load_crawl_config(Namespace(**base), {}, tmp_path)


def _png_fetch(log, fail=()):
    def handler(request):
        if not request.url.path.endswith(".png"):
            return httpx.Response(404)
        log.append(str(request.url))
        if request.url.path in fail:
            return httpx.Response(404)
        return httpx.Response(200, content=b"\x89PNG\r\n\x1a\n" + b"0" * 3000)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_pending_assets_are_downloaded_against_the_final_url_and_uploaded():
    pages = _pages({START: (200, [])})
    server = FakeServer(pending={START: ["img/a.png"]})
    log = []
    summary = run_crawl(_config(), server.client(), _png_fetch(log), lambda: FakeRenderer(pages))
    assert log == ["https://h.example/img/a.png"]
    assert len(server.assets) == 1
    assert summary.assets_uploaded == 1 and summary.failures == []


def test_dry_run_downloads_nothing():
    pages = _pages({START: (200, [])})
    server = FakeServer(pending={START: ["img/a.png"]})
    log = []
    run_crawl(_config(dry_run=True), server.client(), _png_fetch(log), lambda: FakeRenderer(pages))
    assert log == [] and server.assets == []


def test_failed_download_adds_failure_and_run_continues():
    pages = _pages({START: (200, ["/docs/b"]), f"{START}/b": (200, [])})
    server = FakeServer(pending={START: ["/bad.png"], f"{START}/b": ["/good.png"]})
    log = []
    summary = run_crawl(
        _config(), server.client(), _png_fetch(log, fail=("/bad.png",)), lambda: FakeRenderer(pages)
    )
    assert summary.ingested == 2
    assert summary.assets_uploaded == 1
    assert len(summary.failures) == 1 and "404" in summary.failures[0]
