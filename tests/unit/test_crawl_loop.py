import pytest

from vectorist.cli.crawl.crawl import crawl
from vectorist.cli.crawl.render import RenderedPage, RenderError
from vectorist.cli.crawl.urls import Scope
from vectorist.cli.errors import CrawlError

SCOPE = Scope("https://h.example/docs")
START = "https://h.example/docs"


def _page(url, body="<p>text</p>", links=(), status=200, final=None):
    anchors = "".join(f'<a href="{href}">l</a>' for href in links)
    return RenderedPage(final_url=final or url, status=status, html=f"<body>{body}{anchors}</body>")


class FakeRenderer:
    def __init__(self, pages):
        self.pages = pages
        self.closed = False

    def render(self, url):
        entry = self.pages[url]
        if isinstance(entry, Exception):
            raise entry
        return entry

    def close(self):
        self.closed = True


def _run(pages, sitemap=None, allowed=lambda _u: True, **overrides):
    renderers = []

    def factory():
        renderers.append(FakeRenderer(pages))
        return renderers[-1]

    params = dict(
        scope=SCOPE, start_url=START, sitemap_urls=sitemap, robots_allowed=allowed,
        renderer_factory=factory, keep_query=False, max_pages=100, delay=0,
        concurrency=1, sleep=lambda _s: None,
    )
    params.update(overrides)
    result = crawl(**params)
    assert all(r.closed for r in renderers)
    return result


def test_follows_links_in_scope_and_dedupes():
    pages = {
        START: _page(START, links=["/docs/a", "/docs/a/", "/docs/b#x", "/blog/x", "mailto:a@b", "/docs/logo.png"]),
        "https://h.example/docs/a": _page("https://h.example/docs/a", links=["/docs/b"]),
        "https://h.example/docs/b": _page("https://h.example/docs/b"),
    }
    result = _run(pages)
    assert sorted(p.final_url for p in result.pages) == [
        START, "https://h.example/docs/a", "https://h.example/docs/b",
    ]
    assert result.failures == [] and not result.truncated


def test_sitemap_urls_are_used_without_link_following():
    pages = {
        "https://h.example/docs/a": _page("https://h.example/docs/a", links=["/docs/never"]),
    }
    result = _run(pages, sitemap=["https://h.example/docs/a"])
    assert [p.final_url for p in result.pages] == ["https://h.example/docs/a"]


def test_max_pages_truncates():
    pages = {
        START: _page(START, links=["/docs/a", "/docs/b"]),
        "https://h.example/docs/a": _page("https://h.example/docs/a"),
        "https://h.example/docs/b": _page("https://h.example/docs/b"),
    }
    result = _run(pages, max_pages=2)
    assert len(result.pages) == 2 and result.truncated


def test_page_failures_do_not_stop_the_crawl():
    pages = {
        START: _page(START, links=["/docs/slow", "/docs/gone", "/docs/empty", "/docs/ok"]),
        "https://h.example/docs/slow": RenderError("Timeout 30000ms exceeded"),
        "https://h.example/docs/gone": _page("https://h.example/docs/gone", status=404),
        "https://h.example/docs/empty": _page("https://h.example/docs/empty", body=""),
        "https://h.example/docs/ok": _page("https://h.example/docs/ok"),
    }
    result = _run(pages)
    assert sorted(p.final_url for p in result.pages) == [START, "https://h.example/docs/ok"]
    assert len(result.failures) == 3


def test_redirect_out_of_scope_is_skipped_and_counted():
    pages = {
        START: _page(START, links=["/docs/moved"]),
        "https://h.example/docs/moved": _page("https://h.example/docs/moved", final="https://h.example/login"),
    }
    result = _run(pages)
    assert [p.final_url for p in result.pages] == [START]
    assert result.out_of_scope == 1


def test_redirect_to_seen_page_is_not_duplicated():
    pages = {
        START: _page(START, links=["/docs/alias"]),
        "https://h.example/docs/alias": _page("https://h.example/docs/alias", final=START),
    }
    assert len(_run(pages).pages) == 1


def test_robots_disallowed_are_skipped_and_counted():
    pages = {START: _page(START, links=["/docs/private/x", "/docs/ok"]),
             "https://h.example/docs/ok": _page("https://h.example/docs/ok")}
    result = _run(pages, allowed=lambda u: "/private" not in u)
    assert result.robots_skipped == 1 and len(result.pages) == 2


def test_concurrency_renders_every_page_and_closes_every_renderer():
    links = [f"/docs/p{i}" for i in range(6)]
    pages = {START: _page(START, links=links)}
    pages.update({f"https://h.example{l}": _page(f"https://h.example{l}") for l in links})
    result = _run(pages, concurrency=3)
    assert len(result.pages) == 7


def test_renderer_setup_failure_is_fatal():
    def factory():
        raise CrawlError("no chromium")

    with pytest.raises(CrawlError, match="no chromium"):
        crawl(
            scope=SCOPE, start_url=START, sitemap_urls=None, robots_allowed=lambda _u: True,
            renderer_factory=factory, keep_query=False, max_pages=5, delay=0,
            concurrency=1, sleep=lambda _s: None,
        )
