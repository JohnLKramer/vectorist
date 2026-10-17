import httpx
import pytest

from vectorist.cli.crawl.discovery import discover_sitemap, load_robots
from vectorist.cli.crawl.urls import Scope
from vectorist.cli.errors import CrawlError

SCOPE = Scope("https://h.example/docs")


def _client(routes: dict[str, tuple[int, str]]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        status, body = routes.get(str(request.url), (404, ""))
        return httpx.Response(status, text=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _urlset(*urls: str) -> str:
    items = "".join(f"<url><loc>{u}</loc></url>" for u in urls)
    return f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{items}</urlset>'


def test_robots_allow_and_disallow():
    http = _client({"https://h.example/robots.txt": (200, "User-agent: *\nDisallow: /docs/private\n")})
    robots = load_robots(http, SCOPE)
    assert robots.allowed("https://h.example/docs/a")
    assert not robots.allowed("https://h.example/docs/private/x")


def test_missing_robots_allows_everything():
    robots = load_robots(_client({}), SCOPE)
    assert robots.allowed("https://h.example/docs/anything")


def test_robots_server_error_is_fatal():
    with pytest.raises(CrawlError):
        load_robots(_client({"https://h.example/robots.txt": (500, "")}), SCOPE)


def test_sitemap_filters_scope_and_dedupes():
    http = _client({"https://h.example/sitemap.xml": (200, _urlset(
        "https://h.example/docs/a", "https://h.example/docs/a/", "https://h.example/docs-old/b",
        "https://h.example/blog/c#x",
    ))})
    robots = load_robots(http, SCOPE)
    assert discover_sitemap(http, SCOPE, robots, False) == ["https://h.example/docs/a"]


def test_sitemap_index_is_followed():
    index = (
        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<sitemap><loc>https://h.example/s1.xml</loc></sitemap></sitemapindex>"
    )
    http = _client({
        "https://h.example/sitemap.xml": (200, index),
        "https://h.example/s1.xml": (200, _urlset("https://h.example/docs/x")),
    })
    robots = load_robots(http, SCOPE)
    assert discover_sitemap(http, SCOPE, robots, False) == ["https://h.example/docs/x"]


def test_sitemap_from_robots_line():
    http = _client({
        "https://h.example/robots.txt": (200, "User-agent: *\nSitemap: https://h.example/custom.xml\n"),
        "https://h.example/custom.xml": (200, _urlset("https://h.example/docs/y")),
    })
    robots = load_robots(http, SCOPE)
    assert discover_sitemap(http, SCOPE, robots, False) == ["https://h.example/docs/y"]


def test_no_sitemap_returns_none():
    http = _client({})
    assert discover_sitemap(http, SCOPE, load_robots(http, SCOPE), False) is None


def test_sitemap_with_nothing_in_scope_falls_back_to_links():
    http = _client({"https://h.example/sitemap.xml": (200, _urlset("https://h.example/blog/a"))})
    assert discover_sitemap(http, SCOPE, load_robots(http, SCOPE), False) is None


def test_malformed_sitemap_is_fatal():
    http = _client({"https://h.example/sitemap.xml": (200, "<urlset><url>")})
    with pytest.raises(CrawlError):
        discover_sitemap(http, SCOPE, load_robots(http, SCOPE), False)
