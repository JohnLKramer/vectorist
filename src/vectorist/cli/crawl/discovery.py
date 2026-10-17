from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable
from urllib.robotparser import RobotFileParser

import httpx

from vectorist.cli.crawl.urls import Scope, normalize
from vectorist.cli.errors import CrawlError

USER_AGENT = "vectorist"
_MAX_SITEMAP_DEPTH = 3


def _get(http: httpx.Client, url: str) -> httpx.Response | None:
    """Return the response, None for 404, and raise for anything else that fails."""
    try:
        response = http.get(url)
    except httpx.HTTPError as exc:
        raise CrawlError(f"GET {url} failed: {exc}") from exc
    if response.status_code == 404:
        return None
    if response.status_code >= 400:
        raise CrawlError(f"GET {url} returned {response.status_code}")
    return response


class Robots:
    def __init__(self, parser: RobotFileParser | None) -> None:
        self._parser = parser

    def allowed(self, url: str) -> bool:
        return self._parser is None or self._parser.can_fetch(USER_AGENT, url)

    @property
    def sitemaps(self) -> list[str]:
        return (self._parser.site_maps() or []) if self._parser is not None else []


def load_robots(http: httpx.Client, scope: Scope) -> Robots:
    response = _get(http, f"{scope.origin}/robots.txt")
    if response is None:
        return Robots(None)
    parser = RobotFileParser()
    parser.parse(response.text.splitlines())
    return Robots(parser)


def _parse_sitemap(text: str, url: str) -> tuple[str, list[str]]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise CrawlError(f"sitemap {url} is not valid XML: {exc}") from exc
    kind = root.tag.rsplit("}", 1)[-1]
    if kind not in ("urlset", "sitemapindex"):
        raise CrawlError(f"sitemap {url} has unexpected root element <{kind}>")
    locs = [
        (el.text or "").strip()
        for el in root.iter()
        if el.tag.rsplit("}", 1)[-1] == "loc" and (el.text or "").strip()
    ]
    return kind, locs


def _collect(http: httpx.Client, url: str, depth: int, out: list[str]) -> bool:
    if depth > _MAX_SITEMAP_DEPTH:
        raise CrawlError(f"sitemap index nesting deeper than {_MAX_SITEMAP_DEPTH} at {url}")
    response = _get(http, url)
    if response is None:
        return False
    kind, locs = _parse_sitemap(response.text, url)
    if kind == "urlset":
        out.extend(locs)
    else:
        for child in locs:
            _collect(http, child, depth + 1, out)
    return True


def discover_sitemap(
    http: httpx.Client, scope: Scope, robots: Robots, keep_query: bool
) -> list[str] | None:
    """In-scope URLs from the sitemap, or None when there is none to use."""
    candidates = robots.sitemaps or [f"{scope.origin}/sitemap.xml"]
    found: list[str] = []
    any_sitemap = False
    for candidate in candidates:
        any_sitemap |= _collect(http, candidate, 0, found)
    if not any_sitemap:
        return None
    urls: list[str] = []
    seen: set[str] = set()
    for loc in found:
        url = normalize(loc, keep_query)
        if url not in seen and scope.contains(url):
            seen.add(url)
            urls.append(url)
    return urls or None
