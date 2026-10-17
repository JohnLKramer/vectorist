from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from vectorist.cli.crawl.render import RenderedPage, RenderError, Renderer
from vectorist.cli.crawl.urls import Scope, is_html_candidate, normalize


@dataclass
class CrawlResult:
    pages: list[RenderedPage] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    robots_skipped: int = 0
    out_of_scope: int = 0
    truncated: bool = False


def _empty_body(soup: BeautifulSoup) -> bool:
    body = soup.body or soup
    for tag in body.find_all(["script", "style", "noscript", "template"]):
        tag.decompose()
    return not body.get_text(strip=True)


def crawl(
    *,
    scope: Scope,
    start_url: str,
    sitemap_urls: list[str] | None,
    robots_allowed: Callable[[str], bool],
    renderer_factory: Callable[[], Renderer],
    keep_query: bool,
    max_pages: int,
    delay: float,
    concurrency: int,
    sleep: Callable[[float], None] = time.sleep,
) -> CrawlResult:
    result = CrawlResult()
    follow_links = sitemap_urls is None
    pending: deque[str] = deque()
    seen: set[str] = set()
    rendered: set[str] = set()

    def enqueue(url: str) -> None:
        url = normalize(url, keep_query)
        if url in seen or not scope.contains(url) or not is_html_candidate(url):
            return
        seen.add(url)
        if not robots_allowed(url):
            result.robots_skipped += 1
            return
        pending.append(url)

    for url in sitemap_urls if sitemap_urls is not None else [start_url]:
        enqueue(url)

    work: queue.Queue[str | None] = queue.Queue()
    done: queue.Queue[tuple[str, RenderedPage | None, object]] = queue.Queue()

    def worker() -> None:
        renderer: Renderer | None = None
        try:
            while (url := work.get()) is not None:
                try:
                    if renderer is None:
                        renderer = renderer_factory()
                    done.put((url, renderer.render(url), None))
                except RenderError as exc:
                    done.put((url, None, str(exc)))
                except Exception as exc:
                    done.put((url, None, exc))
                sleep(delay)
        finally:
            if renderer is not None:
                renderer.close()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    attempted = 0
    try:
        while pending and attempted < max_pages:
            wave = [
                pending.popleft()
                for _ in range(min(concurrency, max_pages - attempted, len(pending)))
            ]
            attempted += len(wave)
            for url in wave:
                work.put(url)
            for _ in wave:
                url, page, error = done.get()
                if isinstance(error, Exception):
                    raise error
                if error is not None:
                    result.failures.append(f"{url}: {error}")
                    continue
                _accept(url, page, scope, keep_query, result, rendered, seen, enqueue, follow_links)
        result.truncated = bool(pending)
    finally:
        for _ in threads:
            work.put(None)
        for thread in threads:
            thread.join()
    return result


def _accept(url, page, scope, keep_query, result, rendered, seen, enqueue, follow_links) -> None:
    final = normalize(page.final_url, keep_query)
    seen.add(final)
    if not scope.contains(final):
        result.out_of_scope += 1
        return
    if not 200 <= page.status < 300:
        result.failures.append(f"{url}: HTTP status {page.status}")
        return
    if final in rendered:
        return
    soup = BeautifulSoup(page.html, "html.parser")
    if _empty_body(BeautifulSoup(page.html, "html.parser")):
        result.failures.append(f"{url}: rendered page has an empty body")
        return
    rendered.add(final)
    result.pages.append(RenderedPage(final_url=final, status=page.status, html=page.html))
    if follow_links:
        for anchor in soup.find_all("a", href=True):
            target = urljoin(page.final_url, anchor["href"])
            if urlsplit(target).scheme in ("http", "https"):
                enqueue(target)
