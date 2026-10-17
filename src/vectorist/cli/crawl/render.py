from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from vectorist.cli.errors import CrawlError


class RenderError(Exception):
    """A single page failed to render; the crawl continues."""


@dataclass
class RenderedPage:
    final_url: str
    status: int
    html: str


class Renderer(Protocol):
    def render(self, url: str) -> RenderedPage: ...

    def close(self) -> None: ...


class PlaywrightRenderer:
    """Headless Chromium. Each instance must be used from the thread that created it."""

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise CrawlError(
                "the crawler needs Playwright: install with `uv sync --extra crawl`"
            ) from exc
        self._timeout_ms = timeout_seconds * 1000
        self._playwright = sync_playwright().start()
        try:
            self._browser = self._playwright.chromium.launch(headless=True)
        except Exception as exc:
            self._playwright.stop()
            raise CrawlError(
                f"could not launch Chromium ({exc}); run `playwright install chromium`"
            ) from exc
        self._page = self._browser.new_page(
            user_agent="vectorist", ignore_https_errors=False
        )

    def render(self, url: str) -> RenderedPage:
        from playwright.sync_api import Error as PlaywrightError

        try:
            response = self._page.goto(
                url, wait_until="networkidle", timeout=self._timeout_ms
            )
            html = self._page.content()
            final_url = self._page.url
        except PlaywrightError as exc:
            raise RenderError(str(exc).splitlines()[0]) from exc
        return RenderedPage(
            final_url=final_url, status=response.status if response else 0, html=html
        )

    def close(self) -> None:
        self._browser.close()
        self._playwright.stop()
