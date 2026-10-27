from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import httpx

from vectorist.cli.assets import load_remote_asset, upload_assets
from vectorist.cli.client import ServerClient
from vectorist.cli.config import CrawlConfig
from vectorist.cli.crawl.crawl import crawl
from vectorist.cli.crawl.discovery import USER_AGENT, discover_sitemap, load_robots
from vectorist.cli.crawl.render import Renderer
from vectorist.cli.crawl.urls import Scope, normalize
from vectorist.cli.errors import CliError, ConfigError
from vectorist.cli.sync import SyncSummary


@dataclass
class CrawlSummary(SyncSummary):
    robots_skipped: int = 0
    out_of_scope: int = 0
    truncated: bool = False


def run_crawl(
    config: CrawlConfig,
    client: ServerClient,
    fetch: httpx.Client,
    renderer_factory: Callable[[], Renderer],
) -> CrawlSummary:
    try:
        scope = Scope(config.url)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    start_url = normalize(config.url, config.keep_query)
    robots = load_robots(fetch, scope)
    sitemap_urls = discover_sitemap(fetch, scope, robots, config.keep_query)
    result = crawl(
        scope=scope,
        start_url=start_url,
        sitemap_urls=sitemap_urls,
        robots_allowed=robots.allowed,
        renderer_factory=renderer_factory,
        keep_query=config.keep_query,
        max_pages=config.max_pages,
        delay=config.delay,
        concurrency=config.concurrency,
    )
    summary = CrawlSummary(
        sent=len(result.pages),
        failures=list(result.failures),
        dry_run=config.dry_run,
        robots_skipped=result.robots_skipped,
        out_of_scope=result.out_of_scope,
        truncated=result.truncated,
    )
    if config.prune:
        if not result.pages:
            raise CliError("--prune refused: no pages were fetched")
        if result.truncated:
            raise CliError(
                f"--prune refused: the crawl stopped at --max-pages ({config.max_pages}) "
                "so the page set is incomplete"
            )
        if result.failures:
            raise CliError(
                f"--prune refused: {len(result.failures)} page(s) failed, so the page "
                "set is incomplete"
            )

    documents = [
        {"source_uri": page.final_url, "content": page.html, "content_type": "text/html"}
        for page in result.pages
    ]
    if not config.dry_run:
        for start in range(0, len(documents), config.batch_size):
            for item in client.post_stream(documents[start : start + config.batch_size]):
                status = item.get("status")
                if status == "failed":
                    error = item.get("error") or {}
                    summary.failures.append(
                        f"{item.get('source_uri')}: {error.get('message', 'failed')}"
                    )
                elif status == "skipped":
                    summary.unchanged += 1
                else:
                    summary.ingested += 1
                refs = item.get("pending_assets") or []
                if refs and status in ("indexed", "skipped"):
                    page_url = item["source_uri"]
                    upload_assets(
                        client,
                        item["document_id"],
                        page_url,
                        refs,
                        lambda ref, page_url=page_url: load_remote_asset(fetch, page_url, ref),
                        summary,
                    )

    if config.prune:
        keep = {page.final_url for page in result.pages}
        for document in client.list_documents(scope.list_prefix):
            uri = document["source_uri"]
            if uri in keep or not scope.contains(uri):
                continue
            if not config.dry_run:
                client.delete_document(document["id"])
            summary.pruned.append(uri)
    return summary
