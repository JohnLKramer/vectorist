from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx

from vectorist.cli.client import ServerClient
from vectorist.cli.config import CliConfig, CrawlConfig, load_config, load_crawl_config
from vectorist.cli.errors import CliError
from vectorist.cli.repo import find_repo_root
from vectorist.cli.crawl.command import CrawlSummary, run_crawl
from vectorist.cli.crawl.discovery import USER_AGENT
from vectorist.cli.crawl.render import PlaywrightRenderer, Renderer
from vectorist.cli.sync import SyncSummary, run_sync


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vectorist-cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sync = sub.add_parser("sync", help="send a repository's Markdown files to the server")
    sync.add_argument("path", nargs="?", default=".", type=Path)
    sync.add_argument("--server")
    sync.add_argument("--include", action="append")
    sync.add_argument("--exclude", action="append")
    sync.add_argument("--repo-id")
    sync.add_argument("--prune", action="store_true")
    sync.add_argument("--dry-run", action="store_true")
    sync.add_argument("--batch-size", type=int)
    crawl = sub.add_parser("crawl", help="crawl a documentation site and send its pages")
    crawl.add_argument("url", nargs="?")
    crawl.add_argument("--server")
    crawl.add_argument("--max-pages", type=int)
    crawl.add_argument("--delay", type=float)
    crawl.add_argument("--concurrency", type=int)
    crawl.add_argument("--keep-query", action="store_true")
    crawl.add_argument("--prune", action="store_true")
    crawl.add_argument("--dry-run", action="store_true")
    crawl.add_argument("--batch-size", type=int)
    return parser


def _default_client(config: CliConfig | CrawlConfig) -> ServerClient:
    return ServerClient(httpx.Client(base_url=config.server, timeout=300.0))


def _print_summary(summary: SyncSummary) -> None:
    prefix = "[dry run] " if summary.dry_run else ""
    if summary.dry_run:
        print(f"{prefix}would send {summary.sent} file(s)")
    else:
        print(
            f"ingested {summary.ingested}, unchanged {summary.unchanged}, "
            f"failed {len(summary.failures)}"
        )
    if summary.assets_uploaded or summary.assets_skipped:
        print(
            f"assets uploaded {summary.assets_uploaded}, "
            f"skipped {summary.assets_skipped}"
        )
    if isinstance(summary, CrawlSummary):
        print(
            f"skipped by robots.txt {summary.robots_skipped}, "
            f"redirected out of scope {summary.out_of_scope}"
        )
        if summary.truncated:
            print("crawl stopped at --max-pages; some pages were not fetched")
    verb = "would prune" if summary.dry_run else "pruned"
    for uri in summary.pruned:
        print(f"{prefix}{verb} {uri}")
    for failure in summary.failures:
        print(f"failed: {failure}", file=sys.stderr)


def _crawl(
    args: argparse.Namespace,
    client_factory: Callable[[CrawlConfig], ServerClient],
    renderer_factory: Callable[[], Renderer] | None,
    fetch_client: httpx.Client | None,
) -> CrawlSummary:
    config = load_crawl_config(args, os.environ, Path.cwd())
    fetch = fetch_client or httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True
    )
    return run_crawl(
        config, client_factory(config), fetch, renderer_factory or PlaywrightRenderer
    )


def main(
    argv: Sequence[str] | None = None,
    client_factory: Callable[..., ServerClient] = _default_client,
    renderer_factory: Callable[[], Renderer] | None = None,
    fetch_client: httpx.Client | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "crawl":
            summary = _crawl(args, client_factory, renderer_factory, fetch_client)
        else:
            root = find_repo_root(args.path)
            config = load_config(args, os.environ, root)
            summary = run_sync(config, client_factory(config), root)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print_summary(summary)
    return 1 if summary.failures else 0


if __name__ == "__main__":
    sys.exit(main())
