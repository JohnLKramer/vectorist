from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx

from vectorist.cli.client import ServerClient
from vectorist.cli.config import CliConfig, load_config
from vectorist.cli.errors import CliError
from vectorist.cli.repo import find_repo_root
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
    return parser


def _default_client(config: CliConfig) -> ServerClient:
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
    verb = "would prune" if summary.dry_run else "pruned"
    for uri in summary.pruned:
        print(f"{prefix}{verb} {uri}")
    for failure in summary.failures:
        print(f"failed: {failure}", file=sys.stderr)


def main(
    argv: Sequence[str] | None = None,
    client_factory: Callable[[CliConfig], ServerClient] = _default_client,
) -> int:
    args = build_parser().parse_args(argv)
    try:
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
