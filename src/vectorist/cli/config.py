from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vectorist.cli.errors import ConfigError

CONFIG_FILENAME = ".vectorist.toml"
DEFAULT_SERVER = "http://localhost:8080"
DEFAULT_INCLUDE = ["*.md"]
DEFAULT_BATCH_SIZE = 50
_FILE_KEYS = {"server", "include", "exclude", "crawl"}
_CRAWL_KEYS = {"url": str, "max_pages": int, "delay": (int, float), "concurrency": int}
DEFAULT_MAX_PAGES = 500
DEFAULT_DELAY = 0.5
DEFAULT_CONCURRENCY = 1


@dataclass(frozen=True)
class CliConfig:
    server: str
    include: list[str]
    exclude: list[str]
    repo_id: str | None
    prune: bool
    dry_run: bool
    batch_size: int


def _read_file(repo_root: Path) -> dict[str, Any]:
    path = repo_root / CONFIG_FILENAME
    if not path.is_file():
        return {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    unknown = sorted(set(data) - _FILE_KEYS)
    if unknown:
        raise ConfigError(f"{path}: unknown key(s): {', '.join(unknown)}")
    for key in ("include", "exclude"):
        value = data.get(key)
        if value is not None and not (
            isinstance(value, list) and all(isinstance(item, str) for item in value)
        ):
            raise ConfigError(f"{path}: '{key}' must be a list of strings")
    if "server" in data and not isinstance(data["server"], str):
        raise ConfigError(f"{path}: 'server' must be a string")
    crawl = data.get("crawl")
    if crawl is not None:
        if not isinstance(crawl, dict):
            raise ConfigError(f"{path}: 'crawl' must be a table")
        unknown = sorted(set(crawl) - set(_CRAWL_KEYS))
        if unknown:
            raise ConfigError(f"{path}: unknown key(s) in [crawl]: {', '.join(unknown)}")
        for key, value in crawl.items():
            expected = _CRAWL_KEYS[key]
            if isinstance(value, bool) or not isinstance(value, expected):
                raise ConfigError(f"{path}: [crawl] '{key}' has the wrong type")
    return data


def load_config(args: Any, env: Mapping[str, str], repo_root: Path) -> CliConfig:
    """Flags override the environment, which overrides the file, then defaults."""
    file_data = _read_file(repo_root)
    server = (
        args.server
        or env.get("VECTORIST_SERVER_URL")
        or file_data.get("server")
        or DEFAULT_SERVER
    )
    batch_size = args.batch_size if args.batch_size is not None else DEFAULT_BATCH_SIZE
    if batch_size < 1:
        raise ConfigError("--batch-size must be at least 1")
    return CliConfig(
        server=server.rstrip("/"),
        include=list(args.include or file_data.get("include") or DEFAULT_INCLUDE),
        exclude=list(args.exclude or file_data.get("exclude") or []),
        repo_id=args.repo_id,
        prune=args.prune,
        dry_run=args.dry_run,
        batch_size=batch_size,
    )


@dataclass(frozen=True)
class CrawlConfig:
    server: str
    url: str
    max_pages: int
    delay: float
    concurrency: int
    keep_query: bool
    prune: bool
    dry_run: bool
    batch_size: int


def load_crawl_config(args: Any, env: Mapping[str, str], cwd: Path) -> CrawlConfig:
    """Flags override the environment, which overrides the file, then defaults.

    A crawl has no repository, so the config file is read from the working directory.
    """
    file_data = _read_file(cwd)
    crawl = file_data.get("crawl", {})
    url = args.url or crawl.get("url")
    if not url:
        raise ConfigError("no URL given: pass one on the command line or set [crawl] url")
    server = (
        args.server
        or env.get("VECTORIST_SERVER_URL")
        or file_data.get("server")
        or DEFAULT_SERVER
    )

    def pick(flag: Any, key: str, default: Any) -> Any:
        if flag is not None:
            return flag
        return crawl.get(key, default)

    max_pages = pick(args.max_pages, "max_pages", DEFAULT_MAX_PAGES)
    delay = pick(args.delay, "delay", DEFAULT_DELAY)
    concurrency = pick(args.concurrency, "concurrency", DEFAULT_CONCURRENCY)
    batch_size = args.batch_size if args.batch_size is not None else DEFAULT_BATCH_SIZE
    if max_pages < 1:
        raise ConfigError("--max-pages must be at least 1")
    if delay < 0:
        raise ConfigError("--delay must not be negative")
    if concurrency < 1:
        raise ConfigError("--concurrency must be at least 1")
    if batch_size < 1:
        raise ConfigError("--batch-size must be at least 1")
    return CrawlConfig(
        server=server.rstrip("/"),
        url=url,
        max_pages=max_pages,
        delay=float(delay),
        concurrency=concurrency,
        keep_query=args.keep_query,
        prune=args.prune,
        dry_run=args.dry_run,
        batch_size=batch_size,
    )
