from __future__ import annotations

import re
import subprocess
from fnmatch import fnmatch
from pathlib import Path

from vectorist.cli.errors import RepoError


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        )
    except FileNotFoundError as exc:
        raise RepoError("git executable not found") from exc
    if result.returncode != 0:
        raise RepoError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def find_repo_root(path: Path) -> Path:
    if not path.is_dir():
        raise RepoError(f"not a directory: {path}")
    return Path(_git(path, "rev-parse", "--show-toplevel").strip())


_SSH_SCP = re.compile(r"^(?:[^@/]+@)?(?P<host>[^:/]+):(?P<path>.+)$")
_URL = re.compile(r"^[a-z][a-z0-9+.-]*://(?:[^@/]+@)?(?P<host>[^:/]+)(?::\d+)?/(?P<path>.+)$")


def normalize_remote(url: str) -> str:
    url = url.strip()
    match = _URL.match(url) or _SSH_SCP.match(url)
    if match is None:
        raise RepoError(f"cannot parse git remote URL: {url!r}; pass --repo-id")
    path = match.group("path").strip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    return f"{match.group('host')}/{path}"


def repo_identity(root: Path, override: str | None) -> str:
    if override:
        return override.strip("/")
    try:
        url = _git(root, "remote", "get-url", "origin")
    except RepoError as exc:
        raise RepoError(
            "repository has no 'origin' remote; pass --repo-id <host>/<owner>/<repo>"
        ) from exc
    return normalize_remote(url)


def discover_files(root: Path, include: list[str], exclude: list[str]) -> list[str]:
    listing = _git(root, "ls-files", "-z")
    paths = [p for p in listing.split("\0") if p]
    selected = [
        p
        for p in paths
        if any(fnmatch(p, g) for g in include) and not any(fnmatch(p, g) for g in exclude)
        and (root / p).is_file()
    ]
    return sorted(selected)


def source_uri(identity: str, relative_path: str) -> str:
    return f"git:{identity}/{relative_path}"


def uri_prefix(identity: str) -> str:
    return f"git:{identity}/"
