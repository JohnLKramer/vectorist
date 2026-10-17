from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

_NON_HTML_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".pdf", ".zip", ".gz",
    ".tgz", ".css", ".js", ".json", ".xml", ".woff", ".woff2", ".ttf", ".mp4", ".mp3",
}


def normalize(url: str, keep_query: bool) -> str:
    """Canonical form used as the document source_uri."""
    parts = urlsplit(url.strip())
    path = parts.path or "/"
    if len(path) > 1:
        path = path.rstrip("/") or "/"
    query = parts.query if keep_query else ""
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def is_html_candidate(url: str) -> bool:
    path = urlsplit(url).path.lower()
    dot = path.rfind(".")
    return dot <= path.rfind("/") or path[dot:] not in _NON_HTML_EXTENSIONS


class Scope:
    """Same scheme and host as the start URL, path under the start URL's path.

    A start URL whose last segment has a file extension scopes to its directory.
    """

    def __init__(self, start_url: str) -> None:
        parts = urlsplit(normalize(start_url, keep_query=False))
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError(f"not an http(s) URL: {start_url!r}")
        self.scheme = parts.scheme
        self.netloc = parts.netloc
        path = "" if parts.path == "/" else parts.path
        parent, _, last = path.rpartition("/")
        # A start URL that names a file (docs/index.html) scopes to its directory.
        self.path = parent if "." in last else path

    @property
    def origin(self) -> str:
        return f"{self.scheme}://{self.netloc}"

    @property
    def list_prefix(self) -> str:
        """Prefix for listing server documents; narrow it again with contains()."""
        return f"{self.origin}{self.path or '/'}"

    def contains(self, url: str) -> bool:
        parts = urlsplit(url)
        if parts.scheme != self.scheme or parts.netloc != self.netloc:
            return False
        if not self.path:
            return True
        return parts.path == self.path or parts.path.startswith(self.path + "/")
