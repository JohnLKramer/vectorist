from __future__ import annotations

import base64
import binascii
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import unquote, urljoin, urlsplit

import httpx

from vectorist.cli.client import ServerClient
from vectorist.cli.errors import ServerError

if TYPE_CHECKING:
    from vectorist.cli.sync import SyncSummary

MAX_ASSET_BYTES = 5_000_000
CRAWL_MIN_ASSET_BYTES = 2048


class AssetSkipped(Exception):
    """The reference is remote and is not uploaded."""


class AssetError(Exception):
    """The reference cannot be loaded as an image."""


def sniff_image_type(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _checked(data: bytes, what: str) -> tuple[bytes, str]:
    if len(data) > MAX_ASSET_BYTES:
        raise AssetError(f"{what} is larger than {MAX_ASSET_BYTES} bytes")
    content_type = sniff_image_type(data)
    if content_type is None:
        raise AssetError(f"{what} is not a supported image (PNG, JPEG, GIF, WebP)")
    return data, content_type


def decode_data_uri(ref: str) -> tuple[bytes, str]:
    header, sep, payload = ref.partition(",")
    if not sep or not header.startswith("data:") or not header.endswith(";base64"):
        raise AssetError("malformed data: URI (expected data:<type>;base64,<data>)")
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AssetError(f"malformed data: URI ({exc})") from exc
    return _checked(data, "data: URI")


def load_local_asset(root: Path, doc_path: str, ref: str) -> tuple[bytes, str]:
    if ref.startswith("data:"):
        return decode_data_uri(ref)
    parts = urlsplit(ref)
    if parts.scheme in ("http", "https"):
        raise AssetSkipped(ref)
    if parts.scheme:
        raise AssetError(f"unsupported reference scheme {parts.scheme!r}")
    rel = unquote(parts.path)
    base = Path(root)
    target = base / rel.lstrip("/") if rel.startswith("/") else base / PurePosixPath(doc_path).parent / rel
    try:
        resolved = target.resolve()
        if not resolved.is_relative_to(base.resolve()):
            raise AssetError(f"{ref} points outside the repository")
        data = resolved.read_bytes()
    except (OSError, ValueError) as exc:
        # ValueError: the path has an embedded NUL byte (a "%00" in the ref).
        detail = exc.strerror if isinstance(exc, OSError) and exc.strerror else exc
        raise AssetError(f"cannot read {ref}: {detail}") from exc
    return _checked(data, ref)


def load_remote_asset(fetch: httpx.Client, page_url: str, ref: str) -> tuple[bytes, str]:
    if ref.startswith("data:"):
        return decode_data_uri(ref)
    url = urljoin(page_url, ref)
    scheme = urlsplit(url).scheme
    if scheme not in ("http", "https"):
        raise AssetError(f"unsupported reference scheme {scheme!r}")
    try:
        response = fetch.get(url)
    except httpx.HTTPError as exc:
        raise AssetError(f"cannot download {url}: {exc}") from exc
    if not response.is_success:
        raise AssetError(f"cannot download {url}: HTTP {response.status_code}")
    data = response.content
    if len(data) < CRAWL_MIN_ASSET_BYTES:
        raise AssetSkipped("under 2 KB")
    return _checked(data, url)


def upload_assets(
    client: ServerClient,
    document_id: str,
    source_uri: str,
    refs: list[str],
    load: Callable[[str], tuple[bytes, str]],
    summary: SyncSummary,
) -> None:
    for ref in refs:
        try:
            data, content_type = load(ref)
            client.put_asset(document_id, ref, data, content_type)
        except AssetSkipped:
            summary.assets_skipped += 1
        except (AssetError, ServerError) as exc:
            summary.failures.append(f"{source_uri}: {ref}: {exc}")
        else:
            summary.assets_uploaded += 1
