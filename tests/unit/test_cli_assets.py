import base64

import httpx
import pytest

from vectorist.cli.assets import (
    MAX_ASSET_BYTES,
    AssetError,
    AssetSkipped,
    decode_data_uri,
    CRAWL_MIN_ASSET_BYTES,
    load_local_asset,
    load_remote_asset,
    sniff_image_type,
    upload_assets,
)
from vectorist.cli.errors import ServerError
from vectorist.cli.sync import SyncSummary

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 16


def test_sniff_image_type():
    assert sniff_image_type(PNG) == "image/png"
    assert sniff_image_type(b"\xff\xd8\xff\xe0" + b"0" * 8) == "image/jpeg"
    assert sniff_image_type(b"GIF89a" + b"0" * 8) == "image/gif"
    assert sniff_image_type(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert sniff_image_type(b"hello") is None


def test_decode_data_uri():
    uri = "data:image/png;base64," + base64.b64encode(PNG).decode()
    assert decode_data_uri(uri) == (PNG, "image/png")


def test_decode_data_uri_malformed():
    with pytest.raises(AssetError):
        decode_data_uri("data:image/png;base64,!!!not-base64")
    with pytest.raises(AssetError):
        decode_data_uri("data:image/png")


@pytest.fixture
def root(tmp_path):
    (tmp_path / "docs" / "img").mkdir(parents=True)
    (tmp_path / "img").mkdir()
    (tmp_path / "docs" / "img" / "a.png").write_bytes(PNG)
    (tmp_path / "img" / "a.png").write_bytes(PNG + b"root")
    (tmp_path / "docs" / "my pic.png").write_bytes(PNG)
    return tmp_path


def test_relative_ref(root):
    assert load_local_asset(root, "docs/guide.md", "img/a.png") == (PNG, "image/png")


def test_leading_slash_is_repo_root(root):
    data, _ = load_local_asset(root, "docs/guide.md", "/img/a.png")
    assert data == PNG + b"root"


def test_query_and_fragment_ignored(root):
    assert load_local_asset(root, "docs/guide.md", "img/a.png?raw=1#x")[0] == PNG


def test_percent_decoded(root):
    assert load_local_asset(root, "docs/guide.md", "my%20pic.png")[0] == PNG


def test_outside_repo_rejected(root):
    with pytest.raises(AssetError, match="outside"):
        load_local_asset(root, "docs/guide.md", "../../outside.png")


def test_symlink_outside_rejected(root, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp("elsewhere") / "x.png"
    elsewhere.write_bytes(PNG)
    (root / "docs" / "link.png").symlink_to(elsewhere)
    with pytest.raises(AssetError, match="outside"):
        load_local_asset(root, "docs/guide.md", "link.png")


def test_missing_file(root):
    with pytest.raises(AssetError):
        load_local_asset(root, "docs/guide.md", "nope.png")


def test_nul_in_ref_is_an_asset_error(root):
    with pytest.raises(AssetError, match="bad%00.png"):
        load_local_asset(root, "docs/guide.md", "bad%00.png")


def test_http_is_skipped(root):
    with pytest.raises(AssetSkipped):
        load_local_asset(root, "docs/guide.md", "https://example.com/a.png")


def test_non_image_rejected(root):
    (root / "docs" / "t.png").write_bytes(b"just text")
    with pytest.raises(AssetError, match="not a supported image"):
        load_local_asset(root, "docs/guide.md", "t.png")


def test_too_large_rejected(root):
    (root / "docs" / "big.png").write_bytes(PNG + b"0" * MAX_ASSET_BYTES)
    with pytest.raises(AssetError):
        load_local_asset(root, "docs/guide.md", "big.png")


def test_data_ref_through_load_local_asset(root):
    uri = "data:image/png;base64," + base64.b64encode(PNG).decode()
    assert load_local_asset(root, "docs/guide.md", uri) == (PNG, "image/png")


class StubClient:
    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def put_asset(self, document_id, ref, data, content_type):
        self.calls.append((document_id, ref, data, content_type))
        if ref in self.fail_on:
            raise ServerError("PUT returned 409")
        return {"status": "queued"}


def test_upload_assets_counts_and_continues():
    def load(ref):
        if ref == "skip":
            raise AssetSkipped(ref)
        if ref == "bad":
            raise AssetError("outside the repository")
        return PNG, "image/png"

    client = StubClient(fail_on={"srv"})
    summary = SyncSummary()
    upload_assets(client, "d1", "git:r/a.md", ["ok", "skip", "bad", "srv", "ok2"], load, summary)
    assert summary.assets_uploaded == 2
    assert summary.assets_skipped == 1
    assert summary.failures == [
        "git:r/a.md: bad: outside the repository",
        "git:r/a.md: srv: PUT returned 409",
    ]
    assert [c[1] for c in client.calls] == ["ok", "srv", "ok2"]


BIG_PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 3000


def _remote(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_remote_relative_ref_resolves_against_page_url():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, content=BIG_PNG)

    assert load_remote_asset(_remote(handler), "https://h.example/docs/a/page", "../img/x.png") == (
        BIG_PNG,
        "image/png",
    )
    assert seen == ["https://h.example/docs/img/x.png"]


def test_remote_small_image_is_skipped():
    small = b"\x89PNG\r\n\x1a\n" + b"0" * 992
    client = _remote(lambda r: httpx.Response(200, content=small))
    assert CRAWL_MIN_ASSET_BYTES == 2048
    with pytest.raises(AssetSkipped, match="under 2 KB"):
        load_remote_asset(client, "https://h.example/p", "/x.png")


def test_remote_404_is_an_error():
    client = _remote(lambda r: httpx.Response(404))
    with pytest.raises(AssetError, match="404"):
        load_remote_asset(client, "https://h.example/p", "/x.png")


def test_remote_html_body_is_an_error():
    client = _remote(lambda r: httpx.Response(200, content=b"<html>" + b" " * 3000, headers={"content-type": "image/png"}))
    with pytest.raises(AssetError, match="not a supported image"):
        load_remote_asset(client, "https://h.example/p", "/x.png")


def test_remote_non_http_scheme_is_an_error():
    client = _remote(lambda r: httpx.Response(200, content=BIG_PNG))
    with pytest.raises(AssetError, match="scheme"):
        load_remote_asset(client, "https://h.example/p", "ftp://h.example/x.png")


def test_remote_data_uri_ignores_size_floor():
    uri = "data:image/png;base64," + base64.b64encode(PNG).decode()
    client = _remote(lambda r: httpx.Response(500))
    assert load_remote_asset(client, "https://h.example/p", uri) == (PNG, "image/png")
