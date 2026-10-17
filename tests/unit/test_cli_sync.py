import json
import subprocess
from argparse import Namespace

import httpx
import pytest

from vectorist.cli.client import ServerClient
from vectorist.cli.config import load_config
from vectorist.cli.errors import CliError, ServerError
from vectorist.cli.sync import run_sync

ID = "h/o/r"


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    for name in ("a.md", "b.md", "c.md"):
        (tmp_path / name).write_text(f"# {name}")
    _git(tmp_path, "add", ".")
    return tmp_path


def _config(root, **overrides):
    base = dict(
        server=None, include=None, exclude=None, repo_id=ID,
        prune=False, dry_run=False, batch_size=None,
    )
    base.update(overrides)
    return load_config(Namespace(**base), {}, root)


class FakeServer:
    def __init__(self, existing=(), statuses=None, fail=False):
        self.requests = []
        self.existing = list(existing)
        self.statuses = statuses or {}
        self.fail = fail

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail:
            return httpx.Response(500, text="boom")
        if request.url.path == "/v1/ingest/stream":
            lines = [json.loads(l) for l in request.content.decode().splitlines()]
            return httpx.Response(200, json=[
                {"source_uri": l["source_uri"], "document_id": "x",
                 "status": self.statuses.get(l["source_uri"], "indexed"),
                 "error": {"code": "E", "message": "bad"} if self.statuses.get(l["source_uri"]) == "failed" else None}
                for l in lines
            ])
        if request.url.path == "/v1/documents" and request.method == "GET":
            return httpx.Response(200, json={"documents": self.existing, "next_cursor": None})
        if request.method == "DELETE":
            return httpx.Response(204)
        raise AssertionError(request.url)

    def client(self):
        return ServerClient(httpx.Client(transport=httpx.MockTransport(self), base_url="http://t"))

    def posts(self):
        return [r for r in self.requests if r.method == "POST"]

    def deletes(self):
        return [r.url.path for r in self.requests if r.method == "DELETE"]


def test_batches_by_batch_size(repo):
    server = FakeServer()
    summary = run_sync(_config(repo, batch_size=2), server.client(), repo)
    assert len(server.posts()) == 2
    assert summary.ingested == 3 and summary.failures == []


def test_failed_and_skipped_results_are_counted(repo):
    server = FakeServer(statuses={f"git:{ID}/a.md": "failed", f"git:{ID}/b.md": "skipped"})
    summary = run_sync(_config(repo), server.client(), repo)
    assert summary.unchanged == 1 and summary.ingested == 1
    assert len(summary.failures) == 1 and "a.md" in summary.failures[0]


def test_server_error_aborts(repo):
    with pytest.raises(ServerError):
        run_sync(_config(repo), FakeServer(fail=True).client(), repo)


def test_prune_deletes_only_missing_files(repo):
    existing = [
        {"id": "keep", "source_uri": f"git:{ID}/a.md"},
        {"id": "gone", "source_uri": f"git:{ID}/old.md"},
    ]
    server = FakeServer(existing=existing)
    summary = run_sync(_config(repo, prune=True), server.client(), repo)
    assert server.deletes() == ["/v1/documents/gone"]
    assert summary.pruned == [f"git:{ID}/old.md"]


def test_prune_refused_when_no_files(repo):
    server = FakeServer(existing=[{"id": "x", "source_uri": f"git:{ID}/a.md"}])
    with pytest.raises(CliError, match="refused"):
        run_sync(_config(repo, prune=True, include=["*.nomatch"]), server.client(), repo)
    assert server.requests == []


def test_dry_run_sends_nothing_and_deletes_nothing(repo):
    server = FakeServer(existing=[{"id": "gone", "source_uri": f"git:{ID}/old.md"}])
    summary = run_sync(_config(repo, prune=True, dry_run=True), server.client(), repo)
    assert server.posts() == [] and server.deletes() == []
    assert summary.sent == 3 and summary.pruned == [f"git:{ID}/old.md"]


def test_non_utf8_file_is_reported_not_fatal(repo):
    (repo / "bad.md").write_bytes(b"\xff\xfe\x00bad")
    _git(repo, "add", "bad.md")
    summary = run_sync(_config(repo), FakeServer().client(), repo)
    assert summary.ingested == 3
    assert len(summary.failures) == 1 and "bad.md" in summary.failures[0]
