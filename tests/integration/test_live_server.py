import subprocess
import sys

import httpx
import pytest

pytestmark = pytest.mark.live


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _sync(live_server, repo):
    return subprocess.run(
        [
            sys.executable, "-m", "vectorist.cli", "sync", str(repo),
            "--server", live_server.base_url, "--repo-id", "h/o/r", "--prune",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _documents(live_server):
    response = httpx.get(
        f"{live_server.base_url}/v1/documents", params={"source_prefix": "git:h/o/r/"}
    )
    return [d["source_uri"] for d in response.json()["documents"]]


def test_ingest_and_search_over_socket(live_server, fixtures_dir):
    ingest = httpx.post(
        f"{live_server.base_url}/v1/ingest",
        json={"source_uri": "repo/pooling.md", "content": (fixtures_dir / "pooling.md").read_text()},
    )
    assert ingest.status_code == 200
    assert ingest.json()["status"] == "indexed"

    search = httpx.post(
        f"{live_server.base_url}/v1/search", json={"query": "connection pool", "limit": 5}
    )
    assert search.status_code == 200
    assert search.json()["results"][0]["source_uri"] == "repo/pooling.md"


def test_cli_sync_edit_delete_and_prune(live_server, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "keep.md").write_text("# Keep\n\nconnection pooling notes\n")
    (repo / "drop.md").write_text("# Drop\n\nephemeral text\n")
    _git(repo, "add", ".")

    first = _sync(live_server, repo)
    assert first.returncode == 0, first.stderr
    assert _documents(live_server) == ["git:h/o/r/drop.md", "git:h/o/r/keep.md"]

    (repo / "keep.md").write_text("# Keep\n\nrewritten pooling notes\n")
    (repo / "drop.md").unlink()
    _git(repo, "rm", "-q", "--cached", "drop.md")
    second = _sync(live_server, repo)
    assert second.returncode == 0, second.stderr
    assert _documents(live_server) == ["git:h/o/r/keep.md"]

    search = httpx.post(
        f"{live_server.base_url}/v1/search", json={"query": "rewritten pooling", "limit": 5}
    ).json()
    assert any("rewritten" in hit["content_text"] for hit in search["results"])
    assert all(hit["source_uri"] != "git:h/o/r/drop.md" for hit in search["results"])


def test_console_script_resolves():
    script = __import__("pathlib").Path(sys.executable).parent / "vectorist-cli"
    result = subprocess.run([str(script), "--help"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
