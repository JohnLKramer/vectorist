import asyncio
import subprocess

import httpx

from vectorist.api.app import create_app
from vectorist.cli.__main__ import main


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


async def test_sync_edit_delete_and_prune(tmp_path, monkeypatch, cli_client_factory):
    monkeypatch.setenv("VECTORIST_CACHE_PATH", str(tmp_path / "cache.db"))
    monkeypatch.setenv("VECTORIST_VEC_PATH", str(tmp_path / "vec.db"))
    monkeypatch.setenv("VECTORIST_EMBEDDER", "fake")
    monkeypatch.delenv("VECTORIST_VECTOR_BACKEND", raising=False)
    monkeypatch.delenv("VECTORIST_DATABASE_URL", raising=False)
    monkeypatch.delenv("VECTORIST_PG_URL", raising=False)

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "keep.md").write_text("# Keep\n\nconnection pooling notes\n")
    (repo / "drop.md").write_text("# Drop\n\nephemeral text\n")
    _git(repo, "add", ".")
    argv = ["sync", str(repo), "--repo-id", "h/o/r", "--prune"]

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as aclient:
            factory = cli_client_factory(aclient)

            async def run(args):
                return await asyncio.to_thread(main, args, factory)

            assert await run(argv) == 0
            docs = (await aclient.get("/v1/documents?source_prefix=git:h/o/r/")).json()
            assert len(docs["documents"]) == 2

            (repo / "keep.md").write_text("# Keep\n\nrewritten pooling notes\n")
            (repo / "drop.md").unlink()
            _git(repo, "rm", "-q", "--cached", "drop.md")
            assert await run(argv) == 0

            docs = (await aclient.get("/v1/documents?source_prefix=git:h/o/r/")).json()
            assert [d["source_uri"] for d in docs["documents"]] == ["git:h/o/r/keep.md"]
            search = (await aclient.post("/v1/search", json={"query": "rewritten pooling", "limit": 5})).json()
            assert any("rewritten" in hit["content_text"] for hit in search["results"])
            assert all(hit["source_uri"] != "git:h/o/r/drop.md" for hit in search["results"])

