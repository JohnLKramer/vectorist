import asyncio
import subprocess

import httpx

from vectorist.api.app import create_app
from enrich_helpers import wait_for_enrichment
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



PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64


async def test_sync_uploads_repo_images(tmp_path, monkeypatch, cli_client_factory, enrich_env):
    monkeypatch.setenv("VECTORIST_CACHE_PATH", str(tmp_path / "cache.db"))
    monkeypatch.setenv("VECTORIST_VEC_PATH", str(tmp_path / "vec.db"))
    monkeypatch.setenv("VECTORIST_EMBEDDER", "fake")
    monkeypatch.delenv("VECTORIST_VECTOR_BACKEND", raising=False)
    monkeypatch.delenv("VECTORIST_DATABASE_URL", raising=False)
    monkeypatch.delenv("VECTORIST_PG_URL", raising=False)

    repo = tmp_path / "repo"
    (repo / "docs" / "img").mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / "docs" / "a.md").write_text("# A\n\npooling text\n\n![d](img/a.png)\n")
    (repo / "docs" / "img" / "a.png").write_bytes(PNG)
    _git(repo, "add", ".")

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as aclient:
            factory = cli_client_factory(aclient)
            code = await asyncio.to_thread(
                main, ["sync", str(repo), "--repo-id", "h/o/r"], factory
            )
            assert code == 0
            await wait_for_enrichment(aclient)
            search = (await aclient.post("/v1/search", json={"query": "fake description of image", "limit": 10})).json()
            assert any("fake description of image" in h["content_text"] for h in search["results"])
