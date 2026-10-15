# Project Overview

Vectorist ingests structured documents (starting with Markdown) into a vector
store so retrieval-augmented generation and coding agents can search them.
Phase 1 is the server: HTTP ingest, a hierarchical intermediate representation,
boundary-aware chunking, local embeddings, and pluggable storage.

Tech stack: Python, uv, hatchling, FastAPI, uvicorn, pydantic, tree-sitter,
FastEmbed, SQLite, sqlite-vec, PostgreSQL, pgvector, pytest.

## Repository Structure

```text
AGENTS.md          Agent instructions (this file). Keep on main.
roadmap.md         High-level product roadmap. Keep on main.
README.md          One-paragraph project summary and pointers. Keep on main.
LICENSE
wiki/              Functional and product docs. Write these after the code
                   they describe. Keep on main.
pyproject.toml     Package metadata, dependencies, the `dev` extra, pytest config.
src/vectorist/     Server package. Liquibase-formatted SQL migrations ship
                   inside it under `migrations/sql/{sqlite,postgres}/`.
tests/             pytest suite: `unit/`, `integration/`, `fixtures/`.
docs/superpowers/  Ephemeral specs and implementation plans. Do not merge.
```

## Build & Development Commands

```bash
uv sync --extra dev   # pytest, pytest-asyncio, httpx live in the dev extra
uv run pytest         # default suite: SQLite only, no Docker or Postgres
uv run vectorist      # start the HTTP server on 0.0.0.0:8080
```

Plain `uv sync` does not install the `dev` extra and removes it if present.
The server needs `VECTORIST_CACHE_PATH` and `VECTORIST_VEC_PATH`. See
`wiki/server.md` for every setting.

## Architecture Notes

Request flow: HTTP ingest (`api/routes_ingest.py`) → `pipeline/ingest.py`
hashes the content and checks the SQLite document/IR cache (`cache/`) → the
Markdown adapter (`adapters/markdown.py`) builds a hierarchical IR (`ir/`) →
the chunker (`chunking/`: section boundaries, Tree-sitter code splits, sticky
table headers, breadcrumbs) → the embedding provider (`embeddings/`: FastEmbed
or a fake hash provider for tests) → the vector store (`storage/`: `sqlite-vec`
by default, `pgvector` supported) → `/v1/search`.

- `api/app.py` wires the backends from `config.py` settings and fails startup
  on an unknown vector backend or embedder.
- Only changed chunk hashes are re-embedded; removed chunks are deleted.
- The SQLite cache and sqlite-vec store each run all queries on one worker
  thread. A failed job rolls back before the error reaches the caller.
- Embedding runs in `asyncio.to_thread`; the provider protocol is synchronous.
- `/v1/health` pings the cache and vector store and reports `degraded` when a
  check fails.
- `migrations/runner.py` applies the packaged SQL for SQLite; the pgvector
  store applies the Postgres SQL itself.

Cache and vector store are separate protocols. Ingest `content` is the UTF-8
text of one document.

## Testing

Run `uv run pytest`. The default path must pass without Docker. Postgres
tests in `tests/unit/test_pgvector_store.py` stay optional and run only when
`VECTORIST_PG_URL` is set. There are no CI workflows yet.

## Agent Guardrails

- Do not merge `docs/superpowers/` to `main`. Specs and implementation plans
  are working material for the feature branch only.
- Do not merge other ephemeral notes (scratch designs, session logs, draft
  outlines) to `main`.
- Before a branch merges to `main`, delete `docs/superpowers/` on that branch
  so the merge does not keep or reintroduce it. Those files are already in
  local `main` history; removing them on the feature branch is how they leave
  `main`. Do not rewrite history to do this.
- Put functional and product documentation under `wiki/`, and write it only
  after the code it describes exists.
- Put the high-level roadmap in `roadmap.md` only. Do not duplicate it into
  README or wiki pages.
- README stays a short summary plus pointers to `roadmap.md` and `wiki/`.
- Do not add CLI, crawler, HTML/PDF adapters, auth, or async image enrichment
  during Phase 1 server work.
- Do not commit secrets (`.env`, credentials, database files).
