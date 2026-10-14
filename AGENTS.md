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
src/vectorist/     Server package. Not scaffolded yet.
tests/             pytest suite. Not scaffolded yet.
migrations/        Liquibase-formatted SQL. Not scaffolded yet.
docs/superpowers/  Ephemeral specs and implementation plans. Do not merge.
```

## Build & Development Commands

> TODO: no `pyproject.toml` yet. After scaffold, install and test with the
> commands recorded in `pyproject.toml` (expected: `uv sync` and `uv run pytest`).

## Architecture Notes

> TODO: no server code yet. Approved shape, until the code exists:
> HTTP ingest → content hash against a SQLite document/IR cache → Markdown
> adapter builds a hierarchical IR → chunker (section boundaries, code splits,
> sticky table headers, breadcrumbs) → embedding provider → vector store
> (`sqlite-vec` by default, `pgvector` supported) → `/v1/search`.

Cache and vector store are separate protocols. Ingest `content` is the UTF-8
text of one document.

## Testing

> TODO: no tests or CI workflows yet. After scaffold, run `uv run pytest`.
> Default path must pass without Docker. Postgres tests stay optional.

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
