# Handoff: Vectorist server phase 1

Read this file first if you are picking up the work. Then read
`.superpowers/sdd/progress.md` (gitignored; same worktree only) and the
task you are about to run. Do not re-implement a task the ledger marks
complete.

This file is ephemeral working material. It lives under `docs/superpowers/`
and must be deleted on this branch before a merge to `main`. Do not rewrite
history to remove it. Update this file at the end of every task.

## Where the work is

- Worktree: `/Users/john.kramer/source/personal/vectorist/.worktrees/vectorist-server-phase-1`
- Branch: `vectorist-server-phase-1`
- Plan: `docs/superpowers/plans/2026-10-13-vectorist-server.md`
- Spec: `docs/superpowers/specs/2026-10-13-vectorist-server-design.md`
- Ledger: `.superpowers/sdd/progress.md`
- Per-task briefs and reports: `.superpowers/sdd/task-N-brief.md` and `task-N-report.md`
- Workflow: subagent-driven development. One implementer per task, then a
  task review, then fixes for Critical and Important findings. Do not
  dispatch two implementers at once.

HEAD moves as tasks land. Read `git log --oneline -5` rather than trusting a
SHA copied out of an old chat. As of the pool-close fix, Task 11's range
starts at `89673ab`.

## How to resume

1. `git status` and `git log --oneline`. Trust those and the ledger over
   chat memory.
2. Tasks 1–10 are complete and review-clean. Do not redo them.
3. Task 11 is complete (commits `89673ab..f17705a`, reviews clean). Do not
   redo it.
4. Task 14 is complete (commit `31b7037`, review clean). `wiki/server.md`
   describes the server. An API test covers a second identical ingest
   returning `skipped`. Do not redo it.
5. Tasks 1-14 are implemented and each task review is clean. The
   whole-branch review is the remaining gate. After it, use
   `finishing-a-development-branch`. Delete `docs/superpowers/` on this
   branch before any PR to `main`. Do not rewrite history.

## Commands

Default path. No Docker, no Postgres. This is what CI runs:

```bash
uv run pytest -q
```

At this handoff that is 68 passed, 4 skipped. The skipped tests are
`tests/unit/test_pgvector_store.py`, which skip unless `VECTORIST_PG_URL`
is set.

Postgres tests, only when a database is already available:

```bash
VECTORIST_PG_URL=postgresql://postgres:test@localhost:55432/postgres \
  uv run pytest tests/unit/test_pgvector_store.py -q
```

Do not start Docker, pull images, or install Postgres unless the user
asks. A previous fix subagent hung doing that. `asyncpg` has no default
connect timeout, so a missing database waits until the process is killed.

There is no `ruff` on PATH. Do not block on formatting tools that are not
installed. Pytest is the verification.

## What exists

Pipeline shape, through Task 11:

HTTP is not built yet. `__main__.py` already calls
`uvicorn.run("vectorist.api.app:create_app", factory=True, host="0.0.0.0", port=8080)`,
and `pyproject.toml` exposes `vectorist = "vectorist.__main__:main"`.
`src/vectorist/api/` and `src/vectorist/pipeline/` do not exist yet.
Importing the module entrypoint fails until Task 13.

| Piece | Module | Notes |
| :--- | :--- | :--- |
| Hash | `vectorist.hashing.sha256_text` | SHA-256 of the text |
| IR | `vectorist.ir.models`, `vectorist.ir.serialize` | `IRDocument(source_uri, root_id, nodes)` |
| Migrations | `vectorist.migrations.runner.apply_migrations` | SQLite only. Postgres store applies its own SQL |
| Cache | `vectorist.cache.sqlite_cache.SqliteCache(db_path)` | Dedicated thread. Call `initialize()` before use |
| Markdown | `vectorist.adapters.markdown.MarkdownAdapter` | `parse(source_uri, content) -> IRDocument` |
| Chunks | `vectorist.chunking.chunker.Chunker(max_tokens=512)` | `chunk_document(doc) -> list[Chunk]` |
| Embeddings | `FakeEmbeddingProvider`, lazy `FastEmbedProvider` | 384-d. Fake hashes text only; query and document vectors match |
| sqlite-vec | `SqliteVecStore(db_path)` | Default store. Same thread-worker pattern as the cache |
| pgvector | `PgVectorStore(dsn)` | Optional. See the fix below |

`CacheProtocol`: `get_document_by_uri`, `upsert_document`, `save_ir_snapshot`,
`get_ir_snapshot`, `update_status`. There is no get-by-id and no delete.
`DocumentStatus` values are `PENDING`, `PARSED`, `INDEXED`, `FAILED`.
`documents.source_uri` is unique. `ir_snapshots.document_id` references
`documents(id)` with `ON DELETE CASCADE`.

`VectorStoreProtocol`: `initialize`, `upsert_chunks`, `delete_chunks_by_document_id`,
`get_chunk_hashes`, `search(query_vector, limit=10, filters=None)`.
The only filter in use is `filters["source_uri_prefix"]`.

`Chunk` has no id and no hash. `ChunkRecord` does: `id`, `document_id`,
`chunk_index`, `chunk_hash`, `content_text`, `embedding`, `node_types`,
`metadata`. Search results expect `metadata["source_uri"]` and
`metadata["breadcrumbs"]`.

`EmbeddingProviderProtocol`: `dimensions`, `model_id`, `embed_documents`,
`embed_query`. Sync, not async.

Both SQLite classes require `await initialize()` on a running event loop.
Open failures propagate. Neither exposes `close()`. Workers are daemon
threads. `PgVectorStore` also has no `close()`; tests that connect with
`asyncpg` directly can warn about unclosed connections. That warning is
known and not yet fixed.

## Task 11: complete

Commits `89673ab..f17705a`. Reviews:
[first](52864611-a298-42a9-942e-2b7a338ba33c) and
[re-review of the pool close](ef297f40-97b5-41c1-bb4e-e6b47c416108), both
spec compliant and approved.

`PgVectorStore` uses asyncpg, applies `migrations/postgres/*.sql` through
the Liquibase changeset parser, and scores with `1 - (embedding <=> query)`
(cosine distance). Unfiltered search still uses the HNSW index.

The bug: with `enable_seqscan=off`, Postgres answers a prefix-filtered
`ORDER BY embedding <=> query LIMIT n` from the HNSW index. The index
returns `hnsw.ef_search` candidates (default 40) and then applies the
`WHERE`. A prefix match outside that window disappears, and the query
returns no rows. A small table hides this because the planner picks a
sequential scan.

The fix: the filtered query reads the prefix matches inside
`WITH filtered AS MATERIALIZED (...)`, then sorts that result by distance.
The fence stops the planner from driving the query with the HNSW index.
Turning `enable_indexscan` off was tried first and did not change the plan.
Filtered search therefore scans every matching row. That is the correct
result, and it will get slower as a prefix matches more chunks.

The test `test_search_prefix_filter_keeps_match_outside_hnsw_candidate_window`
opens a second store with `?enable_seqscan=off` on the DSN so the planner
would pick the index without the fix. It expects cosine score `0.0` for an
orthogonal vector. The earlier expectation `1 - sqrt(2)` is the L2 score
and was wrong for cosine. The test was run against `pgvector/pgvector:pg17`
(pgvector 0.8.7): 4 passed. The container was removed afterward. The image
may still be on the machine.

Do not "fix" this by raising `hnsw.ef_search`. That still drops matches
past the new cap.

[Review of Task 11](52864611-a298-42a9-942e-2b7a338ba33c) returned spec
compliant and quality approved. It also listed four Important notes. Three
were the missing `close()`: the store, the `store` fixture, and the extra
store inside the HNSW test. `close()` now drops the asyncpg pool, and both
call sites use it. Re-running the four Postgres tests with
`-W error::ResourceWarning` passed, so the leak warning is gone.

The fourth note guessed that `?enable_seqscan=off` on the DSN might not
reach Postgres. It does. Before the CTE fix, that same test returned no
rows, which is the index plan. A sequential scan of those 200 rows would
have returned the prefix match. Leave the DSN parameter in place.

## Open findings to carry forward

From the Task 10 review, still open, not blocking Tasks 12–14:

- `source_uri_prefix` `LIKE` does not escape `%` or `_`.
- sqlite-vec upsert writes metadata and the vector in separate autocommits.
- One sqlite-vec test constructs `FakeEmbeddingProvider` and does not use it.

Prefix-filtered pgvector search is an exact scan of the matching rows.
That is intentional. `PgVectorStore.close()` exists on the concrete class and is not part of
`VectorStoreProtocol`. The API lifespan calls `close()` when the store has
it. SQLite workers are daemon threads and have no `close()`.

## Server as built after Task 13

`create_app()` in `vectorist.api.app` is synchronous. A FastAPI lifespan
initializes the cache, the vector store, and `IngestService`. Tests enter
it with `app.router.lifespan_context`.

`Settings` (`vectorist.config`) reads:

- `VECTORIST_CACHE_PATH` (required)
- `VECTORIST_VEC_PATH` (required)
- `VECTORIST_VECTOR_BACKEND` (`sqlite` default, or `pgvector`)
- `VECTORIST_DATABASE_URL` (required only for `pgvector`)
- `VECTORIST_EMBEDDER` (`fastembed` default, or `fake`)

An unknown backend, or `pgvector` without a database URL, raises at
startup. Any embedder value other than `fake` constructs `FastEmbedProvider`.

Routes: `POST /v1/ingest`, `POST /v1/ingest/stream` (NDJSON),
`GET /v1/documents/{document_id}`, `DELETE /v1/documents/{document_id}`
(204, or 404 when missing), `POST /v1/search`, `GET /v1/health`.
Health currently always returns ok for all four fields. The pipeline-level
skip test exists. An API-level second-ingest `skipped` test does not.
That is Task 14.

Run the server with `uv run vectorist` or `python -m vectorist` (port 8080).

## Task 12: ingestion pipeline

Files the plan names: `src/vectorist/pipeline/ingest.py`,
`src/vectorist/config.py`, `tests/integration/test_ingest_pipeline.py`.

`IngestService.ingest_document(source_uri, content, content_type="text/markdown")`
returns `IngestResult` with status `indexed`, `skipped`, or `failed`
(lowercase). Phase 1 finishes indexing inside the call. Prefer `indexed`
over `accepted`.

Algorithm from the plan:

1. Reject anything other than `text/markdown` with a `PARSE_ERROR` payload.
2. `content_hash = sha256_text(content)`.
3. Look up by `source_uri`. If the hash matches and status is `INDEXED`,
   return `skipped` and keep the existing document id.
4. Otherwise upsert `PENDING`. Reuse the id when the URI already exists;
   use `uuid4` only for a new URI.
5. Parse, `save_ir_snapshot`, status `PARSED`.
6. Chunk. `chunk_hash = sha256_text(content_text + document_id)`.
7. `get_chunk_hashes`. Embed only new or changed chunks. Delete removed
   chunk ids. Upsert the rest.
8. Status `INDEXED`, or `FAILED` with the error message.

The plan does not name the chunk id. Pick a deterministic id and record it
in the task report. Suggested form: hash of `document_id` and `chunk_index`.

`Settings` holds the cache path and the vector DB path. Use
`pydantic-settings`, which is already a dependency. The integration test
uses a temp SQLite cache, sqlite-vec, `FakeEmbeddingProvider`, and
`tests/fixtures/pooling.md`. Ingest twice: the second call is `skipped`,
and a store search returns a chunk. Do not start Postgres for this task.

`SqliteCache` and `SqliteVecStore` each need `initialize()` before use.
They do not share a connection.

## Task 13: FastAPI routes

Files the plan names under `src/vectorist/api/`: `schemas.py`, `app.py`,
`routes_ingest.py`, `routes_documents.py`, `routes_search.py`,
`routes_health.py`, plus `tests/integration/test_api_e2e.py`.

`create_app()` reads settings and wires `SqliteCache` plus `SqliteVecStore`,
or `PgVectorStore` when `VECTORIST_VECTOR_BACKEND=pgvector`. Embedder is
`FakeEmbeddingProvider` when `VECTORIST_EMBEDDER=fake`, otherwise FastEmbed.
The e2e test sets `VECTORIST_CACHE_PATH`, `VECTORIST_VEC_PATH`, and
`VECTORIST_EMBEDDER=fake`, then uses `httpx.AsyncClient` with
`ASGITransport`. Do not download the FastEmbed model in tests.

Endpoints:

- `POST /v1/ingest/stream` — NDJSON body, JSON list of `IngestResult`
- `POST /v1/ingest` — `{source_uri, content, content_type?}`
- `GET /v1/documents/{document_id}`
- `DELETE /v1/documents/{document_id}` — cache row and vector chunks
- `POST /v1/search` — `{query, limit?, filters?}`
- `GET /v1/health`

`CacheProtocol` cannot get or delete by document id yet. Task 13 needs
those operations. Add them to the protocol and `SqliteCache` rather than
reaching into the SQLite connection from the route. Deleting the document
row cascades the IR snapshot. Chunks live in the other database, so also
call `delete_chunks_by_document_id`.

The test in the plan asserts the stream status is `indexed` or `accepted`,
and that search for `"connection pool"` returns at least one hit.

## Task 14: wiki, health, golden path

Only after Tasks 1–13 exist. Write `wiki/server.md` from the code, not
from this plan. Document install, `uv run pytest`, `uv run vectorist` or
`python -m vectorist`, and the env vars the code actually reads:
`VECTORIST_CACHE_PATH`, `VECTORIST_VEC_PATH`, `VECTORIST_VECTOR_BACKEND`,
`VECTORIST_DATABASE_URL`, `VECTORIST_EMBEDDER`.

`/v1/health` returns `{status: ok|degraded, cache: ok, vector_store: ok, embedder: ok}`.
A second identical ingest through the API returns `skipped`. Do not grow
`README.md` past its pointers. Do not add `docs/superpowers/` to the
Task 14 commit. Change `roadmap.md` only if Phase 1 scope changed.

## Do not do

- Do not merge `docs/superpowers/` to `main`, and do not rewrite history.
- Do not add a CLI, crawler, HTML or PDF adapters, auth, or image enrichment.
- Do not commit `.env`, credentials, or database files.
- Do not start Docker or connect to Postgres unless the user asks.
- Do not run the full suite after a comment-only change. Do run the tests
  that cover the change. Say that the full suite was not run, and give the
  command `uv run pytest`.
