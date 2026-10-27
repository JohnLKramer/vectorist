# Testing

`uv run pytest` is the default suite. It uses SQLite and the fake embedder, and it needs no Docker, Postgres, or model download. Heavier checks sit behind pytest markers and `make` targets. A tier whose prerequisite is missing skips with a reason instead of failing, except the embedder tier, which the default run leaves out.

## Tiers

| Tier | Marker | Command | Needs | Covers |
|---|---|---|---|---|
| Default | none | `make test` | nothing | Unit tests and in-process API tests over `httpx.ASGITransport` |
| Backend parity | `backends` | `make test-pg` | Docker for the Postgres run | One test body run against sqlite-vec and pgvector |
| Live server | `live` | `make test-live` | nothing for SQLite | uvicorn and the CLI as real subprocesses over a socket |
| Real embedder | `embedder` | `make test-embedder` | Network on first run | Retrieval ranking with FastEmbed |
| Crawl | `crawl` | `make test-crawl` | Chromium | The crawler against a local HTTP server |

## Backend parity

`tests/integration/test_backend_parity.py` and `test_ingest_failure.py` take the `backend` fixture, which runs each test on `sqlite` and `pgvector`. The pgvector run skips unless `VECTORIST_PG_URL` is set. The fixture copies that value into `VECTORIST_DATABASE_URL`, which is the setting the server reads, and truncates `vector_chunks` before and after each test.

`make test-pg` starts `docker-compose.test.yml` (a pgvector image on port 55432), runs `pytest -m backends` against it, and removes the container and volume afterwards, including when tests fail. To use your own database, set `VECTORIST_PG_URL` and run `uv run pytest -m backends`.

## Live server

`tests/integration/live.py` starts `python -m uvicorn vectorist.api.app:create_app --factory` on a free loopback port and polls `/v1/health` until it answers. If the process exits or does not become ready within 30 seconds, the error includes the server's stderr. Teardown sends SIGTERM and kills the process after 10 seconds.

The tests cover ingest and search over a real socket, `vectorist.cli sync` run as a subprocess, the `vectorist-cli` console script, and restarts on the same data. After a restart, search still works, identical content is `skipped`, and an edit is `indexed`. Four startup failures are checked as real processes: an unknown vector backend, an unknown embedder, pgvector with no database URL, and a missing `VECTORIST_CACHE_PATH`. Each must exit non-zero within 15 seconds and name the setting on stderr. These tests match the current error wording, so a reworded error needs a matching test change.

## Real embedder

`tests/integration/test_retrieval_quality.py` ingests four documents with the `fastembed` embedder and checks that each query ranks the expected document first. It compares rank order only, never scores, so a model change does not break it. The default run leaves this tier out, so `make test-embedder` clears that filter.

The model is `BAAI/bge-small-en-v1.5`. FastEmbed caches it under `$TMPDIR/fastembed_cache`, and the OS can clear that directory, which triggers a new download.

## Crawl

`make test-crawl` installs Chromium through Playwright and then runs the crawl test. Without Chromium, a plain `uv run pytest` skips that test and says to run `playwright install chromium`.

## Failed ingests

A failed ingest is not rolled back. The document row is marked `FAILED` with the error message, and the API returns HTTP 200 with `status: "failed"` and error code `INGEST_FAILED`. Re-sending the same content is not skipped, because skipping needs an `INDEXED` document, so it indexes again. `test_ingest_failure.py` checks this.

When an edited document is re-ingested, removed chunks are deleted before the new chunks are written. If the write then fails, the old chunks can be partly gone. No test asserts this.
