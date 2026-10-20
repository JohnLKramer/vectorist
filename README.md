[License](LICENSE) · [Roadmap](roadmap.md) · [Wiki](wiki/)

[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-GitHub-ea4aaa?style=for-the-badge&logo=github)](https://github.com/sponsors/JohnLKramer)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/johnlkramer)
[![Ko-fi](https://img.shields.io/badge/Ko--fi-F16061?style=for-the-badge&logo=ko-fi&logoColor=white)](https://ko-fi.com/johnlkramer)


<p align="center">
  <img src="images/v-logo.png" alt="Vectorist logo" width="160">
</p>

# vectorist

Your coding agent can't search docs that were never indexed, and a generic
text splitter will cut a function in half or strand a table row without its
header. `vectorist` ingests your Markdown and HTML into a vector store with
chunking that respects headings, code blocks, and tables. It runs on your own
machine, embeds locally, and gives RAG pipelines and coding agents one search
endpoint. No hosted embedding API sees your docs.

## Table of Contents

- [Quickstart](#quickstart)
- [Why vectorist](#why-vectorist)
- [Get documents in](#get-documents-in)
- [Search](#search)
- [Storage and embeddings](#storage-and-embeddings)
- [How it works](#how-it-works)
- [Roadmap](#roadmap)
- [Development](#development)

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra cli

mkdir -p data
export VECTORIST_CACHE_PATH=./data/cache.db
export VECTORIST_VEC_PATH=./data/vec.db

uv run vectorist    # serves on 0.0.0.0:8080
```

The first embed downloads the `BAAI/bge-small-en-v1.5` model, so the first
document takes longer than the rest.

In another shell, send a document and search for it:

```bash
curl -s -X POST http://127.0.0.1:8080/v1/ingest \
  -H 'Content-Type: application/json' \
  -d '{"source_uri":"docs/pooling.md","content":"# Pool\n\nThe connection pool is shared.\n"}'

curl -s -X POST http://127.0.0.1:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"connection pool","limit":5}'
```

Or sync every Markdown file in a Git repository:

```bash
uv run vectorist-cli sync path/to/repo
```

Every setting is listed in [`wiki/server.md`](wiki/server.md).

## Why vectorist

- **Chunks along the document's own structure.** Splits fall on section
  boundaries. Code blocks are split with Tree-sitter, and tables repeat their
  header row in every chunk.
- **Keeps the context with each chunk.** Search results carry a heading
  breadcrumb, the source line range, and the code language.
- **Embeds locally.** FastEmbed runs the model on your machine, so your docs
  stay there.
- **Re-embeds only what changed.** Re-posting an unchanged document is
  skipped. Editing one section re-embeds that section's chunks, and chunks
  that disappeared are deleted.
- **Starts with zero infrastructure.** SQLite and sqlite-vec are the
  defaults. Switch to PostgreSQL with pgvector when you need it.
- **Fails fast.** A missing setting, an unknown backend, or an unknown
  embedder stops startup with an error instead of leaving a half-working
  server up.

## Get documents in

Three ways in, all ending at the same ingest endpoint. Each document is
identified by a `source_uri` you choose or the tool derives, so re-sending it
updates it in place.

### HTTP ingest

`POST /v1/ingest` takes one document. Set `content_type` to `text/html` for a
rendered web page. The HTML adapter indexes the main content and drops
navigation, headers, footers, and scripts. `POST /v1/ingest/stream` takes
newline-delimited JSON for batches, and one bad line doesn't stop the rest.

`GET /v1/documents?source_prefix=...` lists what is indexed, and
`DELETE /v1/documents/{id}` removes a document and its chunks. See
[`wiki/server.md`](wiki/server.md).

### Sync a Git repository

```bash
uv run vectorist-cli sync [PATH] --server http://localhost:8080
uv run vectorist-cli sync --prune --dry-run   # preview deletions
```

The CLI sends the tracked files that match your include globs (`*.md` by
default). Run it again after edits and the server skips whatever hasn't
changed. Documents are stored as `git:<host>/<owner>/<repo>/<path>`, so SSH and
HTTPS clones of the same repo produce the same IDs. `--prune` refuses to run if
nothing matched, so an empty checkout can't wipe a repository's documents. See
[`wiki/cli.md`](wiki/cli.md).

### Crawl a documentation site

```bash
uv sync --extra crawl
uv run playwright install chromium   # one-time browser download

uv run vectorist-cli crawl https://docs.example.com/guide/
```

The crawler renders each page in headless Chromium, so JavaScript-driven docs
sites work. It stays under the start URL, follows `sitemap.xml` when there is
one, and honors `robots.txt`. Sites that need a login aren't supported. See
[`wiki/crawler.md`](wiki/crawler.md).

## Search

`POST /v1/search` embeds your query with the same model used at ingest and
returns the nearest chunks:

```json
{
  "results": [
    {
      "chunk_id": "…",
      "document_id": "…",
      "source_uri": "docs/pooling.md",
      "content_text": "…",
      "score": 0.8,
      "breadcrumbs": ["pooling.md", "Pool"],
      "metadata": {
        "line_range": [3, 3],
        "node_types": ["PARAGRAPH"]
      }
    }
  ]
}
```

A higher `score` means a closer match. Pass `filters.source_uri_prefix` to
search one repository or one site, for example `git:github.com/acme/widgets/`.

## Storage and embeddings

| Setting | Values | Default |
| :--- | :--- | :--- |
| `VECTORIST_VECTOR_BACKEND` | `sqlite`, `pgvector` | `sqlite` |
| `VECTORIST_EMBEDDER` | `fastembed`, `fake` | `fastembed` |

- **`sqlite`** keeps vectors in a sqlite-vec file. Documents and the parsed
  document structure always live in a SQLite cache.
- **`pgvector`** stores vectors in PostgreSQL. Set `VECTORIST_DATABASE_URL`
  as well.
- **`fake`** produces deterministic hash vectors with no model download. It's
  for tests. Don't point it at a store filled with FastEmbed vectors.

Both stores apply their own schema migrations at startup.

## How it works

```text
HTTP ingest / CLI sync / crawler
        │
        ▼
  hash content ──unchanged──▶ skipped
        │
        ▼
 Markdown or HTML adapter ──▶ hierarchical IR (headings, code, tables, media)
        │
        ▼
 chunker (section boundaries, Tree-sitter code splits,
          sticky table headers, breadcrumbs)
        │
        ▼
 embed changed chunks only (FastEmbed)
        │
        ▼
 vector store (sqlite-vec or pgvector) ──▶ /v1/search
```

Ingest hashes each document, and each chunk, to decide what work to skip.
The adapter turns the document into a tree of headings, paragraphs, code
blocks, and tables, and the chunker cuts along that tree instead of at fixed
character counts. Code blocks are split with Tree-sitter grammars for Python,
JavaScript, TypeScript, Go, Rust, and Java.

Documents and their parsed structure live in the SQLite cache. Vectors live in
a separate vector store. `GET /v1/health` pings both and reports `degraded`
when a check fails.

A failed ingest is not rolled back. The document is marked `FAILED` and the
API returns HTTP 200 with `status: "failed"` and an error code, so check the
status field of each response rather than the HTTP code.

## Roadmap

The server and the ingestion clients (repo sync and the documentation-site
crawler) work today. Image and diagram enrichment, connectors for existing
knowledge systems, and a query surface for coding agents are not built yet.
See [`roadmap.md`](roadmap.md) for the full sequence.

## Development

```bash
uv sync --extra dev
uv run pytest         # default suite: SQLite and the fake embedder, no Docker
make test-pg          # SQLite/pgvector parity (needs Docker)
make test-live        # real server and CLI subprocesses
make test-embedder    # retrieval ranking with the real model
make test-crawl       # crawler against a local server (installs Chromium)
```

Plain `uv sync` removes the `dev` extra. Test tiers are described in
[`wiki/testing.md`](wiki/testing.md).
