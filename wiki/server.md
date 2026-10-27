# Vectorist server

Phase 1 serves HTTP ingest and search for Markdown documents. The process loads settings, opens a SQLite document cache and a vector store, builds an embedding provider, and then accepts requests. A failure while opening any of those components stops startup. The process does not stay up with a missing cache, vector store, or embedder.

## Install and test

Python 3.12 or newer. From the repository root:

```bash
uv sync --extra dev
uv run pytest
```

pytest, pytest-asyncio, and httpx are in the `dev` extra. Plain `uv sync` does not install them, and it removes them if they are already present.

`uv run pytest` is the default suite. It uses SQLite and does not need Docker or Postgres. Postgres-backed tests, when present, stay optional. See [testing.md](testing.md) for the other test tiers.

## Run

```bash
uv run vectorist
```

`python -m vectorist` starts the same server. It listens on `0.0.0.0:8080` and builds the app with `create_app`.

Set the environment variables below before starting. Required paths have no defaults.

## Settings

| Variable | Required | Default | Meaning |
| :--- | :--- | :--- | :--- |
| `VECTORIST_CACHE_PATH` | yes | | SQLite file for documents and the intermediate representation |
| `VECTORIST_VEC_PATH` | yes | | SQLite file for the sqlite-vec store. Still required when the vector backend is Postgres |
| `VECTORIST_VECTOR_BACKEND` | no | `sqlite` | `sqlite` or `pgvector` |
| `VECTORIST_DATABASE_URL` | for `pgvector` | | Postgres URL. Startup fails if the backend is `pgvector` and this is unset |
| `VECTORIST_EMBEDDER` | no | `fastembed` | `fastembed` or `fake` |
| `VECTORIST_ENRICHER` | no | `none` | Image description provider: `none`, `fake`, `ocr`, or `claude`. See [enrichment.md](enrichment.md) |
| `VECTORIST_CLAUDE_MODEL` | no | `claude-haiku-5-5` | Model used when the enricher is `claude` |
| `VECTORIST_ENRICH_POLL_SECONDS` | no | `2.0` | Seconds the enrichment worker waits when the job queue is empty, and before it retries a failed attempt |
| `VECTORIST_ENRICH_MAX_ATTEMPTS` | no | `3` | Failed attempts before an image job is marked `FAILED` |
| `VECTORIST_MAX_ASSET_BYTES` | no | `5000000` | Largest image the upload route accepts |

The `sync` and `crawl` commands have their own upload limit of 5,000,000 bytes per image. That limit is fixed and does not follow `VECTORIST_MAX_ASSET_BYTES`.

`sqlite` is the default vector store. `pgvector` uses `VECTORIST_DATABASE_URL`. Any other backend name fails startup. Any embedder other than `fastembed` or `fake` also fails startup. The match is case-sensitive.

The enricher is checked at startup too. An unknown name fails startup, as does `claude` without `ANTHROPIC_API_KEY` or the `claude` extra, and `ocr` without the `ocr` extra. With the default `none`, no worker starts and ingest behaves as it did before.

Both stores apply their schema migrations at startup. The SQL ships inside the package under `vectorist/migrations/sql/`, so an installed copy finds it the same way a source checkout does. Startup fails if that directory has no migration files.

`fastembed` loads `BAAI/bge-small-en-v1.5` (384 dimensions) on the first embed. `fake` builds deterministic 384-dimension hash vectors and is what the test suite sets. Do not point `fake` at a store that was filled with FastEmbed vectors.

Example for the local SQLite path:

```bash
export VECTORIST_CACHE_PATH=./data/cache.db
export VECTORIST_VEC_PATH=./data/vec.db
uv run vectorist
```

## Ingest

`POST /v1/ingest` accepts one Markdown document as JSON:

```bash
curl -s -X POST http://127.0.0.1:8080/v1/ingest \
  -H 'Content-Type: application/json' \
  -d '{"source_uri":"docs/pooling.md","content":"# Pool\n\nThe connection pool is shared.\n"}'
```

| Field | Meaning |
| :--- | :--- |
| `source_uri` | Caller-chosen identity of the document |
| `content` | UTF-8 Markdown text |
| `content_type` | Optional. Defaults to `text/markdown`. Any other value is rejected |

A successful response looks like:

```json
{
  "source_uri": "docs/pooling.md",
  "document_id": "…",
  "status": "indexed",
  "error": null
}
```

`status` is `indexed` when the document was parsed, chunked, and stored in the same request. Posting the same `source_uri` and the same bytes again returns `skipped` and the same `document_id`. A different body for that URI is indexed again under the same id. `failed` means the document was not indexed. `error` is then an object with `code` and `message`. Unsupported content types use `PARSE_ERROR` and a null `document_id`.

### HTML content

Set `content_type` to `text/html` to ingest a rendered web page. Any other type besides `text/markdown` fails with `PARSE_ERROR`.

The HTML adapter indexes the main content only. It uses the first of `<main>`, `<article>` and `[role=main]` that has text. If the page has none of those, it uses `<body>` with `nav`, `header`, `footer`, `aside` and elements whose `role` is `navigation`, `banner` or `contentinfo` removed. `script`, `style`, `noscript` and `template` are always removed. A page with no text left fails with `INGEST_FAILED` rather than being indexed empty.

Headings become sections and breadcrumbs. The first `h1` (else `<title>`) is the document label. `pre` becomes a code block, with the language taken from a `language-*` or `lang-*` class. Tables keep their header row, and `img` becomes a media node. `line_range` in search results is the element's line in the HTML that was posted, so it is only meaningful for the exact content ingested.

`POST /v1/ingest/stream` reads a body of newline-delimited JSON objects with the same fields. Blank lines are ignored. A line that is not JSON, or that fails validation, becomes a `failed` result with `error.code` `INVALID_LINE` and does not stop the rest of the body. The response is a JSON array of the same objects as the single ingest route.

### Images

When the enricher is not `none`, each ingest response also has `pending_assets`, a list of the image `src` values that still need an upload. `PUT /v1/documents/{document_id}/assets?ref=<src>` takes the image and `GET /v1/enrichment` reports job counts. Both are described in [enrichment.md](enrichment.md).

## Documents

`GET /v1/documents?source_prefix=<prefix>` lists documents whose `source_uri` starts with the prefix, ordered by `source_uri`. The prefix is required and is matched literally (`%` and `_` are ordinary characters). `limit` defaults to 500 (1 to 1000); `cursor` is the `next_cursor` from the previous page.

```json
{
  "documents": [
    {"id": "...", "source_uri": "git:github.com/acme/widgets/a.md", "content_hash": "...", "status": "INDEXED"}
  ],
  "next_cursor": null
}
```

`next_cursor` is `null` on the last page. A missing or empty prefix returns 422.

`GET /v1/documents/{document_id}` returns the stored document:

```json
{
  "id": "…",
  "source_uri": "docs/pooling.md",
  "content_hash": "…",
  "status": "INDEXED",
  "error_message": null,
  "raw_content": "# Pool\n\nThe connection pool is shared.\n"
}
```

`status` here is the cache status: `PENDING`, `PARSED`, `INDEXED`, or `FAILED`. An unknown id returns 404.

`DELETE /v1/documents/{document_id}` removes the document and its chunks and returns 204 with an empty body. An unknown id returns 404.

## Search

`POST /v1/search` embeds the query with the same provider used at ingest and returns the nearest chunks:

```bash
curl -s -X POST http://127.0.0.1:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"connection pool","limit":5}'
```

| Field | Meaning |
| :--- | :--- |
| `query` | Search text |
| `limit` | Optional. Defaults to 10. Must be between 1 and 4096. Other values return 422 |
| `filters.source_uri_prefix` | Optional. Keeps hits whose `source_uri` starts with this string |

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

`score` is `1` minus the vector distance, so a closer chunk scores higher. `breadcrumbs` are the heading path stored with the chunk.

`metadata` holds:

| Key | Meaning |
| :--- | :--- |
| `line_range` | First and last source line of the chunk, 1-based and inclusive |
| `node_types` | Types of the IR nodes in the chunk, such as `PARAGRAPH`, `CODE_BLOCK`, `TABLE`, or `MEDIA` |
| `language` | Only for chunks that contain code. The language of the first code block in the chunk, or `unknown` |

Chunks indexed before a server version added a key keep their old metadata until their text changes.

With `filters.source_uri_prefix`, the sqlite-vec store compares the query against every chunk whose `source_uri` matches, so the nearest matching chunks are returned however many other chunks the store holds.

Markdown indented code blocks become code blocks, like fenced ones without a language. Raw HTML blocks are kept as paragraph text. Their tags are not parsed.

## Health

`GET /v1/health` returns:

```json
{
  "status": "ok",
  "cache": "ok",
  "vector_store": "ok",
  "embedder": "ok"
}
```

Each request checks the cache and the vector store through the connections they opened at startup. The SQLite backends run a query on their worker thread. The pgvector backend runs `SELECT 1` on its existing pool. A check that raises or takes longer than two seconds sets that field to `error` and `status` to `degraded`. The response code is still 200. `embedder` is always `ok`, because the model loads on the first embed.

The server does not probe components in the background. A process that failed to initialize the cache, the vector store, or the embedder does not stay up to answer this route.
