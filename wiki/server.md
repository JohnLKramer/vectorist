# Vectorist server

Phase 1 serves HTTP ingest and search for Markdown documents. The process loads settings, opens a SQLite document cache and a vector store, builds an embedding provider, and then accepts requests. A failure while opening any of those components stops startup. The process does not stay up with a missing cache, vector store, or embedder.

## Install and test

Python 3.12 or newer. From the repository root:

```bash
uv sync
uv run pytest
```

`uv run pytest` is the default suite. It uses SQLite and does not need Docker or Postgres. Postgres-backed tests, when present, stay optional.

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

`sqlite` is the default vector store. `pgvector` uses `VECTORIST_DATABASE_URL`. Any other backend name fails startup.

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

`POST /v1/ingest/stream` reads a body of newline-delimited JSON objects with the same fields. Blank lines are ignored. A line that is not JSON, or that fails validation, becomes a `failed` result with `error.code` `INVALID_LINE` and does not stop the rest of the body. The response is a JSON array of the same objects as the single ingest route.

## Documents

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
| `limit` | Optional. Defaults to 10 |
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
      "metadata": {}
    }
  ]
}
```

`score` is `1` minus the vector distance, so a closer chunk scores higher. `breadcrumbs` are the heading path stored with the chunk.

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

After a successful startup every field is `ok`. The response model also allows `status` to be `degraded`. The server does not probe components in the background. A process that failed to initialize the cache, the vector store, or the embedder does not stay up to answer this route.
