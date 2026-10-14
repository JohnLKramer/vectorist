# Vectorist Server Design

**Date:** 2026-10-13  
**Status:** Draft for review  
**Scope:** Phase 1 server — Unified IR, ingestion, chunking, caching, embeddings, pluggable vector storage, search API

---

## 1. Purpose & Value Proposition

Vectorist is a code-centric knowledge ingestion and retrieval server for RAG and coding agents. It ingests structured text documents (starting with Markdown), builds a **Unified Intermediate Representation (IR)** that preserves semantic boundaries, chunks with context injection, and stores embeddings in a pluggable vector backend.

### Problem

Standard RAG chunkers destroy retrieval quality on developer knowledge:

- Fixed-token and recursive splitters cut through code mid-statement and detach explanatory prose from the code it qualifies.
- Pure AST/code indexers ignore Markdown, wikis, tables, and ADRs where code and prose qualify each other.
- Organizational knowledge also lives in tables, HTML, and PDFs; those formats need a shared IR, not one-off parsers.

### Value

1. **Boundary-aware chunking** via a hierarchical IR (sections, prose, code, tables, media).
2. **Hierarchical breadcrumbs** injected into chunk text so agents recover file + conceptual path.
3. **Self-hostable, free embeddings** (local ONNX / FastEmbed) — no API keys required by default.
4. **Zero-config local storage** (SQLite cache + optional `sqlite-vec`) with a path to PostgreSQL/`pgvector` and dedicated vector DBs later.
5. **Adoption path:** CLI/git sync and web crawler in later phases; async image enrichment without blocking text indexing.

### Out of Scope for This Spec (Later Phases)

- Client CLI, git hooks, IDE extensions
- Web crawler and enterprise connectors (Confluence, Notion, Drive)
- Full HTML/PDF adapters (IR is designed to accept them)
- Async multimodal / OCR enrichment workers (IR `MediaNode` stubs only)
- Knowledge graph / symbol link graph
- Multi-tenant ACLs

---

## 2. Goals & Success Criteria

### Goals (Phase 1)

- Stream documents over HTTP; parse Markdown into Unified IR; chunk; embed; store; search.
- Respect semantic boundaries (headings, code AST or best-effort splits, sticky table headers).
- Separate **document/IR cache** from **vector store** behind protocols.
- Default to SQLite for cache (and `sqlite-vec` for vectors); support PostgreSQL + `pgvector` as production vector backend.
- Liquibase-formatted SQL migrations with a zero-JVM embedded runner for local use.

### Success Criteria

- Re-ingesting unchanged content is a no-op (document SHA-256 match → `skipped`).
- Chunks never cross major section boundaries (`H1`/`H2`).
- Oversized code splits at Tree-sitter boundaries when language is known; otherwise at `;`, `}`, `{`, blank lines, and indentation heuristics.
- Oversized tables slice by row groups with caption + header row on every slice.
- End-to-end: ingest a fixture Markdown corpus → `/v1/search` returns expected chunks for known queries (SQLite path, no Docker required).

---

## 3. Architecture Overview

### Approach

**Decoupled staged pipeline with IR snapshotting:**

1. HTTP ingest accepts documents, hashes content, parses to IR, persists IR to cache, returns quickly with status.
2. Chunker + embedding stage produces vectors and upserts into the vector store.
3. Media/heavy enrichment (Phase 3) hooks into the same lifecycle without re-architecture.

### Components

| Component | Responsibility |
| :--- | :--- |
| Ingestion Gateway (FastAPI) | HTTP streaming, hashing, status responses |
| Format Adapters | Map raw bytes/text → Unified IR (`text/markdown` first) |
| IR Builder | Hierarchical document graph |
| Boundary Chunker | Pack units, specialized splitters, breadcrumb injection |
| Embedding Provider | Pluggable; default FastEmbed / ONNX (`BAAI/bge-small-en-v1.5`, 384-dim) |
| Cache (`CacheProtocol`) | Documents, hashes, IR snapshots — **SQLite default** |
| Vector Store (`VectorStoreProtocol`) | Chunks + embeddings + search — **sqlite-vec default**, **pgvector** supported |
| Search API | Embed query + similarity search + filters |

### Data Flow

```
Client (NDJSON stream)
    → FastAPI /v1/ingest/stream
    → SHA-256(content) vs cache by source_uri
         match + INDEXED → skipped
         else → Markdown adapter → Unified IR tree
    → Cache: documents + ir_snapshots (SQLite)
    → Boundary chunker (+ Tree-sitter / fallback / sticky tables)
    → Breadcrumb injection
    → Embedding provider (batched)
    → VectorStore upsert
    → status INDEXED
```

Search:

```
Client /v1/search {query, limit, filters}
    → Embed query
    → VectorStore.search
    → chunks + scores + breadcrumbs + metadata
```

---

## 4. Unified IR

### Structural Model

**Hierarchical Document Graph** (DOM/AST-like tree): parent/child/sibling relationships, line ranges, and ancestor breadcrumbs.

### Base Node

```text
node_id          Deterministic id (content-addressed where practical)
node_type        SECTION | PARAGRAPH | CODE_BLOCK | TABLE | MEDIA | LIST
parent_id        Optional parent
children_ids     Ordered children
breadcrumbs      Ancestor path labels (file + headings + scopes)
line_range       (start_line, end_line) in source
raw_content      Text for this node
metadata         Node-specific fields
```

### First-Class Node Types (Phase 1)

| Node | Role |
| :--- | :--- |
| `SectionNode` | Heading level 1–6, title, children |
| `ParagraphNode` / text / list | Prose, list items, blockquotes |
| `CodeBlockNode` | `language`, `detection_source`, optional Tree-sitter AST for splits |
| `TableNode` | `caption`, `headers`, optional `column_types`, `rows` |
| `MediaNode` | `media_type`, `alt_text`, `caption`, `asset_uri_or_hash`, `enrichment_status` (`PENDING` / `ENRICHED` / `FAILED`) |

Additional node types (callouts, definitions, math) are deferred until a concrete consumer needs them.

### Code Language Detection (CodeBlockNode)

Order of precedence:

1. Explicit fence/info string (e.g. ```` ```python ````).
2. File extension when the document is a pure source file (future adapter).
3. Lightweight heuristic (optional); else `language = unknown`.

Tree-sitter grammars load for supported languages (Python, TypeScript/JavaScript, Go, Rust, Java at minimum). Unsupported or unknown languages use the fallback splitter (Section 5).

### IR Persistence

The full IR tree for a document is stored as JSON in the **cache** (`ir_snapshots`), enabling re-chunk / re-embed without re-upload or re-parse.

---

## 5. Chunking & Context Injection

### Rules

- Configurable max chunk size (default **512 tokens**).
- Walk IR depth-first; pack adjacent units under the same section.
- Prefer keeping explanatory prose + following code block together when budget allows.
- **Never** cross major section boundaries (`H1` / `H2`).
- Prepend breadcrumb header before embedding, e.g.  
  `[Context: docs/guides/database.md > Storage Architecture > PostgresPool]`.

### Oversized CodeBlock

1. **Preferred:** Tree-sitter — split at function/class/statement boundaries; never mid-expression or inside string literals.
2. **Fallback (unknown/unsupported language):** best-effort splits on:
   - Statement/block delimiters: `;`, `}`, `{`
   - Blank lines (`\n\n`)
   - Indentation-level changes and line breaks

### Oversized TableNode

- Slice into contiguous **row groups**.
- **Sticky headers:** every slice includes caption (if any), header row, and separator styling in the serialized chunk text.

### MediaNode

- Emit text chunk from caption + alt-text + surrounding prose + asset reference.
- Heavy vision/OCR enrichment is asynchronous in a later phase; Phase 1 leaves `enrichment_status = PENDING`.

---

## 6. Caching vs Vector Storage

These are **separate** concerns and protocols.

### Cache (`CacheProtocol`) — default SQLite

Responsibilities: idempotency, document lifecycle, IR snapshots.

Conceptual schema:

```sql
documents (
  id TEXT PRIMARY KEY,
  source_uri TEXT UNIQUE NOT NULL,
  content_hash TEXT NOT NULL,          -- SHA-256 of raw content
  status TEXT NOT NULL,                -- PENDING | PARSED | INDEXED | FAILED
  error_message TEXT,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
)

ir_snapshots (
  document_id TEXT PRIMARY KEY REFERENCES documents(id),
  ir_tree_json JSON NOT NULL,
  node_count INTEGER NOT NULL,
  snapshot_hash TEXT NOT NULL,
  created_at TIMESTAMP
)
```

Phase 1 **must** retain raw document text in cache (`documents.raw_content` or a side table) so adapter upgrades can re-parse without re-upload.

### Vector Store (`VectorStoreProtocol`) — pluggable

Responsibilities: chunk records, embeddings, similarity search, delete-by-document.

Chunk record fields: `id`, `document_id`, `chunk_index`, `content_text` (with breadcrumbs), `embedding`, `node_types`, `metadata` (source_uri, line_range, language, etc.).

Implementations:

| Backend | Role |
| :--- | :--- |
| `sqlite-vec` | Default zero-config vector index |
| `pgvector` | Production / teams with existing Postgres |
| Others (Qdrant, Milvus, …) | Future adapters behind the same protocol |

Solo developers may use SQLite for **both** cache and vectors. Advanced users keep SQLite (or equivalent) as cache and point vectors at Postgres or a dedicated vector DB.

### Hashing Strategy

1. **Document hash:** `SHA-256(raw content)` keyed by `source_uri`. Match + `INDEXED` → skip.
2. **Chunk hash:** deterministic hash of chunk text (including breadcrumbs) + document identity. Unchanged chunk hashes skip re-embedding when a document changes partially.

---

## 7. DDL / Migrations

- Migrations live as **Liquibase-formatted SQL** under `migrations/sqlite/` and `migrations/postgres/` (or vector-backend-specific dirs as needed).
- **Local/default:** small embedded Python runner applies pending changesets; no JVM required.
- **Enterprise:** same files runnable via Liquibase CLI / init containers against PostgreSQL.

---

## 8. Embeddings

- Protocol: `EmbeddingProviderProtocol` (`embed_documents`, `embed_query`, dimension metadata).
- **Default:** FastEmbed / ONNX, `BAAI/bge-small-en-v1.5`, **384** dimensions, Apache-2.0, locally distributable, no API key.
- Optional paid/cloud providers later behind the same protocol.
- Vector store schema and HNSW/`sqlite-vec` dimensions must match the active provider; changing models implies re-embed (IR cache enables re-index without re-ingest).

---

## 9. HTTP API (Phase 1)

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `POST` | `/v1/ingest/stream` | NDJSON stream of documents |
| `POST` | `/v1/ingest` | Single-document convenience upload |
| `GET` | `/v1/documents/{document_id}` | Metadata / status / optional IR summary |
| `DELETE` | `/v1/documents/{document_id}` | Remove cache + vectors |
| `POST` | `/v1/search` | Semantic search |
| `GET` | `/v1/health` | Liveness + backend readiness |

### Ingest payload semantics

Each NDJSON line (or single ingest body) represents **one document**:

| Field | Required | Meaning |
| :--- | :--- | :--- |
| `source_uri` | Yes | Stable identity (repo-relative path, URL, etc.) |
| `content` | Yes | Full raw document body as **UTF-8 text** (e.g. one Markdown file) |
| `content_type` | No | Adapter selector; default/assume `text/markdown` in Phase 1 |

Phase 1 does **not** put binary PDFs/images in `content`. Binary/media upload shapes come with later adapters.

Example:

```json
{"source_uri": "acme/docs/pooling.md", "content_type": "text/markdown", "content": "# Pooling\n\n..."}
```

### Search request (conceptual)

```json
{
  "query": "how is the connection pool configured?",
  "limit": 8,
  "filters": {"source_uri_prefix": "acme/docs/"}
}
```

### Document lifecycle & errors

States: `PENDING` → `PARSED` → `INDEXED` | `FAILED`.

| Condition | Behavior |
| :--- | :--- |
| Unchanged hash | `skipped` |
| Parse failure | `FAILED`; no partial vectors |
| Unknown code language | Index with fallback splitter |
| Embed / vector write failure | Keep IR; `FAILED` with retryable code |

Structured error example:

```json
{"document_id": "...", "status": "failed", "error": {"code": "PARSE_ERROR", "message": "..."}}
```

---

## 10. Tech Stack (Phase 1)

- Python 3.12+, package/env management with **uv**
- FastAPI for HTTP
- Tree-sitter (Python bindings) + grammar packages for major languages
- FastEmbed / ONNX Runtime for embeddings
- SQLite + JSON for cache; `sqlite-vec` for default vectors
- PostgreSQL + `pgvector` as alternate vector implementation
- Liquibase-formatted SQL + embedded migration runner

---

## 11. Testing Strategy

1. **Unit:** IR construction and chunker fixtures (nested Markdown, tagged/untagged fences, tables, media stubs); assert section isolation, Tree-sitter splits, delimiter fallback, sticky headers, breadcrumbs.
2. **Idempotency:** identical re-ingest → skip; partial edit → only changed chunk hashes re-embed.
3. **Integration (SQLite):** stream → search golden queries without Docker.
4. **Optional Postgres:** same suite when `DATABASE_URL` (or equivalent) is set.
5. **Golden corpus:** small Markdown set with code + tables for retrieval regressions.

---

## 12. Product Roadmap (Context)

The durable roadmap is `roadmap.md` at the repo root. This spec is ephemeral and must not be merged to `main`. Do not copy this section into README or wiki.

| Phase | Focus |
| :--- | :--- |
| **1 (this spec)** | Core IR, Markdown adapter, chunker, cache, embeddings, vector backends, ingest + search |
| **2** | Client CLI / git sync; web crawler for internal docs |
| **3** | Async image/diagram enrichment queue; patch IR + re-embed |
| **4** | Enterprise connectors; hybrid BM25 + vector / MCP agent query surface |

---

## 13. Open Decisions Deferred (Explicitly Out of Phase 1)

- Exact NDJSON progress/SSE response shape beyond per-document status
- Whether cache and `sqlite-vec` share one DB file or two
- Full list of Tree-sitter grammars beyond the minimum set
- Authn/authz for multi-user servers
- Compression / max payload limits for streaming ingest

These may be resolved during implementation planning without changing the architecture above.
