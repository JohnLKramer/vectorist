# Vectorist Server Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 1 Vectorist server that ingests Markdown documents over HTTP, converts them to a Unified IR, boundary-aware chunks them, embeds with local FastEmbed, and stores/searches via pluggable cache + vector backends (SQLite/`sqlite-vec` default, `pgvector` supported).

**Architecture:** Decoupled staged pipeline — FastAPI ingest → SHA-256 idempotency via SQLite cache → Markdown adapter builds hierarchical IR (snapshotted in cache) → boundary chunker with breadcrumbs → embedding provider → `VectorStoreProtocol` upsert; `/v1/search` embeds the query and retrieves top-k chunks.

**Tech Stack:** Python 3.12+, uv, FastAPI, uvicorn, pydantic, tree-sitter + language grammars, fastembed, sqlite-vec, asyncpg/pgvector (optional), pytest, httpx, mistune or markdown-it-py for Markdown parsing.

**Spec:** `docs/superpowers/specs/2026-10-13-vectorist-server-design.md`

## Global Constraints

- Python **3.12+**; manage deps with **uv**.
- Default embeddings: **FastEmbed / ONNX**, model **`BAAI/bge-small-en-v1.5`**, **384** dimensions, no API key.
- Default cache: **SQLite**; default vectors: **`sqlite-vec`**; production vectors: **PostgreSQL + pgvector**.
- Migrations: **Liquibase-formatted SQL** + embedded Python runner (no JVM required locally).
- Chunk default max size: **512 tokens** (approximate with whitespace/word tokenizer in Phase 1 if no tokenizer dependency; document the approximation).
- Ingest `content` is **UTF-8 text for one document**; `content_type` defaults to `text/markdown`.
- Code fallback splits on **`;` `{` `}`**, blank lines, and indentation — never invent mid-token splits when delimiters exist.
- Do not implement CLI, crawler, HTML/PDF adapters, auth, or async OCR in this plan.
- TDD: failing test → implement → pass → commit per task.
- Package import root: `vectorist`.
- `docs/superpowers/` (this plan and the spec) is ephemeral. Do not merge it to `main`. Before opening a PR to `main`, delete `docs/superpowers/` on this branch so the merge removes it. Do not rewrite history.
- Do not merge other ephemeral notes to `main`.
- Durable docs: high-level roadmap only in `roadmap.md`. Functional and product docs only under `wiki/`, and only after the code they describe exists. `README.md` stays a short summary plus pointers to those two. Do not put usage manuals in `README.md`.

---

## File Structure

```text
pyproject.toml
README.md                       # short summary + pointers only
AGENTS.md                       # agent instructions (already present)
roadmap.md                      # high-level roadmap (already present)
wiki/                           # product docs; create in Task 14, after the code
migrations/
  sqlite/
    001_cache_init.sql
    002_vec_init.sql
  postgres/
    001_pgvector_init.sql
src/vectorist/
  __init__.py
  __main__.py                 # uvicorn entry: python -m vectorist
  config.py                   # settings from env
  hashing.py                  # sha256 helpers
  ir/
    __init__.py
    models.py                 # IR node dataclasses / pydantic models
    serialize.py              # IR ↔ JSON for snapshots
  adapters/
    __init__.py
    base.py                   # AdapterProtocol
    markdown.py               # Markdown → IRDocument
    language.py               # code fence / heuristic language detection
  chunking/
    __init__.py
    tokenizer.py              # token estimate
    chunker.py                # tree walk, packing, breadcrumbs
    code_split.py             # tree-sitter + delimiter fallback
    table_split.py            # sticky-header row groups
  embeddings/
    __init__.py
    protocol.py
    fastembed_provider.py
    fake.py                   # deterministic fake for unit tests
  cache/
    __init__.py
    protocol.py
    models.py                 # DocumentRecord, statuses
    sqlite_cache.py
  storage/
    __init__.py
    protocol.py
    models.py                 # ChunkRecord, SearchResult
    sqlite_vec_store.py
    pgvector_store.py
  migrations/
    __init__.py
    runner.py                 # apply liquibase-formatted SQL
  pipeline/
    __init__.py
    ingest.py                 # orchestration: hash → parse → cache → chunk → embed → store
  api/
    __init__.py
    app.py                    # FastAPI factory
    schemas.py                # request/response pydantic models
    routes_ingest.py
    routes_documents.py
    routes_search.py
    routes_health.py
tests/
  conftest.py
  fixtures/
    pooling.md
    wide_table.md
    untagged_code.md
  unit/
    test_hashing.py
    test_ir_models.py
    test_migrations_runner.py
    test_sqlite_cache.py
    test_markdown_adapter.py
    test_language_detect.py
    test_chunker.py
    test_code_split.py
    test_table_split.py
    test_embeddings_fake.py
    test_sqlite_vec_store.py
  integration/
    test_ingest_pipeline.py
    test_api_e2e.py
```

---

### Task 1: Project Scaffold & Tooling

**Files:**
- Create: `pyproject.toml`
- Create: `src/vectorist/__init__.py`
- Create: `src/vectorist/__main__.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: nothing
- Produces: installable package `vectorist`, pytest discovers `tests/`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "vectorist"
version = "0.1.0"
description = "Code-centric document ingestion and vector retrieval for RAG agents"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
  "fastapi>=0.115.0",
  "uvicorn[standard]>=0.32.0",
  "pydantic>=2.9.0",
  "pydantic-settings>=2.6.0",
  "markdown-it-py>=3.0.0",
  "mdit-py-plugins>=0.4.0",
  "tree-sitter>=0.23.0",
  "tree-sitter-python>=0.23.0",
  "tree-sitter-javascript>=0.23.0",
  "tree-sitter-typescript>=0.23.0",
  "tree-sitter-go>=0.23.0",
  "tree-sitter-rust>=0.23.0",
  "tree-sitter-java>=0.23.0",
  "fastembed>=0.4.0",
  "sqlite-vec>=0.1.0",
  "asyncpg>=0.30.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.3.0",
  "pytest-asyncio>=0.24.0",
  "httpx>=0.27.0",
]

[project.scripts]
vectorist = "vectorist.__main__:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/vectorist"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Create package entrypoints**

`src/vectorist/__init__.py`:
```python
__version__ = "0.1.0"
```

`src/vectorist/__main__.py`:
```python
def main() -> None:
    import uvicorn
    uvicorn.run("vectorist.api.app:create_app", factory=True, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
```

`tests/conftest.py`:
```python
from pathlib import Path
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
```

- [ ] **Step 3: Install and verify pytest runs**

Run:
```bash
uv sync --extra dev
uv run pytest -q
```
Expected: `no tests ran` or `0 passed` (empty suite OK). Exit code 5 is acceptable if no tests collected; add a trivial smoke test if needed:

```python
# tests/unit/test_version.py
from vectorist import __version__

def test_version():
    assert __version__ == "0.1.0"
```

Run: `uv run pytest tests/unit/test_version.py -v`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml src/vectorist/__init__.py src/vectorist/__main__.py tests/conftest.py tests/unit/test_version.py uv.lock
git commit -m "chore: scaffold vectorist Python package with uv and pytest"
```

---

### Task 2: Hashing Helpers & IR Models

**Files:**
- Create: `src/vectorist/hashing.py`
- Create: `src/vectorist/ir/__init__.py`
- Create: `src/vectorist/ir/models.py`
- Create: `src/vectorist/ir/serialize.py`
- Create: `tests/unit/test_hashing.py`
- Create: `tests/unit/test_ir_models.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `sha256_text(text: str) -> str` (64-char hex)
  - `NodeType` enum: `SECTION | PARAGRAPH | CODE_BLOCK | TABLE | MEDIA | LIST`
  - `IRNode` / specialized models with fields from spec
  - `IRDocument(source_uri: str, root: IRNode, nodes: dict[str, IRNode])`
  - `ir_to_json(doc: IRDocument) -> str` / `ir_from_json(data: str) -> IRDocument`

- [ ] **Step 1: Write failing hashing test**

```python
# tests/unit/test_hashing.py
from vectorist.hashing import sha256_text

def test_sha256_text_is_stable_hex():
    h1 = sha256_text("# Hello\n")
    h2 = sha256_text("# Hello\n")
    assert h1 == h2
    assert len(h1) == 64
    assert h1 != sha256_text("# Hello\n ")
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `uv run pytest tests/unit/test_hashing.py -v`  
Expected: FAIL (`ModuleNotFoundError` or import error)

- [ ] **Step 3: Implement hashing**

```python
# src/vectorist/hashing.py
import hashlib

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run hashing test — PASS**

Run: `uv run pytest tests/unit/test_hashing.py -v`  
Expected: PASS

- [ ] **Step 5: Write failing IR model tests**

```python
# tests/unit/test_ir_models.py
from vectorist.ir.models import (
    CodeBlockNode,
    IRDocument,
    NodeType,
    ParagraphNode,
    SectionNode,
)
from vectorist.ir.serialize import ir_from_json, ir_to_json

def test_round_trip_ir_document():
    para = ParagraphNode(
        node_id="p1",
        parent_id="s1",
        children_ids=[],
        breadcrumbs=["doc.md", "Intro"],
        line_range=(3, 4),
        raw_content="Hello",
        metadata={},
    )
    section = SectionNode(
        node_id="s1",
        parent_id=None,
        children_ids=["p1"],
        breadcrumbs=["doc.md", "Intro"],
        line_range=(1, 4),
        raw_content="Intro",
        metadata={},
        level=1,
        title="Intro",
    )
    code = CodeBlockNode(
        node_id="c1",
        parent_id="s1",
        children_ids=[],
        breadcrumbs=["doc.md", "Intro"],
        line_range=(5, 10),
        raw_content="print(1)",
        metadata={},
        language="python",
        detection_source="fence",
    )
    section.children_ids = ["p1", "c1"]
    doc = IRDocument(
        source_uri="doc.md",
        root_id="s1",
        nodes={n.node_id: n for n in (section, para, code)},
    )
    restored = ir_from_json(ir_to_json(doc))
    assert restored.source_uri == "doc.md"
    assert restored.nodes["c1"].node_type == NodeType.CODE_BLOCK
    assert restored.nodes["c1"].language == "python"
```

- [ ] **Step 6: Implement IR models + serialize (minimal to pass)**

Use pydantic `BaseModel` with `node_type` discriminator. Required fields per spec:

```python
# src/vectorist/ir/models.py (sketch — implement fully)
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field

class NodeType(str, Enum):
    SECTION = "SECTION"
    PARAGRAPH = "PARAGRAPH"
    CODE_BLOCK = "CODE_BLOCK"
    TABLE = "TABLE"
    MEDIA = "MEDIA"
    LIST = "LIST"

class IRNodeBase(BaseModel):
    node_id: str
    node_type: NodeType
    parent_id: str | None
    children_ids: list[str]
    breadcrumbs: list[str]
    line_range: tuple[int, int]
    raw_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

class SectionNode(IRNodeBase):
    node_type: Literal[NodeType.SECTION] = NodeType.SECTION
    level: int
    title: str

class ParagraphNode(IRNodeBase):
    node_type: Literal[NodeType.PARAGRAPH] = NodeType.PARAGRAPH

class ListNode(IRNodeBase):
    node_type: Literal[NodeType.LIST] = NodeType.LIST
    ordered: bool = False

class CodeBlockNode(IRNodeBase):
    node_type: Literal[NodeType.CODE_BLOCK] = NodeType.CODE_BLOCK
    language: str  # "python" | "unknown" | ...
    detection_source: str  # "fence" | "extension" | "heuristic" | "unknown"

class TableNode(IRNodeBase):
    node_type: Literal[NodeType.TABLE] = NodeType.TABLE
    caption: str | None = None
    headers: list[str]
    column_types: list[str] = Field(default_factory=list)
    rows: list[list[str]]

class MediaNode(IRNodeBase):
    node_type: Literal[NodeType.MEDIA] = NodeType.MEDIA
    media_type: str
    alt_text: str = ""
    caption: str = ""
    asset_uri_or_hash: str = ""
    enrichment_status: str = "PENDING"

IRNode = SectionNode | ParagraphNode | ListNode | CodeBlockNode | TableNode | MediaNode

class IRDocument(BaseModel):
    source_uri: str
    root_id: str
    nodes: dict[str, IRNode]
```

`serialize.py`: `model_dump_json` / parse with tagged union on `node_type`.

- [ ] **Step 7: Run IR tests — PASS**

Run: `uv run pytest tests/unit/test_ir_models.py tests/unit/test_hashing.py -v`  
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/vectorist/hashing.py src/vectorist/ir tests/unit/test_hashing.py tests/unit/test_ir_models.py
git commit -m "feat: add content hashing and Unified IR models"
```

---

### Task 3: Liquibase-Formatted Migrations Runner

**Files:**
- Create: `migrations/sqlite/001_cache_init.sql`
- Create: `migrations/sqlite/002_vec_init.sql`
- Create: `migrations/postgres/001_pgvector_init.sql`
- Create: `src/vectorist/migrations/__init__.py`
- Create: `src/vectorist/migrations/runner.py`
- Create: `tests/unit/test_migrations_runner.py`

**Interfaces:**
- Consumes: path to migrations dir, DB connection (sqlite3)
- Produces: `apply_migrations(conn, migrations_dir: Path) -> list[str]` applied changeset ids; `_schema_migrations(id, applied_at)` table

- [ ] **Step 1: Write migration SQL files**

`migrations/sqlite/001_cache_init.sql`:
```sql
--liquibase formatted sql
--changeset vectorist:001-cache-init
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source_uri TEXT UNIQUE NOT NULL,
    content_hash TEXT NOT NULL,
    raw_content TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ir_snapshots (
    document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    ir_tree_json TEXT NOT NULL,
    node_count INTEGER NOT NULL,
    snapshot_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

`migrations/sqlite/002_vec_init.sql`:
```sql
--liquibase formatted sql
--changeset vectorist:002-vec-init
CREATE TABLE IF NOT EXISTS chunk_metadata (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_hash TEXT NOT NULL,
    content_text TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunk_metadata_document_id
ON chunk_metadata(document_id);
```

(Note: `sqlite-vec` virtual table creation happens in `SqliteVecStore.initialize()` because `CREATE VIRTUAL TABLE ... USING vec0` requires the extension loaded — document this in runner comments.)

`migrations/postgres/001_pgvector_init.sql`:
```sql
--liquibase formatted sql
--changeset vectorist:001-pgvector-init
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS vector_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_hash TEXT NOT NULL,
    content_text TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    metadata JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_vector_chunks_document_id
ON vector_chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_vector_chunks_embedding
ON vector_chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

- [ ] **Step 2: Write failing runner test**

```python
# tests/unit/test_migrations_runner.py
import sqlite3
from pathlib import Path
from vectorist.migrations.runner import apply_migrations

ROOT = Path(__file__).resolve().parents[2]

def test_apply_sqlite_cache_migrations(tmp_path: Path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    applied = apply_migrations(conn, ROOT / "migrations" / "sqlite")
    assert "vectorist:001-cache-init" in applied
    assert "vectorist:002-vec-init" in applied
    # idempotent
    applied2 = apply_migrations(conn, ROOT / "migrations" / "sqlite")
    assert applied2 == []
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    names = {r[0] for r in rows}
    assert "documents" in names
    assert "ir_snapshots" in names
    assert "chunk_metadata" in names
    assert "_schema_migrations" in names
```

- [ ] **Step 3: Run — FAIL, then implement runner**

Runner behavior:
1. Ensure `_schema_migrations(id TEXT PRIMARY KEY, applied_at TEXT)`.
2. Scan `*.sql` sorted by filename.
3. Parse `--changeset <author>:<id>` lines; split file into changesets.
4. Skip already-applied ids; execute SQL; record id.

- [ ] **Step 4: Run test — PASS; Commit**

```bash
git add migrations src/vectorist/migrations tests/unit/test_migrations_runner.py
git commit -m "feat: add Liquibase-formatted SQL migration runner"
```

---

### Task 4: SQLite Document/IR Cache

**Files:**
- Create: `src/vectorist/cache/protocol.py`
- Create: `src/vectorist/cache/models.py`
- Create: `src/vectorist/cache/sqlite_cache.py`
- Create: `src/vectorist/cache/__init__.py`
- Create: `tests/unit/test_sqlite_cache.py`

**Interfaces:**
- Consumes: migrations runner, `IRDocument`, `sha256_text`
- Produces:
```python
class DocumentStatus(str, Enum):
    PENDING = "PENDING"
    PARSED = "PARSED"
    INDEXED = "INDEXED"
    FAILED = "FAILED"

@dataclass
class DocumentRecord:
    id: str
    source_uri: str
    content_hash: str
    raw_content: str
    status: DocumentStatus
    error_message: str | None = None

class CacheProtocol(Protocol):
    async def get_document_by_uri(self, source_uri: str) -> DocumentRecord | None: ...
    async def upsert_document(self, doc: DocumentRecord) -> None: ...
    async def save_ir_snapshot(self, document_id: str, ir_tree: IRDocument) -> None: ...
    async def get_ir_snapshot(self, document_id: str) -> IRDocument | None: ...
    async def update_status(self, document_id: str, status: DocumentStatus, error_message: str | None = None) -> None: ...
```

`SqliteCache(db_path: Path)` runs migrations on `initialize()`.

- [ ] **Step 1: Failing tests — upsert, get by uri, IR snapshot round-trip, status update**

```python
# tests/unit/test_sqlite_cache.py
import pytest
from vectorist.cache.models import DocumentRecord, DocumentStatus
from vectorist.cache.sqlite_cache import SqliteCache
from vectorist.ir.models import IRDocument, ParagraphNode

@pytest.mark.asyncio
async def test_upsert_and_get_document(tmp_path):
    cache = SqliteCache(tmp_path / "cache.db")
    await cache.initialize()
    doc = DocumentRecord(
        id="d1",
        source_uri="a.md",
        content_hash="abc",
        raw_content="# A",
        status=DocumentStatus.PENDING,
    )
    await cache.upsert_document(doc)
    got = await cache.get_document_by_uri("a.md")
    assert got is not None
    assert got.content_hash == "abc"
    assert got.raw_content == "# A"
```

Add second test for `save_ir_snapshot` / `get_ir_snapshot` with a tiny `IRDocument`.

- [ ] **Step 2: Implement `SqliteCache` to pass**
- [ ] **Step 3: Commit**

```bash
git add src/vectorist/cache tests/unit/test_sqlite_cache.py
git commit -m "feat: add SQLite document and IR cache"
```

---

### Task 5: Markdown Adapter & Language Detection

**Files:**
- Create: `src/vectorist/adapters/base.py`
- Create: `src/vectorist/adapters/language.py`
- Create: `src/vectorist/adapters/markdown.py`
- Create: `tests/fixtures/pooling.md`
- Create: `tests/fixtures/untagged_code.md`
- Create: `tests/unit/test_language_detect.py`
- Create: `tests/unit/test_markdown_adapter.py`

**Interfaces:**
```python
class AdapterProtocol(Protocol):
    def parse(self, source_uri: str, content: str) -> IRDocument: ...

def detect_code_language(info_string: str | None, source_uri: str = "") -> tuple[str, str]:
    """Returns (language, detection_source)."""

class MarkdownAdapter:
    def parse(self, source_uri: str, content: str) -> IRDocument: ...
```

Detection order: fence info string → (optional extension from `source_uri`) → `("unknown", "unknown")`.

Fixture `pooling.md` must include: H1/H2, prose, fenced python block, markdown table, image with alt text.

- [ ] **Step 1: Write fixtures + failing adapter tests**

Assert:
- Root/section hierarchy reflects headings
- Fenced `python` → `CodeBlockNode.language == "python"`, `detection_source == "fence"`
- Untagged fence → `language == "unknown"`
- Table headers/rows populated
- Image → `MediaNode` with alt text, `enrichment_status == "PENDING"`
- `line_range` present on nodes

- [ ] **Step 2: Implement `detect_code_language` and `MarkdownAdapter` using markdown-it-py token stream; build hierarchical `SectionNode` stack**
- [ ] **Step 3: Tests PASS; Commit**

```bash
git add src/vectorist/adapters tests/fixtures tests/unit/test_language_detect.py tests/unit/test_markdown_adapter.py
git commit -m "feat: parse Markdown into Unified IR with code language detection"
```

---

### Task 6: Token Estimate, Core Chunker, Breadcrumbs

**Files:**
- Create: `src/vectorist/chunking/tokenizer.py`
- Create: `src/vectorist/chunking/chunker.py`
- Create: `src/vectorist/chunking/__init__.py`
- Create: `tests/unit/test_chunker.py`

**Interfaces:**
```python
def estimate_tokens(text: str) -> int:
    """Whitespace/word approximation used until a model tokenizer is wired."""

@dataclass
class Chunk:
    chunk_index: int
    content_text: str          # includes breadcrumb header
    breadcrumbs: list[str]
    node_ids: list[str]
    node_types: list[str]
    line_range: tuple[int, int]
    metadata: dict[str, Any]

class Chunker:
    def __init__(self, max_tokens: int = 512): ...
    def chunk_document(self, doc: IRDocument) -> list[Chunk]: ...
```

Breadcrumb header format (exact):
```text
[Context: { ' > '.join(breadcrumbs) }]
```

Rules to test:
- Never merge nodes across different H1/H2 section subtrees
- Prefer packing paragraph + following code under same section when under budget
- Every chunk `content_text` starts with `[Context: `

- [ ] **Step 1: Failing tests for section isolation + breadcrumb prefix using hand-built `IRDocument`**
- [ ] **Step 2: Implement packing walk; call stubs that raise `NotImplementedError` for oversized code/table — handle those in Tasks 7–8**
- [ ] **Step 3: For oversized plain paragraphs, split on blank lines / sentences as last resort**
- [ ] **Step 4: PASS; Commit**

```bash
git add src/vectorist/chunking tests/unit/test_chunker.py
git commit -m "feat: add boundary-aware chunker with breadcrumb injection"
```

---

### Task 7: Code Splitters (Tree-sitter + Delimiter Fallback)

**Files:**
- Create: `src/vectorist/chunking/code_split.py`
- Create: `tests/unit/test_code_split.py`
- Modify: `src/vectorist/chunking/chunker.py` to call code splitter for oversized `CodeBlockNode`

**Interfaces:**
```python
def split_code(
    source: str,
    language: str,
    max_tokens: int,
    estimate_tokens=estimate_tokens,
) -> list[str]:
    """Return ordered code slices each <= max_tokens when possible."""
```

- [ ] **Step 1: Failing tests**

```python
def test_fallback_splits_on_braces_and_semicolons():
    src = "function a(){ return 1; }\nfunction b(){ return 2; }\n"
    parts = split_code(src, language="unknown", max_tokens=8)
    assert len(parts) >= 2
    assert all("function" in p or "return" in p for p in parts)

def test_python_treesitter_does_not_split_mid_function_when_budget_allows_whole_functions():
    src = "def a():\n    return 1\n\ndef b():\n    return 2\n"
    parts = split_code(src, language="python", max_tokens=estimate_tokens("def a():\n    return 1\n") + 1)
    assert any("def a" in p for p in parts)
```

Fallback algorithm (required):
1. Prefer split points at lines ending with `;` or `}` or starting/ending with `{`
2. Prefer blank-line boundaries
3. Prefer indentation decreases (dedent)
4. Never split a line mid-token if a delimiter boundary exists nearby

Tree-sitter: parse with matching grammar; collect start byte of top-level functions/classes/methods; pack whole nodes until budget exceeded.

- [ ] **Step 2: Implement + wire into chunker**
- [ ] **Step 3: PASS; Commit**

```bash
git add src/vectorist/chunking/code_split.py src/vectorist/chunking/chunker.py tests/unit/test_code_split.py
git commit -m "feat: split oversized code with Tree-sitter and delimiter fallback"
```

---

### Task 8: Table Sticky-Header Splitting

**Files:**
- Create: `src/vectorist/chunking/table_split.py`
- Create: `tests/fixtures/wide_table.md`
- Create: `tests/unit/test_table_split.py`
- Modify: `src/vectorist/chunking/chunker.py`

**Interfaces:**
```python
def split_table(
    node: TableNode,
    max_tokens: int,
    rows_per_group: int | None = None,
) -> list[str]:
    """Each string includes caption + header row + row group."""
```

Exact continued header pattern:
```markdown
[Table: {caption} (Continued - Rows {start} to {end})]
| {h1} | {h2} | ... |
|---|---|---|
| row cells |
```

- [ ] **Step 1: Failing test — 40-row table with max_tokens forcing ≥2 slices; each slice contains header names**
- [ ] **Step 2: Implement + wire oversized `TableNode` in chunker**
- [ ] **Step 3: PASS; Commit**

```bash
git add src/vectorist/chunking/table_split.py src/vectorist/chunking/chunker.py tests/fixtures/wide_table.md tests/unit/test_table_split.py
git commit -m "feat: slice oversized tables with sticky headers"
```

---

### Task 9: Embedding Provider Protocol + Fake + FastEmbed

**Files:**
- Create: `src/vectorist/embeddings/protocol.py`
- Create: `src/vectorist/embeddings/fake.py`
- Create: `src/vectorist/embeddings/fastembed_provider.py`
- Create: `src/vectorist/embeddings/__init__.py`
- Create: `tests/unit/test_embeddings_fake.py`

**Interfaces:**
```python
class EmbeddingProviderProtocol(Protocol):
    @property
    def dimensions(self) -> int: ...
    @property
    def model_id(self) -> str: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...

class FakeEmbeddingProvider:
    """Deterministic hash-based unit vectors of length 384 for tests."""

class FastEmbedProvider:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"): ...
```

- [ ] **Step 1: Test fake provider — same text → same vector; dim == 384; query path works**
- [ ] **Step 2: Implement fake + protocol**
- [ ] **Step 3: Implement `FastEmbedProvider` (lazy-load model)**
- [ ] **Step 4: Optional smoke test marked `@pytest.mark.network` or skip if model download undesired in CI — default CI uses Fake**
- [ ] **Step 5: Commit**

```bash
git add src/vectorist/embeddings tests/unit/test_embeddings_fake.py
git commit -m "feat: add embedding provider protocol with FastEmbed and fake"
```

---

### Task 10: `sqlite-vec` Vector Store

**Files:**
- Create: `src/vectorist/storage/protocol.py`
- Create: `src/vectorist/storage/models.py`
- Create: `src/vectorist/storage/sqlite_vec_store.py`
- Create: `src/vectorist/storage/__init__.py`
- Create: `tests/unit/test_sqlite_vec_store.py`

**Interfaces:**
```python
@dataclass
class ChunkRecord:
    id: str
    document_id: str
    chunk_index: int
    chunk_hash: str
    content_text: str
    embedding: list[float]
    node_types: list[str]
    metadata: dict[str, Any]

@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    source_uri: str
    content_text: str
    score: float
    breadcrumbs: list[str]
    metadata: dict[str, Any]

class VectorStoreProtocol(Protocol):
    async def initialize(self) -> None: ...
    async def upsert_chunks(self, chunks: list[ChunkRecord]) -> None: ...
    async def delete_chunks_by_document_id(self, document_id: str) -> None: ...
    async def get_chunk_hashes(self, document_id: str) -> dict[str, str]: ...  # chunk_id -> hash for incremental embed
    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]: ...
```

`filters` Phase 1: support `source_uri_prefix: str` via metadata JSON field.

- [ ] **Step 1: Failing test — initialize, upsert 2 chunks with Fake embeddings, search finds nearest**
- [ ] **Step 2: Implement store: load `sqlite_vec`, create `vec0` virtual table `float[384]`, join to `chunk_metadata`**
- [ ] **Step 3: PASS; Commit**

```bash
git add src/vectorist/storage tests/unit/test_sqlite_vec_store.py
git commit -m "feat: add sqlite-vec vector store implementation"
```

---

### Task 11: `pgvector` Vector Store

**Files:**
- Create: `src/vectorist/storage/pgvector_store.py`
- Create: `tests/unit/test_pgvector_store.py` (skip if no `VECTORIST_PG_URL`)

**Interfaces:**
- Consumes: same `VectorStoreProtocol`
- Produces: `PgVectorStore(dsn: str)` applying `migrations/postgres`

- [ ] **Step 1: Implement `PgVectorStore` with asyncpg; cosine distance `1 - (embedding <=> $1)` or pgvector `<#>` / `<=>` as appropriate for cosine**
- [ ] **Step 2: Test skipped by default:**

```python
import os
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("VECTORIST_PG_URL"),
    reason="VECTORIST_PG_URL not set",
)
```

- [ ] **Step 3: Commit**

```bash
git add src/vectorist/storage/pgvector_store.py tests/unit/test_pgvector_store.py
git commit -m "feat: add pgvector vector store implementation"
```

---

### Task 12: Ingestion Pipeline Orchestration

**Files:**
- Create: `src/vectorist/pipeline/ingest.py`
- Create: `src/vectorist/config.py`
- Create: `tests/integration/test_ingest_pipeline.py`

**Interfaces:**
```python
@dataclass
class IngestResult:
    source_uri: str
    document_id: str | None
    status: str  # accepted | skipped | failed | indexed
    error: dict[str, str] | None = None

class IngestService:
    def __init__(
        self,
        cache: CacheProtocol,
        vector_store: VectorStoreProtocol,
        embedder: EmbeddingProviderProtocol,
        adapter: AdapterProtocol | None = None,
        chunker: Chunker | None = None,
    ): ...

    async def ingest_document(
        self,
        source_uri: str,
        content: str,
        content_type: str = "text/markdown",
    ) -> IngestResult: ...
```

Algorithm (exact):
1. Reject non-`text/markdown` with `PARSE_ERROR` in Phase 1.
2. `content_hash = sha256_text(content)`.
3. Lookup cache by `source_uri`; if hash matches and status is `INDEXED` → `skipped`.
4. Upsert document `PENDING` with new id (uuid4) or reuse id if uri exists.
5. Parse via adapter → `save_ir_snapshot` → status `PARSED`.
6. Chunk IR → compute `chunk_hash` per chunk (`sha256_text(content_text + document_id)`).
7. Load existing chunk hashes; embed only new/changed chunks; delete removed chunks for document; upsert.
8. Status `INDEXED` or `FAILED` with message.

- [ ] **Step 1: Integration test with temp SQLite cache + sqlite-vec + FakeEmbeddingProvider + pooling.md — ingest twice, second skipped; search-like store query returns chunk**
- [ ] **Step 2: Implement `IngestService` + `Settings` (paths for cache/vec DB)**
- [ ] **Step 3: PASS; Commit**

```bash
git add src/vectorist/pipeline src/vectorist/config.py tests/integration/test_ingest_pipeline.py
git commit -m "feat: orchestrate ingest pipeline with idempotent hashing"
```

---

### Task 13: FastAPI Routes (Ingest, Documents, Search, Health)

**Files:**
- Create: `src/vectorist/api/schemas.py`
- Create: `src/vectorist/api/app.py`
- Create: `src/vectorist/api/routes_ingest.py`
- Create: `src/vectorist/api/routes_documents.py`
- Create: `src/vectorist/api/routes_search.py`
- Create: `src/vectorist/api/routes_health.py`
- Create: `tests/integration/test_api_e2e.py`

**Interfaces:**
- `create_app() -> FastAPI` wires Settings → SqliteCache + SqliteVecStore (or Pg if `VECTORIST_VECTOR_BACKEND=pgvector`) + embedder (Fake if `VECTORIST_EMBEDDER=fake`, else FastEmbed)
- Endpoints per spec:
  - `POST /v1/ingest/stream` — body is NDJSON lines; response JSON list of `IngestResult`
  - `POST /v1/ingest` — JSON `{source_uri, content, content_type?}`
  - `GET /v1/documents/{document_id}`
  - `DELETE /v1/documents/{document_id}` — cache row + `delete_chunks_by_document_id`
  - `POST /v1/search` — `{query, limit?, filters?}`
  - `GET /v1/health`

- [ ] **Step 1: Write e2e test with `httpx.AsyncClient` + `ASGITransport`, `VECTORIST_EMBEDDER=fake`, temp DB paths via env**

```python
async def test_ingest_stream_and_search(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTORIST_CACHE_PATH", str(tmp_path / "cache.db"))
    monkeypatch.setenv("VECTORIST_VEC_PATH", str(tmp_path / "vec.db"))
    monkeypatch.setenv("VECTORIST_EMBEDDER", "fake")
    from vectorist.api.app import create_app
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        body = (
            '{"source_uri":"repo/pooling.md","content_type":"text/markdown","content":'
            + json.dumps((fixtures_dir / "pooling.md").read_text())
            + "}\n"
        )
        r = await client.post("/v1/ingest/stream", content=body, headers={"Content-Type": "application/x-ndjson"})
        assert r.status_code == 200
        assert r.json()[0]["status"] in {"indexed", "accepted"}
        s = await client.post("/v1/search", json={"query": "connection pool", "limit": 5})
        assert s.status_code == 200
        assert len(s.json()["results"]) >= 1
```

- [ ] **Step 2: Implement routes + app factory**
- [ ] **Step 3: PASS; Commit**

```bash
git add src/vectorist/api tests/integration/test_api_e2e.py
git commit -m "feat: expose FastAPI ingest, document, search, and health APIs"
```

---

### Task 14: Wiki, Health Semantics, Golden Fixture Polish

Write this task only after Tasks 1–13 have landed. Product documentation describes the code that exists, not the plan.

**Files:**
- Create: `wiki/server.md` (usage: how to run, env vars, ingest and search)
- Modify: `roadmap.md` only if Phase 1 scope changed while implementing
- Modify: tests/fixtures as needed for reliable fake-embed retrieval (ensure query terms appear in chunk text)
- Create: `tests/integration/test_idempotency_api.py` if not already covered
- Do not expand `README.md` beyond the existing pointers to `roadmap.md` and `wiki/`

**Requirements:**
- `wiki/server.md` documents the commands and settings the code actually uses: install, `uv run pytest`, `uv run vectorist` (or `python -m vectorist`), and env vars (`VECTORIST_CACHE_PATH`, `VECTORIST_VEC_PATH`, `VECTORIST_VECTOR_BACKEND`, `VECTORIST_DATABASE_URL`, `VECTORIST_EMBEDDER`)
- `/v1/health` returns `{status: ok|degraded, cache: ok, vector_store: ok, embedder: ok}`
- Second identical ingest via API returns `skipped`
- Do not add `docs/superpowers/` to this commit

- [ ] **Step 1: Add/adjust tests for skip + health**
- [ ] **Step 2: Write `wiki/server.md` from the implemented server**
- [ ] **Step 3: Run full suite**

```bash
uv run pytest -v
```
Expected: all non-skipped tests PASS

- [ ] **Step 4: Commit**

```bash
git add wiki/server.md roadmap.md tests
git commit -m "docs: document Phase 1 server usage and verify golden ingest path"
```

---

## Self-Review (Plan vs Spec)

| Spec requirement | Task |
| :--- | :--- |
| Hierarchical Unified IR (Section/Paragraph/Code/Table/Media) | Task 2, 5 |
| Markdown adapter | Task 5 |
| Language detection fence → unknown | Task 5 |
| Tree-sitter + delimiter fallback (`;{}` blank indent) | Task 7 |
| Sticky table headers | Task 8 |
| Breadcrumb injection | Task 6 |
| SHA-256 skip + chunk-level hash skip | Task 12 |
| SQLite IR cache with raw_content | Task 3, 4 |
| Cache ≠ vector store protocols | Task 4, 10 |
| sqlite-vec default | Task 10 |
| pgvector adapter | Task 11 |
| FastEmbed 384-dim default + fake for tests | Task 9 |
| Liquibase SQL + embedded runner | Task 3 |
| HTTP ingest stream/single, documents, search, health | Task 13 |
| E2E SQLite path without Docker | Task 13, 14 |
| Product docs in `wiki/` after code; roadmap stays in `roadmap.md` | Task 14 |
| `docs/superpowers/` not merged to `main` | Global Constraints; delete on branch before PR |
| MediaNode stub PENDING | Task 5 |
| No CLI/crawler/HTML/PDF/auth | Honored (omitted) |

**Placeholder scan:** No TBD/TODO steps; interfaces named consistently (`CacheProtocol`, `VectorStoreProtocol`, `IngestService.ingest_document`, `Chunk`, `ChunkRecord`).

**Type consistency notes for implementers:**
- API `accepted` may be returned only if async worker split is introduced; Phase 1 may complete indexing inside the request and return `indexed` (or `skipped` / `failed`). Prefer **`indexed`** for successful sync completion to match integration tests.
- `IRDocument.root_id` + `nodes` map is the canonical tree shape (not a nested-only tree).
- Chunk hashes live on `ChunkRecord.chunk_hash`; vector store exposes `get_chunk_hashes` for incremental embed.
