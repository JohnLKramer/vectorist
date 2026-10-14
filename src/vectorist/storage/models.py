from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
