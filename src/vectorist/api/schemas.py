from typing import Any, Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source_uri: str
    content: str
    content_type: str = "text/markdown"


class IngestResponse(BaseModel):
    source_uri: str
    document_id: str | None
    status: str
    error: dict[str, str] | None = None


class DocumentResponse(BaseModel):
    id: str
    source_uri: str
    content_hash: str
    status: str
    error_message: str | None
    raw_content: str


class DocumentSummaryResponse(BaseModel):
    id: str
    source_uri: str
    content_hash: str
    status: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummaryResponse]
    next_cursor: str | None


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(10, ge=1, le=4096)
    filters: dict[str, Any] | None = None


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    source_uri: str
    content_text: str
    score: float
    breadcrumbs: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[SearchHit]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    cache: str
    vector_store: str
    embedder: str
