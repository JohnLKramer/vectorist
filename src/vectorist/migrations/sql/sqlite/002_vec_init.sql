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
