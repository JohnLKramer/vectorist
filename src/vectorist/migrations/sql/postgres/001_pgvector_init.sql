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
