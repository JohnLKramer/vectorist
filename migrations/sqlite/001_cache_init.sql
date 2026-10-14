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
