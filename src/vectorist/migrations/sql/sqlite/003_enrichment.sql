--liquibase formatted sql
--changeset vectorist:003-enrichment
CREATE TABLE IF NOT EXISTS assets (
    hash TEXT PRIMARY KEY,
    content_type TEXT NOT NULL,
    size INTEGER NOT NULL,
    data BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS document_assets (
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ref TEXT NOT NULL,
    asset_hash TEXT NOT NULL,
    PRIMARY KEY (document_id, ref)
);

CREATE INDEX IF NOT EXISTS idx_document_assets_asset_hash
    ON document_assets(asset_hash);

CREATE TABLE IF NOT EXISTS enrichments (
    asset_hash TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (asset_hash, provider_id)
);

CREATE TABLE IF NOT EXISTS enrichment_jobs (
    asset_hash TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
