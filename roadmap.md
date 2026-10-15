# Roadmap

High-level product sequence. Feature behavior and usage live in `wiki/` after
the code exists. Working specs and implementation plans stay off `main`.

## Phase 1 — Server

Markdown ingest over HTTP, hierarchical intermediate representation, boundary-aware
chunking, local embeddings, SQLite document cache, pluggable vector storage
(SQLite by default, PostgreSQL supported), and a search API.

## Phase 2 — Ingestion clients

Command-line sync for repositories, and a crawler for internal documentation sites.

## Phase 3 — Heavy content

Background enrichment for images and diagrams. Text indexing stays available
while that work runs.

## Phase 4 — Organization connectors and agent query

Connectors for existing knowledge systems, and a query surface aimed at coding
agents (including hybrid keyword and vector retrieval).
