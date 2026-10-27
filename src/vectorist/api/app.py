from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from vectorist.api.routes_assets import router as assets_router
from vectorist.api.routes_documents import router as documents_router
from vectorist.api.routes_enrichment import router as enrichment_router
from vectorist.api.routes_health import router as health_router
from vectorist.api.routes_ingest import router as ingest_router
from vectorist.api.routes_search import router as search_router
from vectorist.cache.sqlite_cache import SqliteCache
from vectorist.config import Settings
from vectorist.embeddings.fake import FakeEmbeddingProvider
from vectorist.embeddings.protocol import EmbeddingProviderProtocol
from vectorist.enrichment.factory import build_enricher
from vectorist.enrichment.worker import EnrichmentWorker
from vectorist.pipeline.ingest import IngestService
from vectorist.storage.pgvector_store import PgVectorStore
from vectorist.storage.protocol import VectorStoreProtocol
from vectorist.storage.sqlite_vec_store import SqliteVecStore


def _build_store(settings: Settings) -> VectorStoreProtocol:
    if settings.vector_backend == "sqlite":
        return SqliteVecStore(settings.vec_path)
    if settings.vector_backend == "pgvector":
        if not settings.database_url:
            raise RuntimeError(
                "VECTORIST_DATABASE_URL is required when "
                "VECTORIST_VECTOR_BACKEND=pgvector"
            )
        return PgVectorStore(settings.database_url)
    raise RuntimeError(f"unsupported vector backend: {settings.vector_backend}")


def _build_embedder(settings: Settings) -> EmbeddingProviderProtocol:
    if settings.embedder == "fake":
        return FakeEmbeddingProvider()
    if settings.embedder == "fastembed":
        from vectorist.embeddings.fastembed_provider import FastEmbedProvider

        return FastEmbedProvider()
    raise RuntimeError(f"unsupported embedder: {settings.embedder}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    embedder = _build_embedder(settings)
    enricher = build_enricher(settings)
    cache = SqliteCache(settings.cache_path)
    store = _build_store(settings)
    worker: EnrichmentWorker | None = None
    try:
        await cache.initialize()
        await store.initialize()
        app.state.cache = cache
        app.state.store = store
        app.state.embedder = embedder
        app.state.enricher = enricher
        app.state.max_asset_bytes = settings.max_asset_bytes
        ingest = IngestService(
            cache,
            store,
            embedder,
            enricher_id=enricher.provider_id if enricher else None,
        )
        app.state.ingest = ingest
        if enricher is not None:
            worker = EnrichmentWorker(
                cache,
                ingest,
                enricher,
                settings.enrich_poll_seconds,
                settings.enrich_max_attempts,
            )
            await worker.recover()
            worker.start()
        yield
    finally:
        if worker is not None:
            await worker.stop()
        close = getattr(store, "close", None)
        if close is not None:
            await close()


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(ingest_router)
    app.include_router(documents_router)
    app.include_router(assets_router)
    app.include_router(enrichment_router)
    app.include_router(search_router)
    app.include_router(health_router)
    return app
