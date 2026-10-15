from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from vectorist.api.routes_documents import router as documents_router
from vectorist.api.routes_health import router as health_router
from vectorist.api.routes_ingest import router as ingest_router
from vectorist.api.routes_search import router as search_router
from vectorist.cache.sqlite_cache import SqliteCache
from vectorist.config import Settings
from vectorist.embeddings.fake import FakeEmbeddingProvider
from vectorist.embeddings.protocol import EmbeddingProviderProtocol
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
    from vectorist.embeddings.fastembed_provider import FastEmbedProvider

    return FastEmbedProvider()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    cache = SqliteCache(settings.cache_path)
    store = _build_store(settings)
    try:
        await cache.initialize()
        await store.initialize()
        embedder = _build_embedder(settings)
        app.state.cache = cache
        app.state.store = store
        app.state.embedder = embedder
        app.state.ingest = IngestService(cache, store, embedder)
        yield
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            await close()


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(ingest_router)
    app.include_router(documents_router)
    app.include_router(search_router)
    app.include_router(health_router)
    return app
