from vectorist.config import Settings


def test_settings_reads_cache_and_vec_paths(monkeypatch, tmp_path):
    cache_path = tmp_path / "cache.db"
    vec_path = tmp_path / "vec.db"
    monkeypatch.setenv("VECTORIST_CACHE_PATH", str(cache_path))
    monkeypatch.setenv("VECTORIST_VEC_PATH", str(vec_path))

    settings = Settings()

    assert settings.cache_path == cache_path
    assert settings.vec_path == vec_path
    assert settings.vector_backend == "sqlite"
    assert settings.database_url is None
    assert settings.embedder == "fastembed"


def test_settings_reads_backend_embedder_and_database_url(monkeypatch, tmp_path):
    monkeypatch.setenv("VECTORIST_CACHE_PATH", str(tmp_path / "cache.db"))
    monkeypatch.setenv("VECTORIST_VEC_PATH", str(tmp_path / "vec.db"))
    monkeypatch.setenv("VECTORIST_VECTOR_BACKEND", "pgvector")
    monkeypatch.setenv("VECTORIST_DATABASE_URL", "postgresql://example/vectorist")
    monkeypatch.setenv("VECTORIST_EMBEDDER", "fake")

    settings = Settings()

    assert settings.vector_backend == "pgvector"
    assert settings.database_url == "postgresql://example/vectorist"
    assert settings.embedder == "fake"
