from vectorist.config import Settings


def test_settings_reads_cache_and_vec_paths(monkeypatch, tmp_path):
    cache_path = tmp_path / "cache.db"
    vec_path = tmp_path / "vec.db"
    monkeypatch.setenv("VECTORIST_CACHE_PATH", str(cache_path))
    monkeypatch.setenv("VECTORIST_VEC_PATH", str(vec_path))

    settings = Settings()

    assert settings.cache_path == cache_path
    assert settings.vec_path == vec_path
