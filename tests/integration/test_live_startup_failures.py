import pytest

import live

pytestmark = pytest.mark.live

EXIT_TIMEOUT = 15.0


def _env(tmp_path, **overrides):
    env = live.base_env()
    env["VECTORIST_CACHE_PATH"] = str(tmp_path / "cache.db")
    env["VECTORIST_VEC_PATH"] = str(tmp_path / "vec.db")
    env["VECTORIST_EMBEDDER"] = "fake"
    env.update(overrides)
    return env


@pytest.mark.parametrize(
    ("overrides", "removed", "setting"),
    [
        ({"VECTORIST_VECTOR_BACKEND": "nope"}, (), "vector backend"),
        ({"VECTORIST_EMBEDDER": "nope"}, (), "embedder"),
        ({"VECTORIST_ENRICHER": "bogus"}, (), "enricher"),
        ({"VECTORIST_VECTOR_BACKEND": "pgvector"}, (), "VECTORIST_DATABASE_URL"),
        ({}, ("VECTORIST_CACHE_PATH",), "cache_path"),
    ],
    ids=["unknown-backend", "unknown-embedder", "unknown-enricher", "pgvector-no-url", "missing-cache-path"],
)
def test_startup_fails_loudly(tmp_path, overrides, removed, setting):
    env = _env(tmp_path, **overrides)
    for key in removed:
        del env[key]
    process = live.start_server(env, live.free_port())
    try:
        code = live.wait_exit(process, EXIT_TIMEOUT)
        stderr = process.stderr()
    finally:
        process.close()
    assert code != 0
    assert setting in stderr
