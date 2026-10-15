# tests/unit/test_migrations_runner.py
import sqlite3
from pathlib import Path

import pytest

import vectorist
from vectorist.migrations.runner import apply_migrations, packaged_migrations_dir

PACKAGE_ROOT = Path(vectorist.__file__).resolve().parent


def test_apply_sqlite_cache_migrations(tmp_path: Path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    applied = apply_migrations(conn, packaged_migrations_dir("sqlite"))
    assert "vectorist:001-cache-init" in applied
    assert "vectorist:002-vec-init" in applied
    # idempotent
    applied2 = apply_migrations(conn, packaged_migrations_dir("sqlite"))
    assert applied2 == []
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    names = {r[0] for r in rows}
    assert "documents" in names
    assert "ir_snapshots" in names
    assert "chunk_metadata" in names
    assert "_schema_migrations" in names


@pytest.mark.parametrize(
    ("dialect", "first_file"),
    [("sqlite", "001_cache_init.sql"), ("postgres", "001_pgvector_init.sql")],
)
def test_migrations_resolve_inside_the_installed_package(dialect, first_file):
    directory = packaged_migrations_dir(dialect)
    assert directory.is_relative_to(PACKAGE_ROOT)
    assert (directory / first_file).is_file()


def test_unknown_migrations_dialect_fails():
    with pytest.raises(FileNotFoundError, match="oracle"):
        packaged_migrations_dir("oracle")


def test_apply_migrations_fails_when_directory_has_no_sql(tmp_path: Path):
    conn = sqlite3.connect(tmp_path / "t.db")
    with pytest.raises(FileNotFoundError, match="no migration files"):
        apply_migrations(conn, tmp_path / "empty")
