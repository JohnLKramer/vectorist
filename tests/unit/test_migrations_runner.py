# tests/unit/test_migrations_runner.py
import sqlite3
from pathlib import Path
from vectorist.migrations.runner import apply_migrations

ROOT = Path(__file__).resolve().parents[2]

def test_apply_sqlite_cache_migrations(tmp_path: Path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    applied = apply_migrations(conn, ROOT / "migrations" / "sqlite")
    assert "vectorist:001-cache-init" in applied
    assert "vectorist:002-vec-init" in applied
    # idempotent
    applied2 = apply_migrations(conn, ROOT / "migrations" / "sqlite")
    assert applied2 == []
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    names = {r[0] for r in rows}
    assert "documents" in names
    assert "ir_snapshots" in names
    assert "chunk_metadata" in names
    assert "_schema_migrations" in names
