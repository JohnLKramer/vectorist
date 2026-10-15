"""Liquibase-formatted SQL migration runner (zero JVM).

Scans ``*.sql`` files in ``migrations_dir``, parses ``--changeset author:id`` markers,
and applies pending SQL against a sqlite3 connection. Tracks applied ids in
``_schema_migrations``.

The SQL ships inside the package under ``vectorist/migrations/sql/<dialect>``;
``packaged_migrations_dir`` resolves it from wherever the package is installed.

Note: sqlite-vec ``CREATE VIRTUAL TABLE ... USING vec0`` is not run here; that
happens in ``SqliteVecStore.initialize()`` after the extension is loaded.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

_CHANGESET_RE = re.compile(r"^--changeset\s+(\S+:\S+)\s*$", re.MULTILINE)

_SCHEMA_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS _schema_migrations (
    id TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
)
"""


def packaged_migrations_dir(dialect: str) -> Path:
    directory = Path(str(resources.files("vectorist.migrations") / "sql" / dialect))
    if not directory.is_dir():
        raise FileNotFoundError(
            f"packaged migrations not found for {dialect}: {directory}"
        )
    return directory


def migration_files(migrations_dir: Path) -> list[Path]:
    files = sorted(migrations_dir.glob("*.sql"))
    if not files:
        raise FileNotFoundError(f"no migration files in {migrations_dir}")
    return files


def _ensure_schema_migrations(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_MIGRATIONS_DDL)
    conn.commit()


def _applied_ids(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT id FROM _schema_migrations").fetchall()
    return {row[0] for row in rows}


def _parse_changesets(sql_text: str) -> list[tuple[str, str]]:
    """Return (changeset_id, sql_body) pairs in file order."""
    matches = list(_CHANGESET_RE.finditer(sql_text))
    if not matches:
        return []

    result: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        changeset_id = match.group(1)
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(sql_text)
        body = sql_text[body_start:body_end].strip()
        if body:
            result.append((changeset_id, body))
        else:
            result.append((changeset_id, ""))
    return result


def apply_migrations(conn: sqlite3.Connection, migrations_dir: Path) -> list[str]:
    """Apply pending Liquibase-formatted SQL migrations; return newly applied changeset ids."""
    files = migration_files(migrations_dir)
    _ensure_schema_migrations(conn)
    already = _applied_ids(conn)

    applied: list[str] = []
    now = datetime.now(timezone.utc).isoformat()

    for path in files:
        sql_text = path.read_text(encoding="utf-8")
        for changeset_id, body in _parse_changesets(sql_text):
            if changeset_id in already:
                continue
            if body:
                conn.executescript(body)
            conn.execute(
                "INSERT INTO _schema_migrations (id, applied_at) VALUES (?, ?)",
                (changeset_id, now),
            )
            conn.commit()
            already.add(changeset_id)
            applied.append(changeset_id)

    return applied
