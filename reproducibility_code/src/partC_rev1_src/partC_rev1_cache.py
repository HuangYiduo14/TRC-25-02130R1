"""SQLite cache helpers for partC_rev1 preprocessing outputs."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pandas as pd


def partC_rev1_connect_sqlite(path: str | Path) -> sqlite3.Connection:
    sqlite_path = Path(path).expanduser()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(sqlite_path, timeout=300.0)
    conn.execute("PRAGMA busy_timeout=300000")
    return conn


def _partC_rev1_scalarize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, set)) else False:
        return None
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (list, tuple, dict, set)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def partC_rev1_make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a shallow copy with SQLite/GPKG-safe, case-insensitively unique column names."""

    out = df.copy()
    used: set[str] = set()
    new_columns: list[str] = []
    changed = False
    for raw_column in out.columns:
        base = str(raw_column) if str(raw_column) else "column"
        candidate = base
        suffix = 1
        while candidate.lower() in used:
            candidate = f"{base}_partC_rev1_dup{suffix}"
            suffix += 1
        used.add(candidate.lower())
        new_columns.append(candidate)
        changed = changed or candidate != raw_column
    if changed:
        out.columns = new_columns
    return out


def _partC_rev1_sqlite_ready(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(partC_rev1_make_unique_columns(df))
    if "geometry" in out.columns:
        geom = out["geometry"]
        out["partC_rev1_geom_wkb"] = geom.map(
            lambda g: getattr(g, "wkb_hex", None) if g is not None and not pd.isna(g) else None
        )
        out = out.drop(columns=["geometry"])
    for column in out.columns:
        if str(out[column].dtype) == "geometry":
            out[column] = out[column].map(lambda g: getattr(g, "wkb_hex", None) if g is not None else None)
        elif out[column].dtype == "object":
            out[column] = out[column].map(_partC_rev1_scalarize)
    return out


def partC_rev1_write_table_sqlite(
    conn: sqlite3.Connection,
    table_name: str,
    df: pd.DataFrame,
    if_exists: str = "replace",
) -> None:
    if not table_name.startswith("partC_rev1_"):
        raise ValueError(f"SQLite table name must start with partC_rev1_: {table_name}")
    ready = _partC_rev1_sqlite_ready(df)
    for attempt in range(6):
        try:
            ready.to_sql(table_name, conn, if_exists=if_exists, index=False)
            conn.commit()
            return
        except Exception as exc:
            if "database is locked" not in str(exc).lower() or attempt == 5:
                raise
            time.sleep(min(60.0, 5.0 * (2**attempt)))


def partC_rev1_read_table_sqlite(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    return pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)


def partC_rev1_list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    return [str(row[0]) for row in rows]


def partC_rev1_write_metadata(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS partC_rev1_cache_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO partC_rev1_cache_metadata (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (str(key), json.dumps(value, sort_keys=True, default=str)),
    )
    conn.commit()
