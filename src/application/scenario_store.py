"""Per-user scenario storage (T67a): a ``ScenarioStore`` protocol and a local SQLite backend.

No streamlit here. Every query is scoped by ``user_id`` and parameterised, so one user can never read, rename or
delete another user's scenario. Contexts are stored as ``RunContext.model_dump_json()`` and validated on load
through ``scenario_io`` (unknown fields are ignored, an invalid or corrupted document raises ``ScenarioError``).

The schema version lives in ``PRAGMA user_version``; ``MIGRATIONS`` maps a version to the function that upgrades
the database *to* that version, so a later schema change is one more entry.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from src.application.run_context import RunContext
from src.application.scenario_io import ScenarioError, context_from_json

DEFAULT_USER_ID = "local"
DB_FILENAME = "scenarios.db"
MAX_NAME_LENGTH = 80
RESERVED_PREFIX = "__"  # names kept for internal rows (see LAST_CONTEXT_NAME)
LAST_CONTEXT_NAME = "__last_context__"


class ScenarioStoreError(ValueError):
    """A refused operation (bad name, duplicate name, unknown scenario). The message is user-presentable."""


@dataclass(frozen=True)
class UserInfo:
    id: str
    display_name: str
    created_at: str


@dataclass(frozen=True)
class ScenarioSummary:
    id: int
    name: str
    created_at: str
    updated_at: str
    top_strategy_keys: list[str]


class ScenarioStore(Protocol):
    def ensure_user(self, user_id: str, display_name: str | None = None) -> UserInfo: ...
    def list_users(self) -> list[UserInfo]: ...
    def list_scenarios(self, user_id: str) -> list[ScenarioSummary]: ...
    def save(self, user_id: str, name: str, ctx: RunContext, strategy_keys: list[str] | None = None) -> int: ...
    def load(self, user_id: str, scenario_id: int) -> tuple[ScenarioSummary, RunContext]: ...
    def delete(self, user_id: str, scenario_id: int) -> bool: ...
    def rename(self, user_id: str, scenario_id: int, new_name: str) -> None: ...
    def save_last_context(self, user_id: str, ctx: RunContext) -> None: ...
    def load_last_context(self, user_id: str) -> RunContext | None: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_name(name: str, *, internal: bool = False) -> str:
    name = (name or "").strip()
    if not name:
        raise ScenarioStoreError("Le nom du scénario est vide.")
    if len(name) > MAX_NAME_LENGTH:
        raise ScenarioStoreError(f"Nom trop long ({MAX_NAME_LENGTH} caractères maximum).")
    if name.startswith(RESERVED_PREFIX) and not internal:
        raise ScenarioStoreError(f"Un nom ne peut pas commencer par « {RESERVED_PREFIX} ».")
    return name


# --------------------------------------------------------------------------- #
# Schema and migrations
# --------------------------------------------------------------------------- #
def _migrate_to_1(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE users (
            id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE scenarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            run_context_json TEXT NOT NULL,
            top_strategy_keys_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (user_id, name)
        );
        CREATE INDEX idx_scenarios_user ON scenarios(user_id);
        """
    )


MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {1: _migrate_to_1}
SCHEMA_VERSION = max(MIGRATIONS)


def _migrate(conn: sqlite3.Connection) -> None:
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current > SCHEMA_VERSION:
        raise ScenarioStoreError(f"Base de scénarios plus récente que l'application (version {current}).")
    for version in range(current + 1, SCHEMA_VERSION + 1):
        MIGRATIONS[version](conn)
        conn.execute(f"PRAGMA user_version = {int(version)}")  # PRAGMA takes no bound parameter; int() is safe
    conn.commit()


# --------------------------------------------------------------------------- #
# SQLite backend
# --------------------------------------------------------------------------- #
class SQLiteScenarioStore:
    """One short-lived connection per call (safe across Streamlit's threads)."""

    def __init__(self, data_dir: str | os.PathLike[str]) -> None:
        self.path = Path(data_dir) / DB_FILENAME
        self._prepare_file()
        with closing(self._connect()) as conn:
            _migrate(conn)

    def _prepare_file(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass  # not supported on this OS / filesystem
        if not self.path.exists():
            try:
                os.close(os.open(self.path, os.O_CREAT | os.O_WRONLY, 0o600))
            except OSError:
                pass  # sqlite creates it below
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # --- users ---
    def ensure_user(self, user_id: str, display_name: str | None = None) -> UserInfo:
        user_id = (user_id or "").strip()
        if not user_id:
            raise ScenarioStoreError("Identifiant utilisateur vide.")
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users (id, display_name, created_at) VALUES (?, ?, ?)",
                (user_id, (display_name or user_id).strip() or user_id, _now()),
            )
            conn.commit()
            row = conn.execute("SELECT id, display_name, created_at FROM users WHERE id = ?", (user_id,)).fetchone()
        return UserInfo(*row)

    def list_users(self) -> list[UserInfo]:
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT id, display_name, created_at FROM users ORDER BY created_at, id").fetchall()
        return [UserInfo(*r) for r in rows]

    # --- scenarios ---
    @staticmethod
    def _keys(text: str) -> list[str]:
        try:
            data = json.loads(text)
        except ValueError:
            return []
        return [str(k) for k in data] if isinstance(data, list) else []

    def list_scenarios(self, user_id: str) -> list[ScenarioSummary]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, name, created_at, updated_at, top_strategy_keys_json FROM scenarios "
                "WHERE user_id = ? AND substr(name, 1, ?) != ? ORDER BY updated_at DESC, id DESC",
                (user_id, len(RESERVED_PREFIX), RESERVED_PREFIX),
            ).fetchall()
        return [ScenarioSummary(r[0], r[1], r[2], r[3], self._keys(r[4])) for r in rows]

    def _upsert(self, user_id: str, name: str, ctx: RunContext, strategy_keys: list[str] | None) -> int:
        self.ensure_user(user_id)
        now = _now()
        ctx_json = ctx.model_dump_json()
        keys_json = json.dumps([str(k) for k in (strategy_keys or [])], ensure_ascii=False)
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO scenarios (user_id, name, run_context_json, top_strategy_keys_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (user_id, name) DO UPDATE SET run_context_json = excluded.run_context_json, "
                "top_strategy_keys_json = excluded.top_strategy_keys_json, updated_at = excluded.updated_at",
                (user_id, name, ctx_json, keys_json, now, now),
            )
            conn.commit()
            row = conn.execute("SELECT id FROM scenarios WHERE user_id = ? AND name = ?", (user_id, name)).fetchone()
        return int(row[0])

    def save(self, user_id: str, name: str, ctx: RunContext, strategy_keys: list[str] | None = None) -> int:
        """Create the scenario, or overwrite the one with the same name for this user. Returns its id."""
        return self._upsert(user_id, _clean_name(name), ctx, strategy_keys)

    def load(self, user_id: str, scenario_id: int) -> tuple[ScenarioSummary, RunContext]:
        """Raises ``ScenarioStoreError`` when unknown (or not this user's) and ``ScenarioError`` when corrupted."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT id, name, created_at, updated_at, top_strategy_keys_json, run_context_json FROM scenarios "
                "WHERE user_id = ? AND id = ?",
                (user_id, scenario_id),
            ).fetchone()
        if row is None:
            raise ScenarioStoreError("Scénario introuvable.")
        summary = ScenarioSummary(row[0], row[1], row[2], row[3], self._keys(row[4]))
        return summary, context_from_json(row[5])

    def delete(self, user_id: str, scenario_id: int) -> bool:
        with closing(self._connect()) as conn:
            cur = conn.execute("DELETE FROM scenarios WHERE user_id = ? AND id = ?", (user_id, scenario_id))
            conn.commit()
        return cur.rowcount > 0

    def rename(self, user_id: str, scenario_id: int, new_name: str) -> None:
        new_name = _clean_name(new_name)
        with closing(self._connect()) as conn:
            try:
                cur = conn.execute(
                    "UPDATE scenarios SET name = ?, updated_at = ? WHERE user_id = ? AND id = ?",
                    (new_name, _now(), user_id, scenario_id),
                )
            except sqlite3.IntegrityError as exc:
                raise ScenarioStoreError(f"Un scénario nommé « {new_name} » existe déjà.") from exc
            conn.commit()
        if cur.rowcount == 0:
            raise ScenarioStoreError("Scénario introuvable.")

    # --- last used context ---
    def save_last_context(self, user_id: str, ctx: RunContext) -> None:
        self._upsert(user_id, _clean_name(LAST_CONTEXT_NAME, internal=True), ctx, None)

    def load_last_context(self, user_id: str) -> RunContext | None:
        """The last saved context, or None when there is none or it can no longer be read."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT run_context_json FROM scenarios WHERE user_id = ? AND name = ?",
                (user_id, LAST_CONTEXT_NAME),
            ).fetchone()
        if row is None:
            return None
        try:
            return context_from_json(row[0])
        except ScenarioError:
            return None


def default_store() -> SQLiteScenarioStore:
    """The store at ``APP_DATA_DIR`` (looked up at call time so tests can redirect ``defaults.DATA_DIR``)."""
    from src.core import defaults

    return SQLiteScenarioStore(defaults.DATA_DIR)
