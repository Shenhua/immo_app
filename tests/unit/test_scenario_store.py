"""T67a: per-user scenario store (SQLite)."""

from __future__ import annotations

import json
import os
import sqlite3
import stat

import pytest

from src.application.run_context import RunContext
from src.application.scenario_io import ScenarioError
from src.application.scenario_store import (
    SCHEMA_VERSION,
    ScenarioStoreError,
    SQLiteScenarioStore,
    default_store,
)


@pytest.fixture
def store(tmp_path) -> SQLiteScenarioStore:
    return SQLiteScenarioStore(tmp_path / "db")


def _ctx(**kw) -> RunContext:
    base = {"apport": 123_456, "cf_cible": -150.0, "horizon_ans": 20, "tmi_pct": 30.0, "sci_is": True}
    return RunContext(**{**base, **kw})


def test_crud(store):
    sid = store.save("alice", "Mon plan", _ctx(), ["a", "b"])
    [item] = store.list_scenarios("alice")
    assert (item.id, item.name, item.top_strategy_keys) == (sid, "Mon plan", ["a", "b"])
    store.rename("alice", sid, "Plan B")
    assert store.list_scenarios("alice")[0].name == "Plan B"
    assert store.delete("alice", sid) is True
    assert store.list_scenarios("alice") == []
    assert store.delete("alice", sid) is False


def test_save_same_name_overwrites(store):
    a = store.save("alice", "x", _ctx(apport=1000))
    b = store.save("alice", "x", _ctx(apport=2000))
    assert a == b
    assert len(store.list_scenarios("alice")) == 1
    assert store.load("alice", a)[1].apport == 2000


def test_rename_to_existing_name_is_refused(store):
    store.save("alice", "one", _ctx())
    two = store.save("alice", "two", _ctx())
    with pytest.raises(ScenarioStoreError):
        store.rename("alice", two, "one")


@pytest.mark.parametrize("bad", ["", "   ", "x" * 200, "__last_context__"])
def test_bad_names_refused(store, bad):
    with pytest.raises(ScenarioStoreError):
        store.save("alice", bad, _ctx())


def test_round_trip_non_default_context(store):
    ctx = _ctx(max_properties=2, apport=87_000, vacance_pct=7.5)
    sid = store.save("alice", "rt", ctx)
    _, loaded = store.load("alice", sid)
    assert loaded == ctx
    assert loaded != RunContext()


def test_user_isolation(store):
    sid = store.save("alice", "secret", _ctx())
    store.save("bob", "mine", _ctx())
    assert [s.name for s in store.list_scenarios("bob")] == ["mine"]
    with pytest.raises(ScenarioStoreError):
        store.load("bob", sid)
    assert store.delete("bob", sid) is False
    with pytest.raises(ScenarioStoreError):
        store.rename("bob", sid, "stolen")
    assert store.list_scenarios("alice")[0].name == "secret"  # untouched
    assert store.load_last_context("bob") is None


def test_sql_injection_in_user_id_and_name_is_inert(store):
    store.save("alice", "ok", _ctx())
    assert store.list_scenarios("x' OR '1'='1") == []
    store.save("alice", "'); DROP TABLE scenarios; --", _ctx())
    assert len(store.list_scenarios("alice")) == 2


def test_unknown_fields_ignored_and_corrupted_json_handled(store, tmp_path):
    sid = store.save("alice", "old", _ctx())
    bad = store.save("alice", "bad", _ctx())
    garbage = store.save("alice", "garbage", _ctx())
    doc = json.loads(_ctx().model_dump_json())
    doc["field_from_the_future"] = 42
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE scenarios SET run_context_json = ? WHERE id = ?", (json.dumps(doc), sid))
        conn.execute("UPDATE scenarios SET run_context_json = ? WHERE id = ?", (json.dumps({"apport": "abc"}), bad))
        conn.execute("UPDATE scenarios SET run_context_json = ? WHERE id = ?", ("{not json", garbage))
    assert store.load("alice", sid)[1] == _ctx()
    with pytest.raises(ScenarioError):
        store.load("alice", bad)
    with pytest.raises(ScenarioError):
        store.load("alice", garbage)
    assert len(store.list_scenarios("alice")) == 3  # listing never parses the context


def test_last_context(store):
    assert store.load_last_context("alice") is None
    store.save_last_context("alice", _ctx(apport=5000))
    store.save_last_context("alice", _ctx(apport=6000))
    assert store.load_last_context("alice").apport == 6000
    assert store.list_scenarios("alice") == []  # internal row is not a user scenario
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE scenarios SET run_context_json = '[1]'")
    assert store.load_last_context("alice") is None


def test_users(store):
    store.ensure_user("alice", "Alice D.")
    store.ensure_user("alice", "other name")  # first display name wins
    store.ensure_user("bob")
    assert {u.id: u.display_name for u in store.list_users()} == {"alice": "Alice D.", "bob": "bob"}
    with pytest.raises(ScenarioStoreError):
        store.ensure_user("  ")


def test_schema_version_and_reopen(tmp_path):
    first = SQLiteScenarioStore(tmp_path)
    first.save("alice", "keep", _ctx())
    again = SQLiteScenarioStore(tmp_path)  # migration is idempotent
    assert [s.name for s in again.list_scenarios("alice")] == ["keep"]
    with sqlite3.connect(again.path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_newer_database_is_refused(tmp_path):
    SQLiteScenarioStore(tmp_path)
    with sqlite3.connect(tmp_path / "scenarios.db") as conn:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(ScenarioStoreError):
        SQLiteScenarioStore(tmp_path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_db_is_user_only(tmp_path):
    s = SQLiteScenarioStore(tmp_path / "private")
    assert stat.S_IMODE(os.stat(s.path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(s.path.parent).st_mode) == 0o700


def test_default_store_uses_data_dir_from_conftest(tmp_path):
    assert default_store().path.parent == tmp_path / "data_user"
