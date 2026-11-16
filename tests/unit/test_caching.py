"""T42: row selection and tab switches must not recompute a search or a simulation already computed."""

from __future__ import annotations

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.application.run_context import RunContext
from src.ui import app_controller, cached


@pytest.fixture
def counters(monkeypatch):
    st.cache_data.clear()
    calls = {"simulate": 0, "search": 0, "bricks": 0}

    real_sim = app_controller.simulate_long_term_strategy
    real_find = app_controller.StrategyFinder.find_strategies
    real_bricks = cached.create_investment_bricks

    def sim(*a, **k):
        calls["simulate"] += 1
        return real_sim(*a, **k)

    def find(self, *a, **k):
        calls["search"] += 1
        return real_find(self, *a, **k)

    def bricks(*a, **k):
        calls["bricks"] += 1
        return real_bricks(*a, **k)

    monkeypatch.setattr(app_controller, "simulate_long_term_strategy", sim)
    monkeypatch.setattr(app_controller.StrategyFinder, "find_strategies", find)
    monkeypatch.setattr(cached, "create_investment_bricks", bricks)
    yield calls
    st.cache_data.clear()


def _run_search(timeout=180):
    at = AppTest.from_file("../../app.py", default_timeout=timeout)
    at.run()
    next(n for n in at.sidebar.number_input if n.label.startswith("Cash-Flow")).set_value(-500.0)
    at.run()
    next(b for b in at.sidebar.button if "Lancer" in b.label).click()
    at.run()
    return at


def test_selection_and_reruns_do_not_recompute(counters):
    at = _run_search()
    assert not at.exception
    assert counters["search"] == 1
    assert counters["bricks"] == 1
    assert len(at.session_state["strategies"]) >= 2
    assert counters["simulate"] == 1  # strategy 0

    # Plain reruns (tab switch, widget interaction): nothing recomputed.
    for _ in range(3):
        at.run()
    assert counters["simulate"] == 1
    assert counters["search"] == 1
    assert counters["bricks"] == 1

    # Select another row, then go back: one simulation per distinct strategy.
    for idx in (1, 0, 1, 0):
        at.session_state["selected_strategy_idx"] = idx
        at.run()
        assert not at.exception
    assert counters["simulate"] == 2
    assert counters["search"] == 1


def test_cached_simulation_returns_independent_copies(counters):
    at = _run_search()
    strategy = at.session_state["strategies"][0]
    ctx = RunContext.model_validate_json(at.session_state["run_context"].model_dump_json())

    df1 = app_controller.simulate_selected_strategy(strategy, ctx)
    assert isinstance(df1, pd.DataFrame)
    df1.iloc[0, 0] = -1e12
    df1.attrs["tax_regimes"] = "MUTATED"
    df2 = app_controller.simulate_selected_strategy(strategy, ctx)
    assert df2.iloc[0, 0] != -1e12
    assert df2.attrs["tax_regimes"] != "MUTATED"


def test_archetype_loading_is_cached(monkeypatch):
    st.cache_data.clear()
    n = {"c": 0}
    real = app_controller.load_archetypes_from_disk

    def counting(path):
        n["c"] += 1
        return real(path)

    monkeypatch.setattr(app_controller, "load_archetypes_from_disk", counting)
    path = "data/archetypes_2026_v1.json"
    a = cached.cached_archetypes_from_file(*cached.file_signature(path))
    a[0]["nom"] = "MUTATED"
    b = cached.cached_archetypes_from_file(*cached.file_signature(path))
    assert n["c"] == 1
    assert b[0]["nom"] != "MUTATED"
    st.cache_data.clear()
