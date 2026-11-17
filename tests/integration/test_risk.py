"""T64: risk scenarios, break-even indicators and Monte Carlo (pure functions)."""

from __future__ import annotations

import pytest

from src.application.run_context import RunContext
from src.application.services.risk import (
    break_even_appreciation,
    break_even_vacancy,
    monte_carlo,
    monte_carlo_draws,
    run_scenarios,
)
from src.application.services.sensitivity import Deltas, run_sensitivity
from tests.integration.test_search_detail_parity import _run_search

CTX = RunContext(apport=80000, cf_cible=-500)


@pytest.fixture(scope="module")
def strategy():
    return _run_search(CTX)


def test_central_equals_baseline(strategy):
    sc = run_scenarios(strategy, CTX)
    assert sc["Central"] == run_sensitivity(strategy, Deltas(), CTX)


def test_scenarios_ordered(strategy):
    sc = run_scenarios(strategy, CTX)
    p, c, o = sc["Pessimiste"], sc["Central"], sc["Optimiste"]
    assert p["tri_annuel"] <= c["tri_annuel"] <= o["tri_annuel"]
    assert p["enrichissement_net"] <= c["enrichissement_net"] <= o["enrichissement_net"]


def test_break_even_vacancy_hits_target(strategy):
    base_cf = run_sensitivity(strategy, Deltas(), CTX)["cf_stab_mensuel"]
    target = base_cf - 40.0  # reachable by vacancy alone
    be = break_even_vacancy(strategy, CTX, target=target)
    assert be is not None
    cf = run_sensitivity(strategy, Deltas(vacance_pts=be["delta_pts"]), CTX)["cf_stab_mensuel"]
    assert cf == pytest.approx(target, abs=1.0)


def test_break_even_vacancy_not_bracketed(strategy):
    assert break_even_vacancy(strategy, CTX, target=1e9) is None


def test_break_even_appreciation(strategy):
    be = break_even_appreciation(strategy, CTX)
    if be is not None:
        w = run_sensitivity(strategy, Deltas(appreciation_pts=be["delta_pts"]), CTX)["enrichissement_net"]
        assert abs(w) <= 1.0


def test_monte_carlo_deterministic_and_ordered(strategy):
    assert monte_carlo_draws(50) == monte_carlo_draws(50)
    a = monte_carlo(strategy, CTX, n=30, seed=7)
    b = monte_carlo(strategy, CTX, n=30, seed=7)
    assert a == b
    for key in ("tri", "enrichissement_net"):
        p = a[key]
        assert p["p10"] <= p["p50"] <= p["p90"]
