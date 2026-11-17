"""T30: mode-aware cash-flow acceptance and ranking."""

import pytest

from src.application.services.allocator import PortfolioAllocator
from src.application.services.strategy_finder import StrategyFinder, StrategyScorer
from src.core.glossary import cf_accepts, cf_penalty

# (mode, cf, target, tol, accepted, penalty)
CASES = [
    ("min", 150.0, 100.0, 10.0, True, 0.0),  # surplus: accepted, never penalised
    ("min", 100.0, 100.0, 10.0, True, 0.0),  # exactly at target
    ("min", 95.0, 100.0, 10.0, False, 5.0),  # deficit within tol: rejected
    ("min", 99.99, 100.0, 100.0, False, 0.01),
    ("target", 150.0, 100.0, 10.0, False, 50.0),  # surplus outside tol rejected
    ("target", 95.0, 100.0, 10.0, True, 5.0),
    ("target", 110.0, 100.0, 10.0, True, 10.0),
    ("target", 80.0, 100.0, 10.0, False, 20.0),
]


@pytest.mark.parametrize("mode,cf,target,tol,ok,pen", CASES)
def test_glossary_helpers(mode, cf, target, tol, ok, pen):
    assert cf_accepts(cf, target, tol, mode) is ok
    assert cf_penalty(cf, target, mode) == pytest.approx(pen)


@pytest.mark.parametrize("mode,cf,target,tol,ok,pen", CASES)
def test_allocator_agrees_with_glossary(mode, cf, target, tol, ok, pen):
    assert PortfolioAllocator(mode)._accept_cf(cf, target, tol) is ok


def _strats(cfs):
    return [
        {
            "name": f"s{i}",
            "cf_monthly_stab": cf,
            "enrich_net": 1.0,
            "tri_annuel": 0.05,
            "cap_eff": 1.0,
            "dscr_y1": 1.2,
            "tier": 1,
        }
        for i, cf in enumerate(cfs)
    ]


def test_proximity_min_mode():
    # s0: deficit, s1: on target, s2: surplus
    strategies = _strats([-50.0, 100.0, 200.0])
    StrategyScorer(mode_cf="min").score_strategies(strategies, 100.0)
    p = [s["cf_proximity"] for s in strategies]
    assert p[1] == p[2] == 1.0
    assert p[0] < p[1]


def test_proximity_target_mode():
    strategies = _strats([-50.0, 100.0, 200.0])
    StrategyScorer(mode_cf="target").score_strategies(strategies, 100.0)
    p = [s["cf_proximity"] for s in strategies]
    assert p[1] > p[2]
    assert p[1] > p[0]


def test_min_mode_higher_surplus_not_lower():
    strategies = _strats([100.0, 150.0, 300.0])
    StrategyScorer(mode_cf="min").score_strategies(strategies, 100.0)
    p = [s["cf_proximity"] for s in strategies]
    assert p[0] <= p[1] <= p[2]


@pytest.mark.parametrize(
    "mode,last,first",
    [
        ("min", "s0", None),  # deficit ranks last; surplus not below on-target
        ("target", "s0", "s1"),  # on-target first
    ],
)
def test_rank_strategies_cf_tiebreak(mode, last, first):
    strategies = _strats([-50.0, 100.0, 200.0])
    StrategyScorer(mode_cf=mode).score_strategies(strategies, 100.0)
    for s in strategies:  # equalise everything but cf_proximity
        s["finance_score"] = 0.5
        s["balanced_score"] = 0.5
        s["dscr_norm"] = 0.5
    finder = StrategyFinder.__new__(StrategyFinder)
    ranked = finder.rank_strategies(strategies, "cashflow")
    assert ranked[-1]["name"] == last
    if first:
        assert ranked[0]["name"] == first
    else:
        assert {ranked[0]["name"], ranked[1]["name"]} == {"s1", "s2"}
