"""T32: absolute scoring and the Pareto front."""

from __future__ import annotations

import pytest

from src.application.services.scoring import assign_pareto_ranks
from src.application.services.strategy_finder import StrategyFinder, StrategyScorer
from src.core.defaults import preset_weights


def _strat(name: str, **kw) -> dict:
    base = {
        "details": [{"nom": name, "duree_credit_mois": 240}],
        "enrich_net": 100_000.0,
        "tri_annuel": 6.0,
        "dscr_stab": 1.2,
        "apport_total": 50_000.0,
        "qual_score": 60.0,
        "cf_monthly_stab": 0.0,
        "tier": 1,
    }
    base.update(kw)
    return base


def test_score_does_not_depend_on_the_candidate_pool():
    a = _strat("A")
    pool_1 = [a.copy(), _strat("B", enrich_net=1e6, tri_annuel=11.0)]
    pool_2 = [a.copy(), _strat("C", enrich_net=-5e4, tri_annuel=-2.0), _strat("D", dscr_stab=3.0)]
    scorer = StrategyScorer(mode_cf="min")
    scorer.score_strategies(pool_1, 0.0, 20)
    scorer.score_strategies(pool_2, 0.0, 20)
    for key in ("finance_score", "balanced_score", "enrich_norm", "tri_norm", "dscr_norm", "cf_proximity"):
        assert pool_1[0][key] == pytest.approx(pool_2[0][key])


def test_cf_proximity_uses_after_tax_stabilised_cf():
    # Pre-tax CF is on target, after-tax CF is 250 EUR short: the proximity follows the after-tax figure.
    s = _strat("A", cash_flow_final=100.0, cf_monthly_stab=-150.0)
    StrategyScorer(mode_cf="min").score_strategies([s], 100.0, 20)
    assert s["cf_proximity"] == pytest.approx(1.0 - 250.0 / 500.0)
    # Surplus in min mode is not penalised.
    t = _strat("B", cash_flow_final=-900.0, cf_monthly_stab=300.0)
    StrategyScorer(mode_cf="min").score_strategies([t], 100.0, 20)
    assert t["cf_proximity"] == 1.0
    # Target mode penalises the surplus too.
    u = _strat("C", cf_monthly_stab=300.0)
    StrategyScorer(mode_cf="target").score_strategies([u], 100.0, 20)
    assert u["cf_proximity"] == pytest.approx(1.0 - 200.0 / 500.0)


def test_scorer_requires_mode_cf():
    with pytest.raises(TypeError):
        StrategyScorer()  # type: ignore[call-arg]


def test_failed_irr_scores_zero_and_ranks_last_on_the_front():
    ok = _strat("ok")
    bad = _strat("bad", tri_annuel=None)
    StrategyScorer(mode_cf="min").score_strategies([ok, bad], 0.0, 20)
    assert bad["tri_norm"] == 0.0
    assign_pareto_ranks([ok, bad], 0.0, "min")
    assert (ok["pareto_rank"], bad["pareto_rank"]) == (1, 2)


def test_pareto_ranks_on_a_hand_made_set():
    # Objectives: wealth, IRR, DSCR, apport (lower is better), quality, CF penalty (min mode, target 0).
    a = _strat("A", enrich_net=300_000.0, tri_annuel=5.0, apport_total=100_000.0)  # best wealth
    b = _strat("B", enrich_net=100_000.0, tri_annuel=9.0, apport_total=100_000.0)  # best IRR
    c = _strat("C", enrich_net=100_000.0, tri_annuel=5.0, apport_total=20_000.0)  # cheapest
    d = _strat("D", enrich_net=90_000.0, tri_annuel=4.0, apport_total=100_000.0)  # dominated by b (and a)
    e = _strat("E", enrich_net=80_000.0, tri_annuel=3.0, apport_total=100_000.0)  # dominated by d too
    f = _strat(
        "F", enrich_net=300_000.0, tri_annuel=5.0, apport_total=100_000.0, cf_monthly_stab=-80.0
    )  # worse CF than a
    strategies = [a, b, c, d, e, f]
    assign_pareto_ranks(strategies, 0.0, "min")
    assert [s["pareto_rank"] for s in strategies] == [1, 1, 1, 2, 3, 2]


def test_high_wealth_strategy_survives_under_patrimoine():
    """A big-wealth, big-apport, thin-CF strategy is not filtered out and ranks first under Patrimoine."""
    wealth = _strat(
        "W", enrich_net=900_000.0, tri_annuel=7.0, apport_total=200_000.0, dscr_stab=1.0, cf_monthly_stab=-300.0
    )
    cheap = [
        _strat(
            f"S{i}", enrich_net=30_000.0 + i, tri_annuel=8.0, apport_total=20_000.0, dscr_stab=1.4, cf_monthly_stab=50.0
        )
        for i in range(15)
    ]
    strategies = [wealth, *cheap]
    finder = StrategyFinder(bricks=[], apport_disponible=250_000.0, cash_flow_cible=-500.0, mode_cf="min")
    finder.scorer.weights = finder.scorer._normalize_weights(preset_weights("Patrimoine LT"))
    finder.scorer.score_strategies(strategies, -500.0, 25)
    assign_pareto_ranks(strategies, -500.0, "min")
    assert wealth["pareto_rank"] == 1  # best on wealth, never dominated
    ranked = finder.rank_strategies(strategies, "Patrimoine LT", top_n=3)
    assert ranked[0] is wealth


def test_pareto_rank_breaks_ties_between_equal_scores():
    x = _strat("X", pareto_rank=2, finance_score=0.5, balanced_score=0.5, cf_proximity=1.0)
    y = _strat("Y", pareto_rank=1, finance_score=0.5, balanced_score=0.5, cf_proximity=1.0)
    finder = StrategyFinder(bricks=[], apport_disponible=0.0, cash_flow_cible=0.0, mode_cf="min")
    assert finder.rank_strategies([x, y], "Équilibré")[0] is y
