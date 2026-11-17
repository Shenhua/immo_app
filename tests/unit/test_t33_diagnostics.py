"""T33: rejection counts by reason, near misses, relaxation suggestions."""
from __future__ import annotations

import dataclasses

from src.application.run_context import RunContext
from src.application.services.diagnostics import NearMiss, SearchDiagnostics
from src.application.services.strategy_finder import StrategyFinder
from src.domain.models.brick import InvestmentBrick
from src.ui.components.diagnostics import diagnostics_match, near_miss_lines, no_result_lines

_F = {
    "charges_const_mth0": 50.0, "tf_const_mth0": 50.0, "frais_gestion_pct": 0.0, "budget_travaux": 0.0,
    "mobilier": 0.0, "frais_notaire": 7000.0, "taux_annuel_pct": 3.5, "duree_credit_mois": 240,
}


def _bricks() -> list[InvestmentBrick]:
    """A (10 k down payment), B (20 k), C (60 k: over a 50 k budget)."""
    return [
        InvestmentBrick(nom="A", apport_min=10000, prix_achat_bien=100000, capital_emprunte=90000,
                        loyer_mensuel_initial=500, **_F),
        InvestmentBrick(nom="B", apport_min=20000, prix_achat_bien=200000, capital_emprunte=180000,
                        loyer_mensuel_initial=1000, **_F),
        InvestmentBrick(nom="C", apport_min=60000, prix_achat_bien=300000, capital_emprunte=240000,
                        loyer_mensuel_initial=1000, **_F),
    ]


def _run(target: float, max_props: int = 1) -> tuple[StrategyFinder, list]:
    finder = StrategyFinder(_bricks(), 50000, target, max_properties=max_props)
    finder.n_workers = 1
    return finder, finder.find_strategies(eval_params={}, horizon_years=10, top_n=10)


def test_counts_by_reason_on_small_instance():
    finder, strategies = _run(200.0)
    d = finder.last_diagnostics
    assert strategies == []
    # C is over budget, {A, B} exceeds max_props=1, A and B miss the CF target.
    assert d.counts == {"budget": 1, "cf": 2, "max_props": 1, "evaluator": 0, "hcsf": 0}
    assert d.evaluated == 2 and d.accepted == 0
    assert d.top_reason() == "cf"
    by_reason = {n.reason: n for n in d.all_near_misses() if n.reason != "cf"}
    assert by_reason["budget"].names == ("C",)
    assert by_reason["budget"].gap == 10000.0  # 60 k down payment vs 50 k budget
    assert by_reason["max_props"].names == ("A", "B")
    cf = {n.names: n for n in d.near_misses["cf"]}
    assert set(cf) == {("A",), ("B",)}
    assert 0 < cf[("A",)].gap < cf[("B",)].gap  # A is the nearer miss
    assert list(d.near_misses["cf"]) == sorted(d.near_misses["cf"], key=lambda n: n.gap)


def test_accepted_target_has_no_cf_rejection():
    finder, strategies = _run(-250.0)
    d = finder.last_diagnostics
    assert strategies
    assert d.accepted >= 1
    assert d.counts["budget"] == 1


def test_near_misses_keep_only_the_best_n():
    d = SearchDiagnostics(target_cf=0.0, mode_cf="min")
    for i in range(12):
        d.record(NearMiss("cf", (f"P{i}",), float(100 - i), "€/mois"))
    assert d.counts["cf"] == 12
    assert [n.gap for n in d.near_misses["cf"]] == [89.0, 90.0, 91.0, 92.0, 93.0]


def test_relaxation_suggestion_makes_search_succeed():
    finder, strategies = _run(200.0)
    assert strategies == []
    sugg = [s for s in finder.last_diagnostics.suggestions() if s.param == "cf_cible"]
    assert sugg and sugg[0].suggested < 200.0 and sugg[0].count >= 1
    # Apply the suggestion (the same thing the user does in the sidebar) and re-run.
    _, relaxed = _run(sugg[0].suggested)
    assert relaxed


def test_tolerance_suggestion_in_target_mode():
    finder = StrategyFinder(_bricks(), 50000, 200.0, tolerance=10.0, mode_cf="target", max_properties=1)
    finder.n_workers = 1
    assert finder.find_strategies(eval_params={}, horizon_years=10, top_n=10) == []
    sugg = [s for s in finder.last_diagnostics.suggestions() if s.param == "tolerance"]
    assert sugg and sugg[0].suggested > 10.0


def test_other_suggestions_budget_and_max_props():
    finder, _ = _run(200.0)
    by_param = {s.param: s for s in finder.last_diagnostics.suggestions()}
    assert by_param["apport"].suggested == 60000.0
    assert by_param["max_properties"].suggested == 2.0


def test_no_result_lines_name_top_constraint_and_suggest():
    finder, _ = _run(200.0)
    text = "\n".join(no_result_lines(finder.last_diagnostics))
    assert "cash-flow hors de la cible" in text
    assert "CF cible" in text
    assert near_miss_lines(finder.last_diagnostics)


def test_diagnostics_match_guards_stale_state():
    finder, _ = _run(200.0)
    ctx = RunContext(apport=50000, cf_cible=200.0, max_properties=1)
    assert diagnostics_match(finder.last_diagnostics, ctx)
    assert not diagnostics_match(finder.last_diagnostics, ctx.model_copy(update={"cf_cible": 0.0}))
    assert dataclasses.is_dataclass(finder.last_diagnostics)
