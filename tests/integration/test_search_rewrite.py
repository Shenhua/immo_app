"""T31: subset search invariants, determinism, acceptance basis, progress and solver modes."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

import src.application.services.optimizer as optimizer
from src.application.progress import SearchPhase
from src.application.run_context import RunContext
from src.application.services.brick_factory import (
    build_financing_config,
    build_operating_config,
    create_investment_bricks,
)
from src.application.services.optimizer import SearchInvariantError, check_invariants
from src.application.services.strategy_finder import StrategyFinder
from src.core.glossary import cf_accepts
from src.domain.models.archetype import ArchetypeV2

DATA = Path(__file__).resolve().parents[2] / "data" / "archetypes_2026_v1.json"
ALL = [ArchetypeV2(**a).model_dump() for a in json.loads(DATA.read_text(encoding="utf-8"))]


def _finder(ctx: RunContext, archetypes=None, n_workers: int | None = 1) -> StrategyFinder:
    fin = build_financing_config(ctx.to_credit_params())
    op = build_operating_config(
        ctx.gestion_pct,
        ctx.vacance_pct,
        ctx.cfe_par_bien_ann,
        maintenance_m2_an=ctx.maintenance_m2_an,
        gli_pct=ctx.gli_pct,
        mois_travaux=ctx.mois_travaux,
    )
    bricks = create_investment_bricks(ALL if archetypes is None else archetypes, fin, op)
    finder = StrategyFinder(
        bricks=bricks,
        apport_disponible=ctx.apport,
        cash_flow_cible=ctx.cf_cible,
        tolerance=ctx.tolerance,
        qualite_weight=ctx.qualite_weight,
        mode_cf=ctx.mode_cf,
        max_properties=ctx.max_properties,
    )
    finder.n_workers = n_workers
    return finder


def _run(ctx: RunContext, archetypes=None, n_workers: int | None = 1, cb=None):
    finder = _finder(ctx, archetypes, n_workers)
    res = finder.find_strategies(
        eval_params=ctx.to_eval_params(), horizon_years=ctx.horizon_ans, top_n=ctx.top_n, progress_callback=cb
    )
    return finder, res


def _sig(res):
    return [
        (
            tuple(sorted((d["nom"], d["duree_credit_mois"]) for d in s["details"])),
            round(s["apport_total"], 6),
            round(s["balanced_score"], 12),
        )
        for s in res
    ]


def _assert_invariants(res, ctx):
    for s in res:
        names = [d["nom"] for d in s["details"]]
        assert s["apport_total"] <= ctx.apport + 1e-6
        assert len(set(names)) == len(names)
        assert 1 <= len(names) <= ctx.max_properties
        assert cf_accepts(s["cf_monthly_stab"], ctx.cf_cible, ctx.tolerance, ctx.mode_cf)


@pytest.mark.parametrize(
    "ctx",
    [
        RunContext(apport=150000, cf_cible=-900.0, mode_cf="min", max_properties=3),
        RunContext(apport=150000, cf_cible=-900.0, tolerance=150.0, mode_cf="target", max_properties=3),
    ],
    ids=["min", "target"],
)
def test_invariants_and_after_tax_acceptance(ctx):
    _, res = _run(ctx, ALL[:6])
    assert res
    _assert_invariants(res, ctx)


def test_two_runs_identical():
    ctx = RunContext(apport=100000, cf_cible=-700.0, mode_cf="min", max_properties=3)
    assert _sig(_run(ctx, ALL[:6])[1]) == _sig(_run(ctx, ALL[:6])[1])


def test_process_pool_matches_serial():
    ctx = RunContext(apport=150000, cf_cible=-600.0, tolerance=200.0, mode_cf="target", max_properties=2)
    assert _sig(_run(ctx, ALL[:8], n_workers=2)[1]) == _sig(_run(ctx, ALL[:8], n_workers=1)[1])


def test_unfurnished_refund_is_not_rejected_late():
    """Review case: déficit-foncier refund lifts the after-tax CF above the pre-tax one (-724 vs -622 for a
    -800 +/- 100 window). The search must size the down payment on the after-tax CF and accept it."""
    nu = [a for a in ALL if not a.get("meuble")]
    ctx = RunContext(
        apport=200000,
        cf_cible=-800.0,
        tolerance=100.0,
        mode_cf="target",
        max_properties=1,
        regime_foncier="reel",
        tmi_pct=30.0,
    )
    _, res = _run(ctx, nu)
    assert res
    _assert_invariants(res, ctx)
    assert any(s["cf_monthly_stab"] - s["cash_flow_final"] > 20 for s in res), "expected a refund case"


def test_progress_reports_evaluation_for_any_max_props():
    events = []
    ctx = RunContext(apport=120000, cf_cible=-600.0, mode_cf="min", max_properties=5)
    finder, _ = _run(ctx, ALL[:6], cb=events.append)
    ev = [e for e in events if e.phase is SearchPhase.EVALUATION]
    assert ev and ev[-1].items_processed == ev[-1].items_total > 0
    assert finder.solver_mode == "EXHAUSTIVE"


def test_proxy_mode_when_over_budget(monkeypatch):
    monkeypatch.setattr(optimizer, "EVAL_BUDGET", 30)
    ctx = RunContext(apport=200000, cf_cible=-700.0, mode_cf="min", max_properties=3)
    finder, res = _run(ctx, ALL[:8])
    assert finder.solver_mode == "SUBSET_PROXY"
    assert res
    _assert_invariants(res, ctx)


def test_greedy_durations_beyond_exact_bound(monkeypatch):
    """k = 6 properties x 3 durations = 729 > 243 assignments: greedy + one swap pass, still valid."""
    monkeypatch.setattr(optimizer, "EVAL_BUDGET", 60)
    ctx = RunContext(apport=400000, cf_cible=-5000.0, mode_cf="min", max_properties=6)
    cheap = sorted(ALL, key=lambda a: a["prix_m2"] * a["surface"])[:7]
    finder, res = _run(ctx, cheap)
    assert finder.solver_mode == "SUBSET_PROXY"
    assert res
    _assert_invariants(res, ctx)


def test_check_invariants_rejects_bad_results():
    ok = {"apport_total": 10.0, "details": [{"nom": "A"}, {"nom": "B"}]}
    check_invariants([ok], 10.0, 2)
    with pytest.raises(SearchInvariantError):
        check_invariants([{**ok, "apport_total": 11.0}], 10.0, 2)
    with pytest.raises(SearchInvariantError):
        check_invariants([{**ok, "details": [{"nom": "A"}, {"nom": "A"}]}], 10.0, 2)
    with pytest.raises(SearchInvariantError):
        check_invariants([ok], 10.0, 1)


@pytest.mark.skipif(os.environ.get("RUN_SEARCH_BENCH") != "1", reason="benchmark: set RUN_SEARCH_BENCH=1")
def test_empire_mode_under_10_seconds():
    """Acceptance: 18 archetypes x 3 durations = 54 bricks, max 5 properties, under 10 s, deterministic."""
    ctx = RunContext(apport=1000000, cf_cible=-3000.0, mode_cf="min", max_properties=5)
    t = time.perf_counter()
    _, first = _run(ctx, n_workers=None)
    assert time.perf_counter() - t < 10.0
    _assert_invariants(first, ctx)
    assert _sig(_run(ctx, n_workers=None)[1]) == _sig(first)
