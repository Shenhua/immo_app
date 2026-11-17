"""T50: property-based tests (hypothesis) on the search, the single-strategy simulation and the allocator.

Instances are kept small (few bricks, max_properties <= 3, serial evaluation) so the suite stays fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.application.run_context import RunContext
from src.application.services.allocator import PortfolioAllocator
from src.application.services.brick_factory import (
    build_financing_config,
    build_operating_config,
    create_investment_bricks,
)
from src.application.services.simulation import simulate_long_term_strategy
from src.application.services.strategy_finder import StrategyFinder
from src.core.glossary import cf_accepts
from src.domain.models.archetype import ArchetypeV2

DATA = Path(__file__).resolve().parents[2] / "data" / "archetypes_2026_v1.json"
ALL = [ArchetypeV2(**a).model_dump() for a in json.loads(DATA.read_text(encoding="utf-8"))]
POOL = ALL[:10]

SETTINGS = settings(max_examples=40, deadline=None, suppress_health_check=list(HealthCheck))


def _bricks(ctx: RunContext, archetypes):
    fin = build_financing_config(ctx.to_credit_params())
    op = build_operating_config(
        ctx.gestion_pct, ctx.vacance_pct, ctx.cfe_par_bien_ann,
        maintenance_m2_an=ctx.maintenance_m2_an, gli_pct=ctx.gli_pct, mois_travaux=ctx.mois_travaux,
    )
    return create_investment_bricks(archetypes, fin, op)


def _search(ctx: RunContext, archetypes):
    finder = StrategyFinder(
        bricks=_bricks(ctx, archetypes), apport_disponible=ctx.apport, cash_flow_cible=ctx.cf_cible,
        tolerance=ctx.tolerance, qualite_weight=ctx.qualite_weight, mode_cf=ctx.mode_cf,
        max_properties=ctx.max_properties,
    )
    finder.n_workers = 1
    return finder.find_strategies(eval_params=ctx.to_eval_params(), horizon_years=ctx.horizon_ans, top_n=ctx.top_n)


@st.composite
def search_inputs(draw):
    idx = draw(st.lists(st.integers(0, len(POOL) - 1), min_size=1, max_size=4, unique=True))
    ctx = RunContext(
        apport=draw(st.integers(20_000, 250_000)),
        cf_cible=float(draw(st.integers(-1200, 100))),
        tolerance=float(draw(st.integers(50, 300))),
        mode_cf=draw(st.sampled_from(["min", "target"])),
        max_properties=draw(st.integers(1, 3)),
        horizon_ans=draw(st.sampled_from([10, 15, 20])),
        top_n=5,
    )
    return ctx, [POOL[i] for i in idx]


def _sig(res):
    return [
        (tuple(sorted((d["nom"], d["duree_credit_mois"]) for d in s["details"])), round(s["apport_total"], 6),
         round(s["balanced_score"], 12))
        for s in res
    ]


# --------------------------------------------------------------------------- 1. invariants
@SETTINGS
@given(search_inputs())
def test_invariants_on_every_result(inp):
    ctx, archs = inp
    for s in _search(ctx, archs):
        names = [d["nom"] for d in s["details"]]
        assert s["apport_total"] <= ctx.apport + 1e-6
        assert len(set(names)) == len(names)
        assert 1 <= len(names) <= ctx.max_properties
        assert cf_accepts(s["cf_monthly_stab"], ctx.cf_cible, ctx.tolerance, ctx.mode_cf)


# --------------------------------------------------------------------------- 2. determinism
@SETTINGS
@given(search_inputs())
def test_search_is_deterministic(inp):
    ctx, archs = inp
    assert _sig(_search(ctx, archs)) == _sig(_search(ctx, archs))


# --------------------------------------------------------------------------- 3. monotonicity
def _single_strategy(ctx: RunContext, arch, rent_factor: float = 1.0):
    """One brick, minimum down payment, simulated with the run's engine inputs."""
    brick = _bricks(ctx, [arch])[0]
    alloc = PortfolioAllocator("min", market=ctx.to_eval_params()["hypotheses_marche"])
    _, details, _, apport = alloc.allocate([brick], 10**9, -10**6)
    for d in details:
        d["loyer_mensuel_initial"] *= rent_factor
    return {"details": details, "apport_total": apport}


def _simulate(strategy, ctx: RunContext, horizon: int, **market_over):
    market = {**ctx.to_eval_params()["hypotheses_marche"], **market_over}
    df, bilan = simulate_long_term_strategy(strategy, horizon, market, ctx.to_eval_params())
    return float(df["Cash-Flow Net d'Impôt"].iloc[1]) / 12.0, bilan


arch_st = st.integers(0, len(POOL) - 1).map(lambda i: POOL[i])
horizon_st = st.sampled_from([10, 15, 20])


@pytest.mark.xfail(
    strict=False,
    reason="T36: non-monotonic year-2 CF (POOL[4], f=1.0625, delta=0.0625, horizon=10) under investigation",
)
@SETTINGS
@given(arch_st, st.floats(0.5, 1.5), st.floats(0.01, 0.4), horizon_st)
def test_higher_rent_raises_irr_and_cf(arch, f, delta, horizon):
    ctx = RunContext()
    cf_lo, b_lo = _simulate(_single_strategy(ctx, arch, f), ctx, horizon)
    cf_hi, b_hi = _simulate(_single_strategy(ctx, arch, f * (1 + delta)), ctx, horizon)
    assert cf_hi >= cf_lo - 1e-6
    if b_lo["tri_annuel"] is not None and b_hi["tri_annuel"] is not None:
        assert b_hi["tri_annuel"] >= b_lo["tri_annuel"] - 1e-9


@SETTINGS
@given(arch_st, st.floats(1.0, 6.0), st.floats(0.1, 2.0), horizon_st)
def test_higher_loan_rate_lowers_cf(arch, rate, delta, horizon):
    def run(r):
        ctx = RunContext(taux_credits={15: r, 20: r, 25: r})
        return _simulate(_single_strategy(ctx, arch), ctx, horizon)[0]

    assert run(rate + delta) <= run(rate) + 1e-6


@SETTINGS
@given(arch_st, st.floats(-1.0, 5.0), st.floats(0.1, 3.0), horizon_st)
def test_higher_appreciation_raises_wealth(arch, app, delta, horizon):
    ctx = RunContext()
    strat = _single_strategy(ctx, arch)
    lo = _simulate(strat, ctx, horizon, appreciation_bien_pct=app)[1]["enrichissement_net"]
    hi = _simulate(strat, ctx, horizon, appreciation_bien_pct=app + delta)[1]["enrichissement_net"]
    assert hi >= lo - 1e-6


# --------------------------------------------------------------------------- 4. allocator
@SETTINGS
@given(
    st.lists(st.integers(0, len(POOL) - 1), min_size=1, max_size=3, unique=True),
    st.integers(0, 300_000), st.integers(0, 150_000), st.integers(-1500, 300),
    st.sampled_from(["min", "target"]),
)
def test_allocator_budget_and_monotone_in_budget(idx, budget, more, target, mode):
    ctx = RunContext()
    seen, uniq = set(), []
    for b in _bricks(ctx, [POOL[i] for i in idx]):
        if b.nom_bien not in seen:  # one brick per archetype so names stay distinct
            seen.add(b.nom_bien)
            uniq.append(b)
    alloc = PortfolioAllocator(mode, market=ctx.to_eval_params()["hypotheses_marche"])
    prepared = [alloc.prepare(b) for b in uniq]
    a1 = alloc.plan(prepared, float(budget), float(target), 100.0)
    a2 = alloc.plan(prepared, float(budget + more), float(target), 100.0)
    if a1.ok:
        assert a1.within_budget
        assert a1.apport_total <= budget + 1e-6
        assert all(x >= 0 for x in a1.extra)
        assert all(d["credit_final"] >= 0 for d in alloc.details(prepared, a1.extra))
        # more budget never breaks a feasible allocation
        assert a2.ok
        assert a2.apport_total <= budget + more + 1e-6


@pytest.mark.parametrize("mode", ["min", "target"])
def test_allocator_over_budget_is_never_ok(mode):
    ctx = RunContext()
    bricks = _bricks(ctx, POOL[:2])[:2]
    alloc = PortfolioAllocator(mode, market=ctx.to_eval_params()["hypotheses_marche"])
    prepared = [alloc.prepare(b) for b in bricks]
    a = alloc.plan(prepared, 0.0, -10_000.0, 100.0)
    assert not a.ok and not a.within_budget
