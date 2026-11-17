"""T35: the search engine must be built from the same parameters as ``RunContext.build_engine()``.

The guard loops over every RunContext field, so a field added later and not wired into the engine
(``to_tax_params`` / ``market_hypotheses``) fails here instead of silently drifting between search and detail.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.application.run_context import RunContext
from src.application.services.brick_factory import FinancingConfig, OperatingConfig, create_investment_bricks
from src.application.services.simulation import simulate_long_term_strategy
from src.application.services.strategy_finder import EvaluationParams, StrategyFinder
from src.ui.app_controller import load_archetypes

# Fields that legitimately do not reach the simulation engine (objectives, financing, scoring, data).
NOT_ENGINE_FIELDS = {
    "apport": "search budget",
    "cf_cible": "search target",
    "tolerance": "search tolerance",
    "mode_cf": "search / allocation mode",
    "qualite_weight": "scoring",
    "horizon_ans": "passed to the simulation call, not the engine",
    "max_properties": "search size limit",
    "use_full_capital": "search option",
    "top_n": "result count",
    "revenu_net_mensuel": "T60 HCSF eligibility: post-search filter, not the engine",
    "charges_mensuelles": "T60 HCSF eligibility: post-search filter, not the engine",
    "filtrer_hcsf": "T60 HCSF eligibility: post-search filter, not the engine",
    "taux_credits": "financing: baked into the bricks",
    "frais_notaire_pct": "financing: bricks",
    "assurance_ann_pct": "financing: bricks",
    "frais_pret_pct": "financing: bricks",
    "inclure_travaux": "financing: bricks",
    "inclure_reno_ener": "financing: bricks",
    "inclure_mobilier": "financing: bricks",
    "financer_mobilier": "financing: bricks",
    "gestion_pct": "operating costs: bricks",
    "vacance_pct": "operating costs: bricks",
    "maintenance_m2_an": "operating costs: bricks",
    "gli_pct": "operating costs: bricks",
    "mois_travaux": "operating costs: bricks",
    "finance_preset_name": "scoring",
    "finance_weights": "scoring",
    "apply_cap": "archetype filtering",
    "archetypes_version": "data hash",
}

OVERRIDES: dict[str, Any] = {
    "regime_fiscal": "microbic",
    "regime_foncier": "micro",
    "dpe_policy": "ne_pas_renover",
    "nb_owners": 2,
    "ratio_terrain_pct": 33.0,
}


def _non_default(name: str, ctx: RunContext) -> Any:
    if name in OVERRIDES:
        return OVERRIDES[name]
    cur = getattr(ctx, name)
    if isinstance(cur, bool):
        return not cur
    if isinstance(cur, (int, float)):
        return cur + 7
    raise AssertionError(f"no non-default value rule for RunContext.{name}: add it to OVERRIDES or NOT_ENGINE_FIELDS")


def _snapshot(obj: Any, depth: int = 0) -> Any:
    if hasattr(obj, "__dict__") and depth < 4:
        return {k: _snapshot(v, depth + 1) for k, v in vars(obj).items()}
    return obj


@pytest.mark.parametrize("name", list(RunContext.model_fields))
def test_every_run_context_field_reaches_the_search_engine(name):
    base = RunContext()
    if name in NOT_ENGINE_FIELDS:
        value = getattr(base, name)
    else:
        value = _non_default(name, base)
        assert value != getattr(base, name)
    ctx = base.model_copy(update={name: value})

    detail = _snapshot(ctx.build_engine())
    search = _snapshot(EvaluationParams.from_dict(ctx.to_eval_params()).build_engine())
    assert search == detail, f"search engine differs from RunContext.build_engine() for {name}"

    if name not in NOT_ENGINE_FIELDS:
        assert detail != _snapshot(base.build_engine()), (
            f"RunContext.{name} does not change the engine: wire it in to_tax_params / market_hypotheses, "
            "or add it to NOT_ENGINE_FIELDS with a reason"
        )


def test_allowlist_has_no_stale_entries():
    assert set(NOT_ENGINE_FIELDS) <= set(RunContext.model_fields)


def test_dpe_ban_reaches_the_search():
    """E/F/G archetypes without the "Reno E->D" option, ban policy, horizon past the ban: search == detail."""
    ctx = RunContext(apport=80000, cf_cible=-500, dpe_policy="ne_pas_renover", horizon_ans=25, inclure_reno_ener=False)
    archs = [a for a in load_archetypes() if a.get("dpe_initial") in ("E", "F", "G")][:10]
    assert archs
    fin = FinancingConfig(credit_rates={15: 3.5, 20: 3.6, 25: 3.8}, frais_notaire_pct=8.0, assurance_ann_pct=0.35,
                          frais_pret_pct=1.0, inclure_reno_ener=False)
    op = OperatingConfig(frais_gestion_pct=5.0, provision_pct=3.0, cfe_par_bien_ann=500)
    bricks = create_investment_bricks(archs, fin, op)
    finder = StrategyFinder(bricks=bricks, apport_disponible=ctx.apport, cash_flow_cible=ctx.cf_cible,
                            tolerance=500.0, mode_cf="min")
    res = finder.find_strategies(eval_params=ctx.to_eval_params(), horizon_years=ctx.horizon_ans, top_n=1)
    assert res
    _, bilan = simulate_long_term_strategy(res[0], ctx.horizon_ans, ctx.market_hypotheses, ctx.to_tax_params())
    assert res[0]["enrich_net"] == pytest.approx(bilan["enrichissement_net"])

    ctx_reno = ctx.model_copy(update={"dpe_policy": "renover"})
    _, bilan_reno = simulate_long_term_strategy(
        res[0], ctx.horizon_ans, ctx_reno.market_hypotheses, ctx_reno.to_tax_params()
    )
    assert bilan_reno["enrichissement_net"] != pytest.approx(bilan["enrichissement_net"])  # the ban bites
