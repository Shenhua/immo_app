"""T27: the search, the detail view and the sensitivity tab must use the same engine parameters (review C1).

Before the fix the search ignored ``tf_growth_pct`` (it fell back to the charges inflation) while the detail
view and the sensitivity tab used the run's value, so the cards showed a different IRR / wealth than the detail.
"""

from __future__ import annotations

import pytest

from src.application.run_context import RunContext
from src.application.services.brick_factory import FinancingConfig, OperatingConfig, create_investment_bricks
from src.application.services.evaluator import StrategyEvaluator
from src.application.services.simulation import build_engine, simulate_long_term_strategy
from src.application.services.strategy_finder import EvaluationParams, StrategyFinder
from src.ui.app_controller import load_archetypes, simulate_selected_strategy
from src.ui.components.sensitivity import run_sensitivity_simulation

TF_GROWTH = 7.0  # non-default (default is 3 %)

TAX_SETUPS = {
    "lmnp_reel": {"regime_fiscal": "lmnp", "regime_foncier": "reel", "sci_is": False},
    "micro_bic": {"regime_fiscal": "microbic", "regime_foncier": "micro", "sci_is": False},
    "sci_is": {"regime_fiscal": "lmnp", "regime_foncier": "reel", "sci_is": True},
}


def _run_search(ctx: RunContext) -> dict:
    fin = FinancingConfig(credit_rates={15: 3.5, 20: 3.6, 25: 3.8}, frais_notaire_pct=8.0, assurance_ann_pct=0.35, frais_pret_pct=1.0)
    op = OperatingConfig(frais_gestion_pct=5.0, provision_pct=3.0, cfe_par_bien_ann=500)
    bricks = create_investment_bricks(load_archetypes()[:10], fin, op)
    finder = StrategyFinder(
        bricks=bricks, apport_disponible=ctx.apport, cash_flow_cible=ctx.cf_cible, tolerance=500.0, mode_cf="min"
    )
    strategies = finder.find_strategies(eval_params=ctx.to_eval_params(), horizon_years=ctx.horizon_ans, top_n=1)
    assert strategies, "the search must return a strategy"
    return strategies[0]


@pytest.mark.parametrize("setup", list(TAX_SETUPS))
def test_search_bilan_equals_detail_bilan(setup):
    ctx = RunContext(tf_growth_pct=TF_GROWTH, apport=80000, cf_cible=-500, **TAX_SETUPS[setup])
    strategy = _run_search(ctx)

    # Search side: the engine the finder builds, driven by the evaluator exactly as in the search.
    ep = EvaluationParams.from_dict(ctx.to_eval_params())
    assert ep.hypotheses_marche["tf_growth_pct"] == TF_GROWTH
    evaluator = StrategyEvaluator(simulator=build_engine(ep.hypotheses_marche, ep.to_tax_params()))
    df_search, bilan_search = evaluator.run_simulation(strategy, ctx.horizon_ans, evaluator.generate_schedules(strategy["details"]))

    # Detail side: what the UI does.
    df_detail = simulate_selected_strategy(strategy, ctx)
    assert df_detail is not None
    _, bilan_detail = simulate_long_term_strategy(strategy, ctx.horizon_ans, ctx.market_hypotheses, ctx.to_tax_params())

    for key in ("enrichissement_net", "tri_annuel", "liquidation_nette"):
        assert bilan_search[key] == pytest.approx(bilan_detail[key]), key
    cf_col = "Cash-Flow Net d'Impôt"
    assert df_search[cf_col].sum() == pytest.approx(df_detail[cf_col].sum())

    # The numbers stored on the search result (what the card shows) are the detail numbers.
    assert strategy["enrich_net"] == pytest.approx(bilan_detail["enrichissement_net"])
    assert strategy["tri_annuel"] == pytest.approx(bilan_detail["tri_annuel"])
    assert strategy["liquidation_nette"] == pytest.approx(bilan_detail["liquidation_nette"])

    # The context builder gives the same engine parameters as the search.
    assert ctx.build_engine().market.tf_growth_pct == TF_GROWTH


def test_tf_growth_changes_the_search_result():
    """The TF slider must reach the search: the same run with another TF growth gives another wealth."""
    low = _run_search(RunContext(tf_growth_pct=0.0, apport=80000, cf_cible=-500))
    high = _run_search(RunContext(tf_growth_pct=TF_GROWTH, apport=80000, cf_cible=-500))
    assert low["enrich_net"] != pytest.approx(high["enrich_net"])


@pytest.mark.parametrize("setup", list(TAX_SETUPS))
def test_sensitivity_at_zero_delta_has_no_impact(setup):
    ctx = RunContext(tf_growth_pct=TF_GROWTH, apport=80000, cf_cible=-500, **TAX_SETUPS[setup])
    strategy = _run_search(ctx)

    res = run_sensitivity_simulation(strategy, 0.0, 0.0, 0.0, ctx)

    assert res["tri_annuel"] - strategy["tri_annuel"] == pytest.approx(0.0, abs=1e-9)
    assert res["liquidation_nette"] - strategy["liquidation_nette"] == pytest.approx(0.0, abs=1e-6)


def test_market_hypotheses_tf_growth_has_no_none_fallback():
    from src.application.services.simulation import MarketHypotheses
    from src.core.defaults import REALISM

    assert MarketHypotheses().tf_growth_pct == REALISM.tf_growth_pct
    assert MarketHypotheses.from_dict({}).tf_growth_pct == REALISM.tf_growth_pct
    assert EvaluationParams.from_dict({}).hypotheses_marche["tf_growth_pct"] == REALISM.tf_growth_pct
