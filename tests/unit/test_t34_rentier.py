"""T34: Rentier mode (use_full_capital) and the tax-correction loop convergence."""

from __future__ import annotations

from src.application.services.strategy_finder import StrategyFinder
from src.domain.models.brick import InvestmentBrick

_F = {
    "charges_const_mth0": 50.0,
    "tf_const_mth0": 50.0,
    "frais_gestion_pct": 0.0,
    "budget_travaux": 0.0,
    "mobilier": 0.0,
    "frais_notaire": 7000.0,
    "taux_annuel_pct": 3.5,
    "duree_credit_mois": 240,
}


def _bricks() -> list[InvestmentBrick]:
    return [
        InvestmentBrick(
            nom="A", apport_min=10000, prix_achat_bien=100000, capital_emprunte=90000, loyer_mensuel_initial=500, **_F
        ),
        InvestmentBrick(
            nom="B", apport_min=20000, prix_achat_bien=200000, capital_emprunte=180000, loyer_mensuel_initial=1000, **_F
        ),
    ]


def _run(budget: float, target: float, full: bool = False, max_props: int = 1) -> list[dict]:
    finder = StrategyFinder(_bricks(), budget, target, max_properties=max_props)
    finder.n_workers = 1
    return finder.find_strategies(eval_params={}, horizon_years=10, top_n=10, use_full_capital_override=full)


def test_rentier_invests_more_and_earns_more_cash_flow():
    classic = _run(60000, -250.0, False)
    rentier = _run(60000, -250.0, True)
    assert classic and rentier
    assert max(s["apport_total"] for s in rentier) > max(s["apport_total"] for s in classic)
    assert max(float(s["cf_monthly_stab"]) for s in rentier) > max(float(s["cf_monthly_stab"]) for s in classic) + 50.0


def test_rentier_respects_budget_and_target_floor():
    target = -100.0
    strategies = _run(60000, target, True)
    assert strategies
    for s in strategies:
        assert s["apport_total"] <= 60000 + 1e-6
        assert float(s["cf_monthly_stab"]) >= target - 1e-6


def test_rentier_full_cash_has_no_phantom_loan():
    # The budget covers the whole loan of A: it is paid in cash, leaving no loan at all (not 1 EUR).
    strategies = _run(150000, 0.0, True)
    cash = [s for s in strategies if [d["nom"] for d in s["details"]] == ["A"]]
    assert cash
    d = cash[0]["details"][0]
    assert d["credit_final"] == 0.0
    assert d["pmt_total"] == 0.0
    assert d["apport_add_bien"] == 90000.0


def test_tax_correction_loop_converges_when_target_is_relaxed():
    # Regression (T33): with 3 rounds B ended 5 EUR/month short of -260 and was rejected, and A at -200
    # ended 0.2 EUR short. Both are reachable within the budget and must now be accepted.
    for target, expected in ((-260.0, {"A", "B"}), (-200.0, {"A"})):
        strategies = _run(50000, target)
        assert {s["details"][0]["nom"] for s in strategies} == expected
        for s in strategies:
            assert float(s["cf_monthly_stab"]) >= target - 1e-9
    b = next(s for s in _run(50000, -260.0) if s["details"][0]["nom"] == "B")
    assert float(b["cf_monthly_stab"]) < -259.0  # B lands on the target


def _real_year2(regime_fiscal: str) -> list[tuple[float, float, float]]:
    """(monthly rent, monthly income tax) of year 2 for each strategy of a default-context real search."""
    from src.application.run_context import RunContext
    from src.application.services.simulation import simulate_long_term_strategy
    from src.ui.app_controller import load_archetypes_from_disk, run_strategy_search

    ctx = RunContext(regime_fiscal=regime_fiscal)
    out = []
    for s in run_strategy_search(load_archetypes_from_disk("data/archetypes_2026_v1.json"), ctx, run_info={}):
        df, _ = simulate_long_term_strategy(
            s, duration_years=ctx.horizon_ans, market_hypotheses=ctx.market_hypotheses, tax_params=ctx.to_tax_params()
        )
        out.append((float(df.iloc[1]["Loyers Bruts"]) / 12, float(df.iloc[1]["Impôt Dû"]) / 12, ctx.tmi_pct))
    return out


def test_year2_income_tax_is_plausible_on_a_real_search():
    # T34 part C. LMNP réel: amortization wipes the taxable result, so no tax.
    lmnp = _real_year2("lmnp")
    assert lmnp
    assert all(tax < 1.0 for _, tax, _ in lmnp)
    # micro-BIC: 50 % abatement, then TMI + social levies (18.6 % since 2026) on what is left.
    from src.core.defaults import TAX_RATES

    micro = _real_year2("microbic")
    assert micro
    for rent, tax, tmi in micro:
        expected = 0.5 * rent * (tmi + TAX_RATES.ps_bic_pct) / 100.0
        assert abs(tax - expected) < 0.02 * expected + 1.0
