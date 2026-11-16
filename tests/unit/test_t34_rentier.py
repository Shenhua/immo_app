"""T34: Rentier mode (use_full_capital) and the tax-correction loop convergence."""
from __future__ import annotations

from src.application.services.strategy_finder import StrategyFinder
from src.domain.models.brick import InvestmentBrick

_F = {
    "charges_const_mth0": 50.0, "tf_const_mth0": 50.0, "frais_gestion_pct": 0.0, "budget_travaux": 0.0,
    "mobilier": 0.0, "frais_notaire": 7000.0, "taux_annuel_pct": 3.5, "duree_credit_mois": 240,
}


def _bricks() -> list[InvestmentBrick]:
    return [
        InvestmentBrick(nom="A", apport_min=10000, prix_achat_bien=100000, capital_emprunte=90000,
                        loyer_mensuel_initial=500, **_F),
        InvestmentBrick(nom="B", apport_min=20000, prix_achat_bien=200000, capital_emprunte=180000,
                        loyer_mensuel_initial=1000, **_F),
    ]


def _run(budget: float, target: float, full: bool, max_props: int = 1) -> list[dict]:
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
