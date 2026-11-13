"""T22: land ratio (ratio_terrain_pct) and amortization durations, end to end."""

from __future__ import annotations

import pytest

from src.application.run_context import RunContext
from src.application.services.brick_factory import (
    FinancingConfig,
    OperatingConfig,
    build_financing_config,
    create_investment_bricks,
)
from src.application.services.simulation import (
    IRACalculator,
    MarketHypotheses,
    SimulationEngine,
    TaxParams,
    simulate_long_term_strategy,
)
from src.application.services.strategy_finder import StrategyFinder
from src.core.defaults import TAX
from src.domain.calculator.financial import generate_amortization_schedule
from src.domain.models.archetype import ArchetypeV2
from src.ui.app_controller import load_archetypes

EVAL_PARAMS = {
    "tmi_pct": 30,
    "regime_fiscal": "lmnp",
    "frais_vente_pct": 6.0,
    "apply_ira": True,
    "ira_cap_pct": 3.0,
    "cfe_par_bien_ann": 150,
    "hypotheses_marche": {"appreciation_bien_pct": 2.0, "revalo_loyer_pct": 1.5, "inflation_charges_pct": 2.0},
    "finance_preset_name": "Équilibré",
    "max_properties": 1,
}


def _furnished_archetype() -> dict:
    """The best-yielding furnished archetype, so that the tax base is positive once amortization shrinks."""
    furnished = [a for a in load_archetypes() if a.get("meuble") and a.get("valeur_mobilier", 0) > 0]
    return dict(max(furnished, key=lambda a: a["loyer_m2"] / a["prix_m2"]))


def _best_strategy(ratio_override: float | None) -> dict:
    fin = FinancingConfig(credit_rates={20: 3.6}, ratio_terrain_pct=ratio_override)
    bricks = create_investment_bricks([_furnished_archetype()], fin, OperatingConfig())
    finder = StrategyFinder(
        bricks=bricks, apport_disponible=200000.0, cash_flow_cible=-5000.0,
        tolerance=500.0, mode_cf="min", qualite_weight=0.25,
    )
    strategies = finder.find_strategies(eval_params=EVAL_PARAMS, horizon_years=20, top_n=1)
    assert strategies
    return strategies[0]


def test_land_ratio_changes_tax_through_strategy_finder() -> None:
    """More land means less amortization, hence more tax, hence a lower after-tax cash flow (StrategyFinder path)."""
    low = _best_strategy(10.0)
    high = _best_strategy(90.0)
    assert low["details"][0]["ratio_terrain_pct"] == 10.0
    assert high["details"][0]["ratio_terrain_pct"] == 90.0
    df_low, _ = simulate_long_term_strategy(low, 20, EVAL_PARAMS["hypotheses_marche"], EVAL_PARAMS)
    df_high, _ = simulate_long_term_strategy(high, 20, EVAL_PARAMS["hypotheses_marche"], EVAL_PARAMS)
    assert df_high["Impôt Dû"].sum() > df_low["Impôt Dû"].sum()
    assert df_high["Cash-Flow Net d'Impôt"].sum() < df_low["Cash-Flow Net d'Impôt"].sum()


def _prop() -> dict:
    return {
        "prix_achat_bien": 100000, "budget_travaux": 10000, "mobilier": 5000, "renovation_energetique": 0,
        "frais_notaire": 8000, "credit_final": 90000, "taux_pret": 3.6, "duree_pret": 25,
        "assurance_ann_pct": 0.35, "loyer_mensuel_initial": 700, "charges_const_mth0": 60,
        "tf_const_mth0": 40, "frais_gestion_pct": 5.0, "provision_pct": 3.0,
    }


def test_25_year_lmnp_deducts_furniture_exactly_once() -> None:
    engine = SimulationEngine(
        MarketHypotheses(), TaxParams(regime_fiscal="lmnp"), IRACalculator(apply_ira=False), ratio_terrain_pct=15.0
    )
    strategy = {"apport_total": 20000, "details": [_prop()]}
    schedules = [generate_amortization_schedule(90000, 3.6, 300, 0.35)]
    df, _ = engine.simulate(strategy, 25, schedules)
    amort = -df["Amortissements"].to_numpy()
    building, works, furniture = 108000 * 0.85 / 30, 10000 / 15, 5000 / 10
    for year, value in enumerate(amort, start=1):
        expected = (building if year <= 30 else 0) + (works if year <= 15 else 0) + (furniture if year <= 10 else 0)
        assert value == pytest.approx(expected, abs=0.01), year
    assert amort.sum() - building * 25 - works * 15 == pytest.approx(5000, abs=0.01)


def test_building_amortization_stops_after_its_duration() -> None:
    engine = SimulationEngine(
        MarketHypotheses(), TaxParams(regime_fiscal="lmnp"), IRACalculator(apply_ira=False),
        amort_immo_years=10, ratio_terrain_pct=0.0,
    )
    schedules = [generate_amortization_schedule(90000, 3.6, 300, 0.35)]
    df, _ = engine.simulate({"apport_total": 20000, "details": [_prop()]}, 12, schedules)
    amort = -df["Amortissements"].to_numpy()
    assert amort[9] > 0
    assert amort[10] == pytest.approx(10000 / 15, abs=0.01)  # building done, works still running


def test_archetype_data_has_land_ratio() -> None:
    archetypes = load_archetypes()
    assert archetypes
    for a in archetypes:
        ArchetypeV2.model_validate(a)
        expected = 20.0 if a["ville"] in ("Lyon", "Annecy") else 15.0
        assert a["ratio_terrain_pct"] == expected, a["nom"]
    assert ArchetypeV2.model_fields["ratio_terrain_pct"].default == TAX.ratio_terrain_pct == 15.0


def test_run_context_override_reaches_bricks_and_unset_uses_archetype() -> None:
    archetype = next(a for a in load_archetypes() if a["ville"] == "Lyon")
    unset = RunContext()
    assert unset.ratio_terrain_pct is None
    bricks = create_investment_bricks(
        [dict(archetype)], build_financing_config(unset.to_credit_params()), OperatingConfig()
    )
    assert {b["ratio_terrain_pct"] for b in bricks} == {20.0}

    ctx = RunContext(ratio_terrain_pct=35.0)
    assert ctx.to_tax_params()["ratio_terrain_pct"] == 35.0
    assert ctx.to_eval_params()["ratio_terrain_pct"] == 35.0
    bricks = create_investment_bricks(
        [dict(archetype)], build_financing_config(ctx.to_credit_params()), OperatingConfig()
    )
    assert {b["ratio_terrain_pct"] for b in bricks} == {35.0}
    assert RunContext.from_dict(ctx.to_dict()).ratio_terrain_pct == 35.0
