"""T62: tax regime comparison (pure service)."""

from __future__ import annotations

from typing import Any

import pytest

from src.application.run_context import RunContext
from src.application.services.regime_comparison import (
    RegimeKey,
    RegimeResult,
    compare_regimes,
    eligible_regimes,
    find_break_even_year,
)
from src.application.services.simulation import simulate_long_term_strategy


def _prop(meuble: bool, name: str = "b", rent: float = 900.0) -> dict[str, Any]:
    return {
        "nom_bien": name,
        "prix_achat_bien": 150_000.0,
        "frais_notaire": 12_000.0,
        "budget_travaux": 0.0,
        "renovation_energetique": 0.0,
        "mobilier": 5_000.0 if meuble else 0.0,
        "credit_final": 140_000.0,
        "taux_pret": 3.5,
        "duree_pret": 20,
        "assurance_ann_pct": 0.3,
        "loyer_mensuel_initial": rent,
        "charges_const_mth0": 100.0,
        "tf_const_mth0": 70.0,
        "frais_gestion_pct": 0.0,
        "provision_pct": 0.0,
        "meuble": meuble,
    }


def _strategy(*props: dict[str, Any]) -> dict[str, Any]:
    return {"apport_total": 30_000.0 * len(props), "details": list(props)}


CTX = RunContext(horizon_ans=10, apply_ira=False)


def _result(label: str, cum: list[float], wealth: float, user: bool = False) -> RegimeResult:
    return RegimeResult(
        key=RegimeKey("lmnp" if user else "microbic", None),
        label=label,
        is_user_choice=user,
        years=list(range(1, len(cum) + 1)),
        tax=[0.0] * len(cum),
        cash_flow=[0.0] * len(cum),
        cumulative_cf=cum,
        tri_annuel=None,
        enrichissement_net=wealth,
        exit_tax=0.0,
    )


def test_furnished_only_has_no_foncier_variants() -> None:
    keys = eligible_regimes(_strategy(_prop(True)))
    assert all(k.foncier is None for k in keys)
    assert {k.label for k in keys} == {"LMNP réel", "Micro-BIC", "SCI à l'IS"}


def test_unfurnished_only_has_no_furnished_variants() -> None:
    keys = eligible_regimes(_strategy(_prop(False)))
    assert all(k.furnished is None for k in keys)
    assert {k.label for k in keys} == {"Foncier réel", "Micro-foncier", "SCI à l'IS"}


def test_mixed_enumerates_product_plus_sci() -> None:
    assert len(eligible_regimes(_strategy(_prop(True), _prop(False)))) == 5


@pytest.mark.parametrize("meuble", [True, False])
def test_user_regime_matches_normal_simulation(meuble: bool) -> None:
    strategy = _strategy(_prop(meuble))
    cmp = compare_regimes(strategy, CTX)
    df, bilan = simulate_long_term_strategy(strategy, CTX.horizon_ans, CTX.market_hypotheses, CTX.to_tax_params())
    assert cmp.user.is_user_choice
    assert cmp.user.enrichissement_net == pytest.approx(bilan["enrichissement_net"])
    assert cmp.user.tri_annuel == bilan["tri_annuel"]
    assert cmp.user.cash_flow == pytest.approx(list(df["Cash-Flow Net d'Impôt"]))
    assert cmp.user.tax == pytest.approx(list(df["Impôt Dû"]))
    assert cmp.user.exit_tax == pytest.approx(bilan["exit_tax"]["total"])
    assert cmp.user.cumulative_cf[-1] == pytest.approx(df["Cash-Flow Net d'Impôt"].sum())


def test_user_sci_choice_is_flagged() -> None:
    cmp = compare_regimes(_strategy(_prop(True)), CTX.model_copy(update={"sci_is": True}))
    assert cmp.user.key.sci_is


def test_best_is_max_enrichissement() -> None:
    cmp = compare_regimes(_strategy(_prop(True), _prop(False)), CTX)
    assert cmp.best_by_wealth.enrichissement_net == max(r.enrichissement_net for r in cmp.results)
    assert cmp.wealth_gain_vs_user >= 0.0


def test_micro_bic_ceiling_warning_is_returned() -> None:
    cmp = compare_regimes(_strategy(_prop(True, rent=8_000.0)), CTX)
    micro = next(r for r in cmp.results if r.key.furnished == "microbic" and not r.key.sci_is)
    assert any("Micro-BIC impossible" in w for w in micro.warnings)


def test_break_even_year_computed() -> None:
    user = _result("user", [10.0, 20.0, 30.0, 40.0], 100.0, user=True)
    best = _result("best", [0.0, 15.0, 35.0, 60.0], 200.0)
    assert find_break_even_year(best, user) == 3  # 35 > 30 first at year 3


def test_break_even_none_when_never_or_same_regime() -> None:
    user = _result("user", [10.0, 20.0], 100.0, user=True)
    behind = _result("best", [0.0, 5.0], 200.0)
    assert find_break_even_year(behind, user) is None
    assert find_break_even_year(user, user) is None
