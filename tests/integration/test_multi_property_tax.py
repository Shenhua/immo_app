"""Multi-property tax regression tests (Phase 2 gate review, scenarios a, b, e, g).

Households with several properties: category independence (furnished vs unfurnished), micro
ceilings applied across properties, déficit foncier and its recapture, SCI IS with several properties.
Explicit euro figures were verified by hand in the review; the rest are invariants.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

from src.application.services.simulation import build_engine
from src.domain.calculator.financial import generate_amortization_schedule

Prop = tuple[dict[str, Any], float]  # (property dict, down payment)
Run = Callable[..., tuple[pd.DataFrame, dict[str, Any]]]


def prop(
    name: str, price: float, rent: float, *, meuble: bool = True, rate: float = 3.5, years: int = 20,
    works: float = 0.0, energy: float = 0.0, furn: float = 0.0, land: float = 15.0, copro: float = 1200,
    tf: float = 1000, vac: float = 0.0, mois: int = 0,
) -> Prop:
    notary = price * 0.08
    cost = price + notary + works + energy + furn
    loan = cost * 0.9
    return {
        "nom_bien": name, "prix_achat_bien": price, "frais_notaire": notary, "budget_travaux": works,
        "renovation_energetique": energy, "mobilier": furn, "ratio_terrain_pct": land, "credit_final": loan,
        "taux_pret": rate, "duree_pret": years, "assurance_ann_pct": 0.3, "loyer_mensuel_initial": rent,
        "charges_const_mth0": copro / 12, "tf_const_mth0": tf / 12, "frais_gestion_pct": 0.0,
        "provision_pct": vac, "meuble": meuble, "mois_travaux": mois, "maintenance_m2_an": 0.0,
        "gli_pct": 0.0, "surface": 40,
    }, cost - loan


@pytest.fixture
def run() -> Run:
    """Run a household through the engine (flat market, TMI 30, no IRA): ``run(props, horizon, tax, market)``."""

    def _run(
        props: list[Prop],
        horizon: int,
        tax: dict[str, Any] | None = None,
        market: dict[str, float] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        m = {"appreciation_bien_pct": 2.0, "revalo_loyer_pct": 0.0, "inflation_charges_pct": 0.0, "tf_growth_pct": 0.0}
        m.update(market or {})
        t: dict[str, Any] = {"tmi_pct": 30.0, "apply_ira": False, "cfe_par_bien_ann": 300, "frais_vente_pct": 5.0}
        t.update(tax or {})
        details = [p for p, _ in props]
        strategy = {"details": details, "apport_total": sum(a for _, a in props)}
        schedules = [
            generate_amortization_schedule(
                float(p["credit_final"]), float(p["taux_pret"]), int(p["duree_pret"]) * 12, float(p["assurance_ann_pct"])
            )
            for p in details
        ]
        return build_engine(m, t).simulate(strategy, horizon, schedules)

    return _run


# --------------------------------------------------------------------------- #
# a: mixed household, category independence
# --------------------------------------------------------------------------- #
F1 = prop("F1", 150000, 750, furn=5000)
F2 = prop("F2", 100000, 550, furn=4000, works=10000)
U1 = prop("U1", 120000, 600, meuble=False, works=30000)


@pytest.mark.parametrize("horizon", [8, 25, 35])
def test_mixed_household_equals_sum_of_categories(run: Run, horizon: int) -> None:
    """Furnished (LMNP réel) and unfurnished (foncier réel) never offset each other."""
    df, b = run([F1, F2, U1], horizon)
    dff, bf = run([F1, F2], horizon)
    dfu, bu = run([U1], horizon)

    assert b["regimes"] == {"furnished": "lmnp_reel", "unfurnished": "foncier_reel"}
    assert df["Impôt Dû"].sum() == pytest.approx(dff["Impôt Dû"].sum() + dfu["Impôt Dû"].sum(), abs=0.01)
    assert b["impot_pv"] == pytest.approx(bf["impot_pv"] + bu["impot_pv"], abs=0.01)
    assert b["enrichissement_net"] == pytest.approx(bf["enrichissement_net"] + bu["enrichissement_net"], abs=0.01)


def test_mixed_household_hand_checked_figures(run: Run) -> None:
    df, b = run([F1, F2, U1], 25)
    y1 = b["fiscal_annuel"][0]
    # Year 1: unfurnished deficit 10,700 imputed on global income at TMI 30 % -> refund 3,210 (CGI 156 I 3).
    assert y1["deficit_imputed_on_global_income"] == pytest.approx(10_700, abs=1)
    assert df.iloc[0]["IR"] == pytest.approx(-3_210, abs=1)
    assert df.iloc[0]["PS"] == pytest.approx(0, abs=0.01)
    # LMNP amortization is capped by the result before amortization; the rest goes to the ARD stock.
    assert y1["amortization_current_year"] == pytest.approx(9_217, abs=1)
    assert y1["amortization_deducted"] == pytest.approx(1_462, abs=1)
    assert y1["ard_stock_end"] == pytest.approx(7_755, abs=1)
    assert y1["foncier_deficit_stock_end"] == pytest.approx(19_678, abs=1)
    # Exit at 25 years: PS only (IR allowance 100 % from year 22).
    assert b["exit_tax"]["ir"] == 0
    assert b["exit_tax"]["total"] == pytest.approx(20_287, abs=1)
    totals = {pp["nom"]: pp["total_tax"] for pp in b["plus_value_par_bien"]}
    assert totals["F1"] == pytest.approx(10_799, abs=1)
    assert totals["F2"] == pytest.approx(6_436, abs=1)
    assert totals["U1"] == pytest.approx(3_052, abs=1)


def test_amortization_addback_only_on_furnished(run: Run) -> None:
    _, b = run([F1, F2, U1], 25)
    addback = {pp["nom"]: pp["amortization_addback"] for pp in b["plus_value_par_bien"]}
    assert addback["F1"] == pytest.approx(90_242, abs=1)
    assert addback["F2"] == pytest.approx(50_297, abs=1)
    assert addback["U1"] == 0


# --------------------------------------------------------------------------- #
# b: micro ceilings across properties
# --------------------------------------------------------------------------- #
def test_micro_bic_ceiling_applies_to_household_receipts(run: Run) -> None:
    three = [prop(f"F{i}", 400000, 2500, furn=10000) for i in range(3)]  # 90,000 > 83,600
    _, b = run(three, 10, {"regime_fiscal": "microbic"})
    assert b["regimes"] == {"furnished": "lmnp_reel"}
    assert any("Micro-BIC impossible" in w for w in b["tax_warnings"])

    df, b = run(three[:2], 10, {"regime_fiscal": "microbic"})  # 60,000 < ceiling
    assert b["regimes"] == {"furnished": "micro_bic"}
    # 50 % allowance, TMI 30 % + PS 18.6 %.
    assert df.iloc[1]["Impôt Dû"] == pytest.approx(60_000 * 0.5 * (0.30 + 0.186), abs=1)


def test_micro_foncier_ceiling_applies_to_household_receipts(run: Run) -> None:
    two = [prop(f"U{i}", 100000, 700, meuble=False) for i in range(2)]  # 16,800 > 15,000
    _, b = run(two, 10, {"regime_foncier": "micro"})
    assert b["regimes"] == {"unfurnished": "foncier_reel"}
    assert any("Micro-foncier impossible" in w for w in b["tax_warnings"])


def test_micro_foncier_ceiling_tracks_rent_indexation(run: Run) -> None:
    two = [prop(f"U{i}", 100000, 600, meuble=False, vac=5) for i in range(2)]
    _, b = run(two, 20, {"regime_foncier": "micro"}, {"revalo_loyer_pct": 1.5})
    assert b["regimes"] == {"unfurnished": "foncier_reel"}
    assert any("Micro-foncier impossible" in w for w in b["tax_warnings"])


def test_micro_bic_base_is_receipts_net_of_vacancy_and_works_months(run: Run) -> None:
    f = prop("F", 150000, 1000, furn=5000, works=20000, vac=5, mois=3)
    df, _ = run([f], 10, {"regime_fiscal": "microbic"})
    assert df.iloc[0]["Loyers Bruts"] == pytest.approx(1000 * 9 * 0.95, abs=0.01)
    assert df.iloc[0]["Impôt Dû"] == pytest.approx(8550 * 0.5 * 0.486, abs=1)
    assert df.iloc[1]["Impôt Dû"] == pytest.approx(11400 * 0.5 * 0.486, abs=1)


def test_micro_bic_minimum_allowance_305(run: Run) -> None:
    df, _ = run([prop("F", 30000, 50, furn=0)], 5, {"regime_fiscal": "microbic"})
    assert df.iloc[0]["Impôt Dû"] == pytest.approx((600 - 305) * 0.486, abs=0.01)


# --------------------------------------------------------------------------- #
# SCI IS with several properties
# --------------------------------------------------------------------------- #
SCI_A = prop("A", 200000, 1100, meuble=False)
SCI_B = prop("B", 150000, 900, furn=5000)


@pytest.mark.parametrize("horizon", [10, 20, 35])
def test_sci_is_household_equals_sum_of_properties(run: Run, horizon: int) -> None:
    df, b = run([SCI_A, SCI_B], horizon, {"sci_is": True})
    da, ba = run([SCI_A], horizon, {"sci_is": True})
    db, bb = run([SCI_B], horizon, {"sci_is": True})

    assert b["regimes"] == {"sci": "sci_is"}
    # Furnished or not, the SCI is one IS category; here no property has losses, so yearly IS is additive.
    assert df["Impôt Dû"].sum() == pytest.approx(da["Impôt Dû"].sum() + db["Impôt Dû"].sum(), abs=0.01)
    # Exit: the boni is taxed on the pooled liquidation, so the pooled exit tax is at least the separate sum.
    assert b["impot_pv"] >= ba["impot_pv"] + bb["impot_pv"] - 0.01


@pytest.mark.parametrize(
    ("horizon", "is_exit", "total_exit"),
    [(10, 22_729, 49_478), (20, 79_030, 160_945), (35, 149_746, 345_427)],
)
def test_sci_is_hand_checked_exit(run: Run, horizon: int, is_exit: float, total_exit: float) -> None:
    _, b = run([SCI_A, SCI_B], horizon, {"sci_is": True})
    assert b["exit_tax"]["is"] == pytest.approx(is_exit, abs=1)
    assert b["exit_tax"]["total"] == pytest.approx(total_exit, abs=1)


# --------------------------------------------------------------------------- #
# Déficit foncier and recapture
# --------------------------------------------------------------------------- #
def test_deficit_foncier_global_cap_and_recapture(run: Run) -> None:
    u1 = prop("U1", 100000, 500, meuble=False, works=5000, energy=15000)
    u2 = prop("U2", 100000, 500, meuble=False, works=5000, energy=15000)
    _, b = run([u1, u2], 3)
    fiscal = b["fiscal_annuel"]
    # Year 1: -41,026 net result; 21,400 (2 x 10,700 energy-works cap) imputed on global income at TMI 30 %.
    assert fiscal[0]["net_foncier_result"] == pytest.approx(-41_026, abs=1)
    assert fiscal[0]["deficit_imputed_on_global_income"] == pytest.approx(21_400, abs=1)
    assert fiscal[0]["tax"] == pytest.approx(-6_420, abs=1)
    assert fiscal[0]["foncier_deficit_stock_end"] == pytest.approx(19_626, abs=1)
    # Selling within 3 years of the imputation claws the imputed deficit back at TMI.
    assert b["plus_value"]["deficit_recapture"] == pytest.approx(6_773.29, abs=1)
    assert b["exit_tax"]["deficit_recapture"] == pytest.approx(6_773, abs=1)


def test_is_furnished_rule() -> None:
    """m2: one furnished rule (explicit `meuble` wins, else `mode_loyer` "nu*" = unfurnished, else furnished)."""
    from src.domain.models.archetype import ArchetypeV2, is_furnished

    assert is_furnished({"meuble": False, "mode_loyer": "meuble_classique"}) is False
    assert is_furnished({"meuble": True, "mode_loyer": "nu"}) is True
    assert is_furnished({"mode_loyer": "nu_classique"}) is False
    assert is_furnished({"mode_loyer": "meuble_boost"}) is True
    assert is_furnished({}) is True
    base = {"nom": "x", "ville": "v", "surface": 30, "prix_m2": 1000, "loyer_m2": 10}
    assert ArchetypeV2(**base, mode_loyer="nu").meuble is False
    assert ArchetypeV2(**base, mode_loyer="meuble_classique").meuble is True
    assert ArchetypeV2(**base, mode_loyer="nu", meuble=True).meuble is True


def test_works_majoration_depends_on_foncier_regime(run: Run) -> None:
    u = prop("U", 100000, 500, meuble=False, works=30000)
    _, micro = run([u], 10, {"regime_foncier": "micro"})
    _, reel = run([u], 10, {"regime_foncier": "reel"})
    # Micro: works were never deducted, so the real works increase the acquisition price.
    assert micro["plus_value_par_bien"][0]["works_majoration"] == pytest.approx(30_000)
    # Réel: deducted works are excluded, only the 15 % flat rate applies.
    assert reel["plus_value_par_bien"][0]["works_majoration"] == pytest.approx(15_000)
