"""T23: tax regime engine (src/domain/tax), carryforwards and edge cases per regime (TAX_SPEC 4, 8)."""

from __future__ import annotations

import pytest

from src.application.services.simulation import simulate_long_term_strategy
from src.domain.tax import (
    DeficitPool,
    ExitContext,
    FoncierReel,
    LmnpReel,
    MicroBic,
    MicroFoncier,
    PropertyExit,
    RegimeId,
    SciIs,
    TaxConfig,
    TaxHousehold,
    TaxState,
    YearContext,
)


def ctx(year=1, pid=0, rent=0.0, charges=0.0, interest=0.0, ins=0.0, works=0.0, energy=0.0,
        building=0.0, works_amort=0.0, furniture=0.0) -> YearContext:
    return YearContext(
        year=year, property_id=pid, rent=rent, operating_charges=charges, interest=interest, loan_insurance=ins,
        works_paid=works, energy_works_paid=energy, amort_building=building, amort_works=works_amort,
        amort_furniture=furniture,
    )


# --------------------------------------------------------------------------- #
# Deficit pool
# --------------------------------------------------------------------------- #
def test_deficit_pool_fifo_and_ten_year_expiry() -> None:
    pool = DeficitPool(life_years=10)
    pool.add(1, 100.0)
    pool.add(2, 50.0)
    assert pool.use(120.0) == 120.0  # oldest first
    assert pool.items == [[2, 30.0]]
    assert pool.expire(12) == 0.0  # usable up to year 2 + 10
    assert pool.expire(13) == 30.0
    assert pool.total == 0.0


def test_unlimited_pool_never_expires() -> None:
    pool = DeficitPool(life_years=None)
    pool.add(1, 10.0)
    assert pool.expire(100) == 0.0 and pool.total == 10.0


# --------------------------------------------------------------------------- #
# LMNP réel
# --------------------------------------------------------------------------- #
def test_lmnp_ard_carried_without_limit_and_used_later() -> None:
    reg, st = LmnpReel(tmi_pct=30.0), TaxState()
    r1 = reg.compute_year(1, [ctx(rent=10_000, charges=4_000, building=10_000)], st)
    assert r1.taxable_base == 0.0 and st.prop(0).ard == pytest.approx(4_000.0)
    # Year 20, no current amortization: the old ARD still offsets the profit.
    r2 = reg.compute_year(20, [ctx(year=20, rent=10_000, charges=4_000)], st)
    assert r2.taxable_base == pytest.approx(2_000.0)
    assert st.prop(0).ard == 0.0


def test_lmnp_amortization_limited_before_prior_deficits() -> None:
    """TAX_SPEC 4.2 / 11.2: amortization is limited to R first, then prior BIC deficits are used."""
    reg, st = LmnpReel(tmi_pct=30.0), TaxState()
    reg.compute_year(1, [ctx(rent=1_000, charges=3_000)], st)  # BIC deficit 2,000
    res = reg.compute_year(2, [ctx(year=2, rent=10_000, charges=4_000, building=5_000)], st)
    # R = 6,000; amortization 5,000 deducted; benefit 1,000 offset by the deficit.
    assert res.detail["amortization_deducted"] == pytest.approx(5_000.0)
    assert res.taxable_base == 0.0
    assert st.deficits.total == pytest.approx(1_000.0)
    assert st.prop(0).ard == 0.0


def test_lmnp_bic_deficit_expires_after_ten_years() -> None:
    reg, st = LmnpReel(tmi_pct=30.0), TaxState()
    reg.compute_year(1, [ctx(rent=0, charges=1_000)], st)
    assert reg.compute_year(12, [ctx(year=12, rent=2_000)], st).taxable_base == pytest.approx(2_000.0)


def test_lmnp_ps_rate_is_18_6() -> None:
    res = LmnpReel(tmi_pct=30.0).compute_year(1, [ctx(rent=1_000)], TaxState())
    assert res.ir == pytest.approx(300.0) and res.ps == pytest.approx(186.0)


def test_lmnp_39c_limit_per_property_then_netting() -> None:
    """Each property's amortization <= its own R; profits and losses then net at foyer level."""
    reg, st = LmnpReel(tmi_pct=30.0), TaxState()
    a = ctx(pid=0, rent=10_000, charges=2_000, building=3_000)  # R 8,000 -> benefit 5,000
    b = ctx(pid=1, rent=1_000, charges=3_000, building=4_000)  # R -2,000 -> loss, amortization all ARD
    res = reg.compute_year(1, [a, b], st)
    assert res.taxable_base == pytest.approx(3_000.0)
    assert st.prop(1).ard == pytest.approx(4_000.0)
    assert st.prop(0).cum_amort_imm == pytest.approx(3_000.0)
    assert st.deficits.total == 0.0


# --------------------------------------------------------------------------- #
# Micro-BIC / micro-foncier
# --------------------------------------------------------------------------- #
def test_micro_bic_minimum_allowance_305() -> None:
    res = MicroBic(tmi_pct=30.0).compute_year(1, [ctx(rent=500)], TaxState())
    assert res.taxable_base == pytest.approx(195.0)


def test_micro_bic_30pct_non_classe_ceiling() -> None:
    m = MicroBic(tmi_pct=30.0, abattement_pct=30.0)
    assert m.ceiling == 15_000.0
    assert m.compute_year(1, [ctx(rent=10_000)], TaxState()).taxable_base == pytest.approx(7_000.0)
    assert MicroBic(tmi_pct=30.0).ceiling == 83_600.0


def test_micro_bic_rejects_old_71pct_rate() -> None:
    with pytest.raises(ValueError):
        MicroBic(tmi_pct=30.0, abattement_pct=71.0)


def test_micro_foncier_base_and_ps() -> None:
    res = MicroFoncier(tmi_pct=30.0).compute_year(1, [ctx(rent=10_000, charges=9_000, building=5_000)], TaxState())
    assert res.taxable_base == pytest.approx(7_000.0)
    assert res.ps == pytest.approx(7_000 * 0.172)


# --------------------------------------------------------------------------- #
# Foncier réel
# --------------------------------------------------------------------------- #
def test_foncier_deficit_cap_excludes_interest_part() -> None:
    reg, st = FoncierReel(tmi_pct=30.0), TaxState()
    # rent 5,000; interest 8,000; charges 2,000 -> deficit 5,000, of which only 2,000 is non-interest.
    res = reg.compute_year(3, [ctx(year=3, rent=5_000, charges=2_000, interest=8_000)], st)
    assert res.detail["deficit_imputed_on_global_income"] == pytest.approx(2_000.0)
    assert res.ir == pytest.approx(-600.0) and res.ps == 0.0
    assert st.deficits.total == pytest.approx(3_000.0)


def test_foncier_deficit_capped_at_10700() -> None:
    reg, st = FoncierReel(tmi_pct=30.0), TaxState()
    res = reg.compute_year(3, [ctx(year=3, rent=1_000, works=31_000)], st)
    assert res.detail["deficit_imputed_on_global_income"] == pytest.approx(10_700.0)
    assert st.deficits.total == pytest.approx(19_300.0)


@pytest.mark.parametrize(("year", "cap"), [(1, 21_400.0), (2, 21_400.0), (3, 10_700.0)])
def test_foncier_energy_works_cap_until_2027(year: int, cap: float) -> None:
    """Energy works paid by 31/12/2027 lift the cap to 21,400 (year 1 = 2026)."""
    reg = FoncierReel(tmi_pct=30.0)
    res = reg.compute_year(year, [ctx(year=year, rent=1_000, works=40_000, energy=20_000)], TaxState())
    assert res.detail["deficit_imputed_on_global_income"] == pytest.approx(cap)


def test_foncier_deficit_only_offsets_fonciers_and_expires() -> None:
    reg, st = FoncierReel(tmi_pct=30.0), TaxState()
    reg.compute_year(1, [ctx(rent=0, interest=5_000)], st)  # interest-only deficit: all carried forward
    assert st.deficits.total == pytest.approx(5_000.0)
    assert reg.compute_year(5, [ctx(year=5, rent=2_000)], st).taxable_base == 0.0
    assert reg.compute_year(12, [ctx(year=12, rent=4_000)], st).taxable_base == pytest.approx(4_000.0)  # expired


def test_foncier_recapture_on_sale_within_three_years() -> None:
    reg, st = FoncierReel(tmi_pct=30.0), TaxState()
    st.global_imputed_by_year = {1: 10_700.0, 2: 0.0}
    assert reg.deficit_recapture(3, st) == pytest.approx(3_210.0)  # sold before 31/12 of year 1 + 3
    assert reg.deficit_recapture(4, st) == 0.0


# --------------------------------------------------------------------------- #
# SCI IS
# --------------------------------------------------------------------------- #
def test_is_reduced_rate_threshold() -> None:
    sci = SciIs()
    assert sci.is_tax(42_500) == pytest.approx(6_375.0)
    assert sci.is_tax(52_500) == pytest.approx(6_375.0 + 2_500.0)
    assert sci.is_tax(-1) == 0.0


def test_sci_amortization_creates_unlimited_losses() -> None:
    sci, st = SciIs(), TaxState()
    res = sci.compute_year(1, [ctx(rent=5_000, building=8_000)], st)
    assert res.detail["accounting_result"] == pytest.approx(-3_000.0) and res.is_ == 0.0
    assert sci.compute_year(40, [ctx(year=40, rent=4_000)], st).taxable_base == pytest.approx(1_000.0)


def test_sci_exit_on_net_book_value_and_pfu_switch() -> None:
    st = TaxState()
    SciIs().compute_year(1, [ctx(rent=0, building=10_000)], st)  # loss 10,000, cum amort 10,000
    prop = PropertyExit(property_id=0, name="a", price=100_000, notary_fees=0, works=0, furniture=0,
                        sale_price=120_000, sale_fees=0)
    ex_ctx = ExitContext(years_held=1, properties=(prop,), debt_repaid=0.0, cum_cash_flow=0.0, apport=100_000)
    ex = SciIs(distribute_at_exit=False).compute_exit(ex_ctx, st)
    assert ex.details["net_book_value"] == pytest.approx(90_000.0)
    assert ex.details["plus_value_taxable_after_losses"] == pytest.approx(20_000.0)  # 30,000 - 10,000 loss
    assert ex.is_ == pytest.approx(3_000.0) and ex.pfu == 0.0
    st2 = TaxState()
    SciIs().compute_year(1, [ctx(rent=0, building=10_000)], st2)
    ex2 = SciIs(distribute_at_exit=True).compute_exit(ex_ctx, st2)
    assert ex2.pfu == pytest.approx((120_000 - 3_000 - 100_000) * 0.314)


# --------------------------------------------------------------------------- #
# Household: one regime per category (TAX_SPEC 8)
# --------------------------------------------------------------------------- #
def test_household_one_regime_per_category() -> None:
    cfg = TaxConfig(furnished_regime=RegimeId.MICRO_BIC, unfurnished_regime=RegimeId.MICRO_FONCIER)
    hh = TaxHousehold.build(cfg, [True, False, True], {"furnished": 20_000, "unfurnished": 10_000})
    assert hh.regime_of(0) is RegimeId.MICRO_BIC and hh.regime_of(2) is RegimeId.MICRO_BIC
    assert hh.regime_of(1) is RegimeId.MICRO_FONCIER
    res = hh.compute_year(1, [ctx(pid=0, rent=10_000), ctx(pid=1, rent=10_000), ctx(pid=2, rent=10_000)])
    assert res.taxable_base == pytest.approx(10_000 + 7_000)
    assert res.ps == pytest.approx(10_000 * 0.186 + 7_000 * 0.172)


def test_micro_bic_ceiling_falls_back_to_reel_with_warning() -> None:
    cfg = TaxConfig(furnished_regime=RegimeId.MICRO_BIC)
    hh = TaxHousehold.build(cfg, [True], {"furnished": 90_000})
    assert hh.regime_of(0) is RegimeId.LMNP_REEL
    assert any("Micro-BIC impossible" in w for w in hh.warnings)
    assert any("LMP" in w for w in hh.warnings)


def test_micro_foncier_ceiling_falls_back_to_reel_with_warning() -> None:
    cfg = TaxConfig(unfurnished_regime=RegimeId.MICRO_FONCIER)
    hh = TaxHousehold.build(cfg, [False], {"unfurnished": 15_001})
    assert hh.regime_of(0) is RegimeId.FONCIER_REEL
    assert any("Micro-foncier impossible" in w for w in hh.warnings)
    assert TaxHousehold.build(cfg, [False], {"unfurnished": 15_000}).regime_of(0) is RegimeId.MICRO_FONCIER


def test_sci_is_overrides_both_categories() -> None:
    hh = TaxHousehold.build(TaxConfig(sci_is=True), [True, False])
    assert hh.regime_of(0) is RegimeId.SCI_IS and hh.regime_of(1) is RegimeId.SCI_IS
    assert hh.amortizes(1)


# --------------------------------------------------------------------------- #
# Through the simulation engine
# --------------------------------------------------------------------------- #
def _prop(meuble: bool, rent: float = 900.0, works: float = 0.0, name: str = "b") -> dict:
    return {
        "nom_bien": name, "prix_achat_bien": 150_000.0, "frais_notaire": 12_000.0, "budget_travaux": works,
        "renovation_energetique": 0.0, "mobilier": 5_000.0 if meuble else 0.0, "credit_final": 140_000.0,
        "taux_pret": 3.5, "duree_pret": 20, "assurance_ann_pct": 0.3, "loyer_mensuel_initial": rent,
        "charges_const_mth0": 100.0, "tf_const_mth0": 70.0, "frais_gestion_pct": 0.0, "provision_pct": 0.0,
        "meuble": meuble,
    }


def test_unfurnished_never_amortizes_even_with_lmnp_choice() -> None:
    """Regression (TAX_SPEC 10 item 7): before T23, regime "lmnp" amortized unfurnished property."""
    strategy = {"apport_total": 30_000.0, "details": [_prop(False)]}
    df, bilan = simulate_long_term_strategy(strategy, 10, {}, {"regime_fiscal": "lmnp", "regime_foncier": "reel"})
    assert (df["Amortissements"] == 0).all()
    assert bilan["regimes"] == {"unfurnished": "foncier_reel"}
    assert bilan["plus_value"]["amortization_addback"] == 0.0


def test_mixed_portfolio_uses_both_categories() -> None:
    strategy = {"apport_total": 60_000.0, "details": [_prop(True, name="m"), _prop(False, name="n")]}
    df, bilan = simulate_long_term_strategy(strategy, 12, {}, {"regime_fiscal": "lmnp", "regime_foncier": "micro"})
    assert bilan["regimes"] == {"furnished": "lmnp_reel", "unfurnished": "micro_foncier"}
    assert [b["nom"] for b in bilan["plus_value_par_bien"]] == ["m", "n"]
    # Only the furnished property is amortized and added back.
    assert bilan["plus_value_par_bien"][0]["amortization_addback"] > 0
    assert bilan["plus_value_par_bien"][1]["amortization_addback"] == 0
    assert df["IR"].sum() + df["PS"].sum() == pytest.approx(df["Impôt Dû"].sum())


def test_engine_sci_is_reports_is_and_pfu() -> None:
    strategy = {"apport_total": 30_000.0, "details": [_prop(False, rent=1_100)]}
    df, bilan = simulate_long_term_strategy(strategy, 15, {}, {"sci_is": True})
    assert bilan["regimes"] == {"sci": "sci_is"}
    assert (df["IR"] == 0).all() and (df["PS"] == 0).all()
    assert (df["Amortissements"] < 0).all()
    assert bilan["exit_tax"]["pfu"] > 0 and bilan["plus_value"]["net_book_value"] > 0
    assert bilan["impot_pv"] == pytest.approx(bilan["exit_tax"]["is"] + bilan["exit_tax"]["pfu"])


def test_engine_micro_bic_ceiling_warning_in_bilan() -> None:
    strategy = {"apport_total": 30_000.0, "details": [_prop(True, rent=8_000)]}  # 96,000 EUR/yr
    _, bilan = simulate_long_term_strategy(strategy, 5, {}, {"regime_fiscal": "microbic"})
    assert bilan["regimes"] == {"furnished": "lmnp_reel"}
    assert bilan["tax_warnings"]
