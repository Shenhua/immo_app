"""T21: plus-value of individuals (TAX_SPEC section 3)."""

from __future__ import annotations

from typing import Any

import pytest

from src.application.run_context import RunContext
from src.application.services.simulation import simulate_long_term_strategy
from src.domain.models.archetype import Archetype
from src.domain.tax.plus_value import (
    PlusValueBreakdown,
    abattement_ir_pct,
    abattement_ps_pct,
    compute_plus_value,
    surtax,
)


@pytest.mark.parametrize(
    ("years", "ir", "ps"),
    [
        (5, 0.0, 0.0),
        (6, 6.0, 1.65),
        (17, 72.0, 19.8),
        (21, 96.0, 26.4),
        (22, 100.0, 28.0),
        (30, 100.0, 100.0),
    ],
)
def test_holding_period_allowances(years: int, ir: float, ps: float) -> None:
    assert abattement_ir_pct(years) == pytest.approx(ir)
    assert abattement_ps_pct(years) == pytest.approx(ps)


@pytest.mark.parametrize(
    ("base_ir", "expected"),
    [
        (50_000, 0.0),
        (55_000, 0.02 * 55_000 - 5_000 / 20),
        (80_000, 1_600.0),
        (105_000, 0.03 * 105_000 - 5_000 / 10),
        (300_000, 18_000.0),
    ],
)
def test_surtax_table(base_ir: float, expected: float) -> None:
    assert surtax(base_ir) == pytest.approx(expected)


def _pv(**kw: Any) -> PlusValueBreakdown:
    args: dict[str, Any] = {
        "price": 200_000.0,
        "notary_fees": 16_000.0,
        "works": 0.0,
        "sale_price": 400_000.0,
        "sale_fees": 20_000.0,
        "years_held": 10,
    }
    args.update(kw)
    return compute_plus_value(**args)


def test_sale_costs_reduce_the_gain() -> None:
    assert _pv(sale_fees=0.0).plus_value_brute - _pv().plus_value_brute == pytest.approx(20_000.0)


def test_flat_rates_take_the_larger_amount() -> None:
    res = _pv(notary_fees=5_000.0, works=10_000.0, years_held=10)
    assert res.acquisition_fees_retained == pytest.approx(15_000.0)  # 7.5 % x 200,000
    assert res.works_majoration == pytest.approx(30_000.0)  # 15 % x 200,000
    # no 15 % flat rate within 5 years: real works only
    assert _pv(works=10_000.0, years_held=5).works_majoration == pytest.approx(10_000.0)


def test_loss_is_floored_at_zero() -> None:
    res = _pv(sale_price=150_000.0, sale_fees=0.0)
    assert res.plus_value_brute == 0.0
    assert res.total_tax == 0.0


def test_addback_only_when_it_applies() -> None:
    applies = _pv(amortization_deducted_imm=50_000.0, amortization_addback_applies=True)
    residence = _pv(amortization_deducted_imm=50_000.0, amortization_addback_applies=False)
    assert applies.amortization_addback == 50_000.0
    assert residence.amortization_addback == 0.0
    assert applies.plus_value_brute - residence.plus_value_brute == pytest.approx(50_000.0)


def test_surtax_split_between_two_owners() -> None:
    kw: dict[str, Any] = {"sale_price": 500_000.0, "sale_fees": 0.0, "years_held": 6}
    one = _pv(**kw, nb_owners=1)
    two = _pv(**kw, nb_owners=2)
    assert one.surtax > 0
    assert two.surtax == pytest.approx(2 * surtax(one.base_ir / 2))
    assert two.surtax < one.surtax
    assert two.ir_pv == one.ir_pv


def _strategy(prices: list[float]) -> dict[str, Any]:
    details = []
    for i, price in enumerate(prices):
        details.append(
            {
                "nom_bien": f"bien{i}",
                "prix_achat_bien": price,
                "frais_notaire": price * 0.08,
                "budget_travaux": 0.0,
                "renovation_energetique": 0.0,
                "mobilier": 0.0,
                "ratio_terrain_pct": 15.0,
                "credit_final": price * 0.8,
                "taux_pret": 3.5,
                "duree_pret": 20,
                "assurance_ann_pct": 0.3,
                "loyer_mensuel_initial": price * 0.005,
                "charges_const_mth0": 50.0,
                "tf_const_mth0": 50.0,
                "frais_gestion_pct": 0.0,
                "provision_pct": 0.0,
            }
        )
    return {"apport_total": sum(prices) * 0.28, "details": details}


def test_no_netting_across_properties() -> None:
    """A property sold at a loss does not shelter the gain of the other (CGI 150 U)."""
    strategy = _strategy([200_000.0, 200_000.0])
    # Second property: bought dear (works already inflate the cost base), sold at a loss.
    strategy["details"][1]["budget_travaux"] = 300_000.0
    _, bilan = simulate_long_term_strategy(
        strategy, 8, {"appreciation_bien_pct": 5.0}, {"regime_fiscal": "microbic", "frais_vente_pct": 6.0}
    )
    per = bilan["plus_value_par_bien"]
    assert len(per) == 2
    assert per[0]["plus_value_brute"] > 0
    assert per[1]["plus_value_brute"] == 0.0
    assert bilan["plus_value"]["plus_value_brute"] == pytest.approx(per[0]["plus_value_brute"])
    assert bilan["impot_pv"] == pytest.approx(per[0]["total_tax"])


def test_residence_services_flag_removes_addback_in_simulation() -> None:
    _, normal = simulate_long_term_strategy(_strategy([250_000.0]), 10, {}, {"regime_fiscal": "lmnp"})
    exempt = _strategy([250_000.0])
    exempt["details"][0]["residence_services"] = True
    _, res = simulate_long_term_strategy(exempt, 10, {}, {"regime_fiscal": "lmnp"})
    assert normal["plus_value"]["amortization_addback"] > 0
    assert res["plus_value"]["amortization_addback"] == 0.0


def test_micro_bic_has_no_addback() -> None:
    _, bilan = simulate_long_term_strategy(_strategy([250_000.0]), 10, {}, {"regime_fiscal": "microbic"})
    assert bilan["plus_value"]["amortization_addback"] == 0.0


def test_nb_owners_plumbing() -> None:
    assert RunContext().nb_owners == 1
    ctx = RunContext(nb_owners=2)
    assert ctx.to_tax_params()["nb_owners"] == 2
    assert ctx.to_eval_params()["nb_owners"] == 2
    assert RunContext.from_dict(ctx.to_dict()).nb_owners == 2
    with pytest.raises(ValueError):
        RunContext(nb_owners=3)  # type: ignore[arg-type]


def test_archetype_residence_services_defaults_false() -> None:
    assert Archetype.model_fields["residence_services"].default is False
