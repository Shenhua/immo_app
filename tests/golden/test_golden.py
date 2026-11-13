"""Golden tests: the app's simulation engine against docs/TAX_SPEC.md reference values.

Each case of ``cases.yaml`` (single property) is mapped to a strategy dict, run through
``simulate_long_term_strategy`` and compared field by field (one pytest test per field).

``KNOWN_GAPS`` lists what the engine cannot match yet, as ``"case_id/field": "Txx"`` (the task
that will fix it). Keys are fnmatch patterns over ``"<case_id>/<field>"``. Every gap becomes
``xfail(strict=True)``: the task that fixes it makes the test pass, which flips it to XPASS
(a failure in strict mode), and that task then removes the entry.

Field paths: ``yearly.<year>.<key>``, ``exit.<key>``, ``wealth``, ``irr_pct``,
``cumulative_cash_flow_after_tax``, ``total_tax_on_income``.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.application.services.simulation import simulate_long_term_strategy

CASES_PATH = Path(__file__).with_name("cases.yaml")
_DOC = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))
CASES: list[dict[str, Any]] = _DOC["cases"]
TOL_EUR = float(_DOC["tolerance_eur"])
TOL_IRR_PCT = float(_DOC["tolerance_irr_pct"])

# Regimes the engine can run today (T23 adds the others).
SUPPORTED_REGIMES = {"lmnp_reel": "lmnp", "micro_bic": "microbic"}

# --------------------------------------------------------------------------- #
# Known gaps: "case_id/field" (fnmatch pattern) -> task that fixes it.
# --------------------------------------------------------------------------- #
KNOWN_GAPS: dict[str, str] = {
    # Regimes that do not exist in the engine yet.
    "C5_*/*": "T23",
    "C6_*/*": "T23",
    "C7_*/*": "T23",
    # Tax state and split not exposed by the engine (ARD stock, deficits, IR/PS split, 18.6 % BIC PS rate).
    "*/yearly.*.ard_stock_end": "T23",
    "*/yearly.*.amortization_deducted": "T23",
    "*/yearly.*.bic_deficit_stock_end": "T23",
    "*/yearly.*.ir": "T23",
    "*/yearly.*.ps": "T23",
    "*/yearly.*.result_before_amortization": "T23",
    "*/yearly.*.accounting_result": "T23",
    "*/yearly.*.amortization_booked": "T23",
    # PS rate: engine uses 17.2 % on BIC income, spec 18.6 % (affects every year with a positive base).
    "C2_*/yearly.20.tax": "T23",
    "C2_*/yearly.20.cash_flow_after_tax": "T23",
    "C4_*/yearly.*.tax": "T23",
    "C4_*/yearly.*.cash_flow_after_tax": "T23",
    "C8_*/yearly.20.tax": "T23",
    "C8_*/yearly.20.cash_flow_after_tax": "T23",
    "C3_*/yearly.25.tax": "T23/T24",
    "C[248]_*/total_tax_on_income": "T23",
    "C2_*/cumulative_cash_flow_after_tax": "T23/T24",
    "C8_*/cumulative_cash_flow_after_tax": "T23/T24",
    "C4_*/cumulative_cash_flow_after_tax": "T23/T24",
    # CFE: charged in year 1 and not indexed today (spec: furnished only, 0 in year 1, indexed).
    "*/yearly.1.operating_charges": "T24",
    "*/yearly.1.cash_flow_before_tax": "T24",
    "*/yearly.1.cash_flow_after_tax": "T24",
    "C1_*/cumulative_cash_flow_after_tax": "T24",
    "C9_*/cumulative_cash_flow_after_tax": "T24",
    "C3_*/yearly.*.operating_charges": "T24",
    "C3_*/yearly.*.cash_flow_before_tax": "T24",
    "C3_*/yearly.*.cash_flow_after_tax": "T24",
    "C3_*/yearly.25.taxable_base": "T24",
    "C3_*/cumulative_cash_flow_after_tax": "T24",
    "C3_*/total_tax_on_income": "T24",
    # Plus-value on exit, not modelled as in the spec (allowances, add-back, surtax, flat rates).
    "*/exit.sale_fees": "T21",
    "*/exit.acquisition_fees_retained": "T21",
    "*/exit.works_majoration": "T21",
    "*/exit.amortization_addback": "T21",
    "*/exit.corrected_acquisition_price": "T21",
    "*/exit.net_sale_price": "T21",
    "*/exit.plus_value_brute": "T21",
    "*/exit.abattement_*": "T21",
    "*/exit.base_*": "T21",
    "*/exit.ir_pv": "T21",
    "*/exit.ps_pv": "T21",
    "*/exit.surtax": "T21",
    "*/exit.deficit_recapture": "T23",
    "*/exit.net_exit_proceeds": "T21",
    # Sale price: engine appreciates from year 2, i.e. price x (1+g)^(H-1) (TAX_SPEC 10, item 21).
    "*/exit.sale_price": "T21",
    # Wealth, IRR and the cash flows that depend on the tax rates (T24 / T23).
    "*/wealth": "T24",
    "*/irr_pct": "T24",
}


def _gap_for(test_id: str) -> str | None:
    for pattern, task in KNOWN_GAPS.items():
        if fnmatchcase(test_id, pattern):
            return task
    return None


# --------------------------------------------------------------------------- #
# Adapter: case inputs -> engine inputs, engine outputs -> flat field dict
# --------------------------------------------------------------------------- #
class Unsupported(Exception):
    """The engine has no equivalent for this case or field yet."""


def _build_strategy(inp: dict[str, Any]) -> dict[str, Any]:
    apport = inp["price"] + inp["notary_fees"] + inp["works"] + inp["furniture"] - inp["loan_amount"]
    prop = {
        "prix_achat_bien": inp["price"],
        "frais_notaire": inp["notary_fees"],
        "budget_travaux": inp["works"],
        "renovation_energetique": 0.0,
        "mobilier": inp["furniture"],
        "ratio_terrain_pct": inp["land_ratio_pct"],
        "credit_final": inp["loan_amount"],
        "taux_pret": inp["loan_rate_pct"],
        "duree_pret": inp["loan_years"],
        "assurance_ann_pct": inp["loan_insurance_pct"],
        # Vacancy: the reference reduces the rent itself (the engine would book it as a charge), so the
        # adapter passes the rent net of vacancy and no provision.
        "loyer_mensuel_initial": inp["rent_monthly"] * (1 - inp["vacancy_pct"] / 100.0),
        # Operating charges: co-ownership + accounting fees as fixed charges, property tax separately.
        "charges_const_mth0": (inp["copro_charges_annual"] + inp["accounting_fees_annual"]) / 12.0,
        "tf_const_mth0": inp["property_tax_annual"] / 12.0,
        "frais_gestion_pct": inp["management_pct"],
        "provision_pct": 0.0,
    }
    return {"apport_total": apport, "details": [prop]}


def _run_engine(case: dict[str, Any]) -> dict[str, float]:
    inp = case["inputs"]
    regime = SUPPORTED_REGIMES.get(case["regime"])
    if regime is None:
        raise Unsupported(f"regime {case['regime']} not implemented in the engine")
    strategy = _build_strategy(inp)
    tax_params = {
        "tmi_pct": inp["tmi_pct"],
        "regime_fiscal": regime,
        "micro_bic_abatt_pct": inp["micro_bic_abattement_pct"],
        "apply_ira": inp["apply_ira"],
        "cfe_par_bien_ann": inp["cfe_annual"],
        "frais_vente_pct": inp["sale_fees_pct"],
        "ratio_terrain_pct": inp["land_ratio_pct"],
    }
    market = {
        "appreciation_bien_pct": inp["appreciation_pct"],
        "revalo_loyer_pct": inp["rent_indexation_pct"],
        "inflation_charges_pct": inp["charges_indexation_pct"],
    }
    horizon = int(inp["horizon_years"])
    df, bilan = simulate_long_term_strategy(strategy, horizon, market, tax_params)

    out: dict[str, float] = {}
    for _, row in df.iterrows():
        y = int(row["Année"])
        interest = float(row["Intérêts & Assurance"])
        tax = float(row["Impôt Dû"])
        cf_after = float(row["Cash-Flow Net d'Impôt"])
        out[f"yearly.{y}.rent"] = float(row["Loyers Bruts"])
        out[f"yearly.{y}.interest_and_insurance"] = interest
        out[f"yearly.{y}.operating_charges"] = -float(row["Charges Déductibles"]) - interest
        out[f"yearly.{y}.amortization_current_year"] = -float(row["Amortissements"])
        out[f"yearly.{y}.taxable_base"] = float(row["Résultat Fiscal"])
        out[f"yearly.{y}.tax"] = tax
        out[f"yearly.{y}.cash_flow_after_tax"] = cf_after
        out[f"yearly.{y}.cash_flow_before_tax"] = cf_after + tax
    last = df.iloc[-1]
    out["exit.sale_price"] = float(last["Valeur Biens"])
    out["exit.crd"] = float(last["Dette"])
    out["exit.ira"] = float(bilan["ira_total"])
    out["exit.net_exit_proceeds"] = float(bilan["liquidation_nette"])
    out["total_tax_on_income"] = float(df["Impôt Dû"].sum())
    out["cumulative_cash_flow_after_tax"] = float(df["Cash-Flow Net d'Impôt"].sum())
    out["wealth"] = float(bilan["enrichissement_net"])
    out["irr_pct"] = float(bilan["tri_annuel"])
    return out


def _flatten_expected(case: dict[str, Any]) -> dict[str, float]:
    exp = case["expected"]
    flat: dict[str, float] = {}
    for year, vals in exp["yearly"].items():
        for key, val in vals.items():
            flat[f"yearly.{year}.{key}"] = val
    for key, val in exp["exit"].items():
        flat[f"exit.{key}"] = val
    for key in ("wealth", "irr_pct", "cumulative_cash_flow_after_tax", "total_tax_on_income"):
        if key in exp:
            flat[key] = exp[key]
    return flat


_ACTUAL_CACHE: dict[str, dict[str, float] | Unsupported] = {}


def _actual(case: dict[str, Any]) -> dict[str, float]:
    cid = case["id"]
    if cid not in _ACTUAL_CACHE:
        try:
            _ACTUAL_CACHE[cid] = _run_engine(case)
        except Unsupported as exc:
            _ACTUAL_CACHE[cid] = exc
    res = _ACTUAL_CACHE[cid]
    if isinstance(res, Unsupported):
        raise res
    return res


def _params() -> list[Any]:
    params = []
    for case in CASES:
        for field, expected in _flatten_expected(case).items():
            test_id = f"{case['id']}/{field}"
            marks = []
            task = _gap_for(test_id)
            if task:
                marks.append(pytest.mark.xfail(strict=True, reason=f"known gap, fixed by {task}"))
            params.append(pytest.param(case, field, expected, id=test_id, marks=marks))
    return params


@pytest.mark.parametrize(("case", "field", "expected"), _params())
def test_golden(case: dict[str, Any], field: str, expected: float) -> None:
    actual = _actual(case)
    assert field in actual, f"{field} is not exposed by the engine"
    tol = TOL_IRR_PCT if field == "irr_pct" else TOL_EUR
    if field.endswith("_pct") and field != "irr_pct":
        tol = 0.01
    assert actual[field] == pytest.approx(expected, abs=tol), f"{field}: got {actual[field]}, expected {expected}"


def test_known_gaps_reference_existing_cases() -> None:
    """A stale pattern (renamed case or field) would silently disable nothing: fail loudly instead."""
    ids = [p.id for p in _params()]
    for pattern in KNOWN_GAPS:
        assert any(fnmatchcase(i, pattern) for i in ids), f"KNOWN_GAPS pattern matches nothing: {pattern}"
