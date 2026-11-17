"""T63: report export (Excel + PDF)."""

from __future__ import annotations

import io
from typing import Any

import pytest
from openpyxl import load_workbook
from pypdf import PdfReader

from src.application.run_context import RunContext
from src.application.services.report import (
    build_report,
    render_excel,
    render_pdf,
    yearly_summary,
)
from src.application.services.simulation import simulate_long_term_strategy


def _prop(name: str, credit: float = 140_000.0, meuble: bool = True) -> dict[str, Any]:
    return {
        "nom_bien": name,
        "prix_achat_bien": 150_000.0,
        "frais_notaire": 12_000.0,
        "budget_travaux": 0.0,
        "renovation_energetique": 0.0,
        "mobilier": 5_000.0 if meuble else 0.0,
        "credit_final": credit,
        "taux_pret": 3.5,
        "duree_pret": 20,
        "assurance_ann_pct": 0.3,
        "loyer_mensuel_initial": 900.0,
        "charges_const_mth0": 100.0,
        "tf_const_mth0": 70.0,
        "frais_gestion_pct": 0.0,
        "provision_pct": 0.0,
        "meuble": meuble,
        "dpe_initial": "F",
    }


CTX = RunContext(horizon_ans=12, apply_ira=False)
STRAT = {"apport_total": 40_000.0, "details": [_prop("Studio Lyon"), _prop("T2 Nantes", meuble=False)]}


def _pdf_text(pdf: bytes) -> str:
    return "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages)


def test_excel_sheets_and_values() -> None:
    model = build_report(STRAT, CTX, name="Stratégie #1")
    wb = load_workbook(io.BytesIO(render_excel(model)))
    assert {"Hypothèses", "Stratégie", "Flux annuels", "Fiscalité", "Échéanciers", "Alertes"} <= set(wb.sheetnames)
    _, bilan = simulate_long_term_strategy(STRAT, 12, CTX.market_hypotheses, CTX.to_tax_params())
    ws = wb["Synthèse"]
    vals = {ws.cell(row=r, column=1).value: ws.cell(row=r, column=2).value for r in range(1, 20)}
    assert vals["Enrichissement net"] == pytest.approx(bilan["enrichissement_net"])
    assert vals["TRI annuel"] == pytest.approx(bilan["tri_annuel"])
    assert wb["Flux annuels"].max_row == 1 + 12
    assert "€" in wb["Flux annuels"]["B2"].number_format
    hyp = {c[0].value: c[1].value for c in wb["Hypothèses"].iter_rows()}
    assert hyp["Horizon (années)"] == 12
    titles = [c.value for c in wb["Échéanciers"]["A"] if isinstance(c.value, str) and c.value.startswith("Crédit")]
    assert len(titles) == 2
    assert any(cat == "DPE" for cat, _ in model.alerts)


def test_pdf_header_name_and_disclaimer() -> None:
    pdf = render_pdf(build_report(STRAT, CTX, name="Stratégie Alpha", include_regimes=True))
    assert pdf.startswith(b"%PDF")
    assert 2 <= len(PdfReader(io.BytesIO(pdf)).pages) <= 4
    text = _pdf_text(pdf)
    assert "Alpha" in text
    assert "ne constitue pas un conseil fiscal" in text


def test_none_irr_renders_na() -> None:
    df, bilan = simulate_long_term_strategy(STRAT, 12, CTX.market_hypotheses, CTX.to_tax_params())
    bilan = {**bilan, "tri_annuel": None, "tri_warning": "IRR calculation failed"}
    model = build_report(STRAT, CTX, df=df, bilan=bilan)
    assert "n/a" in _pdf_text(render_pdf(model))
    wb = load_workbook(io.BytesIO(render_excel(model)))
    ws = wb["Synthèse"]
    tri = next(ws.cell(row=r, column=2).value for r in range(1, 20) if ws.cell(row=r, column=1).value == "TRI annuel")
    assert tri == "n/a"
    assert any(cat == "TRI" for cat, _ in model.alerts)


def test_all_cash_strategy() -> None:
    strat = {"apport_total": 170_000.0, "details": [_prop("Cash", credit=0.0)]}
    model = build_report(strat, CTX)
    assert model.schedules == []
    wb = load_workbook(io.BytesIO(render_excel(model)))
    assert "Échéanciers" in wb.sheetnames
    assert render_pdf(model).startswith(b"%PDF")


def test_yearly_summary_every_five_plus_last() -> None:
    model = build_report(STRAT, CTX)
    assert list(yearly_summary(model.yearly)["Année"]) == [5, 10, 12]
