"""Downloadable report for one strategy (T63): a pure model builder and two renderers (Excel, PDF).

No streamlit import. ``build_report`` assembles a :class:`ReportModel` from the strategy dict, the frozen
``RunContext`` and the simulation result (it runs the simulation itself unless given ``df`` and ``bilan``).
``render_excel`` (openpyxl) and ``render_pdf`` (fpdf2, pure Python: no pango/cairo needed, unlike weasyprint)
only format that model.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from src.application.run_context import RunContext
from src.application.services.simulation import simulate_long_term_strategy
from src.domain.calculator.financial import generate_amortization_schedule
from src.domain.dpe import property_ban_warning

DISCLAIMER = "Simulation indicative, ne constitue pas un conseil fiscal."
NA = "n/a"

# field -> (French label, kind). kind: eur | pct | int | bool | text
ASSUMPTION_LABELS: dict[str, tuple[str, str]] = {
    "apport": ("Apport disponible", "eur"),
    "cf_cible": ("Cash-flow mensuel cible", "eur"),
    "tolerance": ("Tolérance cash-flow", "eur"),
    "mode_cf": ("Mode cash-flow (target/min)", "text"),
    "qualite_weight": ("Poids qualité vs finance", "pct"),
    "horizon_ans": ("Horizon (années)", "int"),
    "max_properties": ("Nombre max de biens", "int"),
    "use_full_capital": ("Utiliser tout le capital", "bool"),
    "top_n": ("Nombre de stratégies retenues", "int"),
    "taux_credits": ("Taux de crédit par durée", "text"),
    "frais_notaire_pct": ("Frais de notaire (%)", "pct"),
    "assurance_ann_pct": ("Assurance emprunteur annuelle (%)", "pct"),
    "frais_pret_pct": ("Frais de prêt (%)", "pct"),
    "inclure_travaux": ("Financer les travaux", "bool"),
    "inclure_reno_ener": ("Financer la rénovation énergétique", "bool"),
    "inclure_mobilier": ("Inclure le mobilier", "bool"),
    "financer_mobilier": ("Financer le mobilier", "bool"),
    "apply_ira": ("Appliquer les IRA", "bool"),
    "ira_cap_pct": ("Plafond IRA (%)", "pct"),
    "cfe_par_bien_ann": ("CFE par bien et par an", "eur"),
    "gestion_pct": ("Frais de gestion (%)", "pct"),
    "vacance_pct": ("Vacance locative (%)", "pct"),
    "frais_vente_pct": ("Frais de vente (%)", "pct"),
    "maintenance_m2_an": ("Entretien (€/m²/an)", "eur"),
    "gli_pct": ("Garantie loyers impayés (%)", "pct"),
    "mois_travaux": ("Mois de travaux (sans loyer)", "int"),
    "tf_growth_pct": ("Croissance taxe foncière (%)", "pct"),
    "dpe_policy": ("Politique DPE", "text"),
    "dpe_cout_m2_par_classe": ("Coût rénovation DPE (€/m²/classe)", "eur"),
    "regime_fiscal": ("Régime meublé", "text"),
    "regime_foncier": ("Régime nu", "text"),
    "sci_is": ("SCI à l'IS", "bool"),
    "tmi_pct": ("Tranche marginale d'imposition (%)", "pct"),
    "ratio_terrain_pct": ("Part terrain (%)", "pct"),
    "nb_owners": ("Nombre de vendeurs (plus-value)", "int"),
    "appreciation_bien_pct": ("Appréciation des biens (%/an)", "pct"),
    "revalo_loyer_pct": ("Revalorisation des loyers (%/an)", "pct"),
    "inflation_charges_pct": ("Inflation des charges (%/an)", "pct"),
    "finance_preset_name": ("Profil de scoring", "text"),
    "finance_weights": ("Pondérations de scoring", "text"),
    "apply_cap": ("Encadrement des loyers", "bool"),
    "archetypes_version": ("Version des données", "text"),
}

PROPERTY_COLUMNS: list[tuple[str, str, str]] = [
    ("nom_bien", "Bien", "text"),
    ("ville", "Ville", "text"),
    ("type_bien", "Type", "text"),
    ("surface", "Surface (m²)", "num"),
    ("dpe_initial", "DPE", "text"),
    ("prix_achat_bien", "Prix d'achat", "eur"),
    ("frais_notaire", "Frais de notaire", "eur"),
    ("budget_travaux", "Travaux", "eur"),
    ("renovation_energetique", "Rénovation énergétique", "eur"),
    ("mobilier", "Mobilier", "eur"),
    ("credit_final", "Crédit", "eur"),
    ("taux_pret", "Taux (%)", "pct"),
    ("duree_pret", "Durée (ans)", "int"),
    ("assurance_ann_pct", "Assurance (%)", "pct"),
    ("loyer_mensuel_initial", "Loyer mensuel", "eur"),
]

YEARLY_FORMATS: dict[str, str] = {"Année": "int"}  # every other yearly column is euros


@dataclass
class ReportModel:
    title: str
    strategy_name: str
    generated_on: str
    kpis: list[tuple[str, Any, str]]  # label, value (None = n/a), kind
    assumptions: list[tuple[str, Any, str]]
    strategy_summary: list[tuple[str, Any, str]]
    properties: pd.DataFrame
    yearly: pd.DataFrame
    fiscal: pd.DataFrame
    exit_tax: list[tuple[str, Any, str]]
    plus_value: pd.DataFrame
    schedules: list[tuple[str, pd.DataFrame]] = field(default_factory=list)
    alerts: list[tuple[str, str]] = field(default_factory=list)  # (category, message)
    regimes: pd.DataFrame | None = None
    disclaimer: str = DISCLAIMER


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _num(v: Any) -> float | None:
    """float or None for None / NaN / non-numeric."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _plain(v: Any) -> Any:
    """numpy scalars -> python."""
    return v.item() if hasattr(v, "item") else v


def _assumption_value(key: str, value: Any, kind: str) -> Any:
    if key == "taux_credits" and isinstance(value, dict):
        return "; ".join(f"{k} ans : {float(v):.2f} %" for k, v in sorted(value.items(), key=lambda kv: int(kv[0])))
    if key == "finance_weights" and isinstance(value, dict):
        return "; ".join(f"{k} : {float(v):.2f}" for k, v in value.items())
    if value is None:
        return NA if kind != "text" else "auto (données du bien)"
    if kind == "bool":
        return "Oui" if value else "Non"
    return _plain(value)


def default_strategy_name(strategy: dict[str, Any], index: int | None = None) -> str:
    base = f"Stratégie #{index}" if index is not None else "Stratégie"
    tax = strategy.get("taxonomy")
    return f"{base} ({tax})" if tax else base


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(i for i in items if i))


def format_value(v: Any, kind: str) -> str:
    """Text formatting shared by the PDF renderer (French style, plain spaces)."""
    if v is None:
        return NA
    if isinstance(v, str):
        return v
    f = _num(v)
    if f is None:
        return NA
    if kind == "eur":
        return f"{f:,.0f} EUR".replace(",", " ")
    if kind == "pct":
        return f"{f:.2f} %"
    if kind == "int":
        return f"{int(round(f))}"
    if kind == "ratio":
        return f"{f:.2f}"
    return f"{f:,.2f}".replace(",", " ").rstrip("0").rstrip(".") if f != int(f) else f"{int(f)}"


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def _loan_schedules(strategy: dict[str, Any]) -> list[tuple[str, pd.DataFrame]]:
    out: list[tuple[str, pd.DataFrame]] = []
    for i, p in enumerate(strategy.get("details", [])):
        principal = float(p.get("credit_final", 0.0) or 0.0)
        if principal <= 0:
            continue  # all-cash property: no loan
        sch = generate_amortization_schedule(
            principal, float(p.get("taux_pret", 0.0)), int(p.get("duree_pret", 20)) * 12,
            float(p.get("assurance_ann_pct", 0.0)),
        )
        df = pd.DataFrame({
            "Mois": sch["mois"], "Capital restant dû (début)": sch["capital_restant_debut"],
            "Intérêts": sch["interet"], "Principal": sch["principal"], "Assurance": sch["assurance"],
            "Mensualité totale": sch["paiement_total"], "Capital restant dû (fin)": sch["capital_restant_fin"],
        })
        out.append((str(p.get("nom_bien", p.get("nom", f"Bien {i + 1}"))), df))
    return out


def build_report(
    strategy: dict[str, Any],
    ctx: RunContext,
    *,
    name: str | None = None,
    df: pd.DataFrame | None = None,
    bilan: dict[str, Any] | None = None,
    include_regimes: bool = False,
) -> ReportModel:
    """Assemble the report model. Runs the detail simulation unless ``df`` and ``bilan`` are given."""
    if df is None or bilan is None:
        df, bilan = simulate_long_term_strategy(
            strategy, ctx.horizon_ans, ctx.market_hypotheses, ctx.to_tax_params()
        )
    name = name or default_strategy_name(strategy)
    details = strategy.get("details", [])

    tri = _num(bilan.get("tri_annuel"))
    kpis: list[tuple[str, Any, str]] = [
        ("Enrichissement net", _num(bilan.get("enrichissement_net")), "eur"),
        ("TRI annuel", tri, "pct"),
        ("Liquidation nette", _num(bilan.get("liquidation_nette")), "eur"),
        ("Impôt de sortie", _num(bilan.get("exit_tax", {}).get("total")), "eur"),
        ("DSCR année 1", _num(bilan.get("dscr_y1")), "ratio"),
        ("DSCR stabilisé", _num(bilan.get("dscr_stab")), "ratio"),
        ("Apport total", _num(strategy.get("apport_total")), "eur"),
        ("Cash-flow mensuel final", _num(strategy.get("cash_flow_final")), "eur"),
    ]

    assumptions = []
    for key, value in ctx.model_dump().items():
        label, kind = ASSUMPTION_LABELS.get(key, (key, "text"))
        assumptions.append((label, _assumption_value(key, value, kind), kind))

    total_credit = sum(float(p.get("credit_final", 0.0) or 0.0) for p in details)
    total_price = sum(float(p.get("prix_achat_bien", 0.0) or 0.0) for p in details)
    summary: list[tuple[str, Any, str]] = [
        ("Nombre de biens", len(details), "int"),
        ("Prix d'achat total", total_price, "eur"),
        ("Apport", _num(strategy.get("apport_total")), "eur"),
        ("Crédits totaux", total_credit, "eur"),
        ("Régimes fiscaux appliqués", ", ".join(f"{k} : {v}" for k, v in bilan.get("regimes", {}).items()) or NA, "text"),
    ]

    props = pd.DataFrame([
        {label: p.get(key) for key, label, _ in PROPERTY_COLUMNS} for p in details
    ], columns=[label for _, label, _ in PROPERTY_COLUMNS])

    fiscal = pd.DataFrame(bilan.get("fiscal_annuel", []))
    ex = bilan.get("exit_tax", {})
    exit_rows: list[tuple[str, Any, str]] = [
        ("Impôt sur le revenu (plus-value)", _num(ex.get("ir")), "eur"),
        ("Prélèvements sociaux (plus-value)", _num(ex.get("ps")), "eur"),
        ("Surtaxe plus-values élevées", _num(ex.get("surtax")), "eur"),
        ("IS à la sortie (SCI)", _num(ex.get("is")), "eur"),
        ("PFU sur boni (SCI)", _num(ex.get("pfu")), "eur"),
        ("Réintégration déficit foncier", _num(ex.get("deficit_recapture")), "eur"),
        ("Total impôt de sortie", _num(ex.get("total")), "eur"),
        ("IRA (indemnités de remboursement anticipé)", _num(bilan.get("ira_total")), "eur"),
    ]
    pv = pd.DataFrame([
        {k: _plain(v) for k, v in row.items()} for row in bilan.get("plus_value_par_bien", [])
    ])

    alerts: list[tuple[str, str]] = []
    for w in bilan.get("tax_warnings", []) or []:
        alerts.append(("Fiscalité", str(w)))
    for w in _dedupe([str(strategy_w) for strategy_w in _strategy_warnings(strategy)]):
        alerts.append(("Stratégie", w))
    for p in details:
        w = property_ban_warning(p, ctx.horizon_ans)
        if w:
            alerts.append(("DPE", f"{p.get('nom_bien', 'Bien')} : {w}"))
    if tri is None:
        alerts.append(("TRI", f"TRI non calculable ({bilan.get('tri_warning') or 'flux instables'}) : affiché n/a."))
    elif bilan.get("tri_warning"):
        alerts.append(("TRI", str(bilan["tri_warning"])))

    regimes = _regime_table(strategy, ctx) if include_regimes else None

    return ReportModel(
        title="Rapport de simulation immobilière",
        strategy_name=name,
        generated_on=date.today().strftime("%d/%m/%Y"),
        kpis=kpis,
        assumptions=assumptions,
        strategy_summary=summary,
        properties=props,
        yearly=df.reset_index(drop=True),
        fiscal=fiscal,
        exit_tax=exit_rows,
        plus_value=pv,
        schedules=_loan_schedules(strategy),
        alerts=alerts,
        regimes=regimes,
    )


def _strategy_warnings(strategy: dict[str, Any]) -> list[str]:
    raw = strategy.get("warnings") or []
    out = [raw] if isinstance(raw, str) else [str(w) for w in raw]
    if strategy.get("tri_warning"):
        out.append(str(strategy["tri_warning"]))
    return out


def _regime_table(strategy: dict[str, Any], ctx: RunContext) -> pd.DataFrame:
    from src.application.services.regime_comparison import compare_regimes

    cmp = compare_regimes(strategy, ctx)
    return pd.DataFrame([
        {
            "Régime": r.label + (" (choisi)" if r.is_user_choice else ""),
            "Enrichissement net": r.enrichissement_net,
            "TRI annuel (%)": _num(r.tri_annuel),
            "Impôt de sortie": r.exit_tax,
            "Impôt cumulé": sum(r.tax),
        }
        for r in cmp.results
    ])


def yearly_summary(yearly: pd.DataFrame, step: int = 5) -> pd.DataFrame:
    """Every ``step`` years plus the last one (PDF table)."""
    if yearly.empty:
        return yearly
    years = yearly["Année"].astype(int)
    last = years.iloc[-1]
    return yearly[(years % step == 0) | (years == last)]


# --------------------------------------------------------------------------- #
# Excel
# --------------------------------------------------------------------------- #
EUR_FMT = '#,##0 "€";[Red]-#,##0 "€"'
EUR2_FMT = '#,##0.00 "€"'
PCT_FMT = '0.00" %"'
_KIND_FMT = {"eur": EUR_FMT, "pct": PCT_FMT, "int": "0", "ratio": "0.00", "num": "#,##0.##"}


def _col_kind(col: str) -> str:
    c = col.lower()
    if "(%)" in c or c.endswith("_pct") or c in {"ir_pct"}:
        return "pct"
    if col in ("Année", "Mois", "year") or "durée" in c:
        return "int"
    if c.startswith("abattement") or "pct" in c:
        return "pct"
    return "eur"


def _write_table(ws: Any, df: pd.DataFrame, start_row: int, kinds: dict[str, str] | None = None) -> int:
    """Write ``df`` with a bold header at ``start_row``; return the next free row."""
    from openpyxl.styles import Font, PatternFill

    head_fill = PatternFill("solid", fgColor="DDE6F0")
    for j, col in enumerate(df.columns, start=1):
        c = ws.cell(row=start_row, column=j, value=str(col))
        c.font = Font(bold=True)
        c.fill = head_fill
    for i, row in enumerate(df.itertuples(index=False), start=start_row + 1):
        for j, (col, v) in enumerate(zip(df.columns, row, strict=True), start=1):
            v = _plain(v)
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                v = None
            cell = ws.cell(row=i, column=j, value=NA if v is None and (kinds or {}).get(col) == "pct" else v)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                kind = (kinds or {}).get(col) or _col_kind(str(col))
                cell.number_format = _KIND_FMT.get(kind, EUR_FMT)
    return start_row + len(df) + 2


def _write_pairs(ws: Any, rows: list[tuple[str, Any, str]], start_row: int = 1) -> int:
    from openpyxl.styles import Font

    for i, (label, value, kind) in enumerate(rows, start=start_row):
        ws.cell(row=i, column=1, value=label).font = Font(bold=True)
        v = _plain(value)
        if v is None:
            v = NA
        c = ws.cell(row=i, column=2, value=v)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            c.number_format = _KIND_FMT.get(kind, "General")
    return start_row + len(rows) + 1


def _autosize(ws: Any, max_width: int = 60) -> None:
    from openpyxl.utils import get_column_letter

    for col in ws.columns:
        width = max((len(str(c.value)) for c in col if c.value is not None), default=8)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_width, max(10, width + 2))


def render_excel(model: ReportModel) -> bytes:
    """Excel workbook: Synthèse, Hypothèses, Stratégie, Flux annuels, Fiscalité, Échéanciers, Alertes."""
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    ws = wb.active
    ws.title = "Synthèse"
    ws["A1"] = model.title
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = f"{model.strategy_name} - généré le {model.generated_on}"
    nxt = _write_pairs(ws, model.kpis, 4)
    ws.cell(row=nxt, column=1, value=model.disclaimer).font = Font(italic=True)
    _autosize(ws)

    ws = wb.create_sheet("Hypothèses")
    _write_pairs(ws, model.assumptions)
    _autosize(ws)

    ws = wb.create_sheet("Stratégie")
    nxt = _write_pairs(ws, model.strategy_summary)
    kinds = {label: kind for _, label, kind in PROPERTY_COLUMNS}
    _write_table(ws, model.properties, nxt + 1, kinds)
    if model.regimes is not None:
        _write_table(ws, model.regimes, nxt + len(model.properties) + 4)
    _autosize(ws)

    ws = wb.create_sheet("Flux annuels")
    _write_table(ws, model.yearly, 1)
    ws.freeze_panes = "B2"
    _autosize(ws)

    ws = wb.create_sheet("Fiscalité")
    nxt = _write_table(ws, model.fiscal, 1, {"year": "int"})
    ws.cell(row=nxt, column=1, value="Impôt de sortie").font = Font(bold=True)
    nxt = _write_pairs(ws, model.exit_tax, nxt + 1)
    if not model.plus_value.empty:
        ws.cell(row=nxt, column=1, value="Plus-value par bien").font = Font(bold=True)
        _write_table(ws, model.plus_value, nxt + 1, {"nom": "text"})
    _autosize(ws)

    ws = wb.create_sheet("Échéanciers")
    row = 1
    if not model.schedules:
        ws.cell(row=1, column=1, value="Aucun crédit (achat comptant).")
    for name, sch in model.schedules:
        ws.cell(row=row, column=1, value=f"Crédit - {name}").font = Font(bold=True)
        row = _write_table(ws, sch, row + 1, {"Mois": "int"})
    _autosize(ws)

    ws = wb.create_sheet("Alertes")
    ws["A1"], ws["B1"] = "Catégorie", "Message"
    ws["A1"].font = ws["B1"].font = Font(bold=True)
    if not model.alerts:
        ws["A2"] = "Aucune alerte."
    for i, (cat, msg) in enumerate(model.alerts, start=2):
        ws.cell(row=i, column=1, value=cat)
        ws.cell(row=i, column=2, value=msg)
    _autosize(ws, 120)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def _latin(text: Any) -> str:
    """fpdf2 core fonts are latin-1: map the few characters outside it."""
    s = str(text)
    for a, b in (("€", "EUR"), (" ", " "), ("\xa0", " "), ("→", "->"), ("–", "-"), ("—", "-"), ("’", "'"),
                 ("≥", ">="), ("≤", "<="), ("×", "x")):
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


def _pdf_table(pdf: Any, header: list[str], rows: list[list[str]], widths: list[float] | None = None) -> None:
    n = len(header)
    usable = pdf.w - pdf.l_margin - pdf.r_margin
    widths = widths or [usable / n] * n
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(221, 230, 240)
    for h, w in zip(header, widths, strict=True):
        pdf.cell(w, 6, _latin(h)[:28], border=1, fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for r in rows:
        for v, w in zip(r, widths, strict=True):
            pdf.cell(w, 5.5, _latin(v)[:34], border=1)
        pdf.ln()
    pdf.ln(2)


def _h(pdf: Any, text: str) -> None:
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _latin(text), new_x="LMARGIN", new_y="NEXT")


def render_pdf(model: ReportModel) -> bytes:
    """2-4 page PDF summary (A4): KPIs, assumptions, properties, yearly table (every 5 years), exit tax, alerts."""
    from fpdf import FPDF

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(12, 12, 12)
    pdf.set_auto_page_break(True, margin=14)
    pdf.set_title(_latin(f"{model.title} - {model.strategy_name}"))
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _latin(model.title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, _latin(f"{model.strategy_name} - {model.generated_on}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    _h(pdf, "Indicateurs clés")
    kp = [[lbl, format_value(v, kind)] for lbl, v, kind in model.kpis]
    _pdf_table(pdf, ["Indicateur", "Valeur"], kp, [90, 60])

    _h(pdf, "Stratégie")
    _pdf_table(pdf, ["Élément", "Valeur"], [[lbl, format_value(v, k)] for lbl, v, k in model.strategy_summary], [60, 126])
    props = model.properties
    if not props.empty:
        cols = ["Bien", "Prix d'achat", "Travaux", "Crédit", "Taux (%)", "Durée (ans)", "Loyer mensuel"]
        kinds = {label: kind for _, label, kind in PROPERTY_COLUMNS}
        rows = [[format_value(r[c], kinds[c]) for c in cols] for _, r in props.iterrows()]
        _pdf_table(pdf, cols, rows, [46, 26, 22, 26, 16, 20, 28])

    _h(pdf, "Hypothèses")
    half = (len(model.assumptions) + 1) // 2
    pdf.set_font("Helvetica", "", 8)
    for lbl, v, k in model.assumptions[:half] + model.assumptions[half:]:
        pdf.cell(95, 4.6, _latin(lbl)[:48], border="B")
        pdf.cell(0, 4.6, _latin(format_value(v, k))[:70], border="B", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    _h(pdf, "Projection annuelle (tous les 5 ans et dernière année)")
    ys = yearly_summary(model.yearly)
    if not ys.empty:
        cols = [c for c in ("Année", "Valeur Biens", "Dette", "Patrimoine Net", "Loyers Bruts", "Impôt Dû",
                            "Cash-Flow Net d'Impôt") if c in ys.columns]
        rows = [[format_value(r[c], "int" if c == "Année" else "eur") for c in cols] for _, r in ys.iterrows()]
        _pdf_table(pdf, cols, rows)

    _h(pdf, "Impôt de sortie")
    _pdf_table(pdf, ["Poste", "Montant"], [[lbl, format_value(v, k)] for lbl, v, k in model.exit_tax], [100, 50])

    _h(pdf, "Alertes")
    pdf.set_font("Helvetica", "", 9)
    if not model.alerts:
        pdf.cell(0, 6, "Aucune alerte.", new_x="LMARGIN", new_y="NEXT")
    for cat, msg in model.alerts:
        pdf.multi_cell(0, 5, _latin(f"[{cat}] {msg}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, _latin(model.disclaimer), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def build_report_files(
    strategy: dict[str, Any], ctx: RunContext, name: str | None = None, include_regimes: bool = True
) -> tuple[bytes, bytes]:
    """(xlsx bytes, pdf bytes) for one strategy."""
    model = build_report(strategy, ctx, name=name, include_regimes=include_regimes)
    return render_excel(model), render_pdf(model)
