"""Sensitivity / stress-test computation (pure: no streamlit import).

The baseline and the stressed case go through the same function with the same ``RunContext``;
they differ only by the deltas, so zero deltas give exactly zero impact.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from src.application.run_context import RunContext
from src.application.services.simulation import simulate_long_term_strategy
from src.core import glossary

METRICS = ("cf_stab_mensuel", "tri_annuel", "enrichissement_net", "dscr_stab")


@dataclass(frozen=True)
class Deltas:
    """Stress applied to a strategy. Percent deltas are relative, point deltas are additive."""

    loyer_pct: float = 0.0
    vacance_pts: float = 0.0
    travaux_pct: float = 0.0
    taux_pts: float = 0.0
    appreciation_pts: float = 0.0
    # Market hypotheses (T64): additive points on the yearly rent indexation / charges inflation.
    revalo_pts: float = 0.0
    inflation_pts: float = 0.0

    def is_zero(self) -> bool:
        return not any(
            (
                self.loyer_pct,
                self.vacance_pts,
                self.travaux_pct,
                self.taux_pts,
                self.appreciation_pts,
                self.revalo_pts,
                self.inflation_pts,
            )
        )


def apply_deltas(strategy: dict[str, Any], deltas: Deltas) -> dict[str, Any]:
    """Return a stressed deep copy of ``strategy`` (the input is never mutated)."""
    modified = copy.deepcopy(strategy)
    extra_total = 0.0
    for brick in modified.get("details", []):
        if deltas.loyer_pct:
            brick["loyer_mensuel_initial"] *= 1.0 + deltas.loyer_pct / 100.0
        if deltas.vacance_pts:
            brick["provision_pct"] = max(0.0, float(brick.get("provision_pct", 0.0) or 0.0) + deltas.vacance_pts)
        if deltas.taux_pts:
            brick["taux_pret"] = max(0.0, float(brick["taux_pret"]) + deltas.taux_pts)
        if deltas.travaux_pct:
            budget = float(brick.get("budget_travaux", 0.0) or 0.0)
            diff = budget * deltas.travaux_pct / 100.0
            brick["budget_travaux"] = budget + diff
            brick["cout_total"] = brick.get("cout_total", 0.0) + diff
            # The loan is kept fixed: the cost overrun is paid by the down payment.
            brick["apport_min"] = brick.get("apport_min", 0.0) + diff
            extra_total += diff
    if extra_total and "apport_total" in modified:
        modified["apport_total"] += extra_total
    return modified


def run_sensitivity(strategy: dict[str, Any], deltas: Deltas, ctx: RunContext) -> dict[str, Any]:
    """Simulate ``strategy`` under ``deltas``; ``tri_annuel`` is None when the IRR failed."""
    modified = apply_deltas(strategy, deltas)
    hypotheses = dict(ctx.market_hypotheses)
    hypotheses["appreciation_bien_pct"] += deltas.appreciation_pts
    hypotheses["revalo_loyer_pct"] += deltas.revalo_pts
    hypotheses["inflation_charges_pct"] += deltas.inflation_pts
    df, bilan = simulate_long_term_strategy(
        modified,
        duration_years=ctx.horizon_ans,
        market_hypotheses=hypotheses,
        tax_params=ctx.to_tax_params(),
    )
    if df.empty:
        return {}
    cf = glossary.calculate_cashflow_metrics(df, 0.0, 0.0)
    return {
        "cf_stab_mensuel": cf["cf_stabilised_monthly"],
        "tri_annuel": bilan.get("tri_annuel"),
        "enrichissement_net": bilan.get("enrichissement_net", 0.0),
        "liquidation_nette": bilan.get("liquidation_nette", 0.0),
        "dscr_stab": glossary.calculate_dscr_metric(df, stabilised=True),
    }


def impacts(base: dict[str, Any], stressed: dict[str, Any]) -> dict[str, float | None]:
    """stressed - base per metric (None when either side is None)."""
    out: dict[str, float | None] = {}
    for k in METRICS:
        a, b = base.get(k), stressed.get(k)
        out[k] = None if a is None or b is None else float(b) - float(a)
    return out


TORNADO_DRIVERS = {
    "Taux de crédit (pt)": "taux_pts",
    "Loyer (%)": "loyer_pct",
    "Vacance (pt)": "vacance_pts",
    "Appréciation (pt)": "appreciation_pts",
}


def tornado_data(strategy: dict[str, Any], ctx: RunContext, step: float = 1.0) -> list[dict[str, Any]]:
    """Impact of -step / +step on each driver, on IRR and stabilised monthly CF."""
    base = run_sensitivity(strategy, Deltas(), ctx)
    rows = []
    for label, field in TORNADO_DRIVERS.items():
        row: dict[str, Any] = {"driver": label}
        for sign, tag in ((-1, "down"), (1, "up")):
            d = impacts(base, run_sensitivity(strategy, Deltas(**{field: sign * step}), ctx))
            row[f"tri_{tag}"] = d["tri_annuel"]
            row[f"cf_{tag}"] = d["cf_stab_mensuel"]
        rows.append(row)
    return rows
