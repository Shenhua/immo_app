"""
Standardized Financial Definitions & Formulas.
Single Source of Truth for business logic calculations.
"""
from typing import TypedDict

import pandas as pd

MIN_MODES = frozenset({"min", "≥"})

# --- Stabilised year (T29, review finding M1) -------------------------------------------------------
# DEFINITION: the "stabilised" cash flow and DSCR are those of SIMULATION YEAR 2 (the first full
# operating year), not year 1.
#
# RATIONALE: every archetype carries works, so year 1 loses ``mois_travaux`` months of rent (one-off
# vacancy while the works run, TAX_SPEC 1.8), cuts year-1 cash flow by several hundred EUR/month and
# the year-1 DSCR by 40-50 %. Year 1 is therefore not representative of the steady state a
# buyer is asking about ("what will this cost me per month?"). Year 2 is the first year with all months
# rented, and it also carries the indexation (rent, charges, taxe foncière growth), so it is the
# honest recurring figure. Year 2 was preferred to "year 1 recomputed without works months" because it
# needs no second simulation pass and does not mix the year-1 one-offs (works deduction under foncier
# réel, no CFE in the creation year) into the steady-state tax.
#
# The same definition is used by: the allocator estimate (``PortfolioAllocator`` computes its cf0 for
# year 2, with the same growth rates), the evaluator acceptance (``cf_accepts`` on the stabilised CF)
# and the scoring (DSCR and cf_proximity). The year-1 figures (with works) stay available, labelled
# "année 1 (avec travaux)", for display only: ``cf_monthly_y1`` and ``dscr_y1``.
STABILISED_YEAR = 2

# Yearly-table column holding the operating charges + interest (negative). Single constant so a column
# rename is a one-line change.
CHARGES_COL = "Charges Déductibles"


def stabilised_row_index(n_years: int) -> int:
    """Zero-based row of the stabilised year in a yearly table (last row if the horizon is 1 year)."""
    return max(0, min(STABILISED_YEAR - 1, n_years - 1))


def cf_penalty(cf: float, target: float, mode_cf: str = "target") -> float:
    """Distance to the cash-flow target (EUR/month), mode-aware.

    - "min": one-sided, ``max(0, target - cf)``. A surplus is never penalised.
    - "target" (precise): ``abs(cf - target)``.
    """
    if mode_cf in MIN_MODES:
        return max(0.0, target - cf)
    return abs(cf - target)


def cf_accepts(cf: float, target: float, tol: float, mode_cf: str = "target") -> bool:
    """Single source of truth for cash-flow acceptance.

    - "min": accept iff ``cf >= target`` (no tolerance below the target).
    - "target" (precise): accept iff ``abs(cf - target) <= tol``.
    """
    if mode_cf in MIN_MODES:
        return bool(cf >= target)
    return bool(abs(cf - target) <= tol)


class CashflowMetrics(TypedDict):
    cf_year_1_monthly: float  # year 1, WITH the works months (display only)
    cf_stabilised_monthly: float  # stabilised year (see STABILISED_YEAR): acceptance, gap, scoring
    cf_avg_5y_monthly: float
    is_acceptable: bool
    gap: float

def calculate_cashflow_metrics(
    df_sim: pd.DataFrame,
    target_cf: float,
    tolerance: float,
    mode_cf: str = "target"
) -> CashflowMetrics:
    """
    Calculate cashflow metrics according to business rules:
    1. Primary: stabilised-year monthly CF (year 2, see STABILISED_YEAR). Acceptance and gap use it.
    2. Secondary: average of the first 5 years.
    Year 1 (with works months) is reported as ``cf_year_1_monthly`` for display only.

    Args:
        mode_cf: "target" (precise proximity) or "min" (maximize above target)
    """
    if df_sim.empty:
        return {
            "cf_year_1_monthly": -99999.0,
            "cf_stabilised_monthly": -99999.0,
            "cf_avg_5y_monthly": -99999.0,
            "is_acceptable": False,
            "gap": 99999.0,
        }

    # Year 1 Monthly (Primary)
    cf_y1_annual = df_sim.iloc[0]["Cash-Flow Net d'Impôt"]
    cf_y1_monthly = cf_y1_annual / 12.0

    # 5-Year Average Monthly (Secondary)
    # limit to available data if < 5 years
    horizon = min(5, len(df_sim))
    cf_5y_annual_avg = df_sim.iloc[:horizon]["Cash-Flow Net d'Impôt"].mean()
    cf_5y_monthly = cf_5y_annual_avg / 12.0

    cf_stab_annual = df_sim.iloc[stabilised_row_index(len(df_sim))]["Cash-Flow Net d'Impôt"]
    cf_stab_monthly = cf_stab_annual / 12.0

    # Acceptance Logic (Phase 16 Check)
    # Mode "target": Absolute precision (abs diff)
    # Mode "min": One-sided, accept iff cf >= target (no tolerance below)

    gap_stab = cf_penalty(cf_stab_monthly, target_cf, mode_cf)
    is_acceptable = cf_accepts(cf_stab_monthly, target_cf, tolerance, mode_cf)

    return {
        "cf_year_1_monthly": float(cf_y1_monthly),
        "cf_stabilised_monthly": float(cf_stab_monthly),
        "cf_avg_5y_monthly": float(cf_5y_monthly),
        "is_acceptable": bool(is_acceptable),
        "gap": float(gap_stab),
    }

def calculate_dscr_metric(df_sim: pd.DataFrame, stabilised: bool = False) -> float:
    """
    Standardized DSCR calculation, for year 1 (default, with works months) or the stabilised year.
    NOI = Rent + Charges (negative) + Interest (add back)
    Debt Service = Capital + Interest
    """
    if df_sim.empty:
        return 0.0

    row = df_sim.iloc[stabilised_row_index(len(df_sim)) if stabilised else 0]
    ds = row.get("Capital Remboursé", 0.0) + row.get("Intérêts & Assurance", 0.0)

    charges = row.get(CHARGES_COL, 0.0)
    interest = row.get("Intérêts & Assurance", 0.0)
    rent = row.get("Loyers Bruts", 0.0)

    # NOI (Net Operating Income) before debt service
    # Charges in DF are negative and include interest.
    # To get Operating Expenses only: |Charges| - Interest
    # So NOI = Rent - (|Charges| - Interest)
    #        = Rent + Charges + Interest  (since Charges is negative)
    noi = rent + charges + interest

    if ds < 1e-9:
        return 10.0 # Infinite DSCR essentially

    return noi / ds

def calculate_enrichment_metrics(
    bilan: dict,
    apport_initial: float,
    horizon: int
) -> dict[str, float]:
    """
    Standardized Wealth Calculation.

    1. Liquidation Nette:
       What you get in your pocket if you sell everything at Horizon.
       (Sale Price - Transaction Costs - Remaining Debt - Exit Tax)

    2. Enrichment Net (wealth, TAX_SPEC 5):
       Liquidation Nette + sum of after-tax yearly cash flows - Initial Capital (Apport)
       (Pure profit over the period, including the cash flow received or funded year by year)

    3. TRI (Internal Rate of Return):
       Already in bilan['tri_global'], but here we confirm standard definition.
    """

    # Extract from existing Bilan dict (which comes from simulation)
    # The simulation engine already computes these, but this function
    # ensures we have a canonical named interface for them.

    liq_nette = float(bilan.get("liquidation_nette", 0.0))
    # The simulation owns the wealth figure (it has the yearly cash flows); never recompute it as
    # liquidation - apport, which would drop the cash flows. Fallback only for a bilan without it.
    enrich_standard = float(bilan.get("enrichissement_net", liq_nette - apport_initial))

    return {
        "liquidation_nette": liq_nette,
        "enrichissement_net": enrich_standard,
        "apport_initial": apport_initial
    }
