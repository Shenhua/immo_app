"""
Standardized Financial Definitions & Formulas.
Single Source of Truth for business logic calculations.
"""
from typing import TypedDict

import pandas as pd

MIN_MODES = frozenset({"min", "≥"})


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
    cf_year_1_monthly: float
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
    1. Primary: Year 1 Monthly (Worst case)
    2. Secondary: Average of First 5 Years

    Args:
        mode_cf: "target" (precise proximity) or "min" (maximize above target)
    """
    if df_sim.empty:
        return {
            "cf_year_1_monthly": -99999.0,
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

    # Acceptance Logic (Phase 16 Check)
    # Mode "target": Absolute precision (abs diff)
    # Mode "min": One-sided, accept iff cf >= target (no tolerance below)

    gap_y1 = cf_penalty(cf_y1_monthly, target_cf, mode_cf)
    is_acceptable = cf_accepts(cf_y1_monthly, target_cf, tolerance, mode_cf)

    return {
        "cf_year_1_monthly": float(cf_y1_monthly),
        "cf_avg_5y_monthly": float(cf_5y_monthly),
        "is_acceptable": bool(is_acceptable),
        "gap": float(gap_y1),
    }

def calculate_dscr_metric(df_sim: pd.DataFrame) -> float:
    """
    Standardized DSCR calculation.
    NOI = Rent + Charges (negative) + Interest (add back)
    Debt Service = Capital + Interest
    """
    if df_sim.empty:
        return 0.0

    row = df_sim.iloc[0]
    ds = row.get("Capital Remboursé", 0.0) + row.get("Intérêts & Assurance", 0.0)

    charges = row.get("Charges Déductibles", 0.0)
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

    2. Enrichment Net:
       Liquidation Nette - Initial Capital (Apport)
       (Pure profit over the period)

    3. TRI (Internal Rate of Return):
       Already in bilan['tri_global'], but here we confirm standard definition.
    """

    # Extract from existing Bilan dict (which comes from simulation)
    # The simulation engine already computes these, but this function
    # ensures we have a canonical named interface for them.

    liq_nette = float(bilan.get("liquidation_nette", 0.0))
    float(bilan.get("enrichissement_net", 0.0))

    # Recalculate enrichment to be safe (Liquidation - Apport)
    # The bilan version might subtract inflation adjusted apport?
    # Standard: Nominal Enrichment = Liquidation - Nominal Apport
    enrich_standard = liq_nette - apport_initial

    return {
        "liquidation_nette": liq_nette,
        "enrichissement_net": enrich_standard,
        "apport_initial": apport_initial
    }
