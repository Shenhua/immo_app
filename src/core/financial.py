"""Financial calculation functions.

Core loan and amortization calculations for real estate investments.
"""

from __future__ import annotations

from typing import Any

import numpy_financial as npf


def calculate_monthly_payment(
    principal: float,
    annual_rate_pct: float,
    duration_months: int,
) -> float:
    """Calculate monthly loan payment (principal + interest only).

    Args:
        principal: Loan amount in €
        annual_rate_pct: Annual interest rate as percentage (e.g., 3.5 for 3.5%)
        duration_months: Loan term in months

    Returns:
        Monthly payment amount in €
    """
    if principal <= 0 or duration_months <= 0:
        return 0.0

    monthly_rate = (annual_rate_pct / 100.0) / 12.0

    if monthly_rate <= 0:
        return principal / duration_months

    return -npf.pmt(monthly_rate, duration_months, principal)


def calculate_insurance(
    principal: float,
    annual_insurance_pct: float,
) -> float:
    """Calculate monthly insurance premium.

    Args:
        principal: Initial loan amount in €
        annual_insurance_pct: Annual insurance rate as percentage

    Returns:
        Monthly insurance amount in €
    """
    if principal <= 0:
        return 0.0
    return (principal * (annual_insurance_pct / 100.0)) / 12.0


def calculate_total_monthly_payment(
    principal: float,
    annual_rate_pct: float,
    duration_months: int,
    annual_insurance_pct: float = 0.36,
) -> tuple[float, float, float]:
    """Calculate complete monthly payment breakdown.

    Args:
        principal: Loan amount in €
        annual_rate_pct: Annual interest rate %
        duration_months: Loan term in months
        annual_insurance_pct: Annual insurance rate %

    Returns:
        Tuple of (P&I payment, insurance, total payment)
    """
    pmt_pi = calculate_monthly_payment(principal, annual_rate_pct, duration_months)
    pmt_ins = calculate_insurance(principal, annual_insurance_pct)
    return pmt_pi, pmt_ins, pmt_pi + pmt_ins


def generate_amortization_schedule(
    principal: float,
    annual_rate_pct: float,
    duration_months: int,
    annual_insurance_pct: float = 0.36,
) -> dict[str, Any]:
    """Generate full loan amortization schedule.

    Args:
        principal: Loan amount in €
        annual_rate_pct: Annual interest rate %
        duration_months: Loan term in months
        annual_insurance_pct: Annual insurance rate %

    Returns:
        Dict with keys:
        - mois: List of month numbers
        - capital_restant_debut: List of start balances
        - interet: List of interest payments
        - principal: List of principal payments
        - assurance: List of insurance payments
        - paiement_total: List of total payments
        - capital_restant_fin: List of end balances
        - pmt_assur: Monthly insurance amount
        - pmt_total: Monthly total payment
        - nmois: Number of months
        And legacy aliases: interets, principals, balances
    """
    if principal <= 0 or duration_months <= 0:
        return {
            "mois": [],
            "capital_restant_debut": [],
            "interet": [],
            "principal": [],
            "assurance": [],
            "paiement_total": [],
            "capital_restant_fin": [],
            "pmt_assur": 0.0,
            "pmt_total": 0.0,
            "interets": [],
            "principals": [],
            "balances": [],
            "nmois": 0,
        }

    monthly_rate = (annual_rate_pct / 100.0) / 12.0
    pmt_pi = calculate_monthly_payment(principal, annual_rate_pct, duration_months)
    pmt_ins = calculate_insurance(principal, annual_insurance_pct)

    schedule = []
    balance = principal

    for month in range(1, duration_months + 1):
        interest = balance * monthly_rate
        principal_payment = pmt_pi - interest
        new_balance = max(0.0, balance - principal_payment)

        schedule.append({
            "mois": month,
            "capital_restant_debut": round(balance, 2),
            "interet": round(interest, 2),
            "principal": round(principal_payment, 2),
            "assurance": round(pmt_ins, 2),
            "paiement_total": round(pmt_pi + pmt_ins, 2),
            "capital_restant_fin": round(new_balance, 2),
        })

        balance = new_balance

    return {
        "mois": list(range(1, duration_months + 1)),
        "capital_restant_debut": [m["capital_restant_debut"] for m in schedule],
        "interet": [m["interet"] for m in schedule],
        "principal": [m["principal"] for m in schedule],
        "assurance": [m["assurance"] for m in schedule],
        "paiement_total": [m["paiement_total"] for m in schedule],
        "capital_restant_fin": [m["capital_restant_fin"] for m in schedule],
        "pmt_assur": pmt_ins,
        "pmt_total": pmt_pi + pmt_ins,
        "interets": [m["interet"] for m in schedule],
        "principals": [m["principal"] for m in schedule],
        "balances": [m["capital_restant_fin"] for m in schedule],
        "nmois": duration_months,
    }


def k_factor(
    annual_rate_pct: float,
    duration_years: int,
    annual_insurance_pct: float
) -> float:
    """Calculate the loan constant (K factor).

    K = (Monthly Payment + Monthly Insurance) / Principal

    Args:
        annual_rate_pct: Annual interest rate %
        duration_years: Loan duration in years
        annual_insurance_pct: Annual insurance rate %

    Returns:
        K factor (ratio of monthly service to principal)
    """
    r = (annual_rate_pct / 100.0) / 12.0
    n = max(1, duration_years * 12)

    base = (1.0 / n) if r == 0 else r / (1.0 - (1.0 + r)**(-n))
    assur = (annual_insurance_pct / 100.0) / 12.0

    return base + assur


def calculate_remaining_balance(
    principal: float,
    annual_rate_pct: float,
    duration_months: int,
    months_paid: int,
) -> float:
    """Calculate remaining loan balance after N months.

    Args:
        principal: Initial loan amount in €
        annual_rate_pct: Annual interest rate %
        duration_months: Original loan term in months
        months_paid: Number of months already paid

    Returns:
        Remaining balance in €
    """
    if months_paid >= duration_months:
        return 0.0

    if months_paid <= 0:
        return principal

    monthly_rate = (annual_rate_pct / 100.0) / 12.0

    if monthly_rate <= 0:
        return principal * (1 - months_paid / duration_months)

    # Calculate remaining balance using the loan amortization formula
    # Balance = P * [(1+r)^n - (1+r)^p] / [(1+r)^n - 1]
    # where P = principal, r = monthly rate, n = total months, p = months paid
    factor_n = (1 + monthly_rate) ** duration_months
    factor_p = (1 + monthly_rate) ** months_paid

    remaining = principal * (factor_n - factor_p) / (factor_n - 1)

    return max(0.0, remaining)

