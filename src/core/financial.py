"""Financial calculation functions.

Core loan and amortization calculations for real estate investments.
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple

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
) -> Tuple[float, float, float]:
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
) -> List[Dict[str, Any]]:
    """Generate full loan amortization schedule.
    
    Args:
        principal: Loan amount in €
        annual_rate_pct: Annual interest rate %
        duration_months: Loan term in months
        annual_insurance_pct: Annual insurance rate %
        
    Returns:
        List of monthly records with:
        - mois: Month number (1-indexed)
        - capital_restant_debut: Balance at start of month
        - interet: Interest payment
        - principal: Principal payment
        - assurance: Insurance payment
        - paiement_total: Total payment
        - capital_restant_fin: Balance at end of month
    """
    if principal <= 0 or duration_months <= 0:
        return []
    
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
    
    return schedule


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
    
    # FV of the remaining balance
    pmt = calculate_monthly_payment(principal, annual_rate_pct, duration_months)
    fv = npf.fv(monthly_rate, months_paid, -pmt, -principal)
    
    return max(0.0, fv)
