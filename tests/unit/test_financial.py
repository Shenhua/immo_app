"""Unit tests for src.core.financial module."""

import pytest
from src.core.financial import (
    calculate_monthly_payment,
    calculate_insurance,
    calculate_total_monthly_payment,
    generate_amortization_schedule,
    calculate_remaining_balance,
)


class TestCalculateMonthlyPayment:
    """Tests for calculate_monthly_payment function."""
    
    def test_standard_loan(self):
        """Test a standard 20-year loan at 3.5%."""
        pmt = calculate_monthly_payment(200000, 3.5, 240)
        # Expected around 1160€/month
        assert 1150 < pmt < 1170
    
    def test_zero_principal(self):
        """Zero principal should return zero payment."""
        assert calculate_monthly_payment(0, 3.5, 240) == 0.0
    
    def test_zero_rate(self):
        """Zero interest rate should return principal/months."""
        pmt = calculate_monthly_payment(120000, 0.0, 120)
        assert pmt == 1000.0
    
    def test_short_term(self):
        """Short term loan should have higher payments."""
        pmt_15 = calculate_monthly_payment(200000, 3.5, 180)
        pmt_25 = calculate_monthly_payment(200000, 3.5, 300)
        assert pmt_15 > pmt_25


class TestCalculateInsurance:
    """Tests for calculate_insurance function."""
    
    def test_standard_insurance(self):
        """Test typical insurance rate."""
        ins = calculate_insurance(200000, 0.36)
        assert ins == 60.0  # (200000 * 0.0036) / 12
    
    def test_zero_principal(self):
        """Zero principal should return zero insurance."""
        assert calculate_insurance(0, 0.36) == 0.0


class TestCalculateTotalMonthlyPayment:
    """Tests for calculate_total_monthly_payment function."""
    
    def test_returns_tuple(self):
        """Should return tuple of (P&I, insurance, total)."""
        pmt_pi, pmt_ins, total = calculate_total_monthly_payment(200000, 3.5, 240, 0.36)
        assert isinstance(pmt_pi, float)
        assert isinstance(pmt_ins, float)
        assert total == pmt_pi + pmt_ins


class TestGenerateAmortizationSchedule:
    """Tests for generate_amortization_schedule function."""
    
    def test_schedule_length(self):
        """Schedule should have correct number of months."""
        schedule = generate_amortization_schedule(100000, 3.0, 120)
        assert schedule["nmois"] == 120
        assert len(schedule["mois"]) == 120
    
    def test_schedule_structure(self):
        """Each month should have required fields."""
        schedule = generate_amortization_schedule(100000, 3.0, 120)
        required_keys = {"mois", "capital_restant_debut", "interet", "principal", 
                         "assurance", "paiement_total", "capital_restant_fin",
                         "nmois", "entries", "balances", "interets", "principals"}
        # Check primary keys exist
        keys = set(schedule.keys())
        assert {"nmois", "pmt_total", "interets", "principals", "balances"}.issubset(keys)
        
        # Check parallelism
        assert len(schedule["interet"]) == 120
        assert len(schedule["capital_restant_fin"]) == 120
    
    def test_final_balance_zero(self):
        """Final balance should be essentially zero."""
        schedule = generate_amortization_schedule(100000, 3.0, 120)
        assert schedule["balances"][-1] < 1.0
    
    def test_empty_for_zero_principal(self):
        """Zero principal should return empty schedule structure."""
        sch = generate_amortization_schedule(0, 3.0, 120)
        assert sch["nmois"] == 0
        assert len(sch["mois"]) == 0


class TestCalculateRemainingBalance:
    """Tests for calculate_remaining_balance function."""
    
    def test_initial_balance(self):
        """At month 0, balance equals principal."""
        bal = calculate_remaining_balance(200000, 3.5, 240, 0)
        assert bal == 200000
    
    def test_final_balance(self):
        """At end of term, balance is zero."""
        bal = calculate_remaining_balance(200000, 3.5, 240, 240)
        assert bal == 0.0
    
    def test_partial_payoff(self):
        """Balance should decrease over time."""
        bal_60 = calculate_remaining_balance(200000, 3.5, 240, 60)
        bal_180 = calculate_remaining_balance(200000, 3.5, 240, 180)
        assert bal_60 > bal_180
