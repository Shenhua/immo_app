"""
Structural mathematical verification tests.

These tests verify core financial math invariants and constraints
that must hold across the entire simulation engine.

Unlike unit tests, these focus on mathematical correctness and
constraint satisfaction rather than code branching.
"""

import pytest
from src.domain.calculator.financial import (
    calculate_monthly_payment,
    calculate_remaining_balance,
    generate_amortization_schedule,
)


class TestAmortizationMathInvariants:
    """Verify that amortization schedules satisfy key mathematical invariants."""

    def test_total_payments_exceed_principal(self):
        """Sum of all P&I payments must exceed the original principal (interest > 0)."""
        principal = 100_000
        rate = 3.6
        months = 300
        schedule = generate_amortization_schedule(principal, rate, months, 0.0)
        total_payments = sum(schedule["paiement_total"])
        assert total_payments > principal, "Total payments must exceed principal (interest must be positive)"

    def test_balance_decreases_monotonically(self):
        """Remaining balance must decrease monotonically over time."""
        schedule = generate_amortization_schedule(100_000, 3.6, 300, 0.0)
        balances = schedule["balances"]
        for i in range(1, len(balances)):
            assert balances[i] <= balances[i - 1], (
                f"Balance at month {i} ({balances[i]:.2f}) increased from {balances[i-1]:.2f}"
            )

    def test_final_balance_near_zero(self):
        """After all payments, remaining balance must be near zero."""
        schedule = generate_amortization_schedule(100_000, 3.6, 300, 0.0)
        final_balance = schedule["balances"][-1]
        assert abs(final_balance) < 1.0, f"Final balance {final_balance:.4f} must be < 1€"

    def test_interest_decreases_over_time(self):
        """Interest component of each payment should trend downward."""
        schedule = generate_amortization_schedule(100_000, 3.6, 300, 0.0)
        first_year_interest = sum(schedule["interet"][:12])
        last_year_interest = sum(schedule["interet"][-12:])
        assert first_year_interest > last_year_interest, (
            "Interest must decrease over the life of the loan"
        )

    def test_schedule_length_matches_duration(self):
        """Generated schedule must have exactly the requested number of rows."""
        for months in [180, 240, 300]:
            schedule = generate_amortization_schedule(100_000, 3.0, months, 0.0)
            assert schedule["nmois"] == months, f"Expected {months} rows, got {schedule['nmois']}"

    def test_zero_rate_loan(self):
        """A 0% rate loan should have no interest component (payment = principal/term)."""
        principal = 120_000
        months = 120
        pmt = calculate_monthly_payment(principal, 0.0, months)
        assert abs(pmt - principal / months) < 0.01, (
            f"0% rate pmt {pmt:.2f} should equal {principal/months:.2f}"
        )

    def test_higher_rate_means_higher_payment(self):
        """Higher interest rate must produce a higher monthly payment."""
        pmt_low = calculate_monthly_payment(100_000, 2.0, 300)
        pmt_high = calculate_monthly_payment(100_000, 4.0, 300)
        assert pmt_high > pmt_low, "Higher rate should yield higher PMT"

    def test_remaining_balance_halfway_exceeds_half_original(self):
        """At halfway point (month 150 of 300), balance should be >50% of principal.
        
        This is because early payments are interest-heavy, so less principal is
        paid off in the first half of the loan term.
        """
        schedule = generate_amortization_schedule(100_000, 3.6, 300, 0.0)
        # Month 150 = index 149 in the balances list
        halfway_balance = schedule["balances"][149]
        assert halfway_balance > 50_000, (
            f"Balance at midpoint {halfway_balance:.0f}€ should be >50k (interest-heavy early payments)"
        )


class TestGoldenMathChecks:
    """Key golden value checks to protect against regression of core formulas."""

    def test_pmt_100k_36pct_25y(self):
        """100 000 € @ 3.6% over 25 years should be ~506 €/month."""
        pmt = calculate_monthly_payment(100_000, 3.6, 300)
        assert abs(pmt - 506) < 1.0, f"Expected ~506€, got {pmt:.2f}€"

    def test_pmt_83426_36pct_25y(self):
        """83 426 € @ 3.6% over 25 years should be ~422 €/month."""
        pmt = calculate_monthly_payment(83_426, 3.6, 300)
        assert abs(pmt - 422) < 2.0, f"Expected ~422€, got {pmt:.2f}€"

    def test_pmt_150k_34pct_20y(self):
        """150 000 € @ 3.4% over 20 years.
        
        numpy_financial.pmt(3.4%/12, 240, -150000) = 862.25€
        """
        pmt = calculate_monthly_payment(150_000, 3.4, 240)
        # Verified via numpy_financial: 862.25€ is the correct value
        assert abs(pmt - 862.25) < 1.0, f"Expected ~862€, got {pmt:.2f}€"

    def test_insurance_100k_35pct(self):
        """Insurance for 100 000 € at 0.35% annual = 29.17 €/month."""
        schedule = generate_amortization_schedule(100_000, 3.6, 300, 0.35)
        # First month insurance: 100_000 * 0.0035 / 12 = 29.17
        first_ins = schedule["assurance"][0]
        assert abs(first_ins - 29.17) < 0.5, f"Expected 29.17€ insurance, got {first_ins:.2f}€"


class TestEdgeCaseBoundaries:
    """Verify graceful handling of boundary conditions."""

    def test_zero_principal_returns_zero(self):
        """Zero principal should produce zero payment."""
        pmt = calculate_monthly_payment(0, 3.6, 300)
        assert pmt == 0.0

    def test_negative_principal_returns_zero(self):
        """Negative principal should return zero (not crash)."""
        pmt = calculate_monthly_payment(-100_000, 3.6, 300)
        assert pmt == 0.0

    def test_zero_months_returns_zero(self):
        """Zero duration should return zero payment."""
        pmt = calculate_monthly_payment(100_000, 3.6, 0)
        assert pmt == 0.0

    def test_remaining_balance_month_zero(self):
        """Remaining balance at month 0 should equal full principal."""
        balance = calculate_remaining_balance(100_000, 3.6, 300, 0)
        assert abs(balance - 100_000) < 1.0
