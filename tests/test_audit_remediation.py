"""
Tests for Zero-Trust Audit Remediation.
Verifies fixes for critical risks and edge cases identified in the audit.
"""
import pytest

from src.application.services.simulation import IRACalculator, MarketHypotheses, SimulationEngine, TaxParams


class TestAuditRemediation:

    def test_simulation_zero_amortization_error(self):
        """Verify explicit ValueError on zero amortization years."""
        market = MarketHypotheses()
        tax = TaxParams()
        ira = IRACalculator()

        with pytest.raises(ValueError, match="amort_immo_years must be > 0"):
            SimulationEngine(market, tax, ira, amort_immo_years=0)
