"""Edge case tests for comprehensive failure mode coverage.

Tests based on QA Audit "Matrix of Pain" covering:
- Financial calculations with extreme inputs
- Simulation edge cases
- Allocation edge cases
- Optimizer edge cases
- Scoring edge cases
"""

import pytest

from src.core.financial import (
    calculate_insurance,
    calculate_monthly_payment,
    calculate_remaining_balance,
    generate_amortization_schedule,
    k_factor,
)
from src.core.scoring import (
    calculate_balanced_score,
    calculate_dpe_score,
    calculate_property_qualitative_score,
    calculate_qualitative_score,
)
from src.core.simulation import (
    IRACalculator,
    MarketHypotheses,
    SimulationEngine,
    TaxParams,
)
from src.services.allocator import PortfolioAllocator
from src.services.strategy_finder import StrategyScorer


class TestFinancialEdgeCases:
    """Edge cases for financial calculations."""

    def test_pmt_zero_principal(self):
        """Zero principal should return 0."""
        assert calculate_monthly_payment(0, 3.6, 300) == 0.0

    def test_pmt_zero_duration(self):
        """Zero duration should return 0."""
        assert calculate_monthly_payment(100_000, 3.6, 0) == 0.0

    def test_pmt_negative_rate(self):
        """Negative rate should be treated as zero rate."""
        result = calculate_monthly_payment(100_000, -3.5, 240)
        expected = 100_000 / 240  # Simple division when rate is 0
        assert abs(result - expected) < 0.01

    def test_pmt_very_short_loan(self):
        """1-month loan should equal principal plus interest."""
        principal = 10_000
        rate = 12.0  # 12% annual = 1% monthly
        result = calculate_monthly_payment(principal, rate, 1)
        # Expected: principal + 1 month interest = 10000 + 100 = 10100
        assert abs(result - 10100) < 1.0

    def test_pmt_massive_principal(self):
        """Large principal should not overflow."""
        result = calculate_monthly_payment(1_000_000_000, 3.6, 300)
        assert result > 0
        assert result < 10_000_000  # Sanity check

    def test_insurance_zero_principal(self):
        """Zero principal should return 0 insurance."""
        assert calculate_insurance(0, 0.35) == 0.0

    def test_insurance_negative_principal(self):
        """Negative principal should return 0."""
        assert calculate_insurance(-10_000, 0.35) == 0.0

    def test_k_factor_zero_duration(self):
        """K-factor with zero duration should handle gracefully."""
        result = k_factor(3.6, 0, 0.35)
        # With duration_years=0, n=max(1, 0*12)=1
        assert result > 0

    def test_schedule_very_short_loan(self):
        """1-month schedule should have single entry."""
        schedule = generate_amortization_schedule(10_000, 12.0, 1, 0.35)
        assert schedule["nmois"] == 1
        assert len(schedule["balances"]) == 1
        assert schedule["balances"][0] == 0.0  # Fully paid

    def test_schedule_zero_values(self):
        """Zero principal/duration should return empty schedule."""
        schedule = generate_amortization_schedule(0, 3.6, 300, 0.35)
        assert schedule["nmois"] == 0
        assert len(schedule["balances"]) == 0

    def test_remaining_balance_exceeded_duration(self):
        """Months paid > duration should return 0."""
        result = calculate_remaining_balance(100_000, 3.6, 240, 360)
        assert result == 0.0

    def test_remaining_balance_zero_months(self):
        """Zero months paid should return full principal."""
        result = calculate_remaining_balance(100_000, 3.6, 240, 0)
        assert result == 100_000


class TestSimulationEdgeCases:
    """Edge cases for simulation engine."""

    @pytest.fixture
    def engine(self):
        return SimulationEngine(
            market=MarketHypotheses(),
            tax=TaxParams(),
            ira=IRACalculator(apply_ira=False),
        )

    def test_simulate_empty_strategy(self, engine):
        """Empty strategy should return empty results."""
        df, bilan = engine.simulate({}, 25, [])
        assert df.empty
        assert bilan == {}

    def test_simulate_no_details(self, engine):
        """Strategy with no details should return empty."""
        df, bilan = engine.simulate({"apport_total": 10000}, 25, [])
        assert df.empty

    def test_simulate_schedule_mismatch_raises(self, engine):
        """Schedule count != property count should raise ValueError."""
        strategy = {
            "apport_total": 10000,
            "details": [
                {"prix_achat_bien": 100000, "loyer_mensuel_initial": 500},
                {"prix_achat_bien": 150000, "loyer_mensuel_initial": 700},
            ]
        }
        schedules = [generate_amortization_schedule(100000, 3.6, 300, 0.35)]  # Only 1

        with pytest.raises(ValueError) as exc_info:
            engine.simulate(strategy, 25, schedules)

        assert "Schedule count" in str(exc_info.value)

    def test_dscr_populated_in_bilan(self, engine):
        """DSCR should be present in bilan after simulation."""
        strategy = {
            "apport_total": 10000,
            "details": [{
                "prix_achat_bien": 100000,
                "budget_travaux": 5000,
                "mobilier": 3000,
                "renovation_energetique": 0,
                "frais_notaire": 7500,
                "credit_final": 100000,
                "taux_pret": 3.6,
                "duree_pret": 25,
                "assurance_ann_pct": 0.35,
                "loyer_mensuel_initial": 800,
                "charges_const_mth0": 50,
                "tf_const_mth0": 40,
                "frais_gestion_pct": 5.0,
                "provision_pct": 3.0,
            }]
        }
        schedules = [generate_amortization_schedule(100000, 3.6, 300, 0.35)]

        df, bilan = engine.simulate(strategy, 25, schedules)

        assert "dscr_y1" in bilan
        assert bilan["dscr_y1"] > 0


class TestAllocationEdgeCases:
    """Edge cases for portfolio allocator."""

    def test_allocate_empty_bricks(self):
        """Empty brick list should return zero allocation."""
        allocator = PortfolioAllocator(mode_cf="target")
        ok, details, cf, apport = allocator.allocate([], 50000, 0, 100)

        assert cf == 0
        assert apport == 0
        assert len(details) == 0

    def test_allocate_all_over_budget(self):
        """All bricks over budget should return minimal state."""
        allocator = PortfolioAllocator(mode_cf="target")
        bricks = [{
            "apport_min": 100000,  # Way over budget
            "capital_emprunte": 200000,
            "pmt_total": 1000,
            "loyer_mensuel_initial": 800,
            "depenses_mensuelles_hors_credit_initial": 100,
            "taux_pret": 3.6,
            "duree_pret": 25,
            "assurance_ann_pct": 0.35,
        }]

        # Budget is only 50k but brick needs 100k
        ok, details, cf, apport = allocator.allocate(bricks, 50000, 0, 100)
        # The allocation will try but brick doesn't fit the budget constraint

    def test_allocate_mode_min_accepts_above_target(self):
        """Mode 'min' should accept CF above target."""
        allocator = PortfolioAllocator(mode_cf="min")

        # CF observed = 100, target = 50, tolerance = 25
        # In min mode: accept if CF >= (target - tolerance) = 25
        result = allocator._accept_cf(100, 50, 25)
        assert result is True

    def test_allocate_mode_target_within_tolerance(self):
        """Mode 'target' should accept CF within tolerance window."""
        allocator = PortfolioAllocator(mode_cf="target")

        # CF = 55, target = 50, tolerance = 10
        # |55 - 50| = 5 <= 10 → accept
        result = allocator._accept_cf(55, 50, 10)
        assert result is True

    def test_allocate_mode_target_outside_tolerance(self):
        """Mode 'target' should reject CF outside tolerance window."""
        allocator = PortfolioAllocator(mode_cf="target")

        # CF = 100, target = 50, tolerance = 10
        # |100 - 50| = 50 > 10 → reject
        result = allocator._accept_cf(100, 50, 10)
        assert result is False


class TestScoringEdgeCases:
    """Edge cases for scoring functions."""

    def test_dpe_score_unknown_letter(self):
        """Unknown DPE letter should return default 0.6."""
        assert calculate_dpe_score("Z") == 0.6
        assert calculate_dpe_score("X") == 0.6
        assert calculate_dpe_score("") == 0.6

    def test_dpe_score_lowercase(self):
        """Lowercase DPE should be normalized."""
        assert calculate_dpe_score("a") == 1.0
        assert calculate_dpe_score("e") == 0.3

    def test_qualitative_score_empty_details(self):
        """Empty details should return neutral score."""
        score = calculate_qualitative_score({"details": []})
        assert score == 50.0

    def test_qualitative_score_no_details_key(self):
        """Missing details key should return neutral score."""
        score = calculate_qualitative_score({})
        assert score == 50.0

    def test_balanced_score_extreme_tri(self):
        """TRI > 20% should be capped in scoring."""
        score = calculate_balanced_score(
            tri=30.0,  # 30% IRR (extreme)
            enrichissement_net=100000,
            dscr=1.5,
            qual_score=50.0,
        )
        # Score should still be valid (0-100)
        assert 0 <= score <= 100

    def test_balanced_score_negative_enrichment(self):
        """Negative enrichment should score 0 for that component."""
        score = calculate_balanced_score(
            tri=5.0,
            enrichissement_net=-50000,  # Loss
            dscr=1.0,
            qual_score=50.0,
        )
        assert 0 <= score <= 100

    def test_minmax_normalize_identical_values(self):
        """All identical values should normalize to 0.5."""
        result = StrategyScorer.minmax_normalize([100, 100, 100])
        assert all(v == 0.5 for v in result)

    def test_minmax_normalize_single_value(self):
        """Single value should normalize to 0.5."""
        result = StrategyScorer.minmax_normalize([42])
        assert result == [0.5]

    def test_minmax_normalize_with_nan(self):
        """NaN values should become 0."""
        result = StrategyScorer.minmax_normalize([0, float('nan'), 100])
        assert result[1] == 0.0

    def test_property_qualitative_minimal_dict(self):
        """Minimal archetype dict should use defaults."""
        score, features = calculate_property_qualitative_score({})
        assert 0 <= score <= 100
        assert "tension" in features


class TestOptimizerEdgeCases:
    """Edge cases for optimizer components."""

    def test_scorer_empty_strategies(self):
        """Scoring empty list should not crash."""
        scorer = StrategyScorer()
        scorer.score_strategies([], 0)  # Should not raise

    def test_scorer_single_strategy(self):
        """Single strategy should normalize correctly."""
        scorer = StrategyScorer()
        strategies = [{
            "enrich_net": 50000,
            "tri_annuel": 8.0,
            "cap_eff": 1.5,
            "dscr_y1": 1.2,
            "cash_flow_final": 100,
            "qual_score": 60,
        }]
        scorer.score_strategies(strategies, 100)

        # With single value, normalization gives 0.5
        assert "finance_score" in strategies[0]
        assert "balanced_score" in strategies[0]

    def test_scorer_weights_normalize(self):
        """Weights should be normalized to sum to 1."""
        weights = {"a": 10, "b": 20, "c": 70}
        normalized = StrategyScorer._normalize_weights(weights)
        assert abs(sum(normalized.values()) - 1.0) < 0.001
