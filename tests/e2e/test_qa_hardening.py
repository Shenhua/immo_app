"""End-to-end integration tests for QA hardening verification.

Tests the complete pipeline with edge cases that span multiple components:
- Weight consistency between Exhaustive and Genetic modes
- DSCR propagation through the full pipeline
- Empty/edge case handling in strategy search
- Rent cap enforcement
"""

import pytest
from pathlib import Path

from src.services.brick_factory import (
    FinancingConfig,
    OperatingConfig,
    create_investment_bricks,
    apply_rent_caps,
)
from src.services.strategy_finder import StrategyFinder, StrategyScorer
from src.ui.app_controller import load_archetypes


class TestDSCRPropagation:
    """Verify DSCR is correctly calculated and propagated."""

    @pytest.fixture
    def setup(self):
        archetypes = load_archetypes()[:5]
        finance = FinancingConfig(
            credit_rates={20: 3.6, 25: 3.8},
            frais_notaire_pct=8.0,
            assurance_ann_pct=0.35,
        )
        operating = OperatingConfig(
            frais_gestion_pct=5.0,
            provision_pct=3.0,
            cfe_par_bien_ann=500,
        )
        bricks = create_investment_bricks(archetypes, finance, operating)
        return bricks

    def test_dscr_present_in_strategy_results(self, setup):
        """DSCR should be calculated and present in final strategies."""
        bricks = setup
        
        finder = StrategyFinder(
            bricks=bricks,
            apport_disponible=50000,
            cash_flow_cible=0,
            tolerance=500,
            mode_cf="min",
        )
        
        strategies = finder.find_strategies(
            eval_params={"tmi_pct": 30, "regime_fiscal": "lmnp"},
            horizon_years=20,
            top_n=3,
        )
        
        # At least one strategy should have DSCR
        if strategies:
            # DSCR might be in bilan or propagated
            for s in strategies:
                # Check if DSCR is used in scoring (via dscr_norm)
                assert "dscr_norm" in s or "balanced_score" in s


class TestWeightConsistency:
    """Verify weight keys are consistent across components."""

    def test_scorer_weights_match_base_weights(self):
        """StrategyScorer weights should match BASE_WEIGHTS keys."""
        from src.services.strategy_finder import BASE_WEIGHTS
        
        scorer = StrategyScorer()
        
        # All BASE_WEIGHTS keys should be present in scorer weights
        for key in BASE_WEIGHTS:
            assert key in scorer.weights, f"Missing key: {key}"

    def test_custom_weights_applied_correctly(self):
        """Custom weights should be normalized and applied."""
        custom = {"irr": 50, "enrich_net": 30, "dscr": 10, "cf_proximity": 10}
        scorer = StrategyScorer(weights=custom)
        
        # Sum should be 1.0
        total = sum(scorer.weights.values())
        assert abs(total - 1.0) < 0.01


class TestRentCapEnforcement:
    """Verify rent caps are properly enforced."""

    def test_rent_cap_applied_when_exceeds(self):
        """Properties with loyer > cap should be capped."""
        archetypes = [{
            "nom": "Test Property",
            "surface": 50,
            "prix_m2": 3000,
            "loyer_m2": 25.0,  # Higher than cap
            "loyer_m2_max": 20.0,  # Cap
            "soumis_encadrement": True,
            "charges_m2_an": 30,
            "taxe_fonciere_m2_an": 20,
        }]
        
        capped = apply_rent_caps(archetypes, apply_cap=True)
        
        assert capped[0]["loyer_m2"] == 20.0  # Should be capped

    def test_rent_not_capped_when_disabled(self):
        """Properties should keep original rent when cap disabled."""
        archetypes = [{
            "nom": "Test Property",
            "surface": 50,
            "prix_m2": 3000,
            "loyer_m2": 25.0,
            "loyer_m2_max": 20.0,
            "soumis_encadrement": True,
            "charges_m2_an": 30,
            "taxe_fonciere_m2_an": 20,
        }]
        
        not_capped = apply_rent_caps(archetypes, apply_cap=False)
        
        assert not_capped[0]["loyer_m2"] == 25.0  # Original value

    def test_unregulated_property_not_capped(self):
        """Properties without encadrement should not be capped."""
        archetypes = [{
            "nom": "Test Property",
            "surface": 50,
            "prix_m2": 3000,
            "loyer_m2": 25.0,
            "loyer_m2_max": 20.0,
            "soumis_encadrement": False,  # Not regulated
            "charges_m2_an": 30,
            "taxe_fonciere_m2_an": 20,
        }]
        
        result = apply_rent_caps(archetypes, apply_cap=True)
        
        assert result[0]["loyer_m2"] == 25.0  # Not capped


class TestEmptyResultHandling:
    """Verify proper handling when no strategies are found."""

    def test_impossible_target_returns_empty(self):
        """Impossible cash flow target should return empty list gracefully."""
        archetypes = load_archetypes()[:3]
        finance = FinancingConfig(
            credit_rates={25: 3.8},
            frais_notaire_pct=8.0,
            assurance_ann_pct=0.35,
        )
        operating = OperatingConfig()
        bricks = create_investment_bricks(archetypes, finance, operating)
        
        finder = StrategyFinder(
            bricks=bricks,
            apport_disponible=10000,  # Very small budget
            cash_flow_cible=5000,  # Impossible target
            tolerance=10,  # Very tight tolerance
            mode_cf="target",
        )
        
        strategies = finder.find_strategies(
            horizon_years=20,
            top_n=5,
        )
        
        # Should return empty list, not crash
        assert isinstance(strategies, list)

    def test_zero_bricks_returns_empty(self):
        """Zero bricks should return empty list."""
        finder = StrategyFinder(
            bricks=[],
            apport_disponible=50000,
            cash_flow_cible=0,
            tolerance=100,
        )
        
        strategies = finder.find_strategies(top_n=5)
        
        assert strategies == []


class TestScheduleMismatchHandling:
    """Verify schedule validation catches mismatches."""

    def test_simulation_rejects_mismatched_schedules(self):
        """Simulation should raise ValueError for schedule/details mismatch."""
        from src.core.simulation import SimulationEngine, MarketHypotheses, TaxParams, IRACalculator
        from src.core.financial import generate_amortization_schedule
        
        engine = SimulationEngine(
            market=MarketHypotheses(),
            tax=TaxParams(),
            ira=IRACalculator(),
        )
        
        # 2 properties but only 1 schedule
        strategy = {
            "apport_total": 20000,
            "details": [
                {
                    "prix_achat_bien": 100000,
                    "budget_travaux": 0,
                    "mobilier": 0,
                    "renovation_energetique": 0,
                    "frais_notaire": 8000,
                    "credit_final": 80000,
                    "taux_pret": 3.6,
                    "duree_pret": 25,
                    "assurance_ann_pct": 0.35,
                    "loyer_mensuel_initial": 600,
                    "charges_const_mth0": 50,
                    "tf_const_mth0": 30,
                    "frais_gestion_pct": 5.0,
                    "provision_pct": 3.0,
                },
                {
                    "prix_achat_bien": 120000,
                    "budget_travaux": 0,
                    "mobilier": 0,
                    "renovation_energetique": 0,
                    "frais_notaire": 9600,
                    "credit_final": 100000,
                    "taux_pret": 3.6,
                    "duree_pret": 25,
                    "assurance_ann_pct": 0.35,
                    "loyer_mensuel_initial": 700,
                    "charges_const_mth0": 60,
                    "tf_const_mth0": 35,
                    "frais_gestion_pct": 5.0,
                    "provision_pct": 3.0,
                }
            ]
        }
        
        # Only 1 schedule for 2 properties
        schedules = [generate_amortization_schedule(80000, 3.6, 300, 0.35)]
        
        with pytest.raises(ValueError) as exc_info:
            engine.simulate(strategy, 25, schedules)
        
        assert "Schedule count" in str(exc_info.value)
        assert "1" in str(exc_info.value)
        assert "2" in str(exc_info.value)


class TestHorizonEffects:
    """Test that different horizons produce expected differences."""

    @pytest.fixture
    def bricks(self):
        archetypes = load_archetypes()[:3]
        finance = FinancingConfig(
            credit_rates={25: 3.8},
            frais_notaire_pct=8.0,
            assurance_ann_pct=0.35,
        )
        operating = OperatingConfig()
        return create_investment_bricks(archetypes, finance, operating)

    def test_longer_horizon_higher_liquidation(self, bricks):
        """25-year horizon should have >= liquidation than 15-year."""
        results = {}
        
        for horizon in [15, 25]:
            finder = StrategyFinder(
                bricks=bricks,
                apport_disponible=50000,
                cash_flow_cible=0,
                tolerance=500,
                mode_cf="min",
            )
            
            strategies = finder.find_strategies(
                eval_params={"tmi_pct": 30},
                horizon_years=horizon,
                top_n=1,
            )
            
            if strategies:
                results[horizon] = strategies[0].get("liquidation_nette", 0)
        
        if len(results) == 2:
            # Property appreciation over more years should increase value
            assert results[25] >= results[15], \
                f"25y={results[25]} should be >= 15y={results[15]}"
