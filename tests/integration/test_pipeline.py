"""Integration tests for the strategy finding pipeline.

Tests the complete flow from archetypes → bricks → strategies → simulation.
"""

import os
import sys

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestStrategyPipeline:
    """Integration tests for the strategy finding pipeline."""

    @pytest.fixture
    def sample_archetypes(self):
        """Load sample archetypes from test data."""
        return [
            {
                "nom": "Studio Paris 11",
                "ville": "Paris",
                "mode_loyer": "meuble_classique",
                "surface": 25.0,
                "prix_m2": 10000.0,
                "loyer_m2": 35.0,
                "loyer_m2_max": 40.0,
                "charges_m2_an": 50.0,
                "taxe_fonciere_m2_an": 30.0,
                "meuble": True,
                "soumis_encadrement": True,
                "dpe_initial": "C",
                "budget_travaux": 5000.0,
                "valeur_mobilier": 3000.0,
                "indice_tension": 0.8,
                "transport_score": 0.9,
                "liquidite_score": 0.7,
            },
            {
                "nom": "T2 Lyon 3",
                "ville": "Lyon",
                "mode_loyer": "meuble_classique",
                "surface": 45.0,
                "prix_m2": 5000.0,
                "loyer_m2": 18.0,
                "charges_m2_an": 35.0,
                "taxe_fonciere_m2_an": 25.0,
                "meuble": True,
                "soumis_encadrement": False,
                "dpe_initial": "D",
                "budget_travaux": 8000.0,
                "valeur_mobilier": 4000.0,
                "indice_tension": 0.6,
                "transport_score": 0.7,
                "liquidite_score": 0.6,
            },
        ]

    def test_create_bricks_from_archetypes(self, sample_archetypes):
        """Test brick generation from archetypes."""
        from src.application.services.brick_factory import FinancingConfig, OperatingConfig, create_investment_bricks

        taux_credits = {15: 3.2, 20: 3.4, 25: 3.6}

        fin_config = FinancingConfig(
            credit_rates=taux_credits,
            frais_notaire_pct=7.5,
            apport_min_pct=10.0,
            assurance_ann_pct=0.36,
            frais_pret_pct=1.0,
            inclure_travaux=True,
            inclure_reno_ener=True,
            inclure_mobilier=True,
            financer_mobilier=True,
        )

        op_config = OperatingConfig(
            frais_gestion_pct=5.0,
            provision_pct=5.0,
            cfe_par_bien_ann=150.0,
        )

        briques = create_investment_bricks(
            archetypes=sample_archetypes,
            finance=fin_config,
            operating=op_config,
        )

        assert len(briques) > 0
        # Should create variants for different loan durations
        assert len(briques) >= len(sample_archetypes)

        # Check brick structure
        first_brick = briques[0]
        assert "nom_bien" in first_brick
        assert "cout_total" in first_brick
        assert "pmt_total" in first_brick
        assert first_brick["cout_total"] > 0

    def test_archetype_validation(self, sample_archetypes):
        """Test archetype validation via Pydantic model."""
        from src.domain.models.archetype import ArchetypeV2

        validated = []
        for a in sample_archetypes:
            obj = ArchetypeV2(**a)
            validated.append(obj)

        assert len(validated) == len(sample_archetypes)
        assert all(isinstance(v, ArchetypeV2) for v in validated)

    def test_qualitative_scoring_integration(self, sample_archetypes):
        """Test qualitative scoring on real archetype data."""
        from src.domain.calculator.scoring import calculate_property_qualitative_score

        for arch in sample_archetypes:
            score, features = calculate_property_qualitative_score(
                arch,
                loyer_m2=arch.get("loyer_m2"),
                loyer_m2_max=arch.get("loyer_m2_max"),
                prix_achat=arch["surface"] * arch["prix_m2"],
                travaux=arch.get("budget_travaux", 0),
            )

            assert 0 <= score <= 100
            assert "tension" in features
            assert "transport" in features
            assert "dpe" in features

    def test_financial_calculations_integration(self):
        """Test financial calculations with realistic values."""
        from src.domain.calculator.financial import (
            calculate_monthly_payment,
            calculate_remaining_balance,
            generate_amortization_schedule,
        )

        # Typical Paris studio financing
        principal = 275000  # 25m² × 10000€ + frais
        rate = 3.5
        duration = 240  # 20 years

        # Monthly payment
        pmt = calculate_monthly_payment(principal, rate, duration)
        assert 1500 < pmt < 1700  # Reasonable range

        # Amortization
        schedule = generate_amortization_schedule(principal, rate, duration)
        assert schedule["nmois"] == 240
        assert len(schedule["mois"]) == 240
        assert schedule["balances"][-1] < 1.0

        # Remaining balance at 10 years
        bal_120 = calculate_remaining_balance(principal, rate, duration, 120)
        assert 100000 < bal_120 < 180000  # About half paid off


class TestLoggingIntegration:
    """Test logging configuration."""

    def test_logger_configuration(self):
        """Test that logger can be configured."""
        from src.core.logging import configure_logging, get_logger

        # Should not raise
        log = configure_logging(level="DEBUG", json_output=False)
        assert log is not None

        named_log = get_logger("test_module")
        assert named_log is not None
