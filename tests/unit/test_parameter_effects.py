"""Parameter effect tests.

Verifies that changing UI parameters actually affects the output.
Based on docs/PARAMETER_VERIFICATION.md.
"""

from typing import Any

import pytest

from src.services.brick_factory import (
    FinancingConfig,
    OperatingConfig,
    create_investment_bricks,
)
from src.services.strategy_finder import StrategyFinder


# Test fixtures
@pytest.fixture
def sample_archetypes() -> list[dict[str, Any]]:
    """Minimal archetypes for testing."""
    return [
        {
            "nom": "Test Apt 1",
            "ville": "Paris",
            "surface": 30,
            "prix_m2": 5000,
            "loyer_m2": 20,
            "charges_m2_an": 30,
            "taxe_fonciere_m2_an": 15,
            "dpe_initial": "C",
            "indice_tension": 0.8,
            "transport_score": 0.9,
            "liquidite_score": 0.7,
            "soumis_encadrement": False,
            "budget_travaux": 5000,
            "valeur_mobilier": 3000,
        },
        {
            "nom": "Test Apt 2",
            "ville": "Lyon",
            "surface": 40,
            "prix_m2": 4000,
            "loyer_m2": 15,
            "charges_m2_an": 25,
            "taxe_fonciere_m2_an": 12,
            "dpe_initial": "D",
            "indice_tension": 0.6,
            "transport_score": 0.7,
            "liquidite_score": 0.5,
            "soumis_encadrement": False,
            "budget_travaux": 8000,
            "valeur_mobilier": 4000,
        },
        {
            "nom": "Test Apt 3",
            "ville": "Marseille",
            "surface": 50,
            "prix_m2": 3000,
            "loyer_m2": 12,
            "charges_m2_an": 20,
            "taxe_fonciere_m2_an": 10,
            "dpe_initial": "B",
            "indice_tension": 0.9,
            "transport_score": 0.5,
            "liquidite_score": 0.8,
            "soumis_encadrement": False,
            "budget_travaux": 3000,
            "valeur_mobilier": 2000,
        },
    ]


@pytest.fixture
def default_finance_config() -> FinancingConfig:
    return FinancingConfig(
        credit_rates={20: 3.4, 25: 3.6},
        frais_notaire_pct=7.5,
        apport_min_pct=0.0,
        assurance_ann_pct=0.36,
        frais_pret_pct=1.0,
    )


@pytest.fixture
def default_operating_config() -> OperatingConfig:
    return OperatingConfig(
        frais_gestion_pct=5.0,
        provision_pct=3.0,
        cfe_par_bien_ann=500.0,
    )


class TestQualScoreVariation:
    """Verify qualitative scores vary between properties."""

    def test_qual_scores_vary_across_bricks(
        self,
        sample_archetypes,
        default_finance_config,
        default_operating_config,
    ):
        """Properties should have different quality scores."""
        bricks = create_investment_bricks(
            sample_archetypes,
            default_finance_config,
            default_operating_config,
        )

        scores = [b.qual_score_bien for b in bricks]
        unique_scores = set(scores)

        # Should have more than one unique score
        assert len(unique_scores) > 1, f"All qual_scores are identical: {scores}"

    def test_qual_scores_in_valid_range(
        self,
        sample_archetypes,
        default_finance_config,
        default_operating_config,
    ):
        """Quality scores should be 0-100."""
        bricks = create_investment_bricks(
            sample_archetypes,
            default_finance_config,
            default_operating_config,
        )

        for brick in bricks:
            score = brick.qual_score_bien
            assert 0 <= score <= 100, f"Score {score} out of range for {brick.nom}"


class TestParameterEffects:
    """Verify parameters have measurable effects."""

    def test_interest_rate_affects_pmt(
        self,
        sample_archetypes,
        default_operating_config,
    ):
        """Higher interest rate should increase monthly payment."""
        low_rate = FinancingConfig(
            credit_rates={25: 3.0},
            frais_notaire_pct=7.5,
        )
        high_rate = FinancingConfig(
            credit_rates={25: 5.0},
            frais_notaire_pct=7.5,
        )

        bricks_low = create_investment_bricks(sample_archetypes, low_rate, default_operating_config)
        bricks_high = create_investment_bricks(sample_archetypes, high_rate, default_operating_config)

        # Compare same property
        pmt_low = bricks_low[0].pmt_total
        pmt_high = bricks_high[0].pmt_total

        assert pmt_high > pmt_low, f"Higher rate should increase PMT: {pmt_low} vs {pmt_high}"

    def test_gestion_pct_affects_expenses(
        self,
        sample_archetypes,
        default_finance_config,
    ):
        """Higher management fee should increase expenses."""
        low_gestion = OperatingConfig(frais_gestion_pct=3.0)
        high_gestion = OperatingConfig(frais_gestion_pct=10.0)

        bricks_low = create_investment_bricks(sample_archetypes, default_finance_config, low_gestion)
        bricks_high = create_investment_bricks(sample_archetypes, default_finance_config, high_gestion)

        exp_low = bricks_low[0].depenses_mensuelles_hors_credit_initial
        exp_high = bricks_high[0].depenses_mensuelles_hors_credit_initial

        assert exp_high > exp_low, f"Higher gestion should increase expenses: {exp_low} vs {exp_high}"


class TestPresetRanking:
    """Verify different presets produce different rankings."""

    def test_preset_changes_affect_ranking(self):
        """Different presets should potentially reorder strategies."""
        finder = StrategyFinder([], 100000, -100)

        strategies = [
            {
                "balanced_score": 0.7,
                "dscr_norm": 0.9,  # High safety
                "tri_norm": 0.4,  # Low IRR
                "cf_proximity": 0.5,
                "finance_score": 0.6,
                "enrich_norm": 0.5,
            },
            {
                "balanced_score": 0.8,
                "dscr_norm": 0.4,  # Low safety
                "tri_norm": 0.9,  # High IRR
                "cf_proximity": 0.5,
                "finance_score": 0.7,
                "enrich_norm": 0.7,
            },
        ]

        # Safety preset should prefer high DSCR
        ranked_safety = finder.rank_strategies(strategies, "Sécurité (DSCR)")
        assert ranked_safety[0]["dscr_norm"] == 0.9

        # IRR preset should prefer high TRI
        ranked_irr = finder.rank_strategies(strategies, "Rendement / IRR")
        assert ranked_irr[0]["tri_norm"] == 0.9
