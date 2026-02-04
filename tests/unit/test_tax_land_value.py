"""Unit tests for Land Value (Terrain) exclusion from amortization base.

Per French LMNP tax law, land value is NOT amortizable - only the building/construction
and improvements (travaux) can be depreciated.
"""

import pytest

from src.application.services.simulation import (
    IRACalculator,
    MarketHypotheses,
    SimulationEngine,
    TaxParams,
)
from src.domain.calculator.financial import generate_amortization_schedule


class TestLandValueExclusion:
    """Tests for land value exclusion from amortization."""

    def test_zero_land_ratio_amortizes_full_property(self):
        """With ratio_terrain_pct=0, should amortize 100% of property + works."""
        engine = SimulationEngine(
            market=MarketHypotheses(),
            tax=TaxParams(regime_fiscal="lmnp"),
            ira=IRACalculator(apply_ira=False),
            ratio_terrain_pct=0.0,  # No land exclusion
        )

        strategy = {
            "apport_total": 20000,
            "details": [{
                "prix_achat_bien": 100000,
                "budget_travaux": 10000,
                "mobilier": 5000,
                "renovation_energetique": 0,
                "frais_notaire": 8000,
                "credit_final": 90000,
                "taux_pret": 3.6,
                "duree_pret": 25,
                "assurance_ann_pct": 0.35,
                "loyer_mensuel_initial": 700,
                "charges_const_mth0": 60,
                "tf_const_mth0": 40,
                "frais_gestion_pct": 5.0,
                "provision_pct": 3.0,
            }]
        }

        schedules = [generate_amortization_schedule(90000, 3.6, 300, 0.35)]
        df, bilan = engine.simulate(strategy, 3, schedules)

        # Expected Amortization Year 1 = (100k + 10k) / 30 years = 3,666.67
        # + Furniture 5k / 10 years = 500
        # Total = 4,166.67 annually
        expected_amort = (110000 / 30) + (5000 / 10)
        
        # Amortissements are displayed as negative in DataFrame
        actual_amort = abs(df.iloc[0]["Amortissements"])
        assert abs(actual_amort - expected_amort) < 10, \
            f"Expected {expected_amort:.2f}, got {actual_amort:.2f}"

    def test_land_ratio_excludes_terrain_from_base(self):
        """With ratio_terrain_pct=20, should exclude 20k from 100k property."""
        engine = SimulationEngine(
            market=MarketHypotheses(),
            tax=TaxParams(regime_fiscal="lmnp"),
            ira=IRACalculator(apply_ira=False),
            ratio_terrain_pct=20.0,  # 20% is land
        )

        strategy = {
            "apport_total": 20000,
            "details": [{
                "prix_achat_bien": 100000,
                "budget_travaux": 10000,
                "mobilier": 5000,
                "renovation_energetique": 0,
                "frais_notaire": 8000,
                "credit_final": 90000,
                "taux_pret": 3.6,
                "duree_pret": 25,
                "assurance_ann_pct": 0.35,
                "loyer_mensuel_initial": 700,
                "charges_const_mth0": 60,
                "tf_const_mth0": 40,
                "frais_gestion_pct": 5.0,
                "provision_pct": 3.0,
            }]
        }

        schedules = [generate_amortization_schedule(90000, 3.6, 300, 0.35)]
        df, bilan = engine.simulate(strategy, 3, schedules)

        # Land = 100k * 20% = 20k (NOT amortizable)
        # Building = 100k - 20k = 80k (amortizable)
        # Base = 80k + 10k works = 90k
        # Amort/year = 90k / 30 = 3,000 + (5k furniture / 10) = 3,500
        expected_amort = (90000 / 30) + (5000 / 10)
        
        # Amortissements are displayed as negative in DataFrame
        actual_amort = abs(df.iloc[0]["Amortissements"])
        assert abs(actual_amort - expected_amort) < 10, \
            f"Expected {expected_amort:.2f}, got {actual_amort:.2f}"

    def test_land_ratio_higher_taxable_income(self):
        """Higher land ratio should lead to lower amortization, thus higher tax."""
        strategy_base = {
            "apport_total": 20000,
            "details": [{
                "prix_achat_bien": 100000,
                "budget_travaux": 10000,
                "mobilier": 5000,
                "renovation_energetique": 0,
                "frais_notaire": 8000,
                "credit_final": 90000,
                "taux_pret": 3.6,
                "duree_pret": 25,
                "assurance_ann_pct": 0.35,
                "loyer_mensuel_initial": 700,
                "charges_const_mth0": 60,
                "tf_const_mth0": 40,
                "frais_gestion_pct": 5.0,
                "provision_pct": 3.0,
            }]
        }

        schedules = [generate_amortization_schedule(90000, 3.6, 300, 0.35)]

        # Scenario A: No land exclusion
        engine_0 = SimulationEngine(
            market=MarketHypotheses(),
            tax=TaxParams(regime_fiscal="lmnp", tmi_pct=30.0),
            ira=IRACalculator(apply_ira=False),
            ratio_terrain_pct=0.0,
        )
        df_0, _ = engine_0.simulate(strategy_base, 3, schedules)

        # Scenario B: 20% land exclusion
        engine_20 = SimulationEngine(
            market=MarketHypotheses(),
            tax=TaxParams(regime_fiscal="lmnp", tmi_pct=30.0),
            ira=IRACalculator(apply_ira=False),
            ratio_terrain_pct=20.0,
        )
        df_20, _ = engine_20.simulate(strategy_base, 3, schedules)

        # Compare absolute amortization values (displayed as negative)
        amort_0 = abs(df_0.iloc[0]["Amortissements"])
        amort_20 = abs(df_20.iloc[0]["Amortissements"])
        
        # Land exclusion should reduce amortization
        assert amort_20 < amort_0, \
            f"Amort with land exclusion ({amort_20:.2f}) should be < without ({amort_0:.2f})"
        
        # With land exclusion, amortization is lower, so taxable base is higher
        # Thus tax should be higher (or equal if still in deficit)
        tax_0 = df_0.iloc[0]["Impôt Dû"]
        tax_20 = df_20.iloc[0]["Impôt Dû"]

        assert tax_20 >= tax_0, \
            f"Tax with land exclusion ({tax_20:.2f}) should be >= tax without ({tax_0:.2f})"
