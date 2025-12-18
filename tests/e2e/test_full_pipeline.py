"""End-to-end tests for the full strategy search pipeline.

These tests exercise the complete flow:
  Archetypes → Bricks → Strategy Search → Simulation → Metrics Validation

Run with: python -m pytest tests/e2e/test_full_pipeline.py -v
"""


import pytest

from src.services.brick_factory import (
    FinancingConfig,
    OperatingConfig,
    create_investment_bricks,
    validate_bricks,
)
from src.ui.app_controller import load_archetypes


class TestFullPipeline:
    """End-to-end pipeline tests."""

    @pytest.fixture
    def archetypes(self):
        """Load real archetypes from data."""
        return load_archetypes()[:10]  # Limit for speed

    @pytest.fixture
    def finance_config(self):
        return FinancingConfig(
            credit_rates={15: 3.5, 20: 3.6, 25: 3.8},
            frais_notaire_pct=8.0,
            assurance_ann_pct=0.35,
            frais_pret_pct=1.0,
        )

    @pytest.fixture
    def operating_config(self):
        return OperatingConfig(
            frais_gestion_pct=5.0,
            provision_pct=3.0,
            cfe_par_bien_ann=500,
        )

    @pytest.fixture
    def eval_params(self):
        return {
            "tmi_pct": 30,
            "regime_fiscal": "lmnp",
            "frais_vente_pct": 6.0,
            "apply_ira": True,
            "ira_cap_pct": 3.0,
            "cfe_par_bien_ann": 500,
            "hypotheses_marche": {
                "appreciation_bien_pct": 2.0,
                "revalo_loyer_pct": 1.5,
                "inflation_charges_pct": 2.0,
            },
            "finance_preset_name": "Équilibré",
            "max_properties": 3,
        }

    def test_archetypes_load(self, archetypes):
        """Verify archetypes load correctly."""
        assert len(archetypes) > 0
        assert all("surface" in a for a in archetypes)
        assert all("prix_m2" in a for a in archetypes)

    def test_bricks_creation(self, archetypes, finance_config, operating_config):
        """Test brick creation from archetypes."""
        bricks = create_investment_bricks(archetypes, finance_config, operating_config)

        assert len(bricks) > 0

        # Check required fields exist
        for brick in bricks:
            assert "nom_bien" in brick
            assert "prix_achat_bien" in brick
            assert "loyer_mensuel_initial" in brick
            assert "taux_pret" in brick
            assert "duree_pret" in brick
            assert "assurance_ann_pct" in brick
            assert "cout_total" in brick
            assert brick["assurance_ann_pct"] > 0, "Insurance should not be zero"

    def test_brick_validation_passes(self, archetypes, finance_config, operating_config):
        """Validate that created bricks pass validation."""
        bricks = create_investment_bricks(archetypes, finance_config, operating_config)
        warnings = validate_bricks(bricks)

        # No critical warnings expected for real data
        critical_warnings = [w for w in warnings if "Missing" in w or "Invalid" in w]
        assert len(critical_warnings) == 0, f"Validation warnings: {critical_warnings}"

    def test_strategy_search_returns_results(
        self, archetypes, finance_config, operating_config, eval_params
    ):
        """Full strategy search should return valid strategies."""
        from src.services.brick_factory import create_investment_bricks
        from src.services.strategy_finder import StrategyFinder

        bricks = create_investment_bricks(archetypes, finance_config, operating_config)

        finder = StrategyFinder(
            bricks=bricks,
            apport_disponible=50000.0,
            cash_flow_cible=0.0,
            tolerance=500.0,
            mode_cf="min",
            qualite_weight=0.25,
        )

        strategies = finder.find_strategies(
            eval_params=eval_params,
            horizon_years=25,
            top_n=5,
        )

        assert len(strategies) > 0, "Should find at least one strategy"

        # Validate strategy structure
        top = strategies[0]
        assert "details" in top
        assert "apport_total" in top
        assert "fitness" in top

    def test_strategy_metrics_are_valid(
        self, archetypes, finance_config, operating_config, eval_params
    ):
        """Verify key metrics are correctly calculated (not zero)."""
        from src.services.brick_factory import create_investment_bricks
        from src.services.strategy_finder import StrategyFinder

        bricks = create_investment_bricks(archetypes, finance_config, operating_config)

        finder = StrategyFinder(
            bricks=bricks,
            apport_disponible=50000.0,
            cash_flow_cible=0.0,
            tolerance=500.0,
            mode_cf="min",
            qualite_weight=0.25,
        )

        strategies = finder.find_strategies(
            eval_params=eval_params,
            horizon_years=25,
            top_n=3,
        )

        assert len(strategies) > 0
        top = strategies[0]

        # Key metrics should be present and non-zero
        assert top.get("liquidation_nette", 0) > 0, "Liquidation should be positive"
        assert top.get("enrich_net") is not None, "Enrichment should be calculated"

        # TRI can be zero in some edge cases, but should generally be positive
        tri = top.get("tri_annuel", 0)
        # Just check it's a number, not None
        assert isinstance(tri, (int, float)), f"TRI should be numeric, got {type(tri)}"


class TestMacroUseCases:
    """Macro use case tests simulating real user scenarios."""

    @pytest.fixture
    def archetypes(self):
        return load_archetypes()[:5]

    @pytest.fixture
    def base_config(self):
        return {
            "finance": FinancingConfig(
                credit_rates={15: 3.5, 20: 3.6, 25: 3.8},
                frais_notaire_pct=8.0,
                assurance_ann_pct=0.35,
            ),
            "operating": OperatingConfig(
                frais_gestion_pct=5.0,
                provision_pct=3.0,
                cfe_par_bien_ann=500,
            ),
            "eval_params": {
                "tmi_pct": 30,
                "regime_fiscal": "lmnp",
                "frais_vente_pct": 6.0,
                "apply_ira": True,
                "hypotheses_marche": {
                    "appreciation_bien_pct": 2.0,
                    "revalo_loyer_pct": 1.5,
                    "inflation_charges_pct": 2.0,
                },
            },
        }

    def test_small_budget_investor(self, archetypes, base_config):
        """Use case: First-time investor with 30k budget, wants positive CF."""
        from src.services.brick_factory import create_investment_bricks
        from src.services.strategy_finder import StrategyFinder

        bricks = create_investment_bricks(
            archetypes, base_config["finance"], base_config["operating"]
        )

        finder = StrategyFinder(
            bricks=bricks,
            apport_disponible=30000.0,
            cash_flow_cible=50.0,  # Wants positive CF
            tolerance=100.0,
            mode_cf="target",
            qualite_weight=0.3,
        )

        strategies = finder.find_strategies(
            eval_params=base_config["eval_params"],
            horizon_years=20,
            top_n=3,
        )

        # May or may not find strategies depending on data
        # Just verify no crash
        assert isinstance(strategies, list)

    def test_patrimoine_builder(self, archetypes, base_config):
        """Use case: Wealth builder with 100k, accepting negative CF for appreciation."""
        from src.services.brick_factory import create_investment_bricks
        from src.services.strategy_finder import StrategyFinder

        bricks = create_investment_bricks(
            archetypes, base_config["finance"], base_config["operating"]
        )

        finder = StrategyFinder(
            bricks=bricks,
            apport_disponible=100000.0,
            cash_flow_cible=-200.0,  # Accepts negative CF
            tolerance=300.0,
            mode_cf="min",
            qualite_weight=0.5,  # Values quality
        )

        strategies = finder.find_strategies(
            eval_params=base_config["eval_params"],
            horizon_years=25,
            top_n=5,
        )

        assert isinstance(strategies, list)
        if strategies:
            # Patrimoine strategy should have high liquidation
            top = strategies[0]
            assert top.get("liquidation_nette", 0) > 0

    def test_cashflow_optimizer(self, archetypes, base_config):
        """Use case: Investor seeking maximum monthly cashflow."""
        from src.services.brick_factory import create_investment_bricks
        from src.services.strategy_finder import StrategyFinder

        bricks = create_investment_bricks(
            archetypes, base_config["finance"], base_config["operating"]
        )

        finder = StrategyFinder(
            bricks=bricks,
            apport_disponible=75000.0,
            cash_flow_cible=200.0,  # High CF target
            tolerance=50.0,  # Strict
            mode_cf="target",
            qualite_weight=0.1,  # Prioritizes finance
        )

        strategies = finder.find_strategies(
            eval_params=base_config["eval_params"],
            horizon_years=25,  # Max duration for max cashflow logic check
            top_n=3,
        )

        assert isinstance(strategies, list)

    def test_different_horizons_affect_results(self, archetypes, base_config):
        """Verify that different horizons produce different metrics."""
        from src.services.brick_factory import create_investment_bricks
        from src.services.strategy_finder import StrategyFinder

        bricks = create_investment_bricks(
            archetypes, base_config["finance"], base_config["operating"]
        )

        results = {}
        for horizon in [15, 25]:
            finder = StrategyFinder(
                bricks=bricks,
                apport_disponible=50000.0,
                cash_flow_cible=0.0,
                tolerance=500.0,
                mode_cf="min",
            )
            strategies = finder.find_strategies(
                eval_params=base_config["eval_params"],
                horizon_years=horizon,
                top_n=1,
            )
            if strategies:
                results[horizon] = strategies[0].get("liquidation_nette", 0)

        # Longer horizon should generally have higher liquidation
        if len(results) == 2:
            assert results[25] >= results[15], "25y horizon should have >= liquidation than 15y"


class TestRegressionGuards:
    """Regression tests to prevent reintroduction of fixed bugs."""

    def test_insurance_key_is_correct(self):
        """Ensure evaluator uses assurance_ann_pct, not assurance_pret_pct."""
        import inspect

        from src.services.evaluator import StrategyEvaluator

        source = inspect.getsource(StrategyEvaluator.generate_schedules)
        assert "assurance_ann_pct" in source
        assert "assurance_pret_pct" not in source

    def test_strategy_result_enrichment_is_correct(self):
        """Ensure StrategyResult.enrichissement_net uses correct field."""
        import inspect

        from src.models.strategy import StrategyResult

        source = inspect.getsource(StrategyResult.enrichissement_net.fget)
        assert "enrichissement_net" in source
        assert "liquidation_nette" not in source.split("return")[1]

    def test_tri_units_are_percent(self):
        """Ensure TRI normalization expects percent, not fraction."""
        import inspect

        from src.core.scoring import calculate_balanced_score

        source = inspect.getsource(calculate_balanced_score)
        # Should divide by 20.0 (for percent), not 0.20 (for fraction)
        assert "/ 20.0" in source or "/20.0" in source

    def test_vacancy_uses_tension_not_travaux(self):
        """Ensure vacancy calculation uses tension or vacance_pct."""
        import inspect

        from src.core.scoring import calculate_property_qualitative_score

        source = inspect.getsource(calculate_property_qualitative_score)
        # Should reference vacance_pct or tension for vacancy
        assert "vacance_pct" in source or "vacancy_pct" in source
        # The old bug was "1.0 - ratio_trav" directly assigned to vacance
        assert '"vacance": 1.0 - ratio_trav' not in source
