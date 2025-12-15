"""Unit tests for src.services.strategy_finder module."""

import pytest
from src.services.strategy_finder import (
    BASE_WEIGHTS,
    EvaluationParams,
    StrategyScorer,
    CombinationGenerator,
    StrategyFinder,
)


class TestEvaluationParams:
    """Tests for EvaluationParams dataclass."""
    
    def test_defaults(self):
        """Should have sensible defaults."""
        p = EvaluationParams()
        assert p.duree_simulation_ans == 25
        assert p.regime_fiscal == "lmnp"
        assert p.tmi_pct == 30.0
    
    def test_from_dict(self):
        """Should create from dictionary."""
        d = {"tmi_pct": 45.0, "regime_fiscal": "microbic"}
        p = EvaluationParams.from_dict(d)
        assert p.tmi_pct == 45.0
        assert p.regime_fiscal == "microbic"


class TestStrategyScorer:
    """Tests for StrategyScorer."""
    
    def test_normalize_weights(self):
        """Weights should sum to 1."""
        scorer = StrategyScorer(weights={"a": 1, "b": 1, "c": 1, "d": 1, "e": 1})
        total = sum(scorer.weights.values())
        assert abs(total - 1.0) < 0.001
    
    def test_minmax_normalize_basic(self):
        """Should normalize to 0-1 range."""
        values = [0, 50, 100]
        result = StrategyScorer.minmax_normalize(values)
        assert result == [0.0, 0.5, 1.0]
    
    def test_minmax_normalize_empty(self):
        """Empty list should return empty."""
        result = StrategyScorer.minmax_normalize([])
        assert result == []
    
    def test_minmax_normalize_with_bounds(self):
        """Should respect explicit bounds."""
        values = [0, 50, 100]
        result = StrategyScorer.minmax_normalize(values, lo=0, hi=200)
        assert result[1] == 0.25  # 50/200
    
    def test_compute_finance_score(self):
        """Should compute weighted score."""
        scorer = StrategyScorer(weights=BASE_WEIGHTS)
        strategy = {
            "enrich_norm": 1.0,
            "tri_norm": 1.0,
            "cap_eff_norm": 1.0,
            "dscr_norm": 1.0,
            "cf_proximity": 1.0,
        }
        score = scorer.compute_finance_score(strategy)
        assert abs(score - 1.0) < 0.001  # All maxed = 1.0
    
    def test_compute_balanced_score(self):
        """Should blend finance and quality."""
        scorer = StrategyScorer(qualite_weight=0.5)
        strategy = {"finance_score": 0.8, "qual_score": 80.0}
        score = scorer.compute_balanced_score(strategy)
        # 0.5 * 0.8 + 0.5 * 0.8 = 0.8
        assert abs(score - 0.8) < 0.001


class TestCombinationGenerator:
    """Tests for CombinationGenerator."""
    
    def test_generates_combinations(self):
        """Should generate valid combinations."""
        bricks = [
            {"nom_bien": "A", "apport_min": 10000},
            {"nom_bien": "B", "apport_min": 20000},
        ]
        gen = CombinationGenerator(max_properties=2)
        combos = gen.generate(bricks, apport_disponible=50000)
        
        # Should have: (A,), (B,), (A, B)
        assert len(combos) == 3
    
    def test_respects_budget(self):
        """Should filter by apport_min."""
        bricks = [
            {"nom_bien": "A", "apport_min": 30000},
            {"nom_bien": "B", "apport_min": 30000},
        ]
        gen = CombinationGenerator(max_properties=2)
        combos = gen.generate(bricks, apport_disponible=40000)
        
        # Only single properties fit, not both
        assert len(combos) == 2
    
    def test_no_duplicates_same_name(self):
        """Should not allow same property twice."""
        bricks = [
            {"nom_bien": "A", "apport_min": 10000},
            {"nom_bien": "A", "apport_min": 15000},  # Same name, different duration
        ]
        gen = CombinationGenerator(max_properties=2)
        combos = gen.generate(bricks, apport_disponible=50000)
        
        # Should only have single-property combos
        assert all(len(c) == 1 for c in combos)


class TestStrategyFinder:
    """Tests for StrategyFinder service."""
    
    def test_initialization(self):
        """Should initialize with provided params."""
        finder = StrategyFinder(
            bricks=[],
            apport_disponible=100000,
            cash_flow_cible=-100,
            tolerance=50,
            qualite_weight=0.25,
        )
        assert finder.apport_disponible == 100000
        assert finder.cash_flow_cible == -100
        assert finder.scorer.qualite_weight == 0.25
    
    def test_dedupe_strategies(self):
        """Should remove duplicates."""
        finder = StrategyFinder([], 100000, -100)
        strategies = [
            {"details": [{"nom_bien": "A", "duree_pret": 20, "apport_final_bien": 50000}]},
            {"details": [{"nom_bien": "A", "duree_pret": 20, "apport_final_bien": 50050}]},  # ~same
            {"details": [{"nom_bien": "B", "duree_pret": 20, "apport_final_bien": 50000}]},  # different
        ]
        deduped = finder.dedupe_strategies(strategies)
        assert len(deduped) == 2
    
    def test_rank_strategies_balanced(self):
        """Should sort by balanced_score for default preset."""
        finder = StrategyFinder([], 100000, -100)
        strategies = [
            {"balanced_score": 0.5, "dscr_norm": 0.5, "tri_norm": 0.5},
            {"balanced_score": 0.9, "dscr_norm": 0.5, "tri_norm": 0.5},
            {"balanced_score": 0.7, "dscr_norm": 0.5, "tri_norm": 0.5},
        ]
        ranked = finder.rank_strategies(strategies, "Équilibré")
        assert ranked[0]["balanced_score"] == 0.9
    
    def test_rank_strategies_securite_dscr(self):
        """Should prioritize DSCR for Sécurité preset."""
        finder = StrategyFinder([], 100000, -100)
        strategies = [
            {"balanced_score": 0.9, "dscr_norm": 0.3, "finance_score": 0.8},
            {"balanced_score": 0.5, "dscr_norm": 0.9, "finance_score": 0.5},
            {"balanced_score": 0.7, "dscr_norm": 0.6, "finance_score": 0.6},
        ]
        ranked = finder.rank_strategies(strategies, "Sécurité (DSCR)")
        assert ranked[0]["dscr_norm"] == 0.9  # Highest DSCR wins
    
    def test_rank_strategies_cashflow(self):
        """Should prioritize CF proximity for Cash-flow preset."""
        finder = StrategyFinder([], 100000, -100)
        strategies = [
            {"balanced_score": 0.9, "cf_proximity": 0.3, "dscr_norm": 0.5},
            {"balanced_score": 0.5, "cf_proximity": 0.9, "dscr_norm": 0.5},
            {"balanced_score": 0.7, "cf_proximity": 0.6, "dscr_norm": 0.5},
        ]
        ranked = finder.rank_strategies(strategies, "Cash-flow d'abord")
        assert ranked[0]["cf_proximity"] == 0.9  # Highest CF proximity wins
    
    def test_rank_strategies_rendement(self):
        """Should prioritize IRR for Rendement preset."""
        finder = StrategyFinder([], 100000, -100)
        strategies = [
            {"balanced_score": 0.9, "tri_norm": 0.3, "finance_score": 0.8},
            {"balanced_score": 0.5, "tri_norm": 0.9, "finance_score": 0.5},
            {"balanced_score": 0.7, "tri_norm": 0.6, "finance_score": 0.6},
        ]
        ranked = finder.rank_strategies(strategies, "Rendement / IRR")
        assert ranked[0]["tri_norm"] == 0.9  # Highest TRI wins
    
    def test_rank_strategies_patrimoine(self):
        """Should prioritize enrichment for Patrimoine preset."""
        finder = StrategyFinder([], 100000, -100)
        strategies = [
            {"balanced_score": 0.9, "enrich_norm": 0.3, "tri_norm": 0.8},
            {"balanced_score": 0.5, "enrich_norm": 0.9, "tri_norm": 0.5},
            {"balanced_score": 0.7, "enrich_norm": 0.6, "tri_norm": 0.6},
        ]
        ranked = finder.rank_strategies(strategies, "Patrimoine LT")
        assert ranked[0]["enrich_norm"] == 0.9  # Highest enrichment wins

