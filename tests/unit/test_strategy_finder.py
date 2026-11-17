"""Unit tests for src.services.strategy_finder module."""

from src.application.services.allocator import PortfolioAllocator
from src.application.services.optimizer import BrickTable, SearchProblem, enumerate_subsets
from src.application.services.strategy_finder import (
    BASE_WEIGHTS,
    EvaluationParams,
    StrategyFinder,
    StrategyScorer,
)
from src.domain.models.brick import InvestmentBrick


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
        scorer = StrategyScorer(weights={"a": 1, "b": 1, "c": 1, "d": 1, "e": 1}, mode_cf="min")
        total = sum(scorer.weights.values())
        assert abs(total - 1.0) < 0.001

    def test_compute_finance_score(self):
        """Should compute weighted score."""
        scorer = StrategyScorer(weights=BASE_WEIGHTS, mode_cf="min")
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
        scorer = StrategyScorer(qualite_weight=0.5, mode_cf="min")
        strategy = {"finance_score": 0.8, "qual_score": 80.0}
        score = scorer.compute_balanced_score(strategy)
        # 0.5 * 0.8 + 0.5 * 0.8 = 0.8
        assert abs(score - 0.8) < 0.001


def _subsets(bricks, budget, max_props):
    """Property subsets (as name tuples) enumerated by the T31 search."""
    problem = SearchProblem(
        budget=budget, target_cf=0.0, tolerance=100.0, horizon=20, max_props=max_props, mode_cf="min"
    )
    allocator = PortfolioAllocator("min")
    table = BrickTable([allocator.prepare(b) for b in bricks], problem, {})
    return [tuple(table.names[p] for p in s) for s in enumerate_subsets(table, budget, max_props)]


class TestSubsetEnumeration:
    """Property-subset enumeration (replaces CombinationGenerator, T31)."""

    def test_generates_combinations(self):
        bricks = [
            InvestmentBrick(
                nom="A", apport_min=10000, prix_achat_bien=100000, capital_emprunte=90000, loyer_mensuel_initial=500
            ),
            InvestmentBrick(
                nom="B", apport_min=20000, prix_achat_bien=200000, capital_emprunte=180000, loyer_mensuel_initial=1000
            ),
        ]
        assert _subsets(bricks, 50000, 2) == [("A",), ("A", "B"), ("B",)]

    def test_respects_budget(self):
        bricks = [
            InvestmentBrick(
                nom="A", apport_min=30000, prix_achat_bien=100000, capital_emprunte=70000, loyer_mensuel_initial=500
            ),
            InvestmentBrick(
                nom="B", apport_min=30000, prix_achat_bien=100000, capital_emprunte=70000, loyer_mensuel_initial=500
            ),
        ]
        assert _subsets(bricks, 40000, 2) == [("A",), ("B",)]

    def test_no_duplicates_same_name(self):
        bricks = [
            InvestmentBrick(
                nom="A", apport_min=10000, prix_achat_bien=100000, capital_emprunte=90000, loyer_mensuel_initial=500
            ),
            InvestmentBrick(
                nom="A",
                apport_min=15000,
                prix_achat_bien=100000,
                capital_emprunte=85000,
                loyer_mensuel_initial=550,
                duree_credit_mois=300,
            ),
        ]
        assert _subsets(bricks, 50000, 2) == [("A",)]

    def test_max_props(self):
        bricks = [
            InvestmentBrick(
                nom=f"P{i}", apport_min=1000, prix_achat_bien=100000, capital_emprunte=99000, loyer_mensuel_initial=500
            )
            for i in range(6)
        ]
        subs = _subsets(bricks, 1e9, 3)
        assert len(subs) == 6 + 15 + 20
        assert max(len(s) for s in subs) == 3


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
        """Should dedupe to keep only best per signature."""
        finder = StrategyFinder([], 100000, -100)
        bricks_a = [InvestmentBrick(nom="A", prix_achat_bien=100000)]
        bricks_b = [InvestmentBrick(nom="B", prix_achat_bien=100000)]

        strategies = [
            {"details": bricks_a, "balanced_score": 0.8},
            {"details": bricks_a, "balanced_score": 0.7},  # ~same sig
            {"details": bricks_b, "balanced_score": 0.9},  # different
        ]
        deduped = finder.dedupe_strategies(strategies)
        # Only 2 distinct signatures: "A" and "B" → keeps best per each
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
