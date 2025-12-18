"""
Unit tests for StrategyEvaluator.

Tests the shared evaluation logic used by both GeneticOptimizer
and ExhaustiveOptimizer.
"""
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd

from src.services.evaluator import StrategyEvaluator


@pytest.fixture
def mock_simulator():
    """Create a mock simulator with realistic return values."""
    simulator = MagicMock()
    
    # Mock simulate to return realistic bilan
    df_mock = pd.DataFrame({
        "Année": [1, 2, 3],
        "Cash-Flow Net d'Impôt": [1200, 1400, 1600],
        "Loyer Brut": [12000, 12360, 12730],
        "Charges Désendettement": [8000, 8000, 8000],
    })
    bilan_mock = {
        "liquidation_nette": 150000.0,
        "enrichissement_net": 80000.0,
        "tri_annuel": 8.5,
        "dscr_y1": 1.2,
    }
    simulator.simulate.return_value = (df_mock, bilan_mock)
    return simulator


@pytest.fixture
def mock_allocator():
    """Create a mock allocator."""
    allocator = MagicMock()
    allocator.mode_cf = "min"
    return allocator


@pytest.fixture
def mock_scorer():
    """Create a mock scorer with default weights."""
    scorer = MagicMock()
    scorer.weights = {
        "irr": 0.25,
        "enrich_net": 0.30,
        "dscr": 0.15,
        "cf_proximity": 0.20,
        "cap_eff": 0.10
    }
    scorer.qualite_weight = 0.5
    return scorer


@pytest.fixture
def evaluator(mock_simulator, mock_allocator, mock_scorer):
    """Create evaluator with all mocked dependencies."""
    return StrategyEvaluator(
        simulator=mock_simulator,
        allocator=mock_allocator,
        scorer=mock_scorer
    )


@pytest.fixture
def sample_details():
    """Sample property details for testing."""
    return [
        {
            "nom_bien": "Studio Lyon",
            "credit_final": 80000,
            "taux_pret": 3.5,
            "duree_pret": 20,
            "assurance_ann_pct": 0.36,
            "qual_score_bien": 70.0,
        },
        {
            "nom_bien": "T2 Grenoble",
            "credit_final": 60000,
            "taux_pret": 3.5,
            "duree_pret": 25,
            "assurance_ann_pct": 0.36,
            "qual_score_bien": 65.0,
        }
    ]


class TestStrategyEvaluatorInit:
    """Tests for StrategyEvaluator initialization and properties."""
    
    def test_init_with_all_dependencies(self, mock_simulator, mock_allocator, mock_scorer):
        """Test initialization with all dependencies."""
        evaluator = StrategyEvaluator(mock_simulator, mock_allocator, mock_scorer)
        
        assert evaluator.simulator is mock_simulator
        assert evaluator.allocator is mock_allocator
        assert evaluator.scorer is mock_scorer
    
    def test_init_minimal(self, mock_simulator):
        """Test initialization with only required simulator."""
        evaluator = StrategyEvaluator(mock_simulator)
        
        assert evaluator.simulator is mock_simulator
        assert evaluator.allocator is None
        assert evaluator.scorer is None
    
    def test_weights_from_scorer(self, evaluator, mock_scorer):
        """Test that weights come from scorer when available."""
        assert evaluator.weights == mock_scorer.weights
    
    def test_weights_default(self, mock_simulator):
        """Test default weights when no scorer."""
        evaluator = StrategyEvaluator(mock_simulator)
        
        assert evaluator.weights == StrategyEvaluator.DEFAULT_WEIGHTS
        assert "irr" in evaluator.weights
    
    def test_quality_weight_from_scorer(self, evaluator, mock_scorer):
        """Test quality weight from scorer."""
        assert evaluator.quality_weight == mock_scorer.qualite_weight
    
    def test_quality_weight_default(self, mock_simulator):
        """Test default quality weight when no scorer."""
        evaluator = StrategyEvaluator(mock_simulator)
        assert evaluator.quality_weight == 0.5
    
    def test_mode_cf_from_allocator(self, evaluator):
        """Test mode_cf from allocator."""
        assert evaluator.mode_cf == "min"
    
    def test_mode_cf_default(self, mock_simulator):
        """Test default mode_cf when no allocator."""
        evaluator = StrategyEvaluator(mock_simulator)
        assert evaluator.mode_cf == "target"


class TestGenerateSchedules:
    """Tests for schedule generation."""
    
    def test_generate_schedules_normal(self, evaluator, sample_details):
        """Test schedule generation for normal properties."""
        schedules = evaluator.generate_schedules(sample_details)
        
        assert len(schedules) == 2
        # Each schedule should have the expected structure
        for sch in schedules:
            assert "mois" in sch or "nmois" in sch
    
    def test_generate_schedules_empty(self, evaluator):
        """Test schedule generation with empty details."""
        schedules = evaluator.generate_schedules([])
        assert schedules == []
    
    def test_generate_schedules_zero_principal(self, evaluator):
        """Test schedule generation for cash purchase (zero credit)."""
        details = [{"nom_bien": "Cash Property", "credit_final": 0, "duree_pret": 20}]
        schedules = evaluator.generate_schedules(details)
        
        assert len(schedules) == 1
        # Schedule should return empty structure for zero principal
        assert schedules[0]["pmt_total"] == 0.0
    
    def test_generate_schedules_missing_fields(self, evaluator):
        """Test schedule generation with missing optional fields."""
        details = [{"credit_final": 50000}]  # Missing other fields
        schedules = evaluator.generate_schedules(details)
        
        assert len(schedules) == 1


class TestCalculateFinanceScore:
    """Tests for finance score calculation."""
    
    def test_finance_score_perfect(self, evaluator):
        """Test finance score with excellent values."""
        bilan = {
            "tri_annuel": 25.0,  # > 20% target
            "enrichissement_net": 250000.0,  # 2.5x ROE
            "dscr_y1": 1.5,  # > 1.3 target
        }
        cf_metrics = {"gap": 0.0}
        
        score = evaluator.calculate_finance_score(bilan, cf_metrics, 100000.0, 100.0)
        
        # Should be close to 1.0
        assert score > 0.9
    
    def test_finance_score_zero(self, evaluator):
        """Test finance score with zero/bad values."""
        bilan = {
            "tri_annuel": 0.0,
            "enrichissement_net": 0.0,
            "dscr_y1": 0.0,
        }
        cf_metrics = {"gap": 500.0}  # Large gap
        
        score = evaluator.calculate_finance_score(bilan, cf_metrics, 100000.0, 100.0)
        
        # Should be close to 0.0
        assert score < 0.1
    
    def test_finance_score_moderate(self, evaluator):
        """Test finance score with moderate values."""
        bilan = {
            "tri_annuel": 10.0,  # 50% of target
            "enrichissement_net": 100000.0,  # 1x ROE
            "dscr_y1": 1.0,  # Below safe level
        }
        cf_metrics = {"gap": 50.0}
        
        score = evaluator.calculate_finance_score(bilan, cf_metrics, 100000.0, 100.0)
        
        # Should be in middle range
        assert 0.3 < score < 0.7


class TestCalculateQualityScore:
    """Tests for quality score calculation."""
    
    def test_quality_score_high(self, evaluator):
        """Test quality score with high-quality properties."""
        details = [
            {"qual_score_bien": 90.0, "cout_total": 100000},
            {"qual_score_bien": 85.0, "cout_total": 100000},
        ]
        
        score = evaluator.calculate_quality_score(details)
        
        # Should be weighted average: (90*100k + 85*100k) / 200k = 87.5
        assert score > 80.0
    
    def test_quality_score_low(self, evaluator):
        """Test quality score with low-quality properties."""
        details = [
            {"qual_score_bien": 30.0, "cout_total": 100000},
            {"qual_score_bien": 25.0, "cout_total": 100000},
        ]
        
        score = evaluator.calculate_quality_score(details)
        
        # Should be low
        assert score < 40.0
    
    def test_quality_score_empty(self, evaluator):
        """Test quality score with empty details."""
        score = evaluator.calculate_quality_score([])
        
        # Should return default (50)
        assert score == 50.0


class TestCalculateCombinedScore:
    """Tests for combined score calculation."""
    
    def test_combined_score_quality_only(self, evaluator):
        """Test combined score with q_w = 1.0 (quality only)."""
        evaluator.scorer.qualite_weight = 1.0
        
        score = evaluator.calculate_combined_score(0.9, 60.0)
        
        # Should be 60/100 = 0.6, ignoring finance=0.9
        assert abs(score - 0.6) < 0.01
    
    def test_combined_score_finance_only(self, evaluator):
        """Test combined score with q_w = 0.0 (finance only)."""
        evaluator.scorer.qualite_weight = 0.0
        
        score = evaluator.calculate_combined_score(0.8, 100.0)
        
        # Should be 0.8, ignoring quality=100
        assert abs(score - 0.8) < 0.01
    
    def test_combined_score_balanced(self, evaluator):
        """Test combined score with q_w = 0.5 (50/50)."""
        evaluator.scorer.qualite_weight = 0.5
        
        score = evaluator.calculate_combined_score(0.8, 60.0)
        
        # Should be 0.5 * 0.8 + 0.5 * 0.6 = 0.4 + 0.3 = 0.7
        expected = 0.5 * 0.8 + 0.5 * (60.0 / 100.0)
        assert abs(score - expected) < 0.01


class TestEvaluateFullPipeline:
    """Integration tests for full evaluation pipeline."""
    
    def test_evaluate_success(self, evaluator, sample_details):
        """Test successful evaluation."""
        strategy = {
            "details": sample_details,
            "apport_total": 50000.0,
        }
        
        with patch.object(evaluator, 'calculate_cf_metrics') as mock_cf:
            mock_cf.return_value = {
                "is_acceptable": True,
                "gap": 20.0,
                "cf_year_1_monthly": 150.0,
                "cf_avg_5y_monthly": 180.0,
            }
            
            is_valid, result = evaluator.evaluate(strategy, -100.0, 100.0, 25)
        
        assert is_valid is True
        assert "fitness" in result
        assert "qual_score" in result
        assert "tri_annuel" in result
        assert "dscr_y1" in result
        assert result["fitness"] > 0
    
    def test_evaluate_empty_details(self, evaluator):
        """Test evaluation with empty details."""
        strategy = {"details": [], "apport_total": 50000.0}
        
        is_valid, result = evaluator.evaluate(strategy, -100.0, 100.0, 25)
        
        assert is_valid is False
        assert "error" in result
    
    def test_evaluate_simulation_failure(self, evaluator, sample_details):
        """Test evaluation when simulation fails."""
        strategy = {"details": sample_details, "apport_total": 50000.0}
        
        # Make simulation raise an exception
        evaluator.simulator.simulate.side_effect = Exception("Simulation error")
        
        is_valid, result = evaluator.evaluate(strategy, -100.0, 100.0, 25)
        
        assert is_valid is False
        assert "error" in result


class TestEdgeCases:
    """Edge case tests."""
    
    def test_finance_score_with_zero_apport(self, evaluator):
        """Test finance score doesn't divide by zero with zero apport."""
        bilan = {"tri_annuel": 10.0, "enrichissement_net": 50000.0, "dscr_y1": 1.0}
        cf_metrics = {"gap": 50.0}
        
        # Should not raise, uses safe_apport = max(1.0, 0) = 1.0
        score = evaluator.calculate_finance_score(bilan, cf_metrics, 0.0, 100.0)
        
        assert score >= 0.0
    
    def test_finance_score_with_zero_tolerance(self, evaluator):
        """Test finance score with zero tolerance uses default."""
        bilan = {"tri_annuel": 10.0, "enrichissement_net": 50000.0, "dscr_y1": 1.0}
        cf_metrics = {"gap": 50.0}
        
        # Should use safe_tol = 100.0
        score = evaluator.calculate_finance_score(bilan, cf_metrics, 100000.0, 0.0)
        
        assert score >= 0.0
    
    def test_finance_score_none_dscr(self, evaluator):
        """Test finance score handles None DSCR."""
        bilan = {"tri_annuel": 10.0, "enrichissement_net": 50000.0, "dscr_y1": None}
        cf_metrics = {"gap": 50.0}
        
        # Should treat None as 0.0
        score = evaluator.calculate_finance_score(bilan, cf_metrics, 100000.0, 100.0)
        
        assert score >= 0.0
