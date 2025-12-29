"""Unit tests for src.domain.calculator.scoring module."""

from src.domain.calculator.scoring import (
    calculate_balanced_score,
    calculate_dpe_score,
    calculate_property_qualitative_score,
    calculate_qualitative_score,
)


class TestCalculateDpeScore:
    """Tests for DPE letter to score conversion."""

    def test_rating_a(self):
        """A rating should be 1.0."""
        assert calculate_dpe_score("A") == 1.0

    def test_rating_d(self):
        """D rating (typical) should be 0.6."""
        assert calculate_dpe_score("D") == 0.6

    def test_rating_g(self):
        """G rating (worst) should be 0.0."""
        assert calculate_dpe_score("G") == 0.0

    def test_lowercase_input(self):
        """Should handle lowercase input."""
        assert calculate_dpe_score("b") == 0.9

    def test_unknown_rating(self):
        """Unknown rating defaults to 0.6."""
        assert calculate_dpe_score("X") == 0.6

    def test_none_input(self):
        """None input defaults to D (0.6)."""
        assert calculate_dpe_score(None) == 0.6


class TestCalculatePropertyQualitativeScore:
    """Tests for single property qualitative scoring."""

    def test_returns_tuple(self):
        """Should return (score, features_dict)."""
        result = calculate_property_qualitative_score({})
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], dict)

    def test_score_range(self):
        """Score should be between 0 and 100."""
        source = {
            "indice_tension": 0.8,
            "transport_score": 0.9,
            "dpe_initial": "B",
            "liquidite_score": 0.8,
        }
        score, _ = calculate_property_qualitative_score(source)
        assert 0 <= score <= 100

    def test_high_quality_property(self):
        """High quality property should score well."""
        source = {
            "indice_tension": 0.95,
            "transport_score": 0.95,
            "dpe_initial": "A",
            "liquidite_score": 0.95,
        }
        score, _ = calculate_property_qualitative_score(source, prix_achat=100000, travaux=0)
        assert score > 80

    def test_low_quality_property(self):
        """Poor quality property should score low."""
        source = {
            "indice_tension": 0.1,
            "transport_score": 0.1,
            "dpe_initial": "G",
            "liquidite_score": 0.1,
        }
        score, _ = calculate_property_qualitative_score(source, prix_achat=100000, travaux=50000)
        assert score < 30


class TestCalculateQualitativeScore:
    """Tests for strategy-level qualitative scoring."""

    def test_empty_strategy(self):
        """Empty strategy defaults to 50."""
        assert calculate_qualitative_score({}) == 50.0

    def test_single_property(self, sample_strategy_data):
        """Single property strategy should use that property's score."""
        strategy = {
            "details": [sample_strategy_data["details"][0]]
        }
        score = calculate_qualitative_score(strategy)
        assert abs(score - 75.0) < 1.0  # Should be close to the qual_score_bien

    def test_weighted_by_cost(self, sample_strategy_data):
        """Score should be weighted by property cost."""
        score = calculate_qualitative_score(sample_strategy_data)
        # Paris (275k, 75 score) + Lyon (180k, 60 score)
        # Expected: (75*275 + 60*180) / 455 ≈ 69
        assert 65 < score < 75


class TestCalculateBalancedScore:
    """Tests for balanced score combining finance and quality."""

    def test_basic_calculation(self):
        """Should combine financial and quality scores."""
        score = calculate_balanced_score(
            tri=0.08,
            enrichissement_net=200000,
            dscr=1.3,
            qual_score=70.0,
            qualite_weight=0.25,
        )
        assert 0 <= score <= 100

    def test_quality_weight_effect(self):
        """Higher quality weight should increase score when qual_score is high."""
        low_weight = calculate_balanced_score(0.10, 300000, 1.5, 90.0, qualite_weight=0.1)
        high_weight = calculate_balanced_score(0.10, 300000, 1.5, 90.0, qualite_weight=0.5)
        # Higher qual_score (90) + higher weight = higher total
        assert high_weight > low_weight
