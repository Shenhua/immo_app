"""T25 scoring fixes: rent-cap margin, real capital efficiency, weights from defaults."""

from unittest.mock import MagicMock

import pytest

from src.application.services.evaluator import StrategyEvaluator
from src.core.defaults import BASE_WEIGHTS, CAP_EFF_BENCHMARK_MAX, SEARCH
from src.domain.calculator.scoring import calculate_property_qualitative_score


class TestRentCapMargin:
    """Rent exactly at the cap is neutral (0.6), above it 0, below it scales up."""

    @staticmethod
    def _enc(loyer, cap=20.0, regulated=True):
        src = {"soumis_encadrement": regulated, "loyer_m2": loyer, "loyer_m2_max": cap}
        return calculate_property_qualitative_score(src)[1]["encadrement"]

    def test_at_cap_is_neutral(self):
        assert self._enc(20.0) == pytest.approx(0.6)
        assert self._enc(20.0) == pytest.approx(self._enc(20.0, regulated=False))

    def test_above_cap_is_zero(self):
        assert self._enc(22.0) == 0.0

    def test_below_cap_scales_up(self):
        assert 0.6 < self._enc(18.0) < self._enc(10.0) <= 1.0


class TestCapitalEfficiencyScore:
    """cap_eff is real capital efficiency, not a copy of s_enrich."""

    @staticmethod
    def _score(weights, enrich, apport=100000.0):
        ev = StrategyEvaluator(MagicMock(), scorer=MagicMock(weights=weights))
        bilan = {"tri_annuel": 0.0, "enrichissement_net": enrich, "dscr_stab": 0.0}
        return ev.calculate_finance_score(bilan, {"gap": 1e9}, apport, 100.0, 20)

    def test_cap_eff_only_weight_uses_benchmark(self):
        w = {"irr": 0.0, "enrich_net": 0.0, "dscr": 0.0, "cf_proximity": 0.0, "cap_eff": 1.0}
        assert self._score(w, 100000.0 * CAP_EFF_BENCHMARK_MAX / 2) == pytest.approx(0.5)
        assert self._score(w, 1e9) == pytest.approx(1.0)

    def test_enrich_and_cap_eff_are_independent(self):
        base = {"irr": 0.0, "enrich_net": 0.0, "dscr": 0.0, "cf_proximity": 0.0, "cap_eff": 0.0}
        e_only = self._score({**base, "enrich_net": 1.0}, 100000.0)
        c_only = self._score({**base, "cap_eff": 1.0}, 100000.0)
        assert e_only == pytest.approx(1.0 / (1.05**20 - 1.0))
        assert c_only == pytest.approx(1.0 / CAP_EFF_BENCHMARK_MAX)

    def test_missing_weight_keys_come_from_base_weights(self):
        assert self._score({}, 100000.0) == pytest.approx(self._score(dict(BASE_WEIGHTS), 100000.0))


def test_quality_weight_default_reads_defaults_module():
    assert StrategyEvaluator(MagicMock()).quality_weight == SEARCH.qualite_weight
