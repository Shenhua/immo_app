"""T37: the Patrimoine preset ranks on absolute EUR of net enrichment; the other presets are unchanged."""

from __future__ import annotations

import random

import pytest

from src.application.services.scoring import assign_pareto_ranks
from src.application.services.strategy_finder import StrategyFinder
from src.core.defaults import preset_weights


def _mk(i: int, r: random.Random) -> dict:
    return {
        "details": [{"nom": f"S{i}", "duree_credit_mois": 240}],
        "enrich_net": r.uniform(-20000, 400000),
        "tri_annuel": r.uniform(0, 12),
        "dscr_stab": r.uniform(0.7, 1.8),
        "apport_total": r.uniform(10000, 150000),
        "qual_score": r.uniform(20, 90),
        "cf_monthly_stab": r.uniform(-400, 200),
        "tier": 1,
    }


def _finder(preset: str, cf_cible: float) -> StrategyFinder:
    finder = StrategyFinder(bricks=[], apport_disponible=250_000.0, cash_flow_cible=cf_cible, mode_cf="min")
    finder.scorer.weights = finder.scorer._normalize_weights(preset_weights(preset))
    return finder


def _ranked(preset: str) -> list[dict]:
    r = random.Random(7)
    strategies = [_mk(i, r) for i in range(40)]
    finder = _finder(preset, -100.0)
    finder.scorer.score_strategies(strategies, -100.0, 25)
    assign_pareto_ranks(strategies, -100.0, "min")
    return finder.rank_strategies(strategies, preset, top_n=40)


# Orderings recorded on the code before T37 (same fixed instance): they must not move.
UNCHANGED = {
    "Équilibré": [
        "S12",
        "S3",
        "S31",
        "S16",
        "S8",
        "S2",
        "S33",
        "S10",
        "S23",
        "S7",
        "S38",
        "S0",
        "S11",
        "S19",
        "S35",
        "S29",
        "S15",
        "S24",
        "S37",
        "S5",
        "S6",
        "S39",
        "S18",
        "S34",
        "S28",
        "S26",
        "S21",
        "S25",
        "S17",
        "S32",
        "S9",
        "S27",
        "S30",
        "S20",
        "S36",
        "S22",
        "S14",
        "S4",
        "S13",
        "S1",
    ],
    "Sécurité (DSCR)": [
        "S12",
        "S31",
        "S16",
        "S8",
        "S15",
        "S3",
        "S18",
        "S19",
        "S36",
        "S28",
        "S10",
        "S0",
        "S32",
        "S17",
        "S5",
        "S26",
        "S38",
        "S11",
        "S35",
        "S14",
        "S23",
        "S37",
        "S21",
        "S6",
        "S4",
        "S30",
        "S39",
        "S7",
        "S29",
        "S34",
        "S13",
        "S27",
        "S2",
        "S20",
        "S33",
        "S24",
        "S9",
        "S25",
        "S1",
        "S22",
    ],
    "Rendement / IRR": [
        "S12",
        "S31",
        "S11",
        "S16",
        "S15",
        "S2",
        "S33",
        "S29",
        "S7",
        "S37",
        "S38",
        "S18",
        "S1",
        "S9",
        "S17",
        "S24",
        "S34",
        "S13",
        "S6",
        "S20",
        "S3",
        "S35",
        "S5",
        "S27",
        "S22",
        "S23",
        "S10",
        "S8",
        "S30",
        "S14",
        "S28",
        "S39",
        "S32",
        "S21",
        "S25",
        "S36",
        "S0",
        "S4",
        "S19",
        "S26",
    ],
    "Cash-flow d'abord": [
        "S12",
        "S8",
        "S19",
        "S18",
        "S36",
        "S28",
        "S26",
        "S38",
        "S11",
        "S23",
        "S4",
        "S39",
        "S7",
        "S34",
        "S13",
        "S27",
        "S2",
        "S33",
        "S9",
        "S32",
        "S10",
        "S14",
        "S30",
        "S17",
        "S15",
        "S0",
        "S6",
        "S3",
        "S24",
        "S16",
        "S31",
        "S5",
        "S35",
        "S21",
        "S22",
        "S25",
        "S1",
        "S20",
        "S29",
        "S37",
    ],
}


@pytest.mark.parametrize("preset", list(UNCHANGED))
def test_other_presets_ordering_unchanged(preset):
    assert [s["details"][0]["nom"] for s in _ranked(preset)] == UNCHANGED[preset]


def test_patrimoine_ranks_by_absolute_euros():
    wealth = [s["enrich_net"] for s in _ranked("Patrimoine LT")]
    assert wealth == sorted(wealth, reverse=True)


def test_patrimoine_more_euros_beats_higher_roe():
    # Small has the higher ROE and IRR (ROE above the target, so enrich_norm saturates); big has far more EUR.
    common = {"dscr_stab": 1.3, "qual_score": 60.0, "cf_monthly_stab": 0.0, "tier": 1}
    small = {
        "details": [{"nom": "small", "duree_credit_mois": 240}],
        "enrich_net": 150_000.0,
        "tri_annuel": 11.0,
        "apport_total": 20_000.0,
        **common,
    }
    big = {
        "details": [{"nom": "big", "duree_credit_mois": 240}],
        "enrich_net": 600_000.0,
        "tri_annuel": 7.0,
        "apport_total": 200_000.0,
        **common,
    }
    finder = _finder("Patrimoine LT", 0.0)
    strategies = [small, big]
    finder.scorer.score_strategies(strategies, 0.0, 25)
    assign_pareto_ranks(strategies, 0.0, "min")
    assert small["enrich_norm"] == pytest.approx(1.0)
    assert finder.rank_strategies(strategies, "Patrimoine LT")[0] is big
