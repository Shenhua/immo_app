"""Absolute scoring and Pareto ranking of strategies (T32).

Single implementation shared by ``StrategyEvaluator`` (per-candidate fitness during the search) and
``StrategyScorer`` (final scores on the result set). Every component is scored on a fixed benchmark range from
``defaults.py``, so a strategy's score never depends on the other candidates of the run.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.defaults import (
    BASE_WEIGHTS,
    CAP_EFF_BENCHMARK_MAX,
    CF_PROXIMITY_BENCHMARK_EUR,
    DSCR_BENCHMARK_MAX,
    DSCR_BENCHMARK_MIN,
    ENRICH_BENCHMARK_ANNUAL_PCT,
    IRR_BENCHMARK_MAX_PCT,
    PARETO_MAX_RANK,
)
from src.core.glossary import cf_penalty


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_components(
    *,
    irr_pct: float | None,
    enrich_net: float,
    apport: float,
    dscr_stab: float | None,
    cf_gap: float,
    horizon: int,
) -> dict[str, float]:
    """Absolute 0..1 components. ``cf_gap`` is the ``glossary.cf_penalty`` of the stabilised after-tax CF."""
    # A failed IRR (None) scores 0, it is never read as 0 %.
    s_irr = 0.0 if irr_pct is None else _clamp01(float(irr_pct) / IRR_BENCHMARK_MAX_PCT)
    roe = float(enrich_net) / max(1.0, float(apport))
    target_roe = (1.0 + ENRICH_BENCHMARK_ANNUAL_PCT / 100.0) ** max(1, int(horizon)) - 1.0
    dscr = float(dscr_stab or 0.0)
    return {
        "tri_norm": s_irr,
        "enrich_norm": _clamp01(roe / target_roe),
        "cap_eff_norm": _clamp01(roe / CAP_EFF_BENCHMARK_MAX),
        "dscr_norm": _clamp01((dscr - DSCR_BENCHMARK_MIN) / (DSCR_BENCHMARK_MAX - DSCR_BENCHMARK_MIN)),
        "cf_proximity": _clamp01(1.0 - max(0.0, float(cf_gap)) / CF_PROXIMITY_BENCHMARK_EUR),
    }


def weighted_finance_score(components: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted sum of the components; weight keys missing from a custom dict fall back to BASE_WEIGHTS."""
    w = {**BASE_WEIGHTS, **weights}
    return (
        w["irr"] * components["tri_norm"]
        + w["enrich_net"] * components["enrich_norm"]
        + w["dscr"] * components["dscr_norm"]
        + w["cf_proximity"] * components["cf_proximity"]
        + w["cap_eff"] * components["cap_eff_norm"]
    )


def blend_quality(finance_score: float, quality_score: float, quality_weight: float) -> float:
    """Blend the 0..1 finance score with the 0..100 quality score."""
    qual = quality_score / 100.0
    if quality_weight >= 1.0:
        return qual
    if quality_weight <= 0.0:
        return finance_score
    return (1.0 - quality_weight) * finance_score + quality_weight * qual


def pareto_objectives(s: dict[str, Any], target_cf: float, mode_cf: str) -> tuple[float, ...]:
    """Objective vector, all maximised: wealth, IRR, DSCR, -down payment, quality, -CF penalty."""
    tri = s.get("tri_annuel")
    cf = s.get("cf_monthly_stab", s.get("cash_flow_final", 0.0))
    return (
        float(s.get("enrich_net", 0.0) or 0.0),
        float("-inf") if tri is None else float(tri),  # failed IRR: worst
        float(s.get("dscr_stab", 0.0) or 0.0),
        -float(s.get("apport_total", 0.0) or 0.0),
        float(s.get("qual_score", 50.0)),
        -cf_penalty(float(cf), target_cf, mode_cf),
    )


def assign_pareto_ranks(strategies: list[dict[str, Any]], target_cf: float, mode_cf: str) -> None:
    """Set ``pareto_rank`` on each strategy (mutates): 1 = non-dominated front, 2 = front once rank 1 is removed...

    Not a filter. Ranks above ``PARETO_MAX_RANK`` are all set to ``PARETO_MAX_RANK + 1``.
    """
    n = len(strategies)
    if n == 0:
        return
    m = np.array([pareto_objectives(s, target_cf, mode_cf) for s in strategies], dtype=float)
    remaining = np.arange(n)
    rank = 1
    while remaining.size and rank <= PARETO_MAX_RANK:
        sub = m[remaining]
        front = []
        for k in range(remaining.size):
            ge = np.all(sub >= sub[k], axis=1)
            gt = np.any(sub > sub[k], axis=1)
            if not np.any(ge & gt):
                front.append(k)
        for k in front:
            strategies[int(remaining[k])]["pareto_rank"] = rank
        remaining = np.delete(remaining, front)
        rank += 1
    for i in remaining:
        strategies[int(i)]["pareto_rank"] = PARETO_MAX_RANK + 1
