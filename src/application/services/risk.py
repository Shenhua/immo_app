"""Risk scenarios, break-even indicators and Monte Carlo (T64; pure: no streamlit import).

Everything goes through ``sensitivity.run_sensitivity``, so the Central scenario is the baseline exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from src.application.run_context import RunContext
from src.application.services.sensitivity import Deltas, run_sensitivity
from src.core.defaults import (
    BREAK_EVEN_APPRECIATION_RANGE,
    BREAK_EVEN_MAX_SIMS,
    BREAK_EVEN_VACANCE_RANGE,
    MONTE_CARLO,
    RISK_SCENARIOS,
)

PERCENTILES = (10, 50, 90)


def scenario_deltas(name: str) -> Deltas:
    return Deltas(**RISK_SCENARIOS[name])


def run_scenarios(strategy: dict[str, Any], ctx: RunContext) -> dict[str, dict[str, Any]]:
    """Key metrics of each named scenario (``tri_annuel`` may be None)."""
    return {name: run_sensitivity(strategy, scenario_deltas(name), ctx) for name in RISK_SCENARIOS}


def bisect(
    f: Callable[[float], float], lo: float, hi: float, max_sims: int = BREAK_EVEN_MAX_SIMS, tol: float = 0.5
) -> float | None:
    """Root of ``f`` on [lo, hi] (monotone), or None when not bracketed. Uses at most ``max_sims`` calls."""
    f_lo, f_hi = f(lo), f(hi)
    sims = 2
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    if f_lo * f_hi > 0:
        return None
    best = lo if abs(f_lo) < abs(f_hi) else hi
    best_f = min(abs(f_lo), abs(f_hi))
    while sims < max_sims:
        mid = (lo + hi) / 2.0
        f_mid = f(mid)
        sims += 1
        if abs(f_mid) < best_f:
            best, best_f = mid, abs(f_mid)
        if abs(f_mid) <= tol:
            return mid
        if f_mid * f_lo < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return best


def _mean_provision(strategy: dict[str, Any]) -> float:
    bricks = strategy.get("details", [])
    if not bricks:
        return 0.0
    return sum(float(b.get("provision_pct", 0.0) or 0.0) for b in bricks) / len(bricks)


def break_even_vacancy(
    strategy: dict[str, Any], ctx: RunContext, target: float | None = None
) -> dict[str, float] | None:
    """Vacancy at which the stabilised monthly CF reaches ``target`` (default: the user's CF target).

    Returns ``{"delta_pts", "vacance_pct"}`` (delta on every brick's provision; absolute value is the mean
    provision plus the delta), or None when the target is not reached inside the search range.
    """
    tgt = ctx.cf_cible if target is None else target

    def f(delta: float) -> float:
        res = run_sensitivity(strategy, Deltas(vacance_pts=delta), ctx)
        return float(res["cf_stab_mensuel"]) - tgt if res else float("nan")

    lo, hi = BREAK_EVEN_VACANCE_RANGE
    root = bisect(f, lo, hi)
    if root is None or root != root:
        return None
    vacance = max(0.0, _mean_provision(strategy) + root)
    if vacance > 100.0:  # beyond 100 % vacancy: the target cannot be reached by vacancy alone
        return None
    return {"delta_pts": root, "vacance_pct": vacance}


def break_even_appreciation(strategy: dict[str, Any], ctx: RunContext) -> dict[str, float] | None:
    """Yearly appreciation at which ``enrichissement_net`` is 0, or None when not bracketed.

    The tolerance is 1 EUR of wealth. Returns ``{"delta_pts", "appreciation_pct"}``.
    """

    def f(delta: float) -> float:
        res = run_sensitivity(strategy, Deltas(appreciation_pts=delta), ctx)
        return float(res["enrichissement_net"]) if res else float("nan")

    lo, hi = BREAK_EVEN_APPRECIATION_RANGE
    root = bisect(f, lo, hi, tol=1.0)
    if root is None or root != root:
        return None
    return {"delta_pts": root, "appreciation_pct": ctx.appreciation_bien_pct + root}


def _truncated_normal(
    rng: np.random.Generator, mean: float, sd: float, bounds: tuple[float, float], n: int
) -> np.ndarray:
    """Normal draws truncated to ``bounds`` by rejection (deterministic given the generator)."""
    lo, hi = bounds
    out = np.empty(0)
    while out.size < n:
        x = rng.normal(mean, sd, size=2 * n)
        out = np.concatenate([out, x[(x >= lo) & (x <= hi)]])
    return out[:n]


def monte_carlo_draws(n: int, seed: int = MONTE_CARLO.seed) -> list[Deltas]:
    """``n`` reproducible draws of (appreciation, rent indexation, vacancy) deltas."""
    n = max(1, min(int(n), MONTE_CARLO.n_max))
    rng = np.random.default_rng(seed)
    appr = _truncated_normal(rng, 0.0, MONTE_CARLO.appreciation_sd, MONTE_CARLO.appreciation_bounds, n)
    reva = _truncated_normal(rng, 0.0, MONTE_CARLO.revalo_sd, MONTE_CARLO.revalo_bounds, n)
    vac = _truncated_normal(rng, MONTE_CARLO.vacance_mean, MONTE_CARLO.vacance_sd, MONTE_CARLO.vacance_bounds, n)
    return [
        Deltas(appreciation_pts=float(a), revalo_pts=float(r), vacance_pts=float(v))
        for a, r, v in zip(appr, reva, vac, strict=True)
    ]


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    p10, p50, p90 = (float(x) for x in np.percentile(values, PERCENTILES))
    return {"p10": p10, "p50": p50, "p90": p90}


def monte_carlo(
    strategy: dict[str, Any], ctx: RunContext, n: int = MONTE_CARLO.n_default, seed: int = MONTE_CARLO.seed
) -> dict[str, Any]:
    """Run the draws. Returns the samples and P10/P50/P90 of IRR and net enrichment.

    Draws whose IRR failed (None) are left out of the IRR percentiles only (``n_irr_failed`` counts them).
    """
    tri: list[float] = []
    enrich: list[float] = []
    failed = 0
    for d in monte_carlo_draws(n, seed):
        res = run_sensitivity(strategy, d, ctx)
        if not res:
            failed += 1
            continue
        enrich.append(float(res["enrichissement_net"]))
        if res.get("tri_annuel") is None:
            failed += 1
        else:
            tri.append(float(res["tri_annuel"]))
    return {
        "n": len(enrich),
        "seed": seed,
        "tri_samples": tri,
        "enrichissement_samples": enrich,
        "tri": _percentiles(tri),
        "enrichissement_net": _percentiles(enrich),
        "n_irr_failed": failed,
    }
