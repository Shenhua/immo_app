"""Portfolio search (T31): property subsets x loan durations, deterministic.

Replaces the brick-level exhaustive search (ThreadPool over up to 1M combinations) and the genetic algorithm.

Pipeline
--------
1. **Prepare** every brick once (``PortfolioAllocator.prepare``): loan constant ``k``, stabilised pre-tax NOI,
   minimum down payment, the dumped dict. Nothing is re-dumped per combination.
2. **Enumerate property subsets** (distinct ``nom``) of size 1..max_props, depth-first over the properties
   sorted by their cheapest minimum down payment, pruning a branch as soon as the cheapest down payments exceed
   the budget. A property can therefore never appear twice in a portfolio, whatever its loan duration.
3. **Screen loan-duration assignments** with vectorised numpy, per subset:
   - exact enumeration of all assignments when ``options ** k <= EXACT_MAX_ASSIGNMENTS`` (3 durations: k <= 5);
   - otherwise greedy: start from the duration with the best stand-alone cash flow for every property, then one
     pass of single-position swaps. This is a local optimum of the proxy, not a global one.
   Each assignment gets the analytic pre-tax allocation, a pre-filter verdict and a cheap proxy (pre-tax return
   on equity over the horizon, ``_roe_proxy``).
4. **Pick the assignments to simulate.** If every assignment that passes the pre-filter fits in
   ``EVAL_BUDGET`` evaluations, all of them are simulated (``solver_mode = "EXHAUSTIVE"``, same candidate set
   as the old exhaustive search). Otherwise each subset keeps its ``M`` best assignments by proxy, with
   ``M = max(1, EVAL_BUDGET // subsets)`` (``solver_mode = "SUBSET_PROXY"``). Bound: on the 18-property data
   set, the proxy picks the evaluator's best duration set for 48-90 % of the subsets and the mean fitness lost
   on the others is under 0.6 point out of 100 (measured during T31).
5. **Evaluate** each picked assignment with the full simulation, serially or in a process pool (see
   ``PARALLEL_MIN_EVALS``), and accept it on the after-tax stabilised cash flow (``glossary.CF_ACCEPTANCE_BASIS``).
   The pre-tax allocation is corrected for the simulated income tax (at most ``TAX_CORRECTION_ROUNDS`` rounds).
6. Results come back in candidate order (subset order, then assignment order), never in completion order.

Every result satisfies: ``apport_total <= budget``, unique ``nom``, ``len(details) <= max_props``.
"""
from __future__ import annotations

import itertools
import math
import os
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pickle import PicklingError
from typing import Any

import numpy as np
import structlog

from src.core.defaults import MARKET, TAX, TAX_RATES
from src.core.glossary import MIN_MODES, cf_accepts
from src.domain.models.archetype import is_furnished
from src.domain.models.brick import InvestmentBrick

from .allocator import PortfolioAllocator, PreparedBrick
from .diagnostics import NearMiss, SearchDiagnostics
from .evaluator import StrategyEvaluator
from .simulation import SimulationEngine

log = structlog.get_logger(__name__)

# Above this many duration assignments per subset the screening switches from exact to greedy + one swap pass.
EXACT_MAX_ASSIGNMENTS = 243  # 3 ** 5
# Simulations allowed before the duration choice falls back to the proxy's top-M per subset.
EVAL_BUDGET = 12_000
# Candidates below this count are simulated serially (a process pool costs ~1 s to start).
PARALLEL_MIN_EVALS = 2_000
MAX_WORKERS = 16  # capped further at cpu_count - 1
# Rounds of "simulate, measure the income tax, re-allocate" per candidate.
TAX_CORRECTION_ROUNDS = 3
# An accepted allocation is re-run with less down payment only if the tax refund frees this much CF (EUR/month).
MIN_REFUND_SAVING = 5.0
# Rows screened per numpy batch (memory bound: rows x k floats per array).
_SCREEN_CHUNK_ROWS = 200_000


class SearchInvariantError(AssertionError):
    """A search result broke a hard invariant (budget, unique properties, size). Always a bug."""


def check_invariants(strategies: Sequence[dict[str, Any]], budget: float, max_props: int) -> None:
    """Raise ``SearchInvariantError`` unless every strategy is within budget, has unique names and fits."""
    for s in strategies:
        details = s.get("details", [])
        names = [d.get("nom", d.get("nom_bien")) if isinstance(d, dict) else d.nom for d in details]
        if float(s.get("apport_total", 0.0)) > budget + 1e-6:
            raise SearchInvariantError(f"over budget: {s.get('apport_total')} > {budget} ({names})")
        if len(set(names)) != len(names):
            raise SearchInvariantError(f"duplicate property in {names}")
        if not 0 < len(names) <= max_props:
            raise SearchInvariantError(f"{len(names)} properties, max {max_props} ({names})")


@dataclass(frozen=True)
class SearchProblem:
    """Everything a worker needs to evaluate a candidate (picklable)."""

    budget: float
    target_cf: float
    tolerance: float
    horizon: int
    max_props: int
    mode_cf: str
    # Rentier mode: spend the whole budget (up to the per-brick cap) to maximise the CF; the target is a floor.
    use_full_capital: bool = False
    # Bounds on the monthly income tax of the stabilised year, used only to widen the pre-tax pre-filter:
    # after-tax CF <= pre-tax CF + refund_max (déficit foncier on global income), >= pre-tax CF - rent * tax_rate_max.
    refund_max: float = 0.0
    tax_rate_max: float = 0.0
    appreciation_pct: float = MARKET.appreciation_bien_pct
    frais_vente_pct: float = TAX.frais_vente_pct


def tax_bounds(tax_params: dict[str, Any]) -> tuple[float, float]:
    """(max monthly refund, max tax rate on gross rent) for the pre-filter margins.

    Refund: an unfurnished deficit under the foncier réel regime is deducted from global income up to
    10,700 EUR, plus 10,700 EUR for energy works until 2027 (CGI 156 I 3°; TAX_RATES), at the TMI.
    LMNP and SCI IS deficits only carry forward, so they never refund. Tax rate: taxable income never exceeds
    gross rent, so tax <= rent x (TMI + highest social levy), or x the IS normal rate for an SCI.
    """
    tmi = float(tax_params.get("tmi_pct", TAX.tmi_pct)) / 100.0
    sci = bool(tax_params.get("sci_is", TAX.sci_is))
    foncier_reel = str(tax_params.get("regime_foncier", TAX.regime_foncier)) == "reel"
    refund = 0.0
    if foncier_reel and not sci:
        refund = (TAX_RATES.deficit_foncier_global_cap + TAX_RATES.deficit_foncier_energy_extra_cap) * tmi / 12.0
    rate = max(tmi + max(TAX_RATES.ps_bic_pct, TAX_RATES.ps_foncier_pct) / 100.0, TAX_RATES.is_normal_pct / 100.0)
    return refund, rate


# --------------------------------------------------------------------------- brick table
class BrickTable:
    """Prepared bricks as numpy columns, grouped by property."""

    def __init__(self, prepared: Sequence[PreparedBrick], problem: SearchProblem, market: dict[str, float]):
        self.prepared = list(prepared)
        n = len(self.prepared)
        self.apport_min = np.array([b.apport_min for b in self.prepared], dtype=float)
        self.capital = np.array([b.capital for b in self.prepared], dtype=float)
        self.k = np.array([b.k for b in self.prepared], dtype=float)
        self.noi = np.array([b.noi_stab for b in self.prepared], dtype=float)
        g_rent = 1.0 + float(market.get("revalo_loyer_pct", MARKET.revalo_loyer_pct)) / 100.0
        rent = np.array([float(b.data.get("loyer_mensuel_initial", 0.0) or 0.0) for b in self.prepared]) * g_rent
        self.unfurnished = np.array([not is_furnished(b.data) for b in self.prepared], dtype=bool)
        self.tax_up = rent * problem.tax_rate_max  # max monthly tax per brick (pre-filter, target mode)

        # ROE proxy constants (pre-tax, see _roe_proxy): value_i is what the brick returns over the horizon at
        # zero debt (NOI + net sale price); debt_i is the debt cost over the horizon per euro still borrowed.
        h = int(problem.horizon)
        growth_sum = sum(g_rent ** (y - 2) for y in range(1, h + 1))
        g_price = (1.0 + problem.appreciation_pct / 100.0) ** h
        price = np.array([float(b.data.get("prix_achat_bien", 0.0) or 0.0) for b in self.prepared])
        self.value = 12.0 * self.noi * growth_sum + price * g_price * (1.0 - problem.frais_vente_pct / 100.0)
        self.debt = np.array([
            b.k * min(b.duree_mois, 12 * h) + _balance_fraction(float(b.data.get("taux_annuel_pct", 0.0)),
                                                                  b.duree_mois, 12 * h)
            for b in self.prepared
        ])
        self.cf0 = self.noi - self.k * self.capital

        # Properties: distinct names, options sorted by duration (then input order). Duplicated (nom, duration)
        # bricks keep the first one.
        options: dict[str, list[int]] = {}
        seen: set[tuple[str, int]] = set()
        for i, b in enumerate(self.prepared):
            if (b.nom, b.duree_mois) in seen:
                continue
            seen.add((b.nom, b.duree_mois))
            options.setdefault(b.nom, []).append(i)
        for opts in options.values():
            opts.sort(key=lambda i: (self.prepared[i].duree_mois, i))
        # Properties by cheapest down payment, then name: the order of enumeration and of every tie.
        self.names = sorted(options, key=lambda nm: (min(self.apport_min[i] for i in options[nm]), nm))
        self.n_props = len(self.names)
        self.m = max((len(v) for v in options.values()), default=0)
        self.options = np.full((self.n_props, max(1, self.m)), -1, dtype=np.int64)
        for p, nm in enumerate(self.names):
            self.options[p, : len(options[nm])] = options[nm]
        self.n_options = np.array([len(options[nm]) for nm in self.names], dtype=np.int64)
        self.prop_min_apport = np.array([min(self.apport_min[i] for i in options[nm]) for nm in self.names])
        # Best stand-alone cash flow option per property (greedy start).
        self.best_cf_option = np.array([
            int(np.argmax([self.cf0[i] for i in options[nm]])) for nm in self.names
        ], dtype=np.int64)
        assert n == len(self.apport_min)


def _balance_fraction(rate_pct: float, n_months: int, paid_months: int) -> float:
    """Outstanding principal after ``paid_months`` of an ``n_months`` annuity, as a fraction of the loan."""
    if n_months <= 0 or paid_months >= n_months:
        return 0.0
    r = rate_pct / 1200.0
    if r == 0:
        return 1.0 - paid_months / n_months
    return ((1 + r) ** n_months - (1 + r) ** paid_months) / ((1 + r) ** n_months - 1)


def enumerate_subsets(table: BrickTable, budget: float, max_props: int,
                      diag: SearchDiagnostics | None = None) -> Iterator[tuple[int, ...]]:
    """Property subsets (indices into ``table.names``) whose cheapest down payments fit the budget.

    Depth-first, properties in ``table.names`` order (cheapest first), so a branch stops at the first property
    that no longer fits. Deterministic. With ``diag``, the pruned extensions are counted by reason (over budget,
    or over the maximum size while fitting the budget).
    """
    costs = table.prop_min_apport
    n = table.n_props

    def rec(start: int, chosen: list[int], spent: float) -> Iterator[tuple[int, ...]]:
        if chosen:
            yield tuple(chosen)
        if len(chosen) >= max_props:
            if diag is not None:
                for p in range(start, n):
                    if spent + costs[p] > budget + 1e-6:
                        break
                    diag.record(NearMiss("max_props", tuple(table.names[i] for i in (*chosen, p)), 1.0, "bien"))
            return
        for p in range(start, n):
            c = spent + costs[p]
            if c > budget + 1e-6:
                if diag is not None:
                    for q in range(p, n):
                        gap = spent + costs[q] - budget
                        diag.add_budget_gap(gap)
                        if q == p:
                            diag.record(NearMiss("budget", tuple(table.names[i] for i in (*chosen, q)), gap, "€"))
                        else:
                            diag.count_only("budget")
                break
            chosen.append(p)
            yield from rec(p + 1, chosen, c)
            chosen.pop()

    yield from rec(0, [], 0.0)


# --------------------------------------------------------------------------- vectorised screening
@dataclass
class Screen:
    """Screening result for rows of brick-index assignments (n, k)."""

    passes: np.ndarray  # pre-filter verdict
    proxy: np.ndarray  # higher is better; -inf when the row does not pass
    cf_gap: np.ndarray  # EUR/month the optimistic CF bound misses the target by (<= 0: not a CF rejection)
    cf_best: np.ndarray  # optimistic reachable CF (EUR/month, after the largest possible refund)


def screen(table: BrickTable, rows: np.ndarray, problem: SearchProblem, max_extra_pct: float) -> Screen:
    """Vectorised analytic allocation + pre-filter + proxy for assignment rows (brick indices, shape (n, k)).

    Mirrors ``PortfolioAllocator.plan``/``_fill`` (same greedy, same tie order) so the proxy is computed at the
    down payment the evaluation will use. The pre-filter is deliberately loose: it only drops rows that cannot
    meet the target on the after-tax basis given the tax bounds of ``SearchProblem``.
    """
    n, k = rows.shape
    amin = table.apport_min[rows].sum(axis=1)
    K = table.k[rows]
    C = table.capital[rows]
    cf0 = table.cf0[rows].sum(axis=1)
    money = problem.budget - amin
    in_budget = money >= -1e-6
    money = np.maximum(money, 0.0)
    target, tol = problem.target_cf, problem.tolerance

    if problem.use_full_capital and problem.mode_cf != "target":
        need = np.full(n, np.inf)
    elif problem.mode_cf == "target":
        need = np.where(cf0 < target - tol, target - cf0, 0.0)
    else:
        need = np.where(cf0 < target, target + PortfolioAllocator.MIN_MODE_EPS - cf0, 0.0)
    X = _fill(K, C, need, money, max_extra_pct)
    extra = X.sum(axis=1)

    # Loose upper bound of the reachable pre-tax CF: every euro at the best k, or every loan repaid.
    cf_max = cf0 + np.minimum((K * C).sum(axis=1), K.max(axis=1) * money)
    refund = np.where(table.unfurnished[rows].any(axis=1), problem.refund_max, 0.0)
    if problem.mode_cf == "target":
        tax_up = table.tax_up[rows].sum(axis=1)
        passes = in_budget & (cf_max + refund >= target - tol) & (cf0 - tax_up <= target + tol)
        cf_gap = np.maximum(target - tol - (cf_max + refund), (cf0 - tax_up) - (target + tol))
    else:
        passes = in_budget & (cf_max + refund >= target)
        cf_gap = target - (cf_max + refund)

    proxy = _roe_proxy(table, rows, X, amin, extra)
    return Screen(passes=passes, proxy=np.where(passes, proxy, -np.inf), cf_gap=cf_gap, cf_best=cf_max + refund)


def _fill(K: np.ndarray, C: np.ndarray, need: np.ndarray, money: np.ndarray, pct: float) -> np.ndarray:
    """Row-wise ``PortfolioAllocator._fill``: extra down payment per column."""
    order = np.argsort(-K, axis=1, kind="stable")
    Ks = np.take_along_axis(K, order, axis=1)
    Cs = np.take_along_axis(C, order, axis=1)
    Xs = np.zeros_like(Ks)
    left = need.astype(float).copy()
    money = money.astype(float).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        for j in range(Ks.shape[1]):
            kj, cj = Ks[:, j], Cs[:, j]
            active = (left > 1e-9) & (money > 1e-9) & (kj > 0) & (cj > 0)
            want = np.where(kj > 0, left / np.where(kj > 0, kj, 1.0), 0.0)
            full = (want >= cj) & (money >= cj)
            x = np.where(full, cj, np.minimum(np.minimum(want, pct * cj), money))
            x = np.where(active, x, 0.0)
            Xs[:, j] = x
            left = left - np.where(active, kj * x, 0.0)
            money = money - x
        for j in range(Ks.shape[1]):
            kj, cj = Ks[:, j], Cs[:, j]
            rest = cj - Xs[:, j]
            act = (left > 1e-9) & (kj > 0) & (rest > 0) & (money >= rest)
            Xs[:, j] = np.where(act, cj, Xs[:, j])
            left = left - np.where(act, kj * rest, 0.0)
            money = money - np.where(act, rest, 0.0)
    X = np.empty_like(Xs)
    np.put_along_axis(X, order, Xs, axis=1)
    return X


def _roe_proxy(table: BrickTable, rows: np.ndarray, X: np.ndarray, amin: np.ndarray, extra: np.ndarray) -> np.ndarray:
    """Pre-tax return on equity over the horizon: (NOI + net sale - debt cost - down payment) / down payment.

    Chosen during T31 among several cheap proxies (shortest/longest loans, smallest down payment, pre-tax IRR,
    a pre-tax replica of the evaluator score) as the one that best predicts which loan durations the evaluator
    ranks first inside a subset.
    """
    borrowed = table.capital[rows] - X
    wealth = table.value[rows].sum(axis=1) - (borrowed * table.debt[rows]).sum(axis=1) - amin - extra
    equity = np.maximum(amin + extra, 1.0)
    return np.asarray(wealth / equity, dtype=float)


# --------------------------------------------------------------------------- evaluation
class CandidateEvaluator:
    """Allocates and simulates one assignment; accepts it on the after-tax stabilised CF."""

    def __init__(self, allocator: PortfolioAllocator, evaluator: StrategyEvaluator, prepared: Sequence[PreparedBrick],
                 problem: SearchProblem):
        self.allocator = allocator
        self.evaluator = evaluator
        self.prepared = list(prepared)
        self.problem = problem
        # Why the last call returned None: (reason, gap >= 0, after-tax CF or nan). None after an accepted call.
        self.last_reject: tuple[str, float, float] | None = None

    def __call__(self, bricks: tuple[int, ...]) -> tuple[tuple[float, ...], float, dict[str, Any]] | None:
        """Return (extra per brick, pre-tax CF estimate, metrics) of the best accepted allocation, or None.

        Round 1 allocates for the target on the pre-tax estimate. If the simulated after-tax CF differs from the
        estimate by the stabilised-year income tax ``t``, the next round allocates for ``target + t``, so the
        after-tax CF lands on the target (or, with a refund, the down payment shrinks). Stops when accepted
        with nothing left to save, when the allocation repeats, or after ``TAX_CORRECTION_ROUNDS``.
        """
        pb = self.problem
        P = [self.prepared[i] for i in bricks]
        bias = 0.0
        best: tuple[tuple[float, ...], float, dict[str, Any]] | None = None
        seen: set[float] = set()
        reject: tuple[str, float, float] | None = None
        self.last_reject = None
        for _ in range(TAX_CORRECTION_ROUNDS):
            plan = self.allocator.plan(P, pb.budget, pb.target_cf + bias, pb.tolerance,
                                       pb.use_full_capital)
            if not plan.within_budget:
                self.last_reject = ("budget", max(0.0, plan.apport_total - pb.budget), math.nan)
                return None
            key = round(plan.apport_total, 2)
            if key in seen:
                break
            seen.add(key)
            details = self.allocator.details(P, plan.extra)
            strategy = {"details": details, "apport_total": plan.apport_total}
            _, metrics = self.evaluator.evaluate(strategy, pb.target_cf, pb.tolerance, pb.horizon)
            if "error" in metrics:
                self.last_reject = None if best is not None else ("evaluator", 0.0, math.nan)
                return best
            after = float(metrics["cf_monthly_stab"])
            accepted = cf_accepts(after, pb.target_cf, pb.tolerance, pb.mode_cf)
            if not accepted:
                gap = pb.target_cf - after if pb.mode_cf in MIN_MODES else abs(after - pb.target_cf) - pb.tolerance
                if reject is None or gap < reject[1]:
                    reject = ("cf", max(0.0, gap), after)
            if accepted and (best is None or metrics["fitness"] > best[2]["fitness"]):
                best = (plan.extra, plan.cf, metrics)
            tax = plan.cf - after  # stabilised monthly income tax (negative = refund)
            new_bias = tax
            if accepted and (sum(plan.extra) <= 0.0 or new_bias >= bias - MIN_REFUND_SAVING):
                break
            if not accepted and abs(new_bias - bias) < 0.01:
                break
            bias = new_bias
        self.last_reject = None if best is not None else reject
        return best


# Worker-process state (process pool only).
_WORKER: CandidateEvaluator | None = None


def _worker_init(prepared: list[PreparedBrick], problem: SearchProblem, mode_cf: str, market: dict[str, float],
                 tax_params: dict[str, Any], scorer: Any) -> None:
    global _WORKER
    import logging

    from .simulation import build_engine

    logging.disable(logging.INFO)
    allocator = PortfolioAllocator(mode_cf, market=market)
    evaluator = StrategyEvaluator(build_engine(market, tax_params), allocator, scorer)
    _WORKER = CandidateEvaluator(allocator, evaluator, prepared, problem)


def _worker_run(chunk: list[tuple[int, ...]]) -> list[Any]:
    """(result or None, rejection or None) per candidate."""
    assert _WORKER is not None
    out = []
    for c in chunk:
        r = _WORKER(c)
        out.append((r, _WORKER.last_reject))
    return out


# --------------------------------------------------------------------------- search
class PortfolioSearch:
    """Deterministic subset search (see module docstring)."""

    def __init__(
        self,
        allocator: PortfolioAllocator,
        simulator: SimulationEngine,
        scorer: Any,
        engine_spec: tuple[dict[str, float], dict[str, Any]] | None = None,
        n_workers: int | None = None,
    ):
        """``engine_spec`` = (market, tax_params) lets worker processes rebuild the engine; without it the
        evaluation is serial. ``n_workers``: None = automatic, 1 = serial."""
        self.allocator = allocator
        self.simulator = simulator
        self.scorer = scorer
        self.engine_spec = engine_spec
        self.n_workers = n_workers
        self.evaluator = StrategyEvaluator(simulator, allocator, scorer)
        self.solver_mode = ""
        self.stats: dict[str, Any] = {}
        self.diagnostics = SearchDiagnostics()

    def solve(
        self,
        all_bricks: Sequence[InvestmentBrick],
        problem: SearchProblem,
        market: dict[str, float] | None = None,
        progress_callback: Callable[[Any], None] | None = None,
    ) -> list[dict[str, Any]]:
        from src.application.progress import SearchPhase, SearchProgress

        market = market or {}
        prepared = [self.allocator.prepare(b) for b in all_bricks]
        table = BrickTable(prepared, problem, market)
        diag = self.diagnostics = SearchDiagnostics(
            budget=problem.budget, target_cf=problem.target_cf, tolerance=problem.tolerance,
            max_props=problem.max_props, mode_cf=problem.mode_cf)

        if progress_callback:
            progress_callback(SearchProgress(phase=SearchPhase.COMBO_GENERATION, phase_index=2,
                                             message="Génération des combinaisons..."))

        t0 = time.perf_counter()
        candidates, exact = self._screen_all(table, problem, diag)
        self.stats["screen_s"] = round(time.perf_counter() - t0, 3)
        self.solver_mode = "EXHAUSTIVE" if exact else "SUBSET_PROXY"
        log.info("search_candidates", candidates=len(candidates), mode=self.solver_mode, **self.stats)

        t0 = time.perf_counter()
        results = self._evaluate(candidates, prepared, problem, market, progress_callback)
        self.stats["evaluate_s"] = round(time.perf_counter() - t0, 3)

        strategies = []
        outcome: dict[tuple[str, ...], list[Any] | None] = {}  # property subset -> rejections, None once accepted
        for bricks, (res, reject) in zip(candidates, results, strict=True):
            names = tuple(sorted({prepared[i].nom for i in bricks}))
            slot = outcome.setdefault(names, [])
            if res is None:
                if reject is not None and slot is not None:
                    slot.append(reject)
                continue
            outcome[names] = None
            extra, cf_pre, metrics = res
            P = [prepared[i] for i in bricks]
            strategies.append(_to_strategy(self.allocator.details(P, extra), cf_pre, metrics))
        self._record_evaluated(diag, outcome)
        check_invariants(strategies, problem.budget, problem.max_props)
        log.info("search_finished", evaluated=len(candidates), accepted=len(strategies))
        return strategies

    @staticmethod
    def _record_evaluated(diag: SearchDiagnostics, outcome: dict[tuple[str, ...], list[Any] | None]) -> None:
        """Subset-level verdicts of the simulated candidates: accepted, or rejected for its smallest CF gap."""
        for names, rejects in outcome.items():
            diag.evaluated += 1
            if rejects is None:
                diag.accepted += 1
                continue
            cf = [r for r in rejects if r[0] == "cf"]
            if cf:
                best = min(cf, key=lambda r: r[1])
                diag.record(NearMiss("cf", names, best[1], "€/mois", cf=best[2]))
                diag.add_cf_gap(names, best[1], estimated=False)
            elif any(r[0] == "evaluator" for r in rejects):
                diag.record(NearMiss("evaluator", names, 0.0, ""))
            else:
                bud = min((r for r in rejects if r[0] == "budget"), key=lambda r: r[1], default=None)
                diag.record(NearMiss("budget", names, bud[1] if bud else 0.0, "€"))
                if bud:
                    diag.add_budget_gap(bud[1])

    # ------------------------------------------------------------------ screening
    def _screen_all(self, table: BrickTable, problem: SearchProblem,
                    diag: SearchDiagnostics | None = None) -> tuple[list[tuple[int, ...]], bool]:
        """Candidates to simulate, in deterministic order, and whether the duration choice was exhaustive."""
        if table.n_props == 0 or problem.max_props <= 0:
            self.stats = {"subsets": 0, "assignments": 0}
            return [], True
        pct = self.allocator.max_extra_apport_pct
        by_k: dict[int, list[tuple[int, ...]]] = {}
        for s in enumerate_subsets(table, problem.budget, problem.max_props, diag):
            by_k.setdefault(len(s), []).append(s)

        # Per subset: passing rows (brick-index tuples) with their proxy, in assignment order.
        sub_rows: list[tuple[np.ndarray, np.ndarray]] = []
        all_exact = True
        n_assign = 0
        for k in sorted(by_k):
            subsets = np.array(by_k[k], dtype=np.int64)
            if max(1, table.m) ** k <= EXACT_MAX_ASSIGNMENTS:
                grid = np.array(list(itertools.product(range(table.m), repeat=k)), dtype=np.int64)
                per_chunk = max(1, _SCREEN_CHUNK_ROWS // len(grid))
                for c0 in range(0, len(subsets), per_chunk):
                    S = subsets[c0:c0 + per_chunk]
                    B = table.options[S[:, None, :], grid[None, :, :]]  # (ns, g, k)
                    ns, g, _ = B.shape
                    rows = B.reshape(ns * g, k)
                    valid = (rows >= 0).all(axis=1)
                    n_assign += int(valid.sum())
                    sc = screen(table, np.where(rows >= 0, rows, 0), problem, pct)
                    ok = (sc.passes & valid).reshape(ns, g)
                    prox = sc.proxy.reshape(ns, g)
                    gaps = np.where(valid, sc.cf_gap, np.inf).reshape(ns, g)
                    B3 = B
                    for i in range(ns):
                        idx = np.flatnonzero(ok[i])
                        if idx.size:
                            sub_rows.append((B3[i, idx], prox[i, idx]))
                        elif diag is not None and np.isfinite(gaps[i]).any():
                            self._screen_reject(diag, table, S[i], float(gaps[i].min()))
            else:
                all_exact = False
                rows = self._greedy_assignments(table, subsets, problem, pct)
                n_assign += len(rows) * (1 + k * max(0, table.m - 1))
                sc = screen(table, rows, problem, pct)
                for ii in np.flatnonzero(sc.passes).tolist():
                    sub_rows.append((rows[ii:ii + 1], sc.proxy[ii:ii + 1]))
                if diag is not None:
                    for ii in np.flatnonzero(~sc.passes).tolist():
                        self._screen_reject(diag, table, subsets[ii], float(sc.cf_gap[ii]), float(sc.cf_best[ii]))

        n_pass = sum(len(r) for r, _ in sub_rows)
        self.stats = {"subsets": sum(len(v) for v in by_k.values()), "assignments": n_assign, "passing": n_pass}
        if n_pass <= EVAL_BUDGET:
            return [tuple(int(x) for x in r) for rows, _ in sub_rows for r in rows], all_exact

        m_keep = max(1, EVAL_BUDGET // max(1, len(sub_rows)))
        out: list[tuple[int, ...]] = []
        for rows, prox in sub_rows:
            if len(rows) > m_keep:
                # best proxy first; ties keep assignment order (stable sort)
                keep = np.sort(np.argsort(-prox, kind="stable")[:m_keep])
                rows = rows[keep]
            out.extend(tuple(int(x) for x in r) for r in rows)
        self.stats["kept_per_subset"] = m_keep
        return out, False

    @staticmethod
    def _screen_reject(diag: SearchDiagnostics, table: BrickTable, subset: np.ndarray, gap: float,
                       cf_best: float | None = None) -> None:
        """A subset none of whose assignments passes the pre-filter: a (lower bound) CF rejection."""
        names = tuple(table.names[int(p)] for p in subset)
        gap = max(0.0, gap)
        diag.record(NearMiss("cf", names, gap, "€/mois", cf=cf_best, estimated=True))
        diag.add_cf_gap(names, gap, estimated=True)

    def _greedy_assignments(self, table: BrickTable, subsets: np.ndarray, problem: SearchProblem,
                            pct: float) -> np.ndarray:
        """One assignment per subset: best stand-alone CF duration everywhere, then one pass of swaps."""
        choice = table.best_cf_option[subsets]  # (ns, k) option positions
        ns, k = subsets.shape
        for j in range(k):
            best_score = np.full(ns, -np.inf)
            best_opt = choice[:, j].copy()
            for o in range(table.m):
                trial = choice.copy()
                trial[:, j] = o
                rows = table.options[subsets, trial]
                valid = (rows >= 0).all(axis=1)
                sc = screen(table, np.where(rows >= 0, rows, 0), problem, pct)
                score = np.where(valid & sc.passes, sc.proxy, -np.inf)
                better = score > best_score  # strict: ties keep the earlier (shorter) option
                best_score = np.where(better, score, best_score)
                best_opt = np.where(better, o, best_opt)
            choice[:, j] = best_opt
        return np.asarray(table.options[subsets, choice], dtype=np.int64)

    # ------------------------------------------------------------------ evaluation
    def _evaluate(self, candidates: list[tuple[int, ...]], prepared: list[PreparedBrick], problem: SearchProblem,
                  market: dict[str, float], progress_callback: Callable[[Any], None] | None) -> list[Any]:
        from src.application.progress import SearchPhase, SearchProgress

        n = len(candidates)
        found = 0

        def report(done: int) -> None:
            if progress_callback:
                progress_callback(SearchProgress(phase=SearchPhase.EVALUATION, phase_index=3, items_processed=done,
                                                 items_total=n, valid_count=found))

        report(0)
        workers = self._workers(n)
        results: list[Any] = []  # (result or None, rejection or None)
        if workers > 1:
            try:
                results = self._evaluate_pool(candidates, prepared, problem, market, workers, report)
            except (BrokenProcessPool, PicklingError, OSError) as e:
                log.warning("process_pool_failed_falling_back_to_serial", error=str(e))
                results = []
        if not results and n:
            ev = CandidateEvaluator(self.allocator, self.evaluator, prepared, problem)
            step = max(1, n // 100)
            for i, c in enumerate(candidates):
                r = ev(c)
                results.append((r, ev.last_reject))
                found += r is not None
                if (i + 1) % step == 0:
                    report(i + 1)
        report(n)
        return results

    def _workers(self, n: int) -> int:
        if self.n_workers is not None:
            w = self.n_workers
        elif n < PARALLEL_MIN_EVALS or self.engine_spec is None:
            w = 1
        else:
            w = min(MAX_WORKERS, (os.cpu_count() or 2) - 1)
        return max(1, min(w, n)) if self.engine_spec is not None else 1

    def _evaluate_pool(self, candidates, prepared, problem, market, workers, report) -> list[Any]:
        assert self.engine_spec is not None
        mkt, tax_params = self.engine_spec
        size = max(1, math.ceil(len(candidates) / (workers * 8)))
        chunks = [candidates[i:i + size] for i in range(0, len(candidates), size)]
        out: list[Any] = []
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                                 initargs=(prepared, problem, self.allocator.mode_cf, mkt, tax_params,
                                           self.scorer)) as ex:
            for part in ex.map(_worker_run, chunks):  # map keeps submission order
                out.extend(part)
                report(len(out))
        return out


def _to_strategy(details: list[dict[str, Any]], cf_pre: float, m: dict[str, Any]) -> dict[str, Any]:
    """Strategy dict with the keys the finder and the UI read (same as the pre-T31 exhaustive search)."""
    return {
        "details": details,
        "apport_total": sum(d["apport_final_bien"] for d in details),
        "cash_flow_final": cf_pre,  # stabilised pre-tax estimate of the allocator (display / scorer)
        "allocation_ok": True,
        "liquidation_nette": m.get("liquidation_nette", 0.0),
        "enrich_net": m.get("enrich_net", 0.0),
        "tri_annuel": m.get("tri_annuel", 0.0),
        "tri_global": m.get("tri_annuel", 0.0),
        "dscr_y1": m.get("dscr_y1", 0.0),
        "dscr_stab": m.get("dscr_stab", 0.0),
        "cf_monthly_stab": m.get("cf_monthly_stab", 0.0),
        "cf_monthly_y1": m.get("cf_monthly_y1", 0.0),
        "cf_monthly_avg": m.get("cf_monthly_avg", 0.0),
        "is_acceptable": bool(m.get("is_acceptable", False)),
        "fitness": m.get("fitness", 0.01),
        "qual_score": m.get("qual_score", 50.0),
    }
