"""Search diagnostics (T33): why portfolios were rejected, the best near misses, and relaxation suggestions.

Collected while the search runs, from numbers it already computes (no extra simulation). The unit is the
**property subset** (a set of distinct properties, whatever its loan durations), which is what the user sees as one
strategy. Reasons:

- ``budget``: the cheapest down payments of the subset already exceed the budget (pruned by the enumeration; the
  count is of the pruned one-property extensions of each visited branch, so a lower bound);
- ``max_props``: a subset that fits the budget but has one property more than the maximum;
- ``cf``: no loan-duration assignment reaches the cash-flow target (pre-filter, optimistic bound) or every
  simulated assignment ends outside it (after-tax CF of the stabilised year, ``glossary.CF_ACCEPTANCE_BASIS``);
- ``evaluator``: the simulation failed.

The relaxation suggestions are computed from the recorded gaps only, with no new search.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

REASONS = ("budget", "cf", "max_props", "evaluator")
REASON_LABELS = {
    "budget": "apport insuffisant (budget dépassé)",
    "cf": "cash-flow hors de la cible",
    "max_props": "nombre maximal de biens atteint",
    "evaluator": "échec de la simulation",
}
NEAR_MISS_N = 5
_CF_MARGIN = 0.10
_GAPS_CAP = 100_000  # memory bound on the stored gap lists


@dataclass(frozen=True)
class NearMiss:
    """A rejected subset and by how much it missed."""

    reason: str
    names: tuple[str, ...]
    gap: float  # always >= 0: EUR/month (cf), EUR (budget), properties (max_props)
    unit: str
    cf: float | None = None  # after-tax stabilised CF (EUR/month) when known
    estimated: bool = False  # True when the gap comes from the optimistic pre-filter, not a simulation

    def describe(self) -> str:
        who = " + ".join(self.names)
        if self.reason == "cf":
            kind = "estimé" if self.estimated else "simulé"
            return f"{who} : {self.gap:,.0f} €/mois hors cible ({kind})".replace(",", " ")
        if self.reason == "budget":
            return f"{who} : {self.gap:,.0f} € au-dessus du budget".replace(",", " ")
        if self.reason == "max_props":
            return f"{who} : {len(self.names)} biens pour un maximum de {len(self.names) - 1}"
        return f"{who} : simulation en échec"


@dataclass(frozen=True)
class Suggestion:
    """A relaxation of one constraint, with its measured effect. ``param`` is a ``RunContext`` field name."""

    reason: str
    param: str  # "cf_cible" | "tolerance" | "apport" | "max_properties"
    current: float
    suggested: float
    effect: str
    count: int
    text: str


@dataclass
class SearchDiagnostics:
    """Rejection counts, near misses and gap lists of one search."""

    budget: float = 0.0
    target_cf: float = 0.0
    tolerance: float = 0.0
    max_props: int = 0
    mode_cf: str = "min"
    top_n_near: int = NEAR_MISS_N
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(REASONS, 0))
    near_misses: dict[str, list[NearMiss]] = field(default_factory=lambda: {r: [] for r in REASONS})
    accepted: int = 0  # subsets with at least one accepted assignment
    evaluated: int = 0  # subsets simulated
    # Gap lists for the suggestions: (gap, names) so a subset is counted once.
    cf_gaps_sim: dict[tuple[str, ...], float] = field(default_factory=dict)
    cf_gaps_est: dict[tuple[str, ...], float] = field(default_factory=dict)
    budget_gaps: list[float] = field(default_factory=list)

    # ------------------------------------------------------------------ recording
    def record(self, near: NearMiss, count: int = 1) -> None:
        """Count ``count`` rejections for ``near.reason`` and keep ``near`` if among the N smallest gaps."""
        self.counts[near.reason] = self.counts.get(near.reason, 0) + count
        lst = self.near_misses.setdefault(near.reason, [])
        if len(lst) < self.top_n_near or near.gap < lst[-1].gap:
            if all(near.names != n.names for n in lst):
                lst.append(near)
                lst.sort(key=lambda n: (n.gap, n.names))
                del lst[self.top_n_near:]

    def count_only(self, reason: str, count: int = 1) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + count

    def add_cf_gap(self, names: tuple[str, ...], gap: float, *, estimated: bool) -> None:
        store = self.cf_gaps_est if estimated else self.cf_gaps_sim
        if len(store) < _GAPS_CAP and (names not in store or gap < store[names]):
            store[names] = gap

    def add_budget_gap(self, gap: float) -> None:
        if len(self.budget_gaps) < _GAPS_CAP:
            self.budget_gaps.append(gap)

    # ------------------------------------------------------------------ reading
    @property
    def total_rejected(self) -> int:
        return sum(self.counts.values())

    def top_reason(self) -> str | None:
        """The reason that eliminated the most subsets (ties: budget, cf, max_props, evaluator order)."""
        best = max(REASONS, key=lambda r: self.counts.get(r, 0))
        return best if self.counts.get(best, 0) > 0 else None

    def all_near_misses(self) -> list[NearMiss]:
        return [n for r in REASONS for n in self.near_misses.get(r, [])]

    def to_dict(self) -> dict[str, Any]:
        return {"counts": dict(self.counts), "accepted": self.accepted, "evaluated": self.evaluated}

    # ------------------------------------------------------------------ suggestions
    def suggestions(self, top_n: int = 10) -> list[Suggestion]:
        """Relaxations that would admit more subsets, the top reason first. Computed from the gaps only."""
        out: list[Suggestion] = []
        top = self.top_reason()
        for reason in REASONS:
            s = {"cf": self._suggest_cf, "budget": self._suggest_budget, "max_props": self._suggest_max_props}.get(
                reason)
            sug = s(top_n) if s else None
            if sug is not None:
                out.append(sug)
        out.sort(key=lambda s: (s.reason != top, -s.count))
        return out

    def _suggest_cf(self, top_n: int) -> Suggestion | None:
        estimated = not self.cf_gaps_sim
        gaps = sorted((self.cf_gaps_est if estimated else self.cf_gaps_sim).values())
        gaps = [g for g in gaps if g > 0]
        if not gaps:
            return None
        k = min(len(gaps), max(1, min(top_n, 5)))
        # 10 % margin: the allocator's after-tax correction (3 rounds) does not always land exactly on the target
        # when the target is relaxed, so the measured gap alone could still leave the strategy a few EUR short.
        need = gaps[k - 1] * (1.0 + _CF_MARGIN)
        if self.mode_cf == "target":
            new_tol = math.ceil((self.tolerance + need) / 10.0) * 10.0
            count = sum(1 for g in gaps if g * (1.0 + _CF_MARGIN) <= new_tol - self.tolerance)
            text = (f"Avec une tolérance de ±{new_tol:,.0f} € au lieu de ±{self.tolerance:,.0f} €, "
                    f"jusqu'à {count} stratégie(s) seraient acceptées").replace(",", " ")
            return Suggestion("cf", "tolerance", self.tolerance, new_tol, "acceptées", count, _hedge(text, estimated))
        new_target = math.floor((self.target_cf - need) / 10.0) * 10.0
        count = sum(1 for g in gaps if g * (1.0 + _CF_MARGIN) <= self.target_cf - new_target)
        text = (f"Avec un CF cible de {new_target:,.0f} €/mois au lieu de {self.target_cf:,.0f} €, "
                f"jusqu'à {count} stratégie(s) seraient acceptées").replace(",", " ")
        return Suggestion("cf", "cf_cible", self.target_cf, new_target, "acceptées", count, _hedge(text, estimated))

    def _suggest_budget(self, top_n: int) -> Suggestion | None:
        gaps = sorted(g for g in self.budget_gaps if g > 0)
        if not gaps:
            return None
        k = min(len(gaps), max(1, top_n))
        step = 5_000.0
        add = math.ceil(gaps[k - 1] / step) * step
        new_budget = self.budget + add
        count = sum(1 for g in gaps if g <= add)
        text = (f"Avec un apport de {new_budget:,.0f} € au lieu de {self.budget:,.0f} €, "
                f"{count} portefeuille(s) de plus seraient examinés").replace(",", " ")
        return Suggestion("budget", "apport", self.budget, new_budget, "examinés", count, text)

    def _suggest_max_props(self, top_n: int) -> Suggestion | None:
        n = self.counts.get("max_props", 0)
        if n <= 0:
            return None
        text = (f"Avec {self.max_props + 1} biens au maximum au lieu de {self.max_props}, "
                f"{n} portefeuille(s) de plus seraient examinés")
        return Suggestion("max_props", "max_properties", float(self.max_props), float(self.max_props + 1),
                          "examinés", n, text)


def _hedge(text: str, estimated: bool) -> str:
    return text + " (estimation avant simulation)" if estimated else text
