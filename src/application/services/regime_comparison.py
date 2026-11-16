"""Tax regime comparison (T62). Pure: no streamlit import.

Runs the same strategy under every *eligible* regime combination with the same ``RunContext``:
furnished properties vary LMNP réel / micro-BIC, unfurnished ones micro-foncier / foncier réel
(TAX_SPEC 4, 8), plus the SCI IS global override. A category with no property is never varied.
Micro ceilings that are exceeded fall back to réel inside the tax engine; the warning is returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.application.run_context import RunContext
from src.application.services.simulation import simulate_long_term_strategy
from src.domain.models.archetype import is_furnished

CF_COL = "Cash-Flow Net d'Impôt"
TAX_COL = "Impôt Dû"
YEAR_COL = "Année"

_FURNISHED_LABELS = {"lmnp": "LMNP réel", "microbic": "Micro-BIC"}
_UNFURNISHED_LABELS = {"reel": "Foncier réel", "micro": "Micro-foncier"}
SCI_LABEL = "SCI à l'IS"


@dataclass(frozen=True)
class RegimeKey:
    """One regime combination. ``furnished`` / ``foncier`` are None when the category is absent."""

    furnished: str | None
    foncier: str | None
    sci_is: bool = False

    @property
    def label(self) -> str:
        if self.sci_is:
            return SCI_LABEL
        parts = []
        if self.furnished is not None:
            parts.append(_FURNISHED_LABELS[self.furnished])
        if self.foncier is not None:
            parts.append(_UNFURNISHED_LABELS[self.foncier])
        return " + ".join(parts)


@dataclass
class RegimeResult:
    key: RegimeKey
    label: str
    is_user_choice: bool
    years: list[int]
    tax: list[float]
    cash_flow: list[float]
    cumulative_cf: list[float]
    tri_annuel: float | None
    enrichissement_net: float
    exit_tax: float
    warnings: list[str] = field(default_factory=list)
    df: pd.DataFrame | None = None
    bilan: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegimeComparison:
    results: list[RegimeResult]
    best_by_wealth: RegimeResult
    best_by_irr: RegimeResult | None
    user: RegimeResult
    break_even_year: int | None  # first year the best regime's cumulative CF exceeds the user's; None if never

    @property
    def wealth_gain_vs_user(self) -> float:
        return self.best_by_wealth.enrichissement_net - self.user.enrichissement_net


def _categories(strategy: dict[str, Any]) -> tuple[bool, bool]:
    """(has furnished, has unfurnished) properties."""
    flags = [is_furnished(p) for p in strategy.get("details", [])]
    return any(flags), (not all(flags)) if flags else False


def user_regime_key(strategy: dict[str, Any], ctx: RunContext) -> RegimeKey:
    """The regime the user's ``ctx`` selects, restricted to the categories present in ``strategy``."""
    has_f, has_u = _categories(strategy)
    fiscal = str(ctx.regime_fiscal).strip().lower()
    if ctx.sci_is or fiscal == "sci_is":
        return RegimeKey(None, None, True)
    return RegimeKey(
        ("microbic" if fiscal == "microbic" else "lmnp") if has_f else None,
        ("micro" if str(ctx.regime_foncier).strip().lower() == "micro" else "reel") if has_u else None,
    )


def eligible_regimes(strategy: dict[str, Any]) -> list[RegimeKey]:
    """Every relevant regime combination for the properties of ``strategy``."""
    has_f, has_u = _categories(strategy)
    f_opts: list[str | None] = ["lmnp", "microbic"] if has_f else [None]
    u_opts: list[str | None] = ["reel", "micro"] if has_u else [None]
    keys = [RegimeKey(f, u) for f in f_opts for u in u_opts]
    keys.append(RegimeKey(None, None, True))
    return keys


def _run(strategy: dict[str, Any], ctx: RunContext, key: RegimeKey, user_key: RegimeKey) -> RegimeResult:
    update: dict[str, Any] = {"sci_is": key.sci_is}
    if not key.sci_is:
        if key.furnished is not None:
            update["regime_fiscal"] = key.furnished
        if key.foncier is not None:
            update["regime_foncier"] = key.foncier
    variant = ctx.model_copy(update=update)
    df, bilan = simulate_long_term_strategy(
        strategy, ctx.horizon_ans, variant.market_hypotheses, variant.to_tax_params()
    )
    cf = [float(v) for v in df[CF_COL]]
    cum: list[float] = []
    total = 0.0
    for v in cf:
        total += v
        cum.append(total)
    return RegimeResult(
        key=key,
        label=key.label,
        is_user_choice=key == user_key,
        years=[int(y) for y in df[YEAR_COL]],
        tax=[float(v) for v in df[TAX_COL]],
        cash_flow=cf,
        cumulative_cf=cum,
        tri_annuel=bilan.get("tri_annuel"),
        enrichissement_net=float(bilan.get("enrichissement_net", 0.0)),
        exit_tax=float(bilan.get("exit_tax", {}).get("total", 0.0)),
        warnings=list(bilan.get("tax_warnings", [])),
        df=df,
        bilan=bilan,
    )


def find_break_even_year(best: RegimeResult, user: RegimeResult) -> int | None:
    """First year where ``best`` cumulative after-tax CF is strictly above ``user``'s (None if never)."""
    if best.key == user.key:
        return None
    for year, b, u in zip(best.years, best.cumulative_cf, user.cumulative_cf, strict=False):
        if b > u + 1e-9:
            return year
    return None


def compare_regimes(strategy: dict[str, Any], ctx: RunContext) -> RegimeComparison:
    """Simulate ``strategy`` under every eligible regime and rank them."""
    user_key = user_regime_key(strategy, ctx)
    results = [_run(strategy, ctx, k, user_key) for k in eligible_regimes(strategy)]
    user = next(r for r in results if r.is_user_choice)
    best = max(results, key=lambda r: r.enrichissement_net)
    with_irr = [r for r in results if r.tri_annuel is not None and r.tri_annuel == r.tri_annuel]
    best_irr = max(with_irr, key=lambda r: float(r.tri_annuel or 0.0)) if with_irr else None
    return RegimeComparison(results, best, best_irr, user, find_break_even_year(best, user))
