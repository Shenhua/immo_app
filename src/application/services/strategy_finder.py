"""Strategy finder service.

Modular strategy finding service extracted from strategy_finder.py.
Provides combination generation, scoring, and ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.defaults import (
    BASE_WEIGHTS,
    DEFAULT_PRESET,
    HCSF,
    MARKET,
    OPERATING,
    PARETO_MAX_RANK,
    REALISM,
    SEARCH,
    TAX,
    PresetId,
)
from src.core.glossary import cf_penalty
from src.core.logging import get_logger
from src.domain.hcsf import hcsf_check
from src.domain.models.brick import InvestmentBrick

from .allocator import PortfolioAllocator
from .diagnostics import NearMiss, SearchDiagnostics
from .optimizer import PortfolioSearch, SearchProblem, check_invariants, tax_bounds
from .scoring import assign_pareto_ranks, score_components

log = get_logger(__name__)


def strategy_key(strategy: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    """Explicit tie-breaker (T31): the sorted (property name, loan months) pairs of a strategy."""
    out = []
    for d in strategy.get("details", []):
        if isinstance(d, dict):
            out.append((str(d.get("nom", d.get("nom_bien", ""))), int(d.get("duree_credit_mois", 0) or 0)))
        else:
            out.append((str(d.nom), int(d.duree_credit_mois)))
    return tuple(sorted(out))


def _default_market() -> dict[str, float]:
    return {
        "appreciation_bien_pct": MARKET.appreciation_bien_pct,
        "revalo_loyer_pct": MARKET.revalo_loyer_pct,
        "inflation_charges_pct": MARKET.inflation_charges_pct,
        "tf_growth_pct": REALISM.tf_growth_pct,
    }


# Legacy flat keys accepted by ``from_dict`` when no ``tax_params`` dict is given (hand-built dicts, tests).
_LEGACY_TAX_KEYS = (
    "tmi_pct",
    "regime_fiscal",
    "regime_foncier",
    "sci_is",
    "apply_ira",
    "ira_cap_pct",
    "cfe_par_bien_ann",
    "frais_vente_pct",
    "ratio_terrain_pct",
    "nb_owners",
    "dpe_policy",
    "dpe_cout_m2_par_classe",
)


@dataclass
class EvaluationParams:
    """Parameters for strategy evaluation.

    The engine parameters (``hypotheses_marche`` and ``tax_params``) are the RunContext-derived dicts kept
    verbatim, so the search builds its engine from exactly what ``RunContext.build_engine()`` uses and a new
    engine field cannot be forgotten here (T35). Missing keys fall back to the defaults inside ``build_engine``.
    """

    duree_simulation_ans: int = SEARCH.horizon_ans
    hypotheses_marche: dict[str, float] = field(default_factory=_default_market)
    tax_params: dict[str, Any] = field(default_factory=dict)
    finance_weights_override: dict[str, float] | None = None
    finance_preset_name: str = DEFAULT_PRESET.label
    # T60: {"revenu_net_mensuel", "charges_mensuelles", "filtrer"}; empty = no HCSF check (direct callers, tests).
    hcsf: dict[str, Any] = field(default_factory=dict)

    def to_tax_params(self) -> dict[str, Any]:
        """Tax / exit / operating parameters for ``build_engine``."""
        return self.tax_params

    def build_engine(self) -> Any:
        """The search engine: same builder and same dicts as ``RunContext.build_engine()``."""
        from .simulation import build_engine

        return build_engine(self.hypotheses_marche, self.tax_params)

    # Read-only views used by the search and the tests (defaults mirror ``build_engine``).
    @property
    def regime_fiscal(self) -> str:
        return str(self.tax_params.get("regime_fiscal", TAX.regime_fiscal))

    @property
    def regime_foncier(self) -> str:
        return str(self.tax_params.get("regime_foncier", TAX.regime_foncier))

    @property
    def sci_is(self) -> bool:
        return bool(self.tax_params.get("sci_is", TAX.sci_is))

    @property
    def tmi_pct(self) -> float:
        return float(self.tax_params.get("tmi_pct", TAX.tmi_pct))

    @property
    def frais_vente_pct(self) -> float:
        return float(self.tax_params.get("frais_vente_pct", TAX.frais_vente_pct))

    @property
    def cfe_par_bien_ann(self) -> float:
        return float(self.tax_params.get("cfe_par_bien_ann", OPERATING.cfe_par_bien_ann))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvaluationParams:
        tax = d.get("tax_params")
        if tax is None:
            tax = {k: d[k] for k in _LEGACY_TAX_KEYS if k in d}
        return cls(
            duree_simulation_ans=d.get("duree_simulation_ans", SEARCH.horizon_ans),
            hypotheses_marche=d.get("hypotheses_marche") or _default_market(),
            tax_params=dict(tax),
            finance_weights_override=d.get("finance_weights_override"),
            finance_preset_name=d.get("finance_preset_name", DEFAULT_PRESET.label),
            hcsf=dict(d.get("hcsf") or {}),
        )


class StrategyScorer:
    """Scores strategies based on financial and qualitative metrics."""

    def __init__(
        self,
        qualite_weight: float = SEARCH.qualite_weight,
        weights: dict[str, float] | None = None,
        *,
        mode_cf: str,
    ):
        # No default for mode_cf: the caller (RunContext) decides, so the scorer and the search agree.
        self.mode_cf = mode_cf
        self.qualite_weight = max(0.0, min(1.0, qualite_weight))
        self.weights = self._normalize_weights(weights or BASE_WEIGHTS)

    @staticmethod
    def _normalize_weights(w: dict[str, float]) -> dict[str, float]:
        """Normalize weights to sum to 1."""
        if not isinstance(w, dict):
            return BASE_WEIGHTS.copy()
        s = sum(max(0.0, float(v)) for v in w.values() if v is not None)
        if s <= 0:
            return BASE_WEIGHTS.copy()
        return {k: max(0.0, float(v)) / s for k, v in w.items()}

    def compute_finance_score(self, strategy: dict[str, Any]) -> float:
        """Compute weighted finance score for a single strategy."""
        w = self.weights
        score: float = (
            w.get("enrich_net", 0.0) * float(strategy.get("enrich_norm", 0.0))
            + w.get("irr", 0.0) * float(strategy.get("tri_norm", 0.0))
            + w.get("cap_eff", 0.0) * float(strategy.get("cap_eff_norm", 0.0))
            + w.get("dscr", 0.0) * float(strategy.get("dscr_norm", 0.0))
            + w.get("cf_proximity", 0.0) * float(strategy.get("cf_proximity", 0.0))
        )
        return score

    def compute_balanced_score(self, strategy: dict[str, Any]) -> float:
        """Combine finance and quality scores."""
        fin = float(strategy.get("finance_score", 0.0))
        qual = float(strategy.get("qual_score", 50.0)) / 100.0
        return (1.0 - self.qualite_weight) * fin + self.qualite_weight * qual

    def score_strategies(
        self, strategies: list[dict[str, Any]], cash_flow_cible: float, horizon: int = SEARCH.horizon_ans
    ) -> None:
        """Add absolute scores to every strategy (mutates in place).

        Each component is scored on the fixed benchmarks of ``defaults.py`` (see ``scoring.py``), so a strategy's
        score does not depend on the other strategies. The CF proximity uses the stabilised after-tax CF
        (``cf_monthly_stab``) and the mode-aware ``glossary.cf_penalty``.
        """
        for s in strategies:
            cf = s.get("cf_monthly_stab", s.get("cash_flow_final", 0.0))
            s.update(
                score_components(
                    irr_pct=s.get("tri_annuel", 0.0),
                    enrich_net=s.get("enrich_net", 0.0),
                    apport=s.get("apport_total", 0.0),
                    dscr_stab=s.get("dscr_stab", 0.0),
                    cf_gap=cf_penalty(float(cf), cash_flow_cible, self.mode_cf),
                    horizon=horizon,
                )
            )
            s["finance_score"] = self.compute_finance_score(s)
            s["balanced_score"] = self.compute_balanced_score(s)


class StrategyFinder:
    """Service for finding optimal investment strategies.

    Orchestrates combination generation, allocation, simulation, and scoring.
    """

    def __init__(
        self,
        bricks: list[InvestmentBrick],
        apport_disponible: float,
        cash_flow_cible: float,
        tolerance: float = 100.0,
        qualite_weight: float = SEARCH.qualite_weight,
        mode_cf: str = "min",
        max_properties: int = 3,
    ):
        self.bricks = bricks
        self.apport_disponible = apport_disponible
        self.cash_flow_cible = cash_flow_cible
        self.tolerance = tolerance
        self.qualite_weight = qualite_weight
        self.mode_cf = mode_cf
        self.max_properties = max_properties
        # Why portfolios were rejected in the last find_strategies call (T33): counts, near misses, suggestions.
        self.last_diagnostics: SearchDiagnostics = SearchDiagnostics()
        # Solver used by the last find_strategies call: "EXHAUSTIVE" (every duration set simulated),
        # "SUBSET_PROXY" (best duration sets per subset by proxy, see optimizer.py) or "" (not run yet).
        self.solver_mode: str = ""
        # Evaluation workers: None = automatic (process pool for large searches), 1 = serial.
        self.n_workers: int | None = None
        self.scorer = StrategyScorer(qualite_weight=qualite_weight, mode_cf=mode_cf)

    def find_strategies(
        self,
        eval_params: dict[str, Any] | None = None,
        horizon_years: int = 25,
        top_n: int = 10,
        use_full_capital_override: bool = False,
        progress_callback: Any = None,
    ) -> list[dict[str, Any]]:
        """Find top strategies matching criteria.

        Orchestrates the full search process:
        1. Generate combinations
        2. Allocate capital
        3. Simulate financial performance
        4. Score and rank

        Args:
            progress_callback: Optional callback function(SearchProgress) for UI updates
        """
        # 1. Setup
        ep = EvaluationParams.from_dict(eval_params or {})

        # Update scorer weights from eval_params
        if ep.finance_weights_override:
            self.scorer.weights = self.scorer._normalize_weights(ep.finance_weights_override)

        engine = ep.build_engine()

        allocator = PortfolioAllocator(self.mode_cf, market=ep.hypotheses_marche)

        # Validate bricks before proceeding (fail-fast)
        from .brick_factory import validate_bricks

        validation_warnings = validate_bricks(self.bricks)
        if validation_warnings:
            log.warning(
                "brick_validation_warnings",
                count=len(validation_warnings),
                first_warning=validation_warnings[0] if validation_warnings else None,
            )

        log.info("strategy_search_started", brick_count=len(self.bricks), apport=self.apport_disponible)

        # Report Phase 1: Brick generation (already done in __init__)
        from src.application.progress import SearchPhase, SearchProgress

        if progress_callback:
            progress_callback(
                SearchProgress(
                    phase=SearchPhase.BRICK_GENERATION,
                    phase_index=1,
                    items_processed=len(self.bricks),
                    items_total=len(self.bricks),
                    message=f"{len(self.bricks)} briques créées",
                )
            )

        # 2. Subset search (T31): deterministic, every result within budget with unique properties.
        refund_max, tax_rate_max = tax_bounds(ep.to_tax_params())
        problem = SearchProblem(
            budget=float(self.apport_disponible),
            target_cf=float(self.cash_flow_cible),
            tolerance=float(self.tolerance),
            horizon=int(horizon_years),
            max_props=int(self.max_properties),
            mode_cf=self.mode_cf,
            use_full_capital=bool(use_full_capital_override),
            refund_max=refund_max,
            tax_rate_max=tax_rate_max,
            appreciation_pct=float(ep.hypotheses_marche.get("appreciation_bien_pct", MARKET.appreciation_bien_pct)),
            frais_vente_pct=float(ep.frais_vente_pct),
        )
        search = PortfolioSearch(
            allocator=allocator,
            simulator=engine,
            scorer=self.scorer,
            engine_spec=(ep.hypotheses_marche, ep.to_tax_params()),
            n_workers=self.n_workers,
        )
        strategies = search.solve(
            self.bricks, problem, market=ep.hypotheses_marche, progress_callback=progress_callback
        )
        self.solver_mode = search.solver_mode
        self.last_diagnostics = search.diagnostics
        log.info("search_solver", mode=self.solver_mode, **search.stats)

        strategies = self._apply_hcsf(strategies, ep.hcsf)
        log.info("optimization_finished", count=len(strategies))

        # Guard against empty results from optimizer (the UI explains why from ``last_diagnostics``)
        if not strategies:
            log.warning(
                "no_strategies_found",
                reason="optimizer_returned_empty",
                brick_count=len(self.bricks),
                budget=self.apport_disponible,
            )
            return []

        # 4. Score & Rank (Post-Optimization)
        # Report Phase 4: Scoring
        if progress_callback:
            progress_callback(
                SearchProgress(
                    phase=SearchPhase.SCORING,
                    phase_index=4,
                    items_total=len(strategies),
                    message="Scoring et classement...",
                )
            )

        # We re-run relative scoring on the "Elite" set to populate UI-friendly normalized fields (A-score, etc.)
        self.scorer.score_strategies(strategies, self.cash_flow_cible, horizon_years)

        # Dedupe
        # Report Phase 5: Deduplication
        if progress_callback:
            progress_callback(
                SearchProgress(
                    phase=SearchPhase.DEDUPLICATION,
                    phase_index=5,
                    items_total=len(strategies),
                    message="Déduplication et filtrage...",
                )
            )

        strategies = self.dedupe_strategies(strategies, top_n=top_n)

        # Pareto rank (1 = front) on every strategy; a marker and a tie-breaker, never a filter.
        assign_pareto_ranks(strategies, self.cash_flow_cible, self.mode_cf)

        top_strategies = self.rank_strategies(strategies, ep.finance_preset_name, top_n)
        check_invariants(top_strategies, self.apport_disponible, self.max_properties)
        log.info("strategy_search_completed", total_evaluated=len(strategies), top_kept=len(top_strategies))

        return top_strategies

    def _apply_hcsf(self, strategies: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
        """T60: attach ``taux_endettement`` / ``hcsf_ok`` to every strategy; drop the unbankable ones if asked.

        Runs on every duration assignment, before the per-subset dedupe, so a subset survives when any of its
        duration sets is bankable. A subset with none is recorded in the diagnostics as an "hcsf" rejection.
        """
        if not cfg:
            return strategies
        income = float(cfg.get("revenu_net_mensuel", HCSF.revenu_net_mensuel))
        debts = float(cfg.get("charges_mensuelles", HCSF.charges_mensuelles))
        keep_only_ok = bool(cfg.get("filtrer", HCSF.filtrer))
        diag = self.last_diagnostics
        diag.revenu_net = income
        out: list[dict[str, Any]] = []
        best_failed: dict[tuple[str, ...], tuple[Any, ...]] = {}
        passed: set[tuple[str, ...]] = set()
        for s in strategies:
            res = hcsf_check(s.get("details", []), income, debts)
            s["taux_endettement"] = res.taux_endettement
            s["hcsf_ok"] = res.hcsf_ok
            names = tuple(sorted(str(d.get("nom", d.get("nom_bien", ""))) for d in s.get("details", [])))
            if res.hcsf_ok or not keep_only_ok:
                out.append(s)
                passed.add(names)
            elif names not in best_failed or res.taux_endettement < best_failed[names][0]:
                best_failed[names] = (res.taux_endettement, res.revenu_requis)
        for names, (ratio, required) in best_failed.items():
            if names in passed:
                continue
            diag.record(NearMiss("hcsf", names, max(0.0, ratio - HCSF.taux_max_pct), "pt", cf=ratio))
            diag.add_hcsf_gap(names, max(0.0, required - income))
        return out

    def _calculate_qualitative_score(self, strategy: dict[str, Any]) -> float:
        """Calculate qualitative score (internal helper)."""
        # Import only as needed to avoid top-level circular dependency if any
        # But ideally we should move scoring logic here fully.
        # For now, using simple weighted average of bricks

        # Details are now list[InvestmentBrick]
        details = strategy.get("details", [])
        total_price = sum(b.prix_achat_bien for b in details)
        if total_price <= 0:
            return 50.0

        score_sum = 0.0
        for b in details:
            qs = b.qual_score_bien or 50.0
            price = b.prix_achat_bien
            score_sum += qs * price

        return float(score_sum / total_price)

    def dedupe_strategies(self, strategies: list[dict[str, Any]], top_n: int = 10) -> list[dict[str, Any]]:
        """Remove near-duplicate strategies.

        Groups by property signature and keeps best per group.
        Labels variations explicitly for UI clarity.
        """
        # Group by signature (property names + count)
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for s in strategies:
            details = s.get("details", [])
            # Details are InvestmentBrick objects or dicts
            sig = tuple(sorted((d.nom if hasattr(d, "nom") else d.get("nom_bien", d.get("nom", ""))) for d in details))
            if sig not in grouped:
                grouped[sig] = []
            grouped[sig].append(s)

        # Sort each group by score descending
        for sig in grouped:
            grouped[sig].sort(
                key=lambda x: (
                    -x.get("balanced_score", -float("inf")),
                    strategy_key(x),  # explicit tie-breaker
                )
            )

        # Selection logic: Strictly keep only the best variation per signature
        # This forces the result list to be composed of DISTINCT strategies.
        # If we want 50 results, we must find 50 distinct property combinations.
        final_list = []

        for _sig, items in grouped.items():
            # Keep only the absolute best (index 0)
            best_item = items[0]
            final_list.append(best_item)

        # Re-sort final list by score
        final_list.sort(key=lambda x: (-x.get("balanced_score", 0), strategy_key(x)))

        return final_list

    def rank_strategies(
        self,
        strategies: list[dict[str, Any]],
        preset_name: PresetId | str = DEFAULT_PRESET,
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """Sort strategies by preset priority.

        The ranking respects the user's profile selection:
        - Primary: Use the weighted finance_score (incorporates user weights)
        - Secondary: Preset-specific tiebreaker metric
        - Tertiary: CF proximity (closer to target is better)
        - Then ``pareto_rank`` (1 = on the Pareto front), then the explicit signature tie-breaker.
        """
        preset = PresetId.parse(preset_name)

        def sort_key(x: dict[str, Any]) -> tuple[Any, ...]:
            pareto = -x.get("pareto_rank", PARETO_MAX_RANK + 1)  # last key before the signature

            # CF proximity as final tiebreaker (higher is better)
            cf_prox = x.get("cf_proximity", 0)
            finance = x.get("finance_score", 0)
            balanced = x.get("balanced_score", 0)

            if preset is PresetId.SECURITY:
                # Safety profile: prioritize DSCR, then finance_score
                return (x.get("dscr_norm", 0), finance, cf_prox, pareto)

            if preset is PresetId.IRR:
                # IRR profile: prioritize TRI, then finance_score
                return (x.get("tri_norm", 0), finance, cf_prox, pareto)

            if preset is PresetId.CASHFLOW:
                # Cash-flow profile: prioritize CF proximity, then DSCR for safety
                return (cf_prox, x.get("dscr_norm", 0), finance, pareto)

            if preset is PresetId.WEALTH:
                # Wealth-building profile: prioritize enrichment, then IRR
                return (x.get("enrich_norm", 0), x.get("tri_norm", 0), balanced, pareto)

            # Default "Équilibré": use balanced_score which combines finance + quality
            return (balanced, finance, cf_prox, pareto)

        # Explicit tie-breaker: pre-sort by signature; the stable reverse sort keeps that order among equal keys.
        by_signature = sorted(strategies, key=strategy_key)
        sorted_strategies = sorted(by_signature, key=sort_key, reverse=True)
        return sorted_strategies[:top_n]
