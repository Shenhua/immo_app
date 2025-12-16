"""Strategy finder service.

Modular strategy finding service extracted from strategy_finder.py.
Provides combination generation, scoring, and ranking.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from src.core.logging import get_logger

log = get_logger(__name__)


# Base financial scoring weights
BASE_WEIGHTS = {
    "enrich_net": 0.30,
    "irr": 0.25,
    "cf_proximity": 0.20,
    "dscr": 0.15,
    "cap_eff": 0.10,
}


@dataclass
class EvaluationParams:
    """Parameters for strategy evaluation."""

    duree_simulation_ans: int = 25
    hypotheses_marche: dict[str, float] = field(default_factory=lambda: {
        "appreciation_bien_pct": 2.0,
        "revalo_loyer_pct": 1.5,
        "inflation_charges_pct": 2.0,
    })
    regime_fiscal: str = "lmnp"
    tmi_pct: float = 30.0
    frais_vente_pct: float = 6.0
    cfe_par_bien_ann: float = 500.0
    apply_ira: bool = True
    ira_cap_pct: float = 3.0
    finance_weights_override: dict[str, float] | None = None
    finance_preset_name: str = "Équilibré (défaut)"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvaluationParams:
        return cls(
            duree_simulation_ans=d.get("duree_simulation_ans", 25),
            hypotheses_marche=d.get("hypotheses_marche", cls.__dataclass_fields__["hypotheses_marche"].default_factory()),
            regime_fiscal=d.get("regime_fiscal", "lmnp"),
            tmi_pct=d.get("tmi_pct", 30.0),
            frais_vente_pct=d.get("frais_vente_pct", 6.0),
            cfe_par_bien_ann=d.get("cfe_par_bien_ann", 500.0),
            apply_ira=d.get("apply_ira", True),
            ira_cap_pct=d.get("ira_cap_pct", 3.0),
            finance_weights_override=d.get("finance_weights_override"),
            finance_preset_name=d.get("finance_preset_name", "Équilibré (défaut)"),
        )


class StrategyScorer:
    """Scores strategies based on financial and qualitative metrics."""

    def __init__(self, qualite_weight: float = 0.25, weights: dict[str, float] | None = None):
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

    @staticmethod
    def minmax_normalize(values: list[float], lo: float | None = None, hi: float | None = None) -> list[float]:
        """Min-max normalize a list of values to 0-1 range."""
        finite_vals = [v for v in values if isfinite(v)]
        if not finite_vals:
            return [0.0] * len(values)

        if lo is None:
            lo = min(finite_vals)
        if hi is None:
            hi = max(finite_vals)

        if hi <= lo:
            return [0.5] * len(values)

        return [
            0.0 if not isfinite(v) else max(0.0, min(1.0, (v - lo) / (hi - lo)))
            for v in values
        ]

    def compute_finance_score(self, strategy: dict[str, Any]) -> float:
        """Compute weighted finance score for a single strategy."""
        w = self.weights
        return (
            w.get("enrich_net", 0.0) * strategy.get("enrich_norm", 0.0) +
            w.get("irr", 0.0) * strategy.get("tri_norm", 0.0) +
            w.get("cap_eff", 0.0) * strategy.get("cap_eff_norm", 0.0) +
            w.get("dscr", 0.0) * strategy.get("dscr_norm", 0.0) +
            w.get("cf_proximity", 0.0) * strategy.get("cf_proximity", 0.0)
        )

    def compute_balanced_score(self, strategy: dict[str, Any]) -> float:
        """Combine finance and quality scores."""
        fin = strategy.get("finance_score", 0.0)
        qual = strategy.get("qual_score", 50.0) / 100.0
        return (1.0 - self.qualite_weight) * fin + self.qualite_weight * qual

    def score_strategies(self, strategies: list[dict[str, Any]], cash_flow_cible: float) -> None:
        """Add normalized scores to all strategies (mutates in place)."""
        if not strategies:
            return

        # Extract raw metrics
        enrich = [s.get("enrich_net", 0.0) for s in strategies]
        irr = [s.get("tri_annuel", 0.0) for s in strategies]
        cap_eff = [s.get("cap_eff", 0.0) for s in strategies]
        dscr = [1.5 if s.get("dscr_y1") is None else max(0.0, float(s.get("dscr_y1", 0.0))) for s in strategies]
        cf_dist = [abs(s.get("cash_flow_final", 0.0) - cash_flow_cible) for s in strategies]
        # CF proximity: closer to target = higher score (use 500€ as reasonable max distance)
        max_cf_dist = 500.0
        cf_prox = [max(0.0, 1.0 - (d / max_cf_dist)) for d in cf_dist]

        # Normalize
        enrich_n = self.minmax_normalize(enrich)
        irr_n = self.minmax_normalize(irr)
        cap_n = self.minmax_normalize([min(v, 10.0) for v in cap_eff], 0.0, 10.0)
        dscr_n = self.minmax_normalize([min(v, 2.5) for v in dscr], 0.0, 2.5)

        # Apply to strategies
        for s, en, ir, ca, ds, cf in zip(strategies, enrich_n, irr_n, cap_n, dscr_n, cf_prox):
            s["enrich_norm"] = en
            s["tri_norm"] = ir
            s["cap_eff_norm"] = ca
            s["dscr_norm"] = ds
            s["cf_proximity"] = cf
            s["finance_score"] = self.compute_finance_score(s)
            s["balanced_score"] = self.compute_balanced_score(s)


class CombinationGenerator:
    """Generates valid property combinations within budget."""

    def __init__(self, max_properties: int = 3):
        self.max_properties = max_properties

    def generate(
        self,
        bricks: list[dict[str, Any]],
        apport_disponible: float,
    ) -> list[tuple[dict[str, Any], ...]]:
        """Generate all valid combinations of bricks.

        Filters:
        - No duplicate properties (same nom_bien)
        - Total apport_min <= apport_disponible
        
        Optimizations:
        - Pre-filter bricks that individually exceed budget
        - Sort by cost for early rejection
        - K-level pruning (if cheapest K properties > budget, skip)
        """
        # P2.1: Pre-filter unaffordable bricks
        affordable_bricks = [b for b in bricks if b.get("apport_min", 0.0) <= apport_disponible]
        
        if len(affordable_bricks) < len(bricks):
            log.debug("pruning_unaffordable_bricks", 
                     original=len(bricks), 
                     remaining=len(affordable_bricks))
        
        if not affordable_bricks:
            log.info("no_affordable_bricks", budget=apport_disponible)
            return []
        
        # Sort by Gross Yield descending (Performance) rather than Cost (Apport)
        # This ensures we test high-potential combinations first before hitting MAX_TOTAL limit.
        def _yield_score(b):
            rent = b.get("loyer_mensuel_initial", 0.0) * 12
            cost = b.get("cout_total", 1.0)
            return rent / max(1.0, cost)

        sorted_bricks = sorted(affordable_bricks, key=_yield_score, reverse=True)
        
        combos = []
        
        # For pruning: Get unique properties by nom_bien, keeping cheapest variant per property
        unique_props = {}
        for b in sorted_bricks:
            nom = b.get("nom_bien", "")
            cost = b.get("apport_min", 0.0)
            if nom not in unique_props or cost < unique_props[nom]:
                unique_props[nom] = cost
        
        # Sorted list of min costs per unique property
        sorted_unique_costs = sorted(unique_props.values())
        
        check_count = 0
        skipped_early = 0
        
        for k in range(1, self.max_properties + 1):
            # Optimization: Smart Pruning
            # If the cheapest K UNIQUE properties cost more than available apport, 
            # then ANY combination of K properties will fail. Stop searching bigger Ks.
            if k <= len(sorted_unique_costs):
                min_cost_k = sum(sorted_unique_costs[:k])
            else:
                # Can't even form k unique properties
                log.info("pruning_not_enough_properties", k=k, available=len(sorted_unique_costs))
                break
                
            if min_cost_k > apport_disponible:
                log.info("pruning_budget", k=k, min_cost=min_cost_k, budget=apport_disponible)
                break
                
            for combo in itertools.combinations(sorted_bricks, k):
                check_count += 1
                
                # Check unique properties
                noms = {c.get("nom_bien") for c in combo if c.get("nom_bien")}
                if len(noms) != len(combo):
                    continue

                # Check budget
                apport_min = sum(c.get("apport_min", 0.0) for c in combo)
                
                if apport_min > apport_disponible:
                    skipped_early += 1
                    continue

                combos.append(combo)
        
        # Log summary
        log.info("combos_generated", 
                count=len(combos), 
                max_props=self.max_properties, 
                budget=apport_disponible, 
                checks=check_count,
                skipped=skipped_early)

        return combos


class StrategyFinder:
    """Service for finding optimal investment strategies.

    Orchestrates combination generation, allocation, simulation, and scoring.
    """

    def __init__(
        self,
        bricks: list[dict[str, Any]],
        apport_disponible: float,
        cash_flow_cible: float,
        tolerance: float = 100.0,
        qualite_weight: float = 0.25,
        mode_cf: str = "target",
        max_properties: int = 3,
    ):
        self.bricks = bricks
        self.apport_disponible = apport_disponible
        self.cash_flow_cible = cash_flow_cible
        self.tolerance = tolerance
        self.qualite_weight = qualite_weight
        self.mode_cf = mode_cf
        self.max_properties = max_properties

        self.combo_generator = CombinationGenerator(max_properties=max_properties)
        self.scorer = StrategyScorer(qualite_weight=qualite_weight)

    def find_strategies(
        self,
        eval_params: dict[str, Any] | None = None,
        horizon_years: int = 25,
        top_n: int = 10,
        use_full_capital_override: bool = False,
    ) -> list[dict[str, Any]]:
        """Find top strategies matching criteria.

        Orchestrates the full search process:
        1. Generate combinations
        2. Allocate capital
        3. Simulate financial performance
        4. Score and rank
        """
        # Dependencies
        from src.core.financial import generate_amortization_schedule
        from src.core.simulation import IRACalculator, MarketHypotheses, SimulationEngine, TaxParams
        from src.services.allocator import PortfolioAllocator

        # 1. Setup
        ep = EvaluationParams.from_dict(eval_params or {})

        # Update scorer weights from eval_params
        if ep.finance_weights_override:
            self.scorer.weights = self.scorer._normalize_weights(ep.finance_weights_override)

        market = MarketHypotheses(
            appreciation_bien_pct=ep.hypotheses_marche["appreciation_bien_pct"],
            revalo_loyer_pct=ep.hypotheses_marche["revalo_loyer_pct"],
            inflation_charges_pct=ep.hypotheses_marche["inflation_charges_pct"],
        )
        tax = TaxParams(
            tmi_pct=ep.tmi_pct,
            regime_fiscal=ep.regime_fiscal,
        )
        ira = IRACalculator(
            apply_ira=ep.apply_ira,
            ira_cap_pct=ep.ira_cap_pct,
        )
        engine = SimulationEngine(
            market=market,
            tax=tax,
            ira=ira,
            cfe_par_bien_ann=ep.cfe_par_bien_ann,
            frais_vente_pct=ep.frais_vente_pct,
        )

        allocator = PortfolioAllocator(self.mode_cf)

        # Validate bricks before proceeding (fail-fast)
        from src.services.brick_factory import validate_bricks
        validation_warnings = validate_bricks(self.bricks)
        if validation_warnings:
            log.warning("brick_validation_warnings", count=len(validation_warnings), 
                        first_warning=validation_warnings[0] if validation_warnings else None)

        log.info("strategy_search_started",
                 brick_count=len(self.bricks),
                 apport=self.apport_disponible)

        # 2. Genetic Algorithm Optimization
        # Replaces legacy Combinatorial Search + ThreadPoolExecutor
        # Implements Expert Recommendation (Phase 2)
        from src.services.optimizer import GeneticOptimizer
        
        # Use aggressive population settings to ensure we find "Needle in Haystack"
        optimizer = GeneticOptimizer(
            population_size=100,
            generations=30,
            elite_size=10,
            allocator=allocator,
            simulator=engine,
            scorer=self.scorer
        )
        
        log.info("ga_optimization_started", 
                 bricks=len(self.bricks), 
                 pop_size=optimizer.pop_size, 
                 generations=optimizer.generations)
                 
        strategies = optimizer.evolve(
            self.bricks,
            self.apport_disponible,
            self.cash_flow_cible,
            self.tolerance,
            horizon=horizon_years
        )
        log.info("ga_optimization_finished", count=len(strategies))

        # 4. Score & Rank (Post-Optimization)
        # We re-run relative scoring on the "Elite" set to populate UI-friendly normalized fields (A-score, etc.)
        self.scorer.score_strategies(strategies, self.cash_flow_cible)

        # Filter out failed allocations
        strategies = [s for s in strategies if s.get("allocation_ok", False)]
        
        # Dedupe
        strategies = self.dedupe_strategies(strategies)

        top_strategies = self.rank_strategies(strategies, ep.finance_preset_name, top_n)
        log.info("strategy_search_completed",
                 total_evaluated=len(strategies),
                 top_kept=len(top_strategies))

        return top_strategies

    def _calculate_qualitative_score(self, strategy: dict[str, Any]) -> float:
        """Calculate qualitative score (internal helper)."""
        # Import only as needed to avoid top-level circular dependency if any
        # But ideally we should move scoring logic here fully.
        # For now, using simple weighted average of bricks

        total_price = sum(b.get("prix_achat_bien", 0) for b in strategy.get("details", []))
        if total_price <= 0:
            return 50.0

        score_sum = 0.0
        for b in strategy.get("details", []):
            qs = b.get("qual_score_bien", 50.0)
            price = b.get("prix_achat_bien", 0)
            score_sum += qs * price

        return score_sum / total_price


    def dedupe_strategies(self, strategies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove near-duplicate strategies.
        Aggressive deduplication with fallback:
        1. Group by 'Signature' (Set of property names).
        2. Keep the best strategies per signature.
        3. If meaningful variety is low, allow multiple variations per signature.
        """
        # Group by signature
        grouped = {}
        for s in strategies:
            details = s.get("details", [])
            sig = tuple(sorted(d.get("nom_bien", "") for d in details))
            if sig not in grouped:
                grouped[sig] = []
            grouped[sig].append(s)

        # Sort each group by score descending
        for sig in grouped:
            grouped[sig].sort(key=lambda x: x.get("balanced_score", -float("inf")), reverse=True)

        # Selection logic
        final_list = []
        
        # If we have very few distinct property sets (e.g. < 3), show more variations per set
        variations_per_sig = 1
        if len(grouped) < 3:
            variations_per_sig = 3

        for sig, items in grouped.items():
            # Keep top N items
            final_list.extend(items[:variations_per_sig])

        return final_list

    def rank_strategies(
        self,
        strategies: list[dict[str, Any]],
        preset_name: str = "Équilibré (défaut)",
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """Sort strategies by preset priority.

        The ranking respects the user's profile selection:
        - Primary: Use the weighted finance_score (incorporates user weights)
        - Secondary: Preset-specific tiebreaker metric
        - Tertiary: CF proximity (closer to target is better)
        """
        name = preset_name.lower()

        def sort_key(x: dict[str, Any]) -> tuple:
            # CF proximity as final tiebreaker (higher is better)
            cf_prox = x.get("cf_proximity", 0)
            finance = x.get("finance_score", 0)
            balanced = x.get("balanced_score", 0)

            # Match French preset names
            if "sécurité" in name or "dscr" in name:
                # Safety profile: prioritize DSCR, then finance_score
                return (x.get("dscr_norm", 0), finance, cf_prox)

            if "rendement" in name or "irr" in name:
                # IRR profile: prioritize TRI, then finance_score
                return (x.get("tri_norm", 0), finance, cf_prox)

            if "cash" in name:
                # Cash-flow profile: prioritize CF proximity, then DSCR for safety
                return (cf_prox, x.get("dscr_norm", 0), finance)

            if "patrimoine" in name:
                # Wealth-building profile: prioritize enrichment, then IRR
                return (x.get("enrich_norm", 0), x.get("tri_norm", 0), balanced)

            # Default "Équilibré": use balanced_score which combines finance + quality
            return (balanced, finance, cf_prox)

        sorted_strategies = sorted(strategies, key=sort_key, reverse=True)
        return sorted_strategies[:top_n]

