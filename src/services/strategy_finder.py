"""Strategy finder service.

Modular strategy finding service extracted from strategy_finder.py.
Provides combination generation, scoring, and ranking.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
import math
from math import isfinite
from typing import Any, Dict

from src.core.logging import get_logger
from src.services.allocator import PortfolioAllocator
from src.services.optimizer import GeneticOptimizer, ExhaustiveOptimizer

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
    """Generates valid property combinations within budget using bounded enumeration."""

    def __init__(self, max_properties: int = 3):
        self.max_properties = max_properties

    def generate(
        self,
        bricks: list[dict[str, Any]],
        apport_disponible: float,
    ) -> list[tuple[dict[str, Any], ...]]:
        """Generate all valid combinations using bounded enumeration.

        This uses recursive branch-and-bound to prune infeasible branches early,
        dramatically reducing the search space for large property counts.
        
        Key optimizations:
        - Pre-filter bricks individually exceeding budget
        - Sort by cost ascending for better pruning
        - Recursive enumeration that tracks running cost and stops when budget exceeded
        - Deduplication by nom_bien within combinations
        """
        # Pre-filter unaffordable bricks
        affordable_bricks = [b for b in bricks if b.get("apport_min", 0.0) <= apport_disponible]
        
        if len(affordable_bricks) < len(bricks):
            log.debug("pruning_unaffordable_bricks", 
                     original=len(bricks), 
                     remaining=len(affordable_bricks))
        
        if not affordable_bricks:
            log.info("no_affordable_bricks", budget=apport_disponible)
            return []
        
        # Sort by cost ASCENDING for better branch-and-bound pruning
        # (low-cost bricks first means we can add more before hitting budget)
        sorted_bricks = sorted(affordable_bricks, key=lambda b: b.get("apport_min", 0.0))
        
        combos = []
        visited_count = [0]  # Use list for mutable in nested function
        pruned_count = [0]
        
        def _enumerate(start_idx: int, current_combo: list, current_cost: float, used_names: set):
            """Recursive bounded enumeration."""
            # Valid combo if non-empty
            if current_combo:
                combos.append(tuple(current_combo))
            
            # Stop if max properties reached
            if len(current_combo) >= self.max_properties:
                return
            
            # Try adding each remaining brick
            for i in range(start_idx, len(sorted_bricks)):
                brick = sorted_bricks[i]
                brick_cost = brick.get("apport_min", 0.0)
                brick_name = brick.get("nom_bien", "")
                
                visited_count[0] += 1
                
                # Pruning 1: Budget exceeded - stop this branch entirely
                # Since bricks are sorted by cost, all subsequent bricks cost >= this one
                if current_cost + brick_cost > apport_disponible:
                    pruned_count[0] += len(sorted_bricks) - i
                    break  # No point checking more expensive bricks
                
                # Pruning 2: Skip duplicate property names within combo
                if brick_name in used_names:
                    continue
                
                # Add brick and recurse
                current_combo.append(brick)
                used_names.add(brick_name)
                
                _enumerate(i + 1, current_combo, current_cost + brick_cost, used_names)
                
                # Backtrack
                current_combo.pop()
                used_names.remove(brick_name)
        
        # Start enumeration
        _enumerate(0, [], 0.0, set())
        
        log.info("combos_generated", 
                count=len(combos), 
                max_props=self.max_properties, 
                budget=apport_disponible, 
                visited=visited_count[0],
                pruned=pruned_count[0])

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

        # 2. Hybrid Solver Logic (Phase 21)
        # Check problem size to decide between Brute Force (Exhaustive) and Genetic Algorithm
        n_bricks = len(self.bricks)
        max_k = self.max_properties
        
        total_combinations = 0
        for k in range(1, max_k + 1):
            total_combinations += math.comb(n_bricks, k)
            
        # Threshold: With smart bounded enumeration, exhaustive is now efficient
        # for most realistic scenarios. Set very high to prefer exhaustive.
        # The actual number of evaluated combos will be much lower due to pruning.
        THRESHOLD_EXHAUSTIVE = 100_000_000  # 100M - effectively always exhaustive
        
        strategies = []
        
        # Always use exhaustive with smart pruning (GA removed)
        if True:  # Was: total_combinations < THRESHOLD_EXHAUSTIVE
            # --- EXHAUSTIVE MODE (v1 Fidelity) ---
            log.info("hybrid_solver_selected", 
                     mode="EXHAUSTIVE", 
                     combos=total_combinations, 
                     threshold=THRESHOLD_EXHAUSTIVE)
            
            optimizer = ExhaustiveOptimizer(
                allocator=allocator,
                simulator=engine,
                scorer=self.scorer
            )
            
            strategies = optimizer.solve(
                all_bricks=self.bricks,
                budget=self.apport_disponible,
                target_cf=self.cash_flow_cible,
                tolerance=self.tolerance,
                horizon=horizon_years,
                max_combinations=THRESHOLD_EXHAUSTIVE, # Safety cap
                max_props=self.max_properties
            )
            
        else:
            # --- GENETIC MODE (v2 Scalability) ---
            # Adaptive GA parameters based on search space size
            # Larger search space = more exploration needed
            if total_combinations > 5_000_000:
                pop_size = 200
                generations = 50
                elite_size = 20
            elif total_combinations > 1_000_000:
                pop_size = 150
                generations = 40
                elite_size = 15
            else:
                pop_size = 100
                generations = 30
                elite_size = 10
                
            log.info("hybrid_solver_selected", 
                     mode="GENETIC", 
                     combos=total_combinations, 
                     threshold=THRESHOLD_EXHAUSTIVE,
                     pop_size=pop_size,
                     generations=generations)
                     
            optimizer = GeneticOptimizer(
                population_size=pop_size,
                generations=generations,
                elite_size=elite_size,
                max_properties=self.max_properties,
                allocator=allocator,
                simulator=engine,
                scorer=self.scorer
            )
            
            # Phase 17 Review: Deep Diversity
            # Request larger pool for dedupe
            pool_size = max(500, top_n * 5)
            
            strategies = optimizer.evolve(
                all_bricks=self.bricks,
                budget=self.apport_disponible,
                target_cf=self.cash_flow_cible,
                tolerance=self.tolerance,
                horizon=horizon_years,
                top_n=pool_size 
            )
            
        log.info("optimization_finished", count=len(strategies))

        # Guard against empty results from optimizer
        if not strategies:
            log.warning("no_strategies_found", 
                        reason="optimizer_returned_empty",
                        brick_count=len(self.bricks),
                        budget=self.apport_disponible)
            return []

        # 4. Score & Rank (Post-Optimization)
        # We re-run relative scoring on the "Elite" set to populate UI-friendly normalized fields (A-score, etc.)
        self.scorer.score_strategies(strategies, self.cash_flow_cible)

        # Tiered ranking instead of hard-filter
        # Tier 1: Feasible (allocation_ok=True)
        # Tier 2: Near-feasible (within 2x tolerance)
        # Tier 3: Infeasible (show a few for transparency)
        for s in strategies:
            if s.get("allocation_ok", False):
                s["tier"] = 1
                s["tier_label"] = "Réalisable"
            else:
                cf_gap = abs(s.get("cash_flow_final", -9999) - self.cash_flow_cible)
                if cf_gap <= self.tolerance * 3:
                    s["tier"] = 2
                    s["tier_label"] = "Proche"
                    s["fitness"] = s.get("fitness", 0) * 0.7  # 30% penalty
                else:
                    s["tier"] = 3
                    s["tier_label"] = "Difficile"
                    s["fitness"] = s.get("fitness", 0) * 0.3  # 70% penalty

        # Sort by tier then fitness (best first within each tier)
        strategies.sort(key=lambda x: (x.get("tier", 3), -x.get("fitness", 0)))
        
        # Log tier distribution
        tier_counts = {1: 0, 2: 0, 3: 0}
        for s in strategies:
            tier_counts[s.get("tier", 3)] = tier_counts.get(s.get("tier", 3), 0) + 1
        log.info("strategy_tiers", tier_1=tier_counts[1], tier_2=tier_counts[2], tier_3=tier_counts[3])
        
        # STRICT FILTER: Remove Tier 3 ("Difficile") strategies
        # User Feedback: "remove results that are totally out of scope"
        # We only keep Tier 1 (Feasible) and Tier 2 (Near-Feasible, within 3x tolerance)
        pre_filter_count = len(strategies)
        strategies = [s for s in strategies if s.get("tier", 3) < 3]
        
        # Log when all strategies were filtered
        if not strategies and pre_filter_count > 0:
            log.warning("all_strategies_infeasible",
                        tier_3_count=tier_counts[3],
                        target_cf=self.cash_flow_cible,
                        tolerance=self.tolerance,
                        hint="Consider relaxing tolerance or adjusting cash flow target")
        
        # Dedupe (preserving tier order)
        strategies = self.dedupe_strategies(strategies, top_n=top_n)

        # Apply Pareto Filter (v1 Logic) to clean up dominated strategies
        # Only apply if we have enough candidates to afford filtering
        if len(strategies) > top_n:
             strategies = self._pareto_filter(strategies)

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


    def dedupe_strategies(self, strategies: list[dict[str, Any]], top_n: int = 10) -> list[dict[str, Any]]:
        """Remove near-duplicate strategies.
        
        Groups by property signature and keeps best per group.
        Labels variations explicitly for UI clarity.
        """
        # Group by signature (property names + count)
        grouped = {}
        for s in strategies:
            details = s.get("details", [])
            sig = tuple(sorted(d.get("nom_bien", "") for d in details))
            if sig not in grouped:
                grouped[sig] = []
            grouped[sig].append(s)

        # Sort each group by tier first, then score descending
        for sig in grouped:
            grouped[sig].sort(key=lambda x: (
                x.get("tier", 3),  # Tier 1 first
                -x.get("balanced_score", -float("inf"))
            ))

        # Selection logic: Strictly keep only the best variation per signature
        # This forces the result list to be composed of DISTINCT strategies.
        # If we want 50 results, we must find 50 distinct property combinations.
        final_list = []
        
        for sig, items in grouped.items():
            # Keep only the absolute best (index 0)
            best_item = items[0]
            final_list.append(best_item)

        # Re-sort final list by tier then score
        final_list.sort(key=lambda x: (x.get("tier", 3), -x.get("balanced_score", 0)))
        
        return final_list

    def _pareto_filter(self, strategies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter out dominated strategies (Pareto Efficiency).
        
        A strategy A dominates B if A is better or equal in all metrics 
        and strictly better in at least one.
        Metrics checked: 
        - Cash Flow (Higher is better, unless target... assuming general 'wealth' for now)
          Actually for Pareto we care about: Yield vs Risk vs Cost.
          Simple v1 Pareto: 
          - Cost (Lower is better)
          - CashFlow (Higher is better)
          - Yield (Higher is better)
        """
        # Sort by cost ascending to make comparison easier
        sorted_s = sorted(strategies, key=lambda x: x.get("apport_total", 0))
        
        keep = []
        
        for cand in sorted_s:
            dominated = False
            c_cost = cand.get("apport_total", 0)
            c_cf = cand.get("cash_flow_final", 0)
            c_tri = cand.get("tri_annuel", 0)
            
            for exist in keep:
                e_cost = exist.get("apport_total", 0)
                e_cf = exist.get("cash_flow_final", 0)
                e_tri = exist.get("tri_annuel", 0)
                
                # Check if 'exist' dominates 'cand'
                # Exist costs less/equal AND has better/equal CF AND better/equal TRI
                if (e_cost <= c_cost) and (e_cf >= c_cf) and (e_tri >= c_tri):
                    # Strict check: is it better in at least one?
                    if (e_cost < c_cost) or (e_cf > c_cf) or (e_tri > c_tri):
                        dominated = True
                        break
            
            if not dominated:
                keep.append(cand)
                
        return keep

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
            # CRITICAL FIX: Always prioritize Tier first!
            # We sort reverse=True (Descending).
            # Tier 1 (Best) > Tier 3 (Worst).
            # So we use negative tier: -1 > -3.
            tier_score = -x.get("tier", 3)
            
            # CF proximity as final tiebreaker (higher is better)
            cf_prox = x.get("cf_proximity", 0)
            finance = x.get("finance_score", 0)
            balanced = x.get("balanced_score", 0)

            # Match French preset names
            if "sécurité" in name or "dscr" in name:
                # Safety profile: prioritize DSCR, then finance_score
                return (tier_score, x.get("dscr_norm", 0), finance, cf_prox)

            if "rendement" in name or "irr" in name:
                # IRR profile: prioritize TRI, then finance_score
                return (tier_score, x.get("tri_norm", 0), finance, cf_prox)

            if "cash" in name:
                # Cash-flow profile: prioritize CF proximity, then DSCR for safety
                return (tier_score, cf_prox, x.get("dscr_norm", 0), finance)

            if "patrimoine" in name:
                # Wealth-building profile: prioritize enrichment, then IRR
                return (tier_score, x.get("enrich_norm", 0), x.get("tri_norm", 0), balanced)

            # Default "Équilibré": use balanced_score which combines finance + quality
            return (tier_score, balanced, finance, cf_prox)

        sorted_strategies = sorted(strategies, key=sort_key, reverse=True)
        return sorted_strategies[:top_n]

