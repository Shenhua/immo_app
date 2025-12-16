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
        
        # Sort by apport_min for more predictable pruning
        sorted_bricks = sorted(affordable_bricks, key=lambda b: b.get("apport_min", 0.0))
        
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

        log.info("strategy_search_started",
                 brick_count=len(self.bricks),
                 apport=self.apport_disponible)

        # 2. Generate combinations
        combos = self.combo_generator.generate(self.bricks, self.apport_disponible)
        log.debug("combinations_generated", count=len(combos))

        strategies = []

        # 3. Process candidates in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        import threading
        
        max_workers = os.cpu_count() or 4
        log.debug("parallel_processing_started", combos=len(combos), workers=max_workers)
        
        # Thread-safe counters
        lock = threading.Lock()
        viable_count = [0]  # Mutable container for thread-safe increment

        def process_combo(combo):
            """Process a single combo and return strategy or None."""
            bricks_copy = [dict(b) for b in combo]
            
            # SIMPLE BUDGET CHECK ONLY
            apport_min_total = sum(b.get("apport_min", 0.0) for b in bricks_copy)
            if apport_min_total > self.apport_disponible:
                return None  # Can't afford this combo
            
            # Determine if we should aggressively deploy capital
            aggressive_deploy = use_full_capital_override

            ok, details, cf_final, apport_used = allocator.allocate(
                bricks_copy,
                self.apport_disponible,
                self.cash_flow_cible,
                self.tolerance,
                use_full_capital=aggressive_deploy
            )

            # SOFT FAILURE: Keep allocation even if target not met
            # Apply penalty in scoring instead of discarding
            strat = {
                "details": details,
                "apport_total": apport_used,
                "patrimoine_acquis": sum(b.get("cout_total", 0.0) for b in details),
                "cash_flow_final": cf_final,
                "allocation_ok": ok,  # Track if target was met
            }

            # Basic Metrics
            total_rent = sum(b.get("loyer_mensuel_initial", 0.0) for b in strat["details"]) * 12
            total_cost = sum(b.get("cout_total", 0.0) for b in strat["details"])
            strat["renta_brute"] = (total_rent / total_cost * 100.0) if total_cost > 0 else 0.0

            # Simulation
            try:
                schedules = [
                    generate_amortization_schedule(
                        float(p["credit_final"]),
                        float(p["taux_pret"]),
                        int(p["duree_pret"]) * 12,
                        float(p["assurance_ann_pct"])
                    ) for p in strat["details"]
                ]

                df_sim, bilan = engine.simulate(strat, horizon_years, schedules)

                # Metrics
                strat["tri_annuel"] = float(bilan.get("tri_annuel", 0.0))
                strat["liquidation_nette"] = float(bilan.get("liquidation_nette", 0.0))

                # DSCR Y1: Net Operating Income / Debt Service
                # NOI = Gross Rent - Operating Expenses (excl. debt service)
                # Note: Charges Déductibles is negative and INCLUDES interest+insurance
                # So we need to add interest back to get operating-only NOI
                if not df_sim.empty:
                    row = df_sim.iloc[0]
                    ds = row["Capital Remboursé"] + row["Intérêts & Assurance"]
                    # Charges are negative, interest is positive
                    # NOI = loyers - |charges| + interest (to undo the interest deduction)
                    charges_with_interest = row["Charges Déductibles"]  # negative
                    interest_assur = row["Intérêts & Assurance"]  # positive
                    noi = row["Loyers Bruts"] + charges_with_interest + interest_assur
                    strat["dscr_y1"] = (noi / ds) if ds > 1e-9 else 0.0
                else:
                    strat["dscr_y1"] = 0.0

                # Qualitative
                strat["qual_score"] = self._calculate_qualitative_score(strat)
                strat["cf_distance"] = abs(strat["cash_flow_final"] - self.cash_flow_cible)

                ap = float(strat.get("apport_total", 1.0)) or 1.0
                strat["cap_eff"] = (strat["liquidation_nette"] - ap) / ap
                strat["enrich_net"] = strat["liquidation_nette"] - ap

                return strat

            except Exception as e:
                log.warning("strategy_simulation_failed", error=str(e))
                return None

        # Early termination: stop after collecting enough viable strategies
        MAX_VIABLE = int(os.getenv("STRATEGY_MAX_VIABLE", "100"))
        MAX_TOTAL = int(os.getenv("STRATEGY_MAX_TOTAL", "500"))
        BATCH_SIZE = max_workers * 2  # Submit in small batches for responsive cancellation
        
        combo_iter = iter(combos)
        active_futures = []
        done = False
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while not done:
                # Submit a batch
                batch_count = 0
                while batch_count < BATCH_SIZE:
                    try:
                        combo = next(combo_iter)
                        active_futures.append(executor.submit(process_combo, combo))
                        batch_count += 1
                    except StopIteration:
                        done = True
                        break
                
                # Process completed futures
                for future in as_completed(active_futures, timeout=0.1):
                    try:
                        result = future.result(timeout=0)
                        if result is not None:
                            with lock:
                                strategies.append(result)
                                if result.get("allocation_ok", False):
                                    viable_count[0] += 1
                    except Exception as e:
                        log.debug("future_result_error", error=str(e))
                
                # Remove completed futures
                active_futures = [f for f in active_futures if not f.done()]
                
                # Check termination conditions
                with lock:
                    current_viable = viable_count[0]
                    current_total = len(strategies)
                
                if current_viable >= MAX_VIABLE or current_total >= MAX_TOTAL:
                    log.info("early_termination", viable=current_viable, total=current_total)
                    break
            
            # Wait for remaining active futures
            for future in as_completed(active_futures):
                try:
                    result = future.result()
                    if result is not None:
                        with lock:
                            strategies.append(result)
                            if result.get("allocation_ok", False):
                                viable_count[0] += 1
                except Exception:
                    pass

        # 4. Score & Rank
        self.scorer.score_strategies(strategies, self.cash_flow_cible)

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

