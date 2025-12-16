"""
Genetic Algorithm Optimizer for Real Estate Portfolios.
Finds optimal combinations of InvestmentBricks to maximize a fitness score 
subject to Budget and Cashflow constraints.
"""
from typing import List, Dict, Any, Tuple, Callable
import random
import copy
import structlog
import numpy as np
from src.services.allocator import PortfolioAllocator
from src.core.simulation import SimulationEngine
from src.models.strategy import Strategy, StrategyResult
from src.core.glossary import (
    calculate_cashflow_metrics,
    calculate_enrichment_metrics
)

log = structlog.get_logger()

class Individual:
    """Represents a candidate portfolio (strategy)."""
    def __init__(self, bricks: List[Dict[str, Any]]):
        self.bricks = bricks
        self.fitness: float = -1.0
        self.stats: Dict[str, Any] = {}
        self.is_valid: bool = False
    
    @property
    def key(self) -> str:
        """Unique signature for deduplication."""
        names = sorted([b["nom_bien"] for b in self.bricks])
        return "|".join(names)

class GeneticOptimizer:
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 20,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        elite_size: int = 5,
        allocator: PortfolioAllocator = None,
        simulator: SimulationEngine = None,
        scorer: Any = None,
        seed: int = 42
    ):
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.seed = seed
        
        self.allocator = allocator
        self.simulator = simulator
        self.scorer = scorer
        
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

    def _generate_random_individual(
        self, 
        all_bricks: List[Dict[str, Any]], 
        budget: float,
        target_cf: float,
        tolerance: float
    ) -> Individual:
        """Create a random valid-ish individual respecting budget."""
        # Simple greedy approach to fill budget randomly
        available = list(all_bricks)
        random.shuffle(available)
        selected = []
        current_cost = 0.0
        
        for b in available:
            cost = b.get("apport_min", 0.0)
            if current_cost + cost <= budget:
                selected.append(b)
                current_cost += cost
            
            # Stop randomly to allow diverse portfolio sizes
            if current_cost > budget * 0.9 and random.random() < 0.5:
                break
                
        return Individual(selected)

    def _evaluate(
        self, 
        ind: Individual, 
        budget: float, 
        target_cf: float, 
        tolerance: float,
        horizon: int
    ) -> None:
        """Calculate fitness for an individual."""
        if not ind.bricks:
            ind.fitness = 0.0
            ind.is_valid = False
            return

        # 1. Allocation
        ok, details, cf_final, apport_used = self.allocator.allocate(
            ind.bricks, budget, target_cf, tolerance
        )
        
        # Store allocated details for later
        ind.stats["allocated_details"] = details
        ind.stats["cash_flow_final"] = cf_final
        ind.stats["apport_total"] = apport_used
        ind.stats["allocation_ok"] = ok
        
        if not ok:
            # Penalize but allow survival if close
            ind.fitness = 0.1
            ind.is_valid = False
            return

        # 2. Simulation
        # Construct strategy dict for simulator
        strat_data = {
            "details": details,
            "apport_total": apport_used
        }
        
        # Generate schedules on the fly
        from src.core.financial import generate_amortization_schedule
        schedules = []
        for p in details:
            # Use credit_final from allocator (already correctly calculated)
            # DO NOT recalculate as (cout_total - apport) — that misses loan fees!
            principal = float(p.get("credit_final", p.get("capital_restant", 0)))
            
            sch = generate_amortization_schedule(
                principal=principal,
                annual_rate_pct=p.get("taux_pret", 0.0),
                duration_months=int(p.get("duree_pret", 20)) * 12,
                annual_insurance_pct=p.get("assurance_ann_pct", 0.0)
            )
            schedules.append(sch)

        try:
            df_sim, bilan = self.simulator.simulate(strat_data, horizon, schedules)
            ind.stats["bilan"] = bilan
            # Note: df_sim not stored to save memory; KPIs extracted below
            
            # 3. Check Constraints (Glossary Standard)
            # Use Allocator mode (Min vs Target) to check acceptability
            mode_cf = self.allocator.mode_cf if self.allocator else "target"
            cf_metrics = calculate_cashflow_metrics(df_sim, target_cf, tolerance, mode_cf=mode_cf)
            ind.stats.update(cf_metrics)

            # Ensure enrichment metrics are present
            enrich_metrics = calculate_enrichment_metrics(
                bilan, 
                strat_data.get("apport_total", 0.0), 
                horizon
            )
            ind.stats.update(enrich_metrics)
            
            if not cf_metrics["is_acceptable"]:
                 ind.fitness = 0.2
                 ind.is_valid = False
                 return

            # 4. Scoring (Absolute Mode)
            # Use absolute targets to avoid "relative normalization" instability
            finance_score = self._calculate_absolute_finance_score(bilan, cf_metrics, target_cf, tolerance)
            
            # Qualitative Score
            # Use the official qualitative scorer if available, else simple proxy
            qual_score = 50.0
            from src.core.scoring import calculate_qualitative_score
            qual_score = calculate_qualitative_score({"details": details})
            
            # Balanced Score (50/50 default or use scorer weights if available)
            # Fitness = Balanced Score (0-100)
            # Fitness = Balanced Score (0-100)
            if ind.fitness < 0:
                # 5. Combined Score
                # Use SCORER's quality weight from parameters (Phase 15.1)
                q_w = self.scorer.qualite_weight if self.scorer else 0.5
                
                # Combine
                if q_w >= 1.0:
                    fitness = qual_score / 100.0
                elif q_w <= 0.0:
                    fitness = finance_score
                else:
                    fitness = (1.0 - q_w) * finance_score + q_w * (qual_score / 100.0)
                
                ind.fitness = max(0.01, fitness * 100) # Scale to 0-100 like legacy
                ind.is_valid = True
            
        except Exception as e:
            log.warning("evaluation_failed", error=str(e))
            ind.fitness = 0.0
            ind.is_valid = False

    def _calculate_absolute_finance_score(self, bilan: dict, cf_metrics: dict, target_cf: float, tolerance: float) -> float:
        """
        Calculate finance score (0.0 to 1.0) using constant absolute scales.
        Implements Expert Report "Absolute Scales" recommendation.
        Respects user-defined weights from StrategyScorer (Phase 15).
        """
        w = self.scorer.weights if self.scorer else {
            "tri": 0.3, "enrich": 0.3, "dscr": 0.2, "cf_proximity": 0.2
        }

        # 1. TRI (Internal Rate of Return)
        # Target: > 20% is excellent (1.0). Old cap (10%) hid unicorn deals (Phase 14.3).
        tri = bilan.get("tri_annuel", 0.0)
        s_tri = max(0.0, min(1.0, tri / 20.0))
        
        # 2. Enrichment (ROE bias for Empire/Growth)
        # Target: 2.0x (doubling equity) over horizon is good baseline.
        enrich = bilan.get("enrichissement_net", 0.0)
        apport_total = cf_metrics.get("apport_total", 1.0)
        if apport_total < 1.0: apport_total = 1.0
        
        roe = enrich / apport_total
        s_enrich = max(0.0, min(1.0, roe / 2.0))

        # 3. DSCR
        # Target: 1.3 is safe (1.0). Old (1.5) was too banking-conservative (Phase 14.2).
        dscr = float(bilan.get("dscr_y1", 0.0) or 0.0)
        s_dscr = max(0.0, min(1.0, dscr / 1.3))
        
        # 4. Cashflow Proximity (Dynamic Tolerance)
        # Score = 1.0 at gap=0. Score = 0.5 at gap=tolerance. Score = 0 at gap=2*tolerance.
        # This respects strict user tolerance settings (Phase 14.1).
        gap = cf_metrics["gap"]
        
        # Use provided tolerance, default to 100 if missing/zero
        safe_tol = tolerance if tolerance > 1.0 else 100.0
        s_cf = max(0.0, 1.0 - (gap / (safe_tol * 2.0)))
        
        # Weighted average using User Weights (Phase 15.2)
        # StrategyScorer keys map: "irr"->s_tri, "enrich_net"->s_enrich
        # "dscr"->s_dscr, "cf_proximity"->s_cf
        
        score = (
            w.get("irr", 0.25) * s_tri +
            w.get("enrich_net", 0.30) * s_enrich +
            w.get("dscr", 0.15) * s_dscr +
            w.get("cf_proximity", 0.20) * s_cf
        )
        return score

    def _individual_to_strategy(self, ind: Individual) -> Dict[str, Any]:
        """Convert Individual to Strategy dict format."""
        # Use stored allocated details
        details = ind.stats.get("allocated_details", ind.bricks)
        bilan = ind.stats.get("bilan", {})
        
        s = {
            "details": details,
            "apport_total": ind.stats.get("apport_total", 0.0),
            "cash_flow_final": ind.stats.get("cash_flow_final", 0.0),
            "allocation_ok": ind.stats.get("allocation_ok", False),
            "fitness": ind.fitness,
            
            # Populate fields expected by UI/Scorer
            # Note: Simulator returns 'tri_annuel', 'enrichissement_net'
            # We map them to standard UI keys 'tri_global', 'enrich_net'
            "liquidation_nette": bilan.get("liquidation_nette", 0.0),
            "enrich_net": bilan.get("enrichissement_net", ind.stats.get("enrichissement_net", 0.0)),
            "tri_annuel": bilan.get("tri_annuel", 0.0),
            "tri_global": bilan.get("tri_annuel", 0.0),
        }
        
        # Add CF metrics
        if "cf_year_1_monthly" in ind.stats:
            s["cf_monthly_y1"] = ind.stats["cf_year_1_monthly"]
            s["cf_monthly_avg"] = ind.stats["cf_avg_5y_monthly"]
            
        return s

    def evolve(
        self,
        all_bricks: List[Dict[str, Any]],
        budget: float,
        target_cf: float,
        tolerance: float,
        horizon: int = 20,
        top_n: int = 100
    ) -> List[Dict[str, Any]]:
        """Main evolution loop."""
        
        # Pre-process bricks for Smart Seeding (Expert Recommendation Part 3)
        # Sort biased pools to ensure initial population isn't just random trash
        # 1. Yield (Gross)
        bricks_yield = sorted(all_bricks, key=lambda b: (b.get("loyer_mensuel_initial",0)*12)/max(1, b.get("cout_total",1)), reverse=True)
        # 2. Quality
        bricks_qual = sorted(all_bricks, key=lambda b: b.get("qual_score_bien", 0), reverse=True)
        # 3. Cost (Cheapest first - helps fit more items)
        bricks_cost = sorted(all_bricks, key=lambda b: b.get("apport_min", 0))
        
        # Initialize Population with diversity
        population = []
        
        # A. High Yield Bias (30%) - Experts want cashflow/efficiency
        for _ in range(int(self.pop_size * 0.3)):
            ind = self._generate_biased_individual(bricks_yield, budget, top_n_percent=0.25)
            population.append(ind)
            
        # B. High Quality Bias (20%) - For Patrimonial
        for _ in range(int(self.pop_size * 0.2)):
            ind = self._generate_biased_individual(bricks_qual, budget, top_n_percent=0.25)
            population.append(ind)
            
        # C. Low Cost Bias (20%) - To fill gaps
        for _ in range(int(self.pop_size * 0.2)):
            ind = self._generate_biased_individual(bricks_cost, budget, top_n_percent=0.40)
            population.append(ind)
            
        # D. Pure Random (30%) - Exploration
        while len(population) < self.pop_size:
            ind = self._generate_random_individual(all_bricks, budget, target_cf, tolerance)
            population.append(ind)
            
        best_fitness = 0.0
        
        # Evolution Loop
        for gen in range(self.generations):
            # 1. Evaluate
            for ind in population:
                self._evaluate(ind, budget, target_cf, tolerance, horizon)
                
            # 2. Sort
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Log progress
            best = population[0]
            if best.fitness > best_fitness:
                best_fitness = best.fitness
                log.debug("new_best_fitness", gen=gen, fitness=f"{best_fitness:.2f}")
                
            # 3. Selection
            # Elitism: Keep best
            next_pop = population[:self.elite_size]
            
            # Tournament / Roulette for the rest
            while len(next_pop) < self.pop_size:
                p1 = self._select_tournament(population)
                p2 = self._select_tournament(population)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self._crossover(p1, p2)
                else:
                    child = copy.deepcopy(p1)
                    
                # Mutation
                self._mutate(child, all_bricks, budget)
                
                next_pop.append(child)
                
            population = next_pop
            
        # Final Evaluation
        for ind in population:
            self._evaluate(ind, budget, target_cf, tolerance, horizon)
            
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Logging Tier distribution (Phase 11.3)
        if population:
            best = population[0]
            log.info("ga_finished", 
                best_fitness=f"{best.fitness:.2f}",
                valid_count=sum(1 for i in population if i.is_valid)
            )

        # Return top valid results formatted as strategies
        # Slice by top_n (Phase 17.1)
        valid_results = [ind for ind in population if ind.is_valid]
        return [self._individual_to_strategy(ind) for ind in valid_results[:top_n]]

    def _select_tournament(self, population: List[Individual], k: int = 3) -> Individual:
        """Select best individual from random tournament."""
        contestants = random.sample(population, min(len(population), k))
        return max(contestants, key=lambda ind: ind.fitness)

    def _crossover(self, p1: Individual, p2: Individual) -> Individual:
        """Uniform crossover: take properties from both parents."""
        # Merge unique bricks
        pool = {b["nom_bien"]: b for b in p1.bricks + p2.bricks}
        all_unique = list(pool.values())
        
        # Randomly select a subset to form a child
        # Try to maintain average length
        if not p1.bricks and not p2.bricks:
            avg_len = 0
        else:
            avg_len = int((len(p1.bricks) + len(p2.bricks)) / 2)
            if avg_len == 0: avg_len = 1
            
        child_bricks = random.sample(all_unique, k=min(len(all_unique), avg_len))
        
        return Individual(child_bricks)

    def _mutate(self, ind: Individual, all_bricks: List[Dict[str, Any]], budget: float):
        """Randomly add/remove/swap properties."""
        if random.random() > self.mutation_rate:
            return

        action = random.choice(["add", "remove", "swap"])
        
        if action == "remove" and len(ind.bricks) > 1:
            ind.bricks.pop(random.randrange(len(ind.bricks)))
            
        elif action == "add":
            # Add a random brick if budget allows (roughly)
            # Precise budget check happens in eval, here we just try
            candidate = random.choice(all_bricks)
            # Avoid dupes
            if candidate["nom_bien"] not in [b["nom_bien"] for b in ind.bricks]:
                ind.bricks.append(candidate)
                
        elif action == "swap" and len(ind.bricks) > 0:
            ind.bricks.pop(random.randrange(len(ind.bricks)))
            candidate = random.choice(all_bricks)
            if candidate["nom_bien"] not in [b["nom_bien"] for b in ind.bricks]:
                ind.bricks.append(candidate)
        
        ind.fitness = -1.0 # Invalidate fitness

    def _generate_biased_individual(
        self, 
        sorted_bricks: List[Dict[str, Any]], 
        budget: float,
        top_n_percent: float = 0.25
    ) -> Individual:
        """Create individual picking mainly from the top N% of the provided list."""
        # Restrict to top percentile
        limit = max(1, int(len(sorted_bricks) * top_n_percent))
        candidates = sorted_bricks[:limit]
        
        # Simple greedy fill from biased pool
        # Shuffle to avoid deterministic same-start
        random.shuffle(candidates)
        
        selected = []
        current_cost = 0.0
        
        for b in candidates:
            cost = b.get("apport_min", 0.0)
            if current_cost + cost <= budget:
                selected.append(b)
                current_cost += cost
            
            # Stop randomly
            if current_cost > budget * 0.9 and random.random() < 0.5:
                break
                
        return Individual(selected)
