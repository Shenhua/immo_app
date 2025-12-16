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
from src.core.glossary import calculate_cashflow_metrics

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
            # Calculate loan amount based on allocation
            cout_total = p.get("cout_total", p.get("prix_achat_bien", 0))
            apport_bien = p.get("apport_final_bien", 0)
            principal = max(0, cout_total - apport_bien)
            
            sch = generate_amortization_schedule(
                principal=principal,
                annual_rate_pct=p.get("taux_pret", 0.0),
                duration_months=int(p.get("duree_pret", 20)) * 12,
                annual_insurance_pct=p.get("assurance_pret_pct", 0.0)
            )
            schedules.append(sch)

        try:
            df_sim, bilan = self.simulator.simulate(strat_data, horizon, schedules)
            ind.stats["bilan"] = bilan
            ind.stats["simulation"] = df_sim # Warning: Memory usage
            
            # 3. Check Constraints (Glossary Standard)
            cf_metrics = calculate_cashflow_metrics(df_sim, target_cf, tolerance)
            ind.stats.update(cf_metrics)
            
            if not cf_metrics["is_acceptable"]:
                 ind.fitness = 0.2
                 ind.is_valid = False
                 return

            # 4. Scoring (Absolute Mode)
            # Use absolute targets to avoid "relative normalization" instability
            finance_score = self._calculate_absolute_finance_score(bilan, cf_metrics, target_cf)
            
            # Qualitative Score
            # Use the official qualitative scorer if available, else simple proxy
            qual_score = 50.0
            from src.core.scoring import calculate_qualitative_score
            qual_score = calculate_qualitative_score({"details": details})
            
            # Balanced Score (50/50 default or use scorer weights if available)
            # Fitness = Balanced Score (0-100)
            if self.scorer:
                w_qual = self.scorer.qualite_weight
                fitness = (1.0 - w_qual) * finance_score + w_qual * (qual_score / 100.0)
            else:
                fitness = 0.5 * finance_score + 0.5 * (qual_score / 100.0)
            
            ind.fitness = max(0.01, fitness * 100) # Scale to 0-100 like legacy
            ind.is_valid = True
            
        except Exception as e:
            log.warning("evaluation_failed", error=str(e))
            ind.fitness = 0.0
            ind.is_valid = False

    def _calculate_absolute_finance_score(self, bilan: dict, cf_metrics: dict, target_cf: float) -> float:
        """
        Calculate finance score (0.0 to 1.0) using constant absolute scales.
        Implements Expert Report "Absolute Scales" recommendation.
        """
        # 1. TRI (Internal Rate of Return)
        # Target: > 10% is excellent (1.0), < 0% is bad (0.0)
        tri = bilan.get("tri_global", 0.0)
        s_tri = max(0.0, min(1.0, tri / 10.0))  # 10% cap
        
        # 2. Enrichment
        # Target: > 200k over horizon is nice? deeply depends on budget.
        # Let's use a arbitrary scale for now or better, Yield on Cost
        enrich = bilan.get("enrichissement_net", 0.0)
        s_enrich = max(0.0, min(1.0, enrich / 200000.0))

        # 3. DSCR
        # Target: 1.5 is safe (1.0)
        dscr = float(bilan.get("dscr_y1", 0.0) or 0.0) # simulation result usually doesn't have dscr_y1 precalc
        # ... actually simulator doesn't compute DSCR, StrategyFinder did.
        # We need to compute it or rely on what's in 'bilan' 
        # (Assuming glossary logic usage earlier or here)
        s_dscr = max(0.0, min(1.0, dscr / 1.5))
        
        # 4. Cashflow Proximity
        # Closer to target is better. 0 dist = 1.0. 
        gap = cf_metrics["gap"]
        s_cf = max(0.0, 1.0 - (gap / 500.0)) # 500 eur tolerance
        
        # Weighted average (Equal weights for now, pure robustness)
        return (0.3 * s_tri) + (0.3 * s_enrich) + (0.2 * s_dscr) + (0.2 * s_cf)

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
            "liquidation_nette": bilan.get("liquidation_nette", 0.0),
            "enrich_net": bilan.get("enrichissement_net", 0.0),
            "tri_annuel": bilan.get("tri_global", 0.0),
            "tri_global": bilan.get("tri_global", 0.0),
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
        horizon: int = 20
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
        n_yield = int(self.pop_size * 0.3)
        for _ in range(n_yield):
            population.append(self._generate_biased_individual(bricks_yield, budget, top_n_percent=0.25))
            
        # B. High Quality Bias (30%) - For the "Patrimonial" angle
        n_qual = int(self.pop_size * 0.3)
        for _ in range(n_qual):
            population.append(self._generate_biased_individual(bricks_qual, budget, top_n_percent=0.25))
            
        # C. Low Cost Bias (20%) - "Small strategies"
        n_cost = int(self.pop_size * 0.2)
        for _ in range(n_cost):
             population.append(self._generate_biased_individual(bricks_cost, budget, top_n_percent=0.4))
             
        # D. Random / Remainder (Exploration)
        while len(population) < self.pop_size:
            population.append(self._generate_random_individual(all_bricks, budget, target_cf, tolerance))
    
        # Evaluator helper
        def run_eval(pop):
            for i in pop:
                if i.fitness < 0: # Check cache/calc
                    self._evaluate(i, budget, target_cf, tolerance, horizon)

        run_eval(population)

        for gen in range(self.generations):
            # Sort by fitness
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Elitism
            next_gen = population[:self.elite_size]
            
            # Breeding
            while len(next_gen) < self.pop_size:
                parent1 = random.choice(population[:20]) # Pick from top 20
                parent2 = random.choice(population[:20])
                
                child = self._crossover(parent1, parent2)
                self._mutate(child, all_bricks, budget)
                next_gen.append(child)
            
            population = next_gen
            run_eval(population)
            
            best = population[0]
            log.info(
                "ga_generation_complete", 
                gen=gen, 
                best_fitness=f"{best.fitness:.2f}",
                valid_count=sum(1 for i in population if i.is_valid)
            )

        # Return top valid results formatted as strategies
        # Recalculate full simulation for top results to ensure comprehensive data
        valid_results = [ind for ind in population if ind.is_valid]
        return [self._individual_to_strategy(ind) for ind in valid_results[:20]]

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
