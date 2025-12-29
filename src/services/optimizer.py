"""
Genetic Algorithm Optimizer for Real Estate Portfolios.
Finds optimal combinations of InvestmentBricks to maximize a fitness score
subject to Budget and Cashflow constraints.
"""
import copy
import random
from typing import Any

import numpy as np
import structlog

from src.core.simulation import SimulationEngine
from src.services.allocator import PortfolioAllocator
from src.services.evaluator import StrategyEvaluator

log = structlog.get_logger()

class Individual:
    """Represents a candidate portfolio (strategy)."""
    def __init__(self, bricks: list[dict[str, Any]]):
        self.bricks = bricks
        self.fitness: float = -1.0
        self.stats: dict[str, Any] = {}
        self.is_valid: bool = False

    @property
    def key(self) -> str:
        """Unique signature for deduplication."""
        names = sorted([b["nom_bien"] for b in self.bricks])
        return "|".join(names)

class GeneticOptimizer:
    """Genetic Algorithm optimizer for finding optimal property portfolios.

    Uses evolutionary strategies to search large combination spaces efficiently:
    - Tournament selection for parent selection
    - Crossover to combine successful strategies
    - Mutation to explore new combinations
    - Elitism to preserve best solutions

    Best for large search spaces (>100K combinations) where exhaustive search
    is infeasible. Falls back from ExhaustiveOptimizer when threshold is exceeded.

    Attributes:
        pop_size: Population size per generation
        generations: Number of evolution cycles
        mutation_rate: Probability of mutation per individual
        crossover_rate: Probability of crossover vs direct copy
        elite_size: Number of top individuals preserved each generation
        max_properties: Maximum properties allowed per portfolio
    """

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 20,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        elite_size: int = 5,
        max_properties: int = 5,
        allocator: PortfolioAllocator = None,
        simulator: SimulationEngine = None,
        scorer: Any = None,
        seed: int = 42
    ):
        """Initialize the Genetic Optimizer.

        Args:
            population_size: Number of individuals per generation (default: 50)
            generations: Number of evolution cycles (default: 20)
            mutation_rate: Probability of mutation (0.0-1.0, default: 0.2)
            crossover_rate: Probability of crossover (0.0-1.0, default: 0.7)
            elite_size: Top N individuals to preserve (default: 5)
            max_properties: Max properties per portfolio (default: 5)
            allocator: PortfolioAllocator for capital allocation
            simulator: SimulationEngine for financial projections
            scorer: StrategyScorer for fitness calculation
            seed: Random seed for reproducibility (default: 42)
        """
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.max_properties = max_properties
        self.seed = seed

        self.allocator = allocator
        self.simulator = simulator
        self.scorer = scorer

        # Create evaluator for shared evaluation logic
        if simulator:
            self.evaluator = StrategyEvaluator(simulator, allocator, scorer)
        else:
            self.evaluator = None

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

    def _generate_random_individual(
        self,
        all_bricks: list[dict[str, Any]],
        budget: float,
        target_cf: float,
        tolerance: float
    ) -> Individual:
        """Create a random valid-ish individual respecting budget and max_properties."""
        # Simple greedy approach to fill budget randomly
        available = list(all_bricks)
        random.shuffle(available)
        selected = []
        current_cost = 0.0

        for b in available:
            if len(selected) >= self.max_properties:
                break
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
        """Calculate fitness for an individual using StrategyEvaluator."""
        if not ind.bricks:
            ind.fitness = 0.0
            ind.is_valid = False
            return

        # 1. Allocation (GA-specific step)
        ok, details, cf_final, apport_used = self.allocator.allocate(
            ind.bricks, budget, target_cf, tolerance
        )

        # Store allocated details
        ind.stats["allocated_details"] = details
        ind.stats["cash_flow_final"] = cf_final
        ind.stats["apport_total"] = apport_used
        ind.stats["allocation_ok"] = ok

        if not ok:
            ind.fitness = 0.1
            ind.is_valid = False
            return

        # 2. Use StrategyEvaluator for simulation and scoring
        strategy = {
            "details": details,
            "apport_total": apport_used
        }

        is_valid, metrics = self.evaluator.evaluate(strategy, target_cf, tolerance, horizon)

        if "error" in metrics:
            ind.fitness = 0.0
            ind.is_valid = False
            return

        # Store metrics
        ind.stats["bilan"] = {
            "liquidation_nette": metrics.get("liquidation_nette", 0.0),
            "enrichissement_net": metrics.get("enrich_net", 0.0),
            "tri_annuel": metrics.get("tri_annuel", 0.0),
            "dscr_y1": metrics.get("dscr_y1", 0.0),
        }
        ind.stats.update({
            k: v for k, v in metrics.items()
            if k not in ["error", "liquidation_nette", "enrich_net"]
        })

        if not metrics.get("is_acceptable", False):
            ind.fitness = 0.2
            ind.is_valid = False
            return

        ind.fitness = metrics.get("fitness", 0.01)
        ind.is_valid = True
        ind.stats["qual_score"] = metrics.get("qual_score", 50.0)



    def _individual_to_strategy(self, ind: Individual) -> dict[str, Any]:
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
            "dscr_y1": bilan.get("dscr_y1", 0.0),
            "qual_score": ind.stats.get("qual_score", 50.0),
        }

        # Add CF metrics
        if "cf_year_1_monthly" in ind.stats:
            s["cf_monthly_y1"] = ind.stats["cf_year_1_monthly"]
            s["cf_monthly_avg"] = ind.stats["cf_avg_5y_monthly"]

        return s

    def evolve(
        self,
        all_bricks: list[dict[str, Any]],
        budget: float,
        target_cf: float,
        tolerance: float,
        horizon: int = 20,
        top_n: int = 100
    ) -> list[dict[str, Any]]:
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
        stagnation_counter = 0

        # Evolution Loop
        for gen in range(self.generations):
            # 1. Evaluate
            for ind in population:
                self._evaluate(ind, budget, target_cf, tolerance, horizon)

            # 2. Sort
            population.sort(key=lambda x: x.fitness, reverse=True)

            # Log progress
            # Log progress
            best = population[0]
            if best.fitness > best_fitness:
                best_fitness = best.fitness
                stagnation_counter = 0
                log.debug("new_best_fitness", gen=gen, fitness=f"{best_fitness:.2f}")
            else:
                stagnation_counter += 1

            # Stagnation Stop (Early Exit)
            if stagnation_counter >= 5 and gen > 10:
                log.info("ga_stagnation_stop", gen=gen, best_fitness=best_fitness)
                break

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

    def _select_tournament(self, population: list[Individual], k: int = 3) -> Individual:
        """Select best individual from random tournament."""
        contestants = random.sample(population, min(len(population), k))
        return max(contestants, key=lambda ind: ind.fitness)

    def _brick_key(self, b: dict[str, Any]) -> str:
        """Generate unique key for a brick (property + loan duration)."""
        return f"{b['nom_bien']}_{b.get('duree_pret', 20)}"

    def _crossover(self, p1: Individual, p2: Individual) -> Individual:
        """Uniform crossover: take properties from both parents."""
        # Merge unique bricks using proper key
        pool = {self._brick_key(b): b for b in p1.bricks + p2.bricks}
        all_unique = list(pool.values())

        # Randomly select a subset to form a child
        # Try to maintain average length, capped by max_properties
        if not p1.bricks and not p2.bricks:
            avg_len = 0
        else:
            avg_len = int((len(p1.bricks) + len(p2.bricks)) / 2)
            if avg_len == 0:
                avg_len = 1

        target_len = min(avg_len, self.max_properties)
        child_bricks = random.sample(all_unique, k=min(len(all_unique), target_len))

        return Individual(child_bricks)

    def _mutate(self, ind: Individual, all_bricks: list[dict[str, Any]], budget: float):
        """Randomly add/remove/swap properties."""
        if random.random() > self.mutation_rate:
            return

        action = random.choice(["add", "remove", "swap"])

        if action == "remove" and len(ind.bricks) > 1:
            ind.bricks.pop(random.randrange(len(ind.bricks)))

        elif action == "add" and len(ind.bricks) < self.max_properties:
            # Add a random brick if budget allows (roughly)
            # Precise budget check happens in eval, here we just try
            candidate = random.choice(all_bricks)
            # Avoid dupes using proper key
            cand_key = self._brick_key(candidate)
            if cand_key not in [self._brick_key(b) for b in ind.bricks]:
                ind.bricks.append(candidate)

        elif action == "swap" and len(ind.bricks) > 0:
            ind.bricks.pop(random.randrange(len(ind.bricks)))
            candidate = random.choice(all_bricks)
            cand_key = self._brick_key(candidate)
            if cand_key not in [self._brick_key(b) for b in ind.bricks]:
                ind.bricks.append(candidate)

        ind.fitness = -1.0 # Invalidate fitness

    def _generate_biased_individual(
        self,
        sorted_bricks: list[dict[str, Any]],
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
            if len(selected) >= self.max_properties:
                break
            cost = b.get("apport_min", 0.0)
            if current_cost + cost <= budget:
                selected.append(b)
                current_cost += cost

            # Stop randomly
            if current_cost > budget * 0.9 and random.random() < 0.5:
                break

        return Individual(selected)

class ExhaustiveOptimizer:
    """
    Deterministic Brute Force Optimizer for small search spaces.
    Uses parallel processing for large search spaces.
    """

    # Threshold for enabling parallel processing
    PARALLEL_THRESHOLD = 5000  # Use parallel if > 5K combos

    def __init__(
        self,
        allocator: PortfolioAllocator,
        simulator: SimulationEngine,
        scorer: Any,
        n_workers: int = None,  # None = auto-detect CPU count
    ):
        """Initialize ExhaustiveOptimizer.

        Args:
            allocator: PortfolioAllocator for capital distribution
            simulator: SimulationEngine for financial projections
            scorer: StrategyScorer for ranking results
            n_workers: Number of parallel workers (None = auto-detect)
        """
        self.allocator = allocator
        self.simulator = simulator
        self.scorer = scorer
        self.n_workers = n_workers

        # Create evaluator for shared evaluation logic
        self.evaluator = StrategyEvaluator(simulator, allocator, scorer)

    def _evaluate_combo(
        self,
        combo: tuple,
        budget: float,
        target_cf: float,
        tolerance: float,
        horizon: int
    ) -> dict[str, Any] | None:
        """Evaluate a single combination. Returns strategy dict or None if invalid."""
        # 1. Allocation
        ok, details, cf_final, apport_used = self.allocator.allocate(
            list(combo), budget, target_cf, tolerance
        )

        if not ok:
            return None

        # 2. Evaluate with StrategyEvaluator
        strategy = {
            "details": details,
            "apport_total": apport_used,
            "cash_flow_final": cf_final,
            "allocation_ok": ok
        }

        is_valid, metrics = self.evaluator.evaluate(strategy, target_cf, tolerance, horizon)

        if "error" in metrics:
            return None

        # Merge metrics
        strategy["liquidation_nette"] = metrics.get("liquidation_nette", 0.0)
        strategy["enrich_net"] = metrics.get("enrich_net", 0.0)
        strategy["tri_annuel"] = metrics.get("tri_annuel", 0.0)
        strategy["tri_global"] = metrics.get("tri_annuel", 0.0)
        strategy["dscr_y1"] = metrics.get("dscr_y1", 0.0)
        strategy["cf_monthly_y1"] = metrics.get("cf_monthly_y1", 0.0)
        strategy["cf_monthly_avg"] = metrics.get("cf_monthly_avg", 0.0)
        strategy["is_acceptable"] = metrics.get("is_acceptable", False)
        strategy["fitness"] = metrics.get("fitness", 0.01)
        strategy["qual_score"] = metrics.get("qual_score", 50.0)

        return strategy

    def solve(
        self,
        all_bricks: list[dict[str, Any]],
        budget: float,
        target_cf: float,
        tolerance: float,
        horizon: int = 20,
        max_combinations: int = 500000,  # Reserved for Genetic mode fallback
        max_props: int = 3,
        progress_callback: Any = None
    ) -> list[dict[str, Any]]:
        """Run exhaustive search using smart bounded enumeration with parallel processing.

        Args:
            progress_callback: Optional callback function(SearchProgress) for UI updates
        """

        # Use CombinationGenerator with budget-aware pruning
        from src.services.strategy_finder import CombinationGenerator

        combo_gen = CombinationGenerator(max_properties=max_props)
        combos = combo_gen.generate(all_bricks, budget)

        n_combos = len(combos)

        log.info("exhaustive_search_started",
                 bricks=len(all_bricks),
                 max_props=max_props,
                 combos_after_pruning=n_combos)

        # Decide between sequential and parallel
        use_parallel = n_combos > self.PARALLEL_THRESHOLD

        if use_parallel:
            candidates = self._solve_parallel(combos, budget, target_cf, tolerance, horizon, progress_callback)
        else:
            candidates = self._solve_sequential(combos, budget, target_cf, tolerance, horizon, progress_callback)

        log.info("exhaustive_search_finished", found=len(candidates))
        return candidates

    def _solve_sequential(
        self,
        combos: list[tuple],
        budget: float,
        target_cf: float,
        tolerance: float,
        horizon: int,
        progress_callback: Any = None
    ) -> list[dict[str, Any]]:
        """Sequential evaluation for small search spaces."""
        from src.ui.progress import SearchPhase, SearchProgress

        candidates = []
        n_combos = len(combos)
        report_interval = max(1, n_combos // 100)  # Report every 1%

        for i, combo in enumerate(combos):
            # Report progress periodically
            if progress_callback and i % report_interval == 0:
                progress_callback(SearchProgress(
                    phase=SearchPhase.EVALUATION,
                    phase_index=3,
                    items_processed=i,
                    items_total=n_combos,
                    valid_count=len(candidates),
                ))

            result = self._evaluate_combo(combo, budget, target_cf, tolerance, horizon)
            if result is not None:
                candidates.append(result)

        # Final progress update
        if progress_callback:
            progress_callback(SearchProgress(
                phase=SearchPhase.EVALUATION,
                phase_index=3,
                items_processed=n_combos,
                items_total=n_combos,
                valid_count=len(candidates),
            ))

        return candidates

    def _solve_parallel(
        self,
        combos: list[tuple],
        budget: float,
        target_cf: float,
        tolerance: float,
        horizon: int,
        progress_callback: Any = None
    ) -> list[dict[str, Any]]:
        """Parallel evaluation for large search spaces using ThreadPoolExecutor.

        Uses threading for parallel processing. While GIL limits true parallelism,
        this still provides speedup for I/O-bound operations.
        """
        import concurrent.futures
        import os

        from src.ui.progress import SearchPhase, SearchProgress

        n_workers = self.n_workers or min(8, (os.cpu_count() or 4))
        n_combos = len(combos)

        log.info("parallel_processing_started",
                 workers=n_workers,
                 combos=n_combos,
                 mode="threaded")

        candidates = []
        processed_count = 0
        report_interval = max(1, n_combos // 100)  # Report every 1%

        # Time-based throttling for UI updates
        import time
        last_report_time = time.time()
        min_report_interval_sec = 0.5  # Update UI at most every 0.5 seconds

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                # Submit all combos for evaluation
                future_to_combo = {
                    executor.submit(
                        self._evaluate_combo,
                        combo, budget, target_cf, tolerance, horizon
                    ): combo
                    for combo in combos
                }

                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_combo):
                    try:
                        result = future.result()
                        if result is not None:
                            candidates.append(result)
                    except Exception as e:
                        log.warning("combo_evaluation_failed", error=str(e))

                    # Report progress with time-based throttling
                    processed_count += 1
                    current_time = time.time()
                    if (progress_callback and
                        processed_count % report_interval == 0 and
                        current_time - last_report_time >= min_report_interval_sec):
                        progress_callback(SearchProgress(
                            phase=SearchPhase.EVALUATION,
                            phase_index=3,
                            items_processed=processed_count,
                            items_total=n_combos,
                            valid_count=len(candidates),
                        ))
                        last_report_time = current_time

            # Final progress update
            if progress_callback:
                progress_callback(SearchProgress(
                    phase=SearchPhase.EVALUATION,
                    phase_index=3,
                    items_processed=n_combos,
                    items_total=n_combos,
                    valid_count=len(candidates),
                ))

            log.info("parallel_processing_finished", found=len(candidates))

        except Exception as e:
            log.warning("parallel_processing_error", error=str(e))
            # Fallback to sequential
            log.info("falling_back_to_sequential")
            return self._solve_sequential(combos, budget, target_cf, tolerance, horizon, progress_callback)

        return candidates

