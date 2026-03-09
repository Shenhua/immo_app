# Architecture Decision Record (ADR): Dual-Optimizer Strategy (Exhaustive vs. Genetic)

**Status:** Accepted
**Date:** 2026-03-06
**Author:** AI Agent

## 1. Context
The core capability of the App Immo Simulation Engine relies on evaluating long-term cash flow, DSCR scores, and IRR calculations across wide arrays of property permutations. Historically, a brute-force exhaustive search iterator mapped all potential portfolio combinations. However, as the `ArchetypeV2` structures expanded the variable space, calculations extending beyond **1,000,000 theoretical combinations** (the `THRESHOLD_EXHAUSTIVE` constant in `strategy_finder.py`) became computationally intractable within Streamlit's synchronous rendering model. Continuing to use a strictly exhaustive strategy blocked real-time iterative rendering updates on the user-facing Streamlit interface due to timeout constraints.

## 2. Decision
We implement a dual-mode optimizer pattern within the application services layer:
1.  **ExhaustiveOptimizer:** An iterator-based, memory-efficient absolute search for small to moderate search spaces (below `THRESHOLD_EXHAUSTIVE = 1,000,000` theoretical combinations). Uses `ThreadPoolExecutor` above a secondary threshold of ~5,000 combinations for parallel brick evaluation. Guarantees the mathematical global optimum within the explored space.
2.  **GeneticOptimizer:** An Evolutionary Algorithm that engages when the search space exceeds `THRESHOLD_EXHAUSTIVE`. Key design details:
    *   **No concurrent thread pooling** — the GA evolution loop evaluates each individual sequentially per generation.
    *   **Smart-Seeded Initial Population:** Initial population of `population_size=50` is biased: 30% high-yield seeded, 20% high-quality seeded, 20% lowest-cost seeded, 30% pure random. This ensures the starting gene pool is not purely random trash.
    *   **Selection & Reproduction:** Tournament selection (k=3 contestants), crossover (`crossover_rate=0.7`), mutation (`mutation_rate=0.2`), elitism (`elite_size=5` top individuals preserved per generation).
    *   **Stagnation Early-Stop:** If best fitness does not improve for 5 consecutive generations AND generation > 10, the evolution loop exits early. This is the actual "circuit breaker" mechanism.
    *   **Generations:** Default `generations=20`. Maximum theoretical evaluations: `pop_size × generations = 50 × 20 = 1,000` individuals.

## 3. Alternative Options Considered
*   **Option A:** Offload exhaustive compute into an asynchronous queueing system using Celery and Redis.
    *   *Pros:* Offloads main web thread; could scale computational horizontal workers.
    *   *Cons:* Fails to constrain the actual compute cost/memory consumption, introduces significant system latency disrupting the immediate real-time interactive user experience, and escalates architectural surface complexity.
*   **Option B:** Pre-filter archetypes using heuristic qualitative culling before pushing variables into the combinations matrix.
    *   *Pros:* Trivially decreases array lengths reaching the optimizer module.
    *   *Cons:* Establishes statistical theoretical blindness—properties with poor raw DPE heuristic values might harbor exceptionally leveraged financial edge cases due to external subsidies, which would incorrectly be filtered prior to mathematical proof processing.

## 4. Consequences
*   **Positive:** UI responsiveness and computation time remain predictable for large configuration loads. The stagnation early-stop (`stagnation_counter >= 5`, after generation 10) prevents runaway computation. The smart-seeded initial population significantly improves convergence speed vs. pure random initialization.
*   **Negative:** When the Genetic Optimizer engages on massive datasets, results yield a "good enough" local optimum along the Pareto boundary — the engine loses the guarantee of surfacing the definitive global mathematical optimum. The quality of results is sensitive to the `population_size`, `generations`, `mutation_rate`, and `crossover_rate` parameters. These should be treated as tunable levers and documented if changed from their defaults.
