# Product Requirements Document (PRD): App Immo - Real Estate Investment Simulator

**Document Status:** Approved
**Author:** AI Technical Lead
**Date:** 2026-03-06

## 1. Overview & Goals
### 1.1 Problem Statement
Real estate investors struggle to rapidly evaluate complex, multi-property portfolios. Calculating 25-year financial simulations—factoring in specific French tax structures (e.g., LMNP, micro-bic), loan amortizations, and dynamic down payments across potentially millions of permutations—is mathematically intensive and error-prone when attempted manually via static spreadsheets.

### 1.2 Target Audience
* **Primary:** Retail real estate investors in the French market seeking highly optimized multi-unit strategies.
* **Secondary:** Wealth managers or financial advisors simulating tax/amortization outcomes for client portfolios.

### 1.3 Core Value Proposition
App Immo is a real-time, algorithmic simulation engine that ingests generic property archetypes and instantly generates, mathematically optimizes, and scores long-term, cash-flow-positive real estate portfolios.

### 1.4 Success Metrics (KPIs)
* **Algorithmic Efficiency:** The engine must optimize spaces of up to 100,000+ permutations and return top-tier results in under 15 seconds.
* **Math Fidelity:** Generate 100% accurate 25-year amortization and tax schedules validated by comprehensive unit tests mapping zero/negative bounds.
* **UX Clarity:** Present complex metrics (Internal Rate of Return [IRR], Debt Service Coverage Ratio [DSCR]) in an interactive, visually immediate UI that requires no technical scripting to operate.

---

## 2. Scope
### 2.1 In Scope
* **Data Ingestion:** Sourcing JSON-based property archetypes representing qualitative fields (DPE, Market Tension).
* **Interactive Frontend:** A robust Streamlit UI separating parameters into a clean sidebar configuration panel.
* **Dual Engines:** Implementation of Exhaustive Optimizer (for spaces below 1,000,000 theoretical combinations) and a Genetic Algorithm Optimizer (Tournament, Crossover, Mutation, Elitism with adaptive population size) for massive state spaces exceeding this threshold.
* **Smart Allocation:** Iterative greedy algorithm dynamically recalculating loan down payments to strictly satisfy minimum cash flow thresholds.
* **Granular Finance Modeling:** 25-year tracking of rent inflation, appreciation, and localized taxation mechanics (LMNP, Micro-bic, SCI IS, Exit Tax).
* **Visualization:** Interactive metric scorecards and Plotly-based waterfall/line charts evaluating performance. A configurable `top_n` parameter (range 10–500, default 50) limits the number of strategies displayed after optimization.

### 2.2 Out of Scope
* Direct external API integrations parsing dynamic listings from property sites (e.g., SeLoger, Leboncoin).
* Cloud persistence of User Accounts (portfolios are session-based or local-disk exports only).
* Mobile application deployments (product is web-optimized).

---

## 3. User Journeys & Functional Requirements

### Scenario 1: Running an Algorithmic Strategy Search
1. **Given:** The user accesses the Streamlit web application.
2. **When:** The user enters a maximum investment budget, selects preferred real estate generic `ArchetypeV2` templates (e.g., Studio, T2) via the sidebar, and adjusts global target limits such as minimum DSCR.
3. **Then:** The `strategy_finder` orchestrator delegates configuration tracking. Based on permutation limits, it routes calculations to the Genetic or Exhaustive sub-engines. A decoupled telemetry object transmits the `SearchProgress` UI loaders.
4. **Data Required:** Budget (Float), Property Limit (Integer), DSCR (Float).
5. **Expected Output:** The main page cascades to display a ranked list of `PortfolioStrategy` combinations, discarding dominated structures via a Pareto efficiency filter out of the UI view.

### Scenario 2: Deep-Dive Simulation Evaluation
1. **Given:** The user reviews their top heuristically-ranked strategies.
2. **When:** The user explicitly toggles a specific `PortfolioStrategy` card.
3. **Then:** The `app_controller` executes a `simulate_selected_strategy` backend request mapping full 25-year `CashflowMetrics`.
4. **Expected Output:** The UI renders dedicated interactive components (via `charts.py` and `results.py`) displaying localized cash flows, loan debt burn-down, enrichment curves, and tax burden visualizations.

---

## 4. Business Logic & Constraints
* **DSCR Validation Constraint:** `DSCR = Net Operating Income / Total Debt Service`. The system tracks DSCR (`dscr_y1`) as a **scoring metric** in the weighted fitness function (weight: 0.15 by default), and normalizes it in the 0–2.5 range for scoring. It is not enforced as a hard binary gate that blocks validation — strategies with low DSCR receive lower scores rather than being discarded.
* **Cash Flow Constraint / Allocator:** During generation loops, the system's allocator *must* aggressively scale upfront capital investment inside bounds (e.g. `MAX_EXTRA_APPORT_PCT = 0.95`) targeting positive continuous initial cash flow. Unresolved combinations must be discarded prior to scoring.
* **Pareto Dominance Principle:** Strategies demonstrating strictly identical or worse risks than peer sets while returning lower yields are labeled "dominated." These are computationally bypassed to reduce user friction.
* **Taxation Execution:** Must strictly model French specificities; notably calculating LMNP (Loueur en Meublé Non Professionnel) depreciation caps and amortizations against the `cout_total` dynamically yearly. The system also supports SCI IS (Société Civile Immobilière à l'Impôt sur les Sociétés) as a selectable tax regime from the UI sidebar (`regime_fiscal` parameter).

---

## 5. Non-Functional Requirements
* **Concurrency:** The `ExhaustiveOptimizer` in `optimizer.py` uses `concurrent.futures.ThreadPoolExecutor` (thread-based) for parallel combo evaluation, activating above a 5,000-combination threshold (`PARALLEL_THRESHOLD = 5000`). Workers auto-scale up to `min(8, cpu_count)`. Note: thread-based parallelism is limited by Python's GIL for pure CPU operations; it provides speedup on mixed I/O+compute workloads. The `GeneticOptimizer` evaluates sequentially within its evolution loop.
* **Layer Isolation:** Absolute strict separation between pure math functions (`domain_layer`) and view components (`ui_presentation`). The pure domain architecture (`pydantic` entities, `numpy_financial` math) is strictly prohibited from executing Streamlit DOM imports.
* **Quality Gates:** Core calculation pipelines (`src/core/glossary.py`, `src/domain/calculator`) require high test isolation verifying pure stateless financial math with isolated `pytest` regression validations against explicit edge cases.
