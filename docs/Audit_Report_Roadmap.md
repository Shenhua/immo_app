# Audit Report & Transformation Roadmap: `app-immo`

## 1. Project Identity Card
* **Project Purpose:** A sophisticated Real Estate Investment Strategy Simulator that helps investors find optimal portfolios of properties by combinatorically analyzing property "bricks" against financial goals (cash flow, yield, wealth creation).
* **Current Stack:** 
    * **Language:** Python 3.10+
    * **Frameworks:** Streamlit (UI)
    * **Data/Math:** Pandas, Numpy, Numpy-financial, Plotly
    * **Quality/DevOps:** Pydantic (Validation), Structlog (Logging), Pytest (Testing), Ruff (Linting), Mypy (Types), Docker.
* **Architecture Overview:** 
    * **Flow:** Raw Archetype Data → `BrickFactory` (Financing) → `StrategyFinder` (Search Orchestration) → `ExhaustiveOptimizer` (Combinations) → `PortfolioAllocator` (Capital Distribution) → `SimulationEngine` (Long-term Projections) → `StrategyScorer` (Ranking) → Streamlit UI.
    * **State Management:** Uses a `SessionManager` to bridge Streamlit's session state with the backend.

---

## 2. Technical Health Report

### Critical Issues
* **Dead Code / Logic Bypass:** In `src/services/strategy_finder.py`, the Genetic Algorithm (GA) is effectively disabled via a hardcoded `if True` (line 362), despite significant investment in the `GeneticOptimizer` class.
* **Logic Duplication:** `GeneticOptimizer` contains its own `_calculate_absolute_finance_score` which duplicates logic in `StrategyEvaluator.calculate_finance_score`. This increases the risk of "score drift" where different solvers produce different results for the same strategy.
* **Weak Typing ("Dict-Driven Design"):** Despite using Pydantic, core services like the `PortfolioAllocator` and `ExhaustiveOptimizer` pass around `dict[str, Any]` for properties and results. This bypasses compile-time checks and makes the codebase brittle to schema changes.
* **Mixed Responsibilities:** `app.py` and `ui/app_controller.py` mix Streamlit UI concerns with business logic orchestration (e.g., building configurations, running simulations).

### Dependency Risks
* **Numpy-Financial:** Reliable, but should be monitored for compatibility with future `numpy` versions.
* **Streamlit Versioning:** The project is pinned to `streamlit<3.0.0`. While safe now, the rapid evolution of Streamlit's API might require a migration strategy soon.

### "Spaghetti Code" Areas
* **`PortfolioAllocator`:** The allocation logic is a complex iterative greedy loop. While robust, it's highly imperative and hard to extend (e.g., adding "Max Debt" as a primary constraint).
* **`StrategyScorer`:** Splitting scoring between `StrategyScorer` and `StrategyEvaluator` makes it unclear where "fitness" actually lives.

### Missing Standards
* **Error Handling:** Relies heavily on "Success Booleans" (`ok, details = ...`) rather than structured exceptions, leading to silent failures or verbose check-if-ok blocks.
* **Type Strictness:** `mypy` is enabled but many core modules are excluded or rely on `Any`.

---

## 3. The Refactoring Blueprint (The "Solid" Plan)

### Proposed Architecture: Clean Hexagonal
We will move towards a structure that separates **Domain Primitives** from **Application Orchestration**.

### Directory Structure
```
src/
├── domain/               # PURE LOGIC (No dependencies)
│   ├── models/           # Strict Pydantic Models (Brick, Strategy, Bilan)
│   ├── calculator/       # Core math (Loan, Tax, CashFlow primitives)
│   └── rules/            # Scoring & Eligibility rules
├── application/          # ORCHESTRATION (Services)
│   ├── services/         # Optimizer, Simulator, Allocator
│   └── handlers/         # Use cases (e.g., RunFullAnalysis)
├── infrastructure/       # EXTERNAL SYSTEMS
│   ├── persistence/      # Archetype JSON/CSV Loaders
│   └── logging/          # Structlog Configuration
└── ui/                   # PRESENTATION
    ├── components/       # Reusable Streamlit widgets
    └── app.py            # Clean entry point (UI only)
```

### Tech Stack Updates
* **Enforce Pydantic V2:** Fully migrate all `dict` usage to Pydantic models with `model_validate`.
* **Structured Exceptions:** Replace `bool` returns with a hierarchy of `ImmoError` classes.

### Best Practices to Implement
* **Common Response Pattern:** All services return a unified `Result[T, E]` or raise structured exceptions.
* **Testing Strategy:** 
    * **Property-Based Testing (Hypothesis):** Use for `PortfolioAllocator` to ensure it never violates budget constraints regardless of input.
    * **Golden Fixtures:** Maintain the `test_golden_validation.py` to prevent regression in financial math.
* **Type Safety:** Set `disallow_untyped_defs = true` for the `domain/` and `application/` layers.

---

## 4. Master Development Plan

### Phase 0: Cleanup & Pruning
* [ ] Remove/Archive `GeneticOptimizer` if Exhaustive is the permanent choice, or restore it correctly.
* [ ] Consolidate scoring logic into a single `ScoringService`.
* [ ] Remove unused `legacy/` fragments or "Zombie" functions in `allocator.py`.

### Phase 1: Domain Hardening (Stabilization)
* [ ] Convert all `dict[str, Any]` signatures to Pydantic models (`InvestmentBrick`, `PortfolioStrategy`).
* [ ] Implement structured error handling (e.g., `AllocationError`, `SimulationError`).
* [ ] Fix the `if True` bypass in `StrategyFinder`.

### Phase 2: Structural Refactoring (Modularization)
* [ ] Move core math from `services/` to `domain/calculator/`.
* [ ] Isolate `SimulationEngine` from UI components.
* [ ] Extract `AppController` into a pure `ApplicationService` that doesn't use `st.session_state` directly.

### Phase 3: Standardization & Polish
* [ ] Enforce strict typing in `pyproject.toml` for `src/domain` and `src/application`.
* [ ] Add `structlog` context managers for better trace-ability of simulations.
* [ ] Implement a `Unit` vs `Integration` test split in the CI pipeline.
