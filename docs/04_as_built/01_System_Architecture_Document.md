# System Architecture Document (As-Built): App Immo

**Document Status:** Living Document
**Last Updated:** 2026-03-06
**Author:** AI Assistant

## 1. Executive Summary
The Real Estate Investment Simulation Engine (App Immo) is a specialized financial modeling application designed to evaluate and optimize property portfolio combinations. The system simulates long-term cash flows, complex localized taxing mechanics (including LMNP and micro-bic), loan amortizations, and market heuristics to identify the most efficient investment property permutations for the end user.

## 2. Global Architecture & Stack
**Architecture Pattern:** Modular Monolith leveraging Domain-Driven Design (DDD), segregating Domain Models, Application Services, Core Utilities, and UI Presentation layers.

**Technology Stack:**
* **Language/Runtimes:** Python 3.10+ (tested on 3.10/3.11/3.12)
* **Frontend:** Streamlit
* **Data Layer / Typings:** Pydantic v2 (BaseModel, computed_field, field_validator)
* **Math & Financial Processing:** `numpy`, `numpy_financial` (IRR, PMT), `pandas`
* **Charts:** Plotly (`plotly.express`, `plotly.graph_objects`)
* **Logging:** `structlog` (JSON + console)
* **Concurrency:** `concurrent.futures.ThreadPoolExecutor`
* **QA & Telemetry:** `pytest`, `pytest-cov`, `mypy`, `ruff`, `ast`

## 3. Component Interaction & Workflows
The macro-level data flow through the system is orchestrated primarily by `app.py` and `ui/app_controller.py`:
1. **User Input / Initialization:** Flow starts at `app.py`. The `SessionManager` initializes the Streamlit global session state. `render_sidebar()` collects user parameters (credit, objectives, scoring).
2. **Domain Mapping & Filtering:** Property data is filtered from `data/archetypes_recale_2025_v2.json`. The controller maps user inputs into strict domain models (`SimulationParams`, `ArchetypeV2`).
3. **Generation:** `brick_factory.py` synthesizes generic archetypes into investable entities (`InvestmentBrick`).
4. **Optimization & Simulation:** The strategy search is orchestrated by `optimizer.py` via `strategy_finder.py`. For small search spaces (< 1M combinations), `ExhaustiveOptimizer` uses `ThreadPoolExecutor` for parallel evaluation. For large spaces (≥ 1M combinations), `GeneticOptimizer` runs a **sequential** evolution loop (no threading). For each candidate, `allocator.py` optimizes down payments to satisfy cash flow targets. Deep tax and amortization schedules are computed via `simulation.py` and pure functions in `domain/calculator/`.
5. **Evaluation:** Strategies are filtered via Pareto-efficiency algorithms (`strategy_finder.py`), scored (`scoring.py`), deduplicated, and ranked.
6. **Result Presentation:** Final `StrategyResult` models hydrate the Streamlit session state, rendering comparative graphs, KPI scorecards, and metric tables on the main page.

## 4. Global Data Dictionary
* **`ArchetypeV2`**: Immutable property template containing foundational characteristics (surface, price/m², rent/m², DPE, scores). Defined in `src/domain/models/archetype.py`.
* **`InvestmentBrick`**: Domain model representing a uniquely configured property variation including computed financing overlays (total cost, PMT total, initial monthly cash flow). Defined in `src/domain/models/brick.py`.
* **`PortfolioStrategy`**: Aggregate model representing a multi-property combination and its aggregated long-term financial trajectory. Defined in `src/domain/models/strategy.py`. Results are serialized as `list[dict]` (not typed instances) throughout the pipeline.
* **`SimulationParams`** / **`TaxParams`**: Models governing algorithm parameters and simulation constants. `SimulationParams` in `src/domain/models/strategy.py`; `TaxParams` in `src/application/services/simulation.py`.
* **`StrategyResult`**: Output model for simulation results and KPIs. Defined in `src/domain/models/strategy.py`.
* **`SearchStats`** / **`SearchProgress`**: UI-layer telemetry dataclasses tracking search execution state. Defined in `src/ui/progress.py` — **not** a core domain model.
* **`CashflowMetrics`**: A `TypedDict` (not a class) in `src/core/glossary.py` defining the standardized output keys for cash flow metrics (DSCR, IRR, net enrichment).

## 5. Detailed Module Breakdown

### `src/core/`
* **Purpose:** Supplies unified, cross-cutting infrastructure. It has no dependencies on the domain layer.
* **Core Logic:** Standardizes exceptions (`exceptions.py`), sets up dynamic formatted logging (`logging.py`), acts as the single source of truth for canonical financial formulas (`glossary.py`), and provides debugging utilities (`debug.py`).

### `src/domain/`
* **Purpose:** Enforces strict formal schemas, bounds, and pure stateless mathematical transformations. Contains no Streamlit or I/O operations.
* **Core Logic:** Uses strictly typed Pydantic models (`models/`). Implements pure calculator patterns (`calculator/`) for financial math (`financial.py`) and standard heuristic computations (`scoring.py`).

### `src/application/`
* **Purpose:** Acts as the use-case layer and core financial computation engine.
* **Core Logic:** Driven by `services/`. It synthesizes raw inputs (`brick_factory.py`), executes long-term logic and mortgage scheduling (`simulation.py`), solves down payment thresholds iteratively (`allocator.py`), and generates multi-million portfolio scenarios (`optimizer.py` and `strategy_finder.py`) through concurrency (Exhaustive) or sequential evolutionary loops (Genetic).

### `src/ui/`
* **Purpose:** Streamlit-specific presentation layer managing interactive components and state mapping.
* **Core Logic:** Orchestrates the app via `app_controller.py`. Session state is managed in `state.py`. UI is decomposed into:
  * `components/` — reusable rendering components: `sidebar.py`, `charts.py`, `results.py`, `progress_display.py`, `filters.py`, `sensitivity.py`
  * `pages/` — multi-page Streamlit page definitions
  * `helpers.py` — shared UI utility functions
  * `progress.py` — `SearchStats` and `SearchProgress` telemetry dataclasses

### `src/utils/`
* **Purpose:** Helper utilities and secondary scripts.
* **Core Logic:** Standalone diagnostic functionality alongside generic project support functions.

## 6. External Integrations
* **Local Data Ingestion:** Relies on local JSON data stores (e.g., `data/archetypes_recale_2025_v2.json`) rather than external databases.
* **Third Party APIs:** No external, latency-bound third-party operational APIs (e.g., Stripe, AWS) are deeply integrated at the core engine level; the system runs computationally localized simulations.

## 7. Known Defects & Technical Debt Register

**High Severity**
* **`allocator.py` [Application Services]:** Conducts fragile exception-based parameter probing (`_get_val()`), utilizing repeated `TypeError` and `KeyError` catches instead of rigorous Pydantic typing.
* **`components/sensitivity.py` [UI Presentation]:** Violates architectural bounds by duplicating system objects and triggering ad-hoc background financial simulations (`simulate_long_term_strategy()`) directly inside the frontend render loop. (`src/ui/components/sensitivity.py`)
* **`app_controller.py` [UI Presentation]:** Explicitly suppresses core engine exceptions during `simulate_selected_strategy`, leaving a silent state failure rather than propagating errors to the UI.
* **`financial.py` [Domain Layer]:** Credit math routines mask zero or negative property limits without raising validation errors, directly passing `0.0` or empty iterators.

**Medium Severity**
* **`optimizer.py` [Application Services]:** Operations within the `GeneticOptimizer` use destructive mutation methods (e.g., `.pop()`), truncating ancestral tracking of the portfolio variants.
* **`strategy_finder.py` [Application Services]:** Drops filtered strategy arrays dynamically during Pareto sweeps without emitting discarded configuration matrices to system loggers, hiding context.
* **`debug.py` [Core & Utils]:** Relies on generic exception blocking to continue execution and uses unreliable array slicers to fetch executing traces.
* **`brick.py` [Domain Layer]:** Contains backwards-compatible hacks overriding built-in `__getitem__` logic into a static class, blurring model consistency.

**Low Severity**
* **`health_check.py`:** This module **does not yet exist**. It is planned but not implemented — see QA Strategy doc §4.1 for the intended design.
* **`app.py` [App Entry]:** Hardcoded application labels (`v27.6`) and magic limits (`top_n=50`) rather than using environment variables.
* **`glossary.py` / `scoring.py` [Core/Domain]:** Multiple hardcoded values for heuristics (DPE baseline `0.6`, Max DSCR `10.0`).
* **`helpers.py` [UI Presentation]:** Embeds raw HTML tag injection directly into global script execution definitions.

## 8. Strategic Enhancement Roadmap
1. **Rigorous Domain Validation Enforcement:** Remove exception-based structural guesswork in `allocator.py` and legacy extraction wrappers in `brick.py`. Enforce explicit `InvestmentBrick` instance typing end-to-end.
2. **Hard Abstraction of Controller Boundaries:** Purge physical calculation iterations from `sensitivity.py`. Re-route ad-hoc model permutations through asynchronous service triggers in `app_controller.py` to ensure read-only UI mappings.
3. **Structured Failure Architecture:** Eliminate generalized `except Exception: pass` implementations in `app_controller.py` and `debug.py`. Force computational failures to wrap into specific `AppImmoError` subclasses for proper UI context prompts.
4. **Extraction of Subsystem Variables:** Transition internal limits, heuristic constants (e.g., `DEFAULT_QUAL_WEIGHTS`, DSCR limits), and algorithm configurations into external environment files or configurations. Expose Pareto-filtration outcomes via `structlog` to provide transparency on rejected strategy matrices.
