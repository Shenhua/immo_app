# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

App Immo is a French real estate investment strategy simulator built with Streamlit. It generates property portfolio combinations from archetypes, runs 25-year financial simulations (IRR, DSCR, cash flow), scores them, and ranks results. Version is tracked in `src/__init__.py`.

## Tech Stack

- **Language**: Python 3.10+ (tested on 3.10/3.11/3.12)
- **Web Framework**: Streamlit
- **Data Models**: Pydantic v2 (BaseModel, computed_field, field_validator)
- **Finance**: numpy-financial (IRR, PMT), numpy, pandas
- **Charts**: Plotly
- **Logging**: structlog (JSON + console)
- **Linting**: Ruff (line-length=120, target py310)
- **Type Checking**: mypy (strict on models, gradual on core)
- **Testing**: pytest + pytest-cov

## Commands

```bash
# Run app
streamlit run app.py

# Install dev deps
pip install -e ".[dev]"

# Tests
pytest tests/ -v                                    # all tests
pytest tests/unit/ -q --tb=no                       # quick unit tests (pre-commit)
pytest tests/unit/test_simulation.py -v             # single file
pytest tests/unit/test_simulation.py::TestTaxParams::test_lmnp_tax_calculation -v  # single test
pytest -k "test_lmnp" -v                            # by keyword
pytest tests/ -v --tb=short --cov=src --cov-report=xml  # with coverage

# Lint & format
ruff check .              # lint
ruff check . --fix        # lint + autofix
ruff format .             # format

# Type check
mypy src/
```

## Architecture

### Layered Structure

```
src/
├── core/           # Cross-cutting: exceptions, glossary (canonical formulas), logging
├── domain/
│   ├── models/     # Pydantic models: ArchetypeV2, InvestmentBrick, PortfolioStrategy
│   └── calculator/ # Pure functions: loan math (financial.py), scoring (scoring.py)
├── application/
│   └── services/   # Use cases: simulation, strategy search, optimization, allocation
├── ui/             # Streamlit-specific: components, pages, state management, charts
└── utils/          # Debug utilities
```

**Key rule**: `domain/` is pure (no Streamlit, no I/O). `core/` has no domain deps.

### Application Flow

`app.py` → `ui/app_controller.py` orchestrates:
1. `SessionManager.initialize()` → init Streamlit session state
2. `render_sidebar()` → collect user parameters (credit, objectives, scoring)
3. `load_archetypes()` → load and validate property data from `data/archetypes_recale_2025_v2.json`
4. `run_strategy_search()` → exhaustive or genetic optimizer (hybrid solver)
5. `simulate_selected_strategy()` → detailed 25-year projection logic
6. `render_result_page()` (in `app.py`) → display results, charts, KPIs

### Strategy Search (Two Modes)

- **ExhaustiveOptimizer**: iterator-based, memory-efficient — used for small search spaces (< 1,000,000 combinations)
- **GeneticOptimizer**: evolutionary algorithm (tournament, crossover, mutation, elitism, smart seeding) — for large spaces (≥ 1,000,000 combinations)
- Both use the pipeline: `BrickFactory` → `PortfolioAllocator` → `StrategyEvaluator`

### Key Domain Models

- **ArchetypeV2**: immutable property template (surface, price/m², rent/m², DPE, scores)
- **InvestmentBrick**: financed property with computed fields (`cout_total`, `pmt_total`, `cash_flow_mensuel_initial`); has legacy `__getitem__` for backward compat
- **PortfolioStrategy**: portfolio of bricks with aggregated scores
- **SimulationParams** / **TaxParams**: all parameters for a simulation run

### Canonical Formulas

`src/core/glossary.py` is the single source of truth for `calculate_cashflow_metrics`, `calculate_dscr_metric`, `calculate_enrichment_metrics`. Always use these, never duplicate the logic.

### Error Hierarchy

All custom exceptions inherit from `AppImmoError` in `src/core/exceptions.py`. Key subclasses: `SimulationError`, `StrategyError`, `NoStrategiesFoundError`, `AllocationError`, `InvalidParameterError`.

## Test Organization

- `tests/unit/` — 18 files, fast, isolated (no I/O). Includes golden value regression tests.
- `tests/integration/` — pipeline flow and invariant checks
- `tests/e2e/` — full pipeline with real data
- `tests/conftest.py` — shared fixtures (`sample_archetype_data`, `sample_strategy_data`)

Pre-commit runs only `tests/unit/` for speed.

## Environment Variables

```
LOGLEVEL=INFO              # DEBUG/INFO/WARNING/ERROR
MAX_EXTRA_APPORT_PCT       # Max extra capital ratio for allocator (default: 0.95)
```

## Ruff Rules

E, W, F (pyflakes), I (isort), B (bugbear), C4, UP (pyupgrade). Ignores: E501, B008, B905.
