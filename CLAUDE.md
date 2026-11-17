# CLAUDE.md

Guidance for Claude Code (and other AI agents) working in this repository.

## What this is
App Immo is a Streamlit app for French individual investors. It searches combinations of rental properties (built from city "archetypes" in `data/archetypes_2026_v1.json`) for a given budget, down payment, cash-flow target and tax setting. It then simulates 10–30 year projections (loans, rents, charges, French income/property tax, exit capital-gains tax) and ranks the strategies.

User manual: `docs/USER_GUIDE.md`. Developer overview: `README.md`.

## Commands (always use the project venv)
```bash
.venv/bin/python -m pytest -q                      # full suite (~15 s)
.venv/bin/python -m pytest -q -m ui                # Streamlit AppTest suite only
.venv/bin/python tests/golden/reference_calc.py --check   # golden tax cases up to date
.venv/bin/python -m ruff check .                   # lint (CI-blocking)
.venv/bin/python -m mypy src/                      # types (CI-blocking; strict on domain/application)
PATH=.venv/bin:$PATH pre-commit run --all-files    # hooks (mypy hook is local → needs venv on PATH)
.venv/bin/streamlit run app.py                     # run the app
```
CI gates: ruff, mypy, and coverage of at least 85 % (`--cov=src`).

## Architecture (layers, dependencies flow downward)
- `src/ui/`: Streamlit only (components, pages, `cached.py` holds the `st.cache_data` wrappers, `state.py`, `sidebar_state.py` holds the widget key ↔ RunContext mapping).
- `src/application/`: orchestration, no streamlit:
  - `run_context.py`: **`RunContext`**, the frozen pydantic model of every parameter;
  - `services/`: search (`strategy_finder.py`, `optimizer.py`, `allocator.py`), `simulation.py` (yearly engine, `build_engine`), `evaluator.py`, `scoring.py`, `sensitivity.py`, `risk.py`, `regime_comparison.py`, `report.py`, `diagnostics.py`, `brick_factory.py`;
  - `scenario_io.py` and `validation.py`.
- `src/domain/`: pure business rules: `tax/` (regime engine: LMNP réel, micro-BIC, micro-foncier, foncier réel, SCI IS; household aggregation; plus-value), `calculator/` (loan maths), `models/`, `dpe.py`, `hcsf.py`.
- `src/core/`: `defaults.py` (**every default value and rate lives here**), `glossary.py` (metric definitions), logging, exceptions.

A test enforces that `src/domain` and `src/application` never import `streamlit` or `src.ui`.

## Rules that are easy to break
- **No literals for defaults or rates in code.** Put them in `src/core/defaults.py`. The sidebar reads defaults from there; a test greps for literals.
- **Tax logic follows `docs/TAX_SPEC.md`.** It is the source of truth; the user decisions are in §11b–§11d. Cite legal sources (CGI/BOFiP/Légifrance) in comments. Never guess a tax rule: ask the user.
- **Golden cases** (`tests/golden/`) must stay green. Expected values come from the standalone `reference_calc.py`, never from the app's own code.
- **Search ↔ detail parity:** every engine-relevant `RunContext` field must reach the search engine. `tests/integration/test_runcontext_engine_parity.py` fails if you add a field without wiring it, or without listing it in `NOT_ENGINE_FIELDS` with a reason.
- **Metric conventions** (`src/core/glossary.py`):
  - acceptance and scoring use the **stabilised year = simulation year 2**, after-tax CF (`CF_ACCEPTANCE_BASIS`);
  - `enrichissement_net` includes the cumulative after-tax CF;
  - a failed IRR is `None`, shown as "n/a", never 0 %.
- **Search invariants:** budget respected, no property twice in a strategy, at most `max_properties`; results are deterministic. The property tests in `tests/property/` check these.
- **Disk writes are opt-in** through env vars (`APP_AUTOSAVE`, `APP_RESULTS_DIR`, `APP_LOG_DIR`, `APP_DATA_DIR`); tests redirect them to tmp.
- UI text is in French. Code, comments and docs are in English.
- The search uses a `ProcessPoolExecutor` for large runs (spawn on macOS), so standalone scripts that call it need an `if __name__ == "__main__":` guard.

## Project history and planning
- `docs/DEV_PLAN.md`: task list with specs (T01–T68).
- `docs/DEV_PROGRESS.md`: status, conductor notes (decisions and gotchas per task), and the log. **Read it before starting planned work.**
- `docs/AGENT_BRIEF.md`: the brief given to development sub-agents (worktree and commit rules).
- `docs/review/phase2/`: Opus review of the tax engine, with its scenario scripts.
- `docs/archive/`: historical notes, not maintained.
