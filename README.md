# App Immo: Real Estate Investment Strategy Simulator

> Version 27.6.0 · A Streamlit app that searches and ranks French rental-property investment strategies.

App Immo is a Streamlit application for French individual investors. You set your down payment (apport), your target monthly cash flow, your horizon and your tax situation. The app then searches combinations of properties from its archetype data (`data/archetypes_2026_v1.json`), simulates each strategy year by year (loans, rents, charges, taxes, exit), and ranks the strategies that meet your cash-flow objective. It is a decision-support tool for exploring trade-offs, not an advisor.

## Features

- **Strategy search** in three modes: **Classique (Levier)** (minimum down payment, maximise return), **Rentier (Max Cash-Flow)** (invest the whole down payment for the highest cash flow) and **Empire (Max Volume)** (as many properties as possible). The **Ciblage Précis** option targets a cash-flow value within a ± tolerance.
- **Five scoring profiles** (for example "Équilibré (défaut)", "Cash-flow d'abord", "Rendement / IRR", "Sécurité (DSCR)", "Patrimoine LT"). Scores use fixed benchmarks, so a strategy gets the same score in every run.
- **Tax regimes**: LMNP réel, micro-BIC, micro-foncier, foncier réel and SCI à l'IS, with one regime per income category (furnished and unfurnished). A **Comparaison fiscale** tab runs every eligible regime on the same strategy.
- **HCSF debt-ratio check**: the debt ratio is computed per strategy against the 35 % limit, and strategies above it can be filtered out. Loans longer than 25 years are blocked.
- **DPE rental bans**: the G (2025), F (2028) and E (2034) bans apply per property. You choose between a renovation budget before the ban year and zero rent from that year. A warning badge is shown.
- **Stress testing**: a sensitivity analysis with a tornado chart, Pessimiste / Central / Optimiste risk scenarios, break-even vacancy and rate, and an optional Monte Carlo simulation (P10 / P50 / P90 bands).
- **Reports**: Excel and PDF export for a selected strategy (assumptions, loan schedules, yearly cash flow and tax tables, exit calculation, risk flags).
- **Shareable link and saved scenarios**: the current inputs are encoded in the URL (`?p=`), so a link reproduces the search. Named scenarios can be downloaded and uploaded as JSON.
- **Search diagnostics**: when few strategies pass, the app shows why candidates were rejected, the near misses, and suggestions for parameters to change.

Interface labels stay in French, for example "Lancer l'analyse" (the search button) and "Ciblage Précis".

## Quick start

Requirements: Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints the local address (by default http://localhost:8501).

For development, install the package in editable mode with the dev extras (pytest, ruff, mypy, hypothesis, pre-commit tools):

```bash
pip install -e ".[dev]"
```

### Docker

```bash
docker compose up
```

The app is then served on http://localhost:8502. The container runs as a non-root user. `./results`, `./data` and `./data_user` (saved scenarios) are mounted from the host. An optional `.env` file is passed to the container as its environment (see [Configuration](#configuration)). Add `--build` after code changes to rebuild the image.

## Project structure

```
app.py                      Streamlit entry point
src/
├── domain/                 Pure financial and tax logic (no Streamlit, no I/O)
│   ├── calculator/         Loan, cash-flow and scoring maths
│   ├── models/             Archetype, brick and strategy models (pydantic)
│   ├── tax/                Tax regimes: base, individuals, household, sci_is, plus_value
│   ├── dpe.py              DPE rental-ban rules
│   └── hcsf.py             HCSF debt-ratio rules
├── application/            Use cases built on the domain
│   ├── services/           simulation, strategy_finder, optimizer, allocator, evaluator,
│   │                       scoring, sensitivity, risk, regime_comparison, diagnostics,
│   │                       report, exporter, brick_factory
│   ├── run_context.py      Frozen RunContext: every input of one run
│   ├── scenario_io.py      Encode and decode scenarios (URL and JSON)
│   ├── validation.py       Input validation (the search button is disabled on errors)
│   └── progress.py         Progress reporting for long searches
├── core/                   Shared configuration and definitions
│   ├── defaults.py         All default values, presets, benchmarks and env-var reads
│   ├── glossary.py         Financial definitions (stabilised year, CF acceptance, enrichment)
│   ├── logging.py          structlog configuration
│   └── exceptions.py       Application exceptions
├── ui/                     Streamlit layer
│   ├── components/         Sidebar, results, charts, sensitivity, regime comparison, diagnostics, scenarios
│   ├── pages/main.py       Main page layout
│   ├── cached.py           st.cache_data wrappers
│   └── state.py            Session state
├── utils/debug.py          Debug-context helpers
data/                       Archetype data (archetypes_2026_v1.json)
docs/                       Tax spec, data sources, development plan and progress
tests/
├── unit/                   Unit tests per module
├── integration/            Pipeline, search parity, RunContext parity, invariants
├── e2e/                    End-to-end pipeline tests
├── golden/                 Reference cases for the tax engine (see below)
├── property/               Hypothesis property tests of the search
└── ui/                     Streamlit AppTest flows (marker `ui`)
```

## How the numbers work

- **Tax rules**: [`docs/TAX_SPEC.md`](docs/TAX_SPEC.md) lists the rules for each regime, with sources (CGI, BOFiP, service-public.fr). Section 11b records the user's decisions, and section 11c the implementation readings. Section 10 lists the known differences between the code and the spec.
- **Golden reference cases**: [`tests/golden/`](tests/golden/) holds `cases.yaml` (cases C1 to C9) and `reference_calc.py`, a standalone calculator with no import from `src/`, so the app's tax code is not used to produce its own reference values. Check that `cases.yaml` is up to date with:

  ```bash
  python tests/golden/reference_calc.py --check   # exit 1 if cases.yaml is stale
  ```

- **Data sources**: [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) gives the source and date for each archetype value. Several values (property taxes, charges, most rent caps) are still marked unverified there. No live data feed is used: the data file is hand-maintained.

Key definitions (see `src/core/glossary.py` and `src/core/defaults.py`):

- **Stabilised year = simulation year 2** (`STABILISED_YEAR`). Year 1 contains the works months with no rent, so the steady-state figures (cash flow, DSCR) are taken from year 2. Year 1 values are kept for display only.
- **Acceptance is on the after-tax cash flow** of the stabilised year (`CF_ACCEPTANCE_BASIS = "stabilised_after_tax"`). In min mode a strategy passes if its cash flow is at least the target. In Ciblage Précis it passes if the cash flow is within the tolerance of the target (`cf_accepts`). The allocator's estimate is pre-tax and only a pre-filter.
- **`enrichissement_net`** = net liquidation proceeds + the sum of the after-tax yearly cash flows − the initial down payment. The cash flows are included, so this is the total profit over the horizon. Do not confuse it with `liquidation_nette` (net proceeds at exit).
- **Scoring is absolute**: each component is (value − low) / (high − low), clamped to 0–1, with fixed benchmarks: IRR 0–12 %, stabilised DSCR 0.8–1.6, capital efficiency (enrichment / down payment) 0–5×, enrichment against a 5 %/year target, and cash-flow proximity (0 at 500 € away). The score is a weighted sum of these components, weighted by the chosen profile.
- **Pareto rank** is computed for every strategy and used as a tie-breaker, not as a filter. Ranks above 10 are grouped together.

## Configuration

Environment variables, read in `src/core/defaults.py` (except `LOGLEVEL` and `APP_LOG_DIR`, read in `src/core/logging.py`):

| Variable | Default | Effect |
|---|---|---|
| `APP_AUTOSAVE` | `0` | Set to `1` to save each search's results and a debug context to `APP_RESULTS_DIR`. |
| `APP_RESULTS_DIR` | `results` | Directory for autosaved results. |
| `APP_DATA_DIR` | `data_user` | Directory of the per-user SQLite database of saved scenarios (`scenarios.db`, created with user-only permissions). Mounted as a volume in Docker Compose. |
| `APP_LOG_DIR` | `<repo>/logs` | Directory for `app.log`. |
| `LOGLEVEL` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING` or `ERROR`. |
| `MAX_EXTRA_APPORT_PCT` | `0.95` | Upper bound on the extra down payment per property, as a fraction of the loan, used by the allocator to prevent full loan payoff at t = 0. |

The app does not read `.env` itself. Export the variables in your shell, or let Docker Compose pass them through its `env_file`. `.env.example` lists them.

**Saved scenarios and profiles.** Scenarios are stored per user in `APP_DATA_DIR/scenarios.db`. Without authentication, the sidebar "Profil" selector picks a local profile (default `local`; no password, so it separates people who share a server, it does not protect against them). Streamlit OIDC login is not configured, but the app is ready for it: when `st.user` reports a logged-in user, their email becomes the user_id and the selector is hidden. To enable it later, add an `[auth]` section to `.streamlit/secrets.toml` (`redirect_uri`, `cookie_secret`, `client_id`, `client_secret`, `server_metadata_url` of your identity provider), see the Streamlit "Authentication" docs, and add a login button (`st.login()`) to the app. Nothing else changes.

All other defaults (market, financing, operating costs, tax rates and ceilings, search settings, realism assumptions, DPE and HCSF parameters, scoring presets and benchmarks, risk scenarios, Monte Carlo settings) live in `src/core/defaults.py`. Change them there, not in the UI modules.

## Development

Run the tests from the repository root with the venv active.

```bash
pytest                                      # full suite (the default run includes the UI tests)
pytest -m ui                                # Streamlit AppTest flows only (tests/ui)
pytest tests/integration tests/e2e          # slower integration and end-to-end tests
python tests/golden/reference_calc.py --check
ruff check .                                # lint
ruff format --check .                       # formatting (also enforced by pre-commit)
mypy src/                                   # type check
```

Pre-commit runs ruff, ruff-format, the EOF and whitespace hooks, YAML and merge-conflict checks, a local mypy hook on `src/`, and a quick pytest run of `tests/unit/`. The mypy and pytest hooks are local (`language: system`), so they use whichever Python is first on `PATH`. Put the venv on `PATH` first:

```bash
pre-commit install
PATH="$PWD/.venv/bin:$PATH" pre-commit run --all-files
```

### CI gates

The workflow is `.github/workflows/ci.yml`. A pull request to `main` must pass:

- `ruff check .`
- `mypy src/` (blocking, not `continue-on-error`)
- unit, golden, UI and property tests on Python 3.10 to 3.13
- integration and end-to-end tests
- coverage of `src` at **85 % or more** (`--cov-fail-under=85`)
- a Docker image build

### Project plan

[`docs/DEV_PLAN.md`](docs/DEV_PLAN.md) is the development plan. [`docs/DEV_PROGRESS.md`](docs/DEV_PROGRESS.md) is the progress tracker, with notes for later tasks. [`docs/AGENT_BRIEF.md`](docs/AGENT_BRIEF.md) is the brief for agents working on tasks. Review notes are in [`docs/review/`](docs/review/).

## Disclaimer

Simulation indicative, ne constitue pas un conseil fiscal ou financier.

(Indicative simulation only. It is not tax or financial advice.)

## License

MIT
