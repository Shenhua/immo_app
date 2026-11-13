# App Immo — Development Plan

Based on the review of 2026-11-13. The phases are in execution order. Each phase relies on the one before it: a clean baseline, then shared foundations, then correct numbers, then correct search, then a correct UI, then new features.

**Agent legend**
| Model | Use for |
|---|---|
| **Haiku** | Mechanical changes with an exact spec: lint fixes, config, renames, small UI fixes with known file:line |
| **Sonnet** | Normal feature or bug work with clear acceptance criteria and moderate reasoning |
| **Opus** | Design decisions, French tax law reasoning, algorithm design, golden reference values, phase-end reviews |

**Rules for every task**
- Work on its own branch.
- Before merging: `pytest` green and `ruff check .` clean. From T52 onward, `mypy src/` must also be clean.
- Bug-fix tasks must add a regression test that fails before the fix.
- **Do not run two tasks that touch the same file in parallel.** The `Files` line lists the files each task touches. `simulation.py` and `strategy_finder.py` are hotspots.
- Open questions go to the user. Do not guess tax rules: cite the source (BOFiP, Légifrance, service-public.fr) in a code comment.

---

## Phase 0 — Baseline and quick safety patches (all Haiku, about 1 day)
Goal: CI green and repo clean, and the UI stops offering options that are known to be broken. Run T01 first. T02–T06 can then run in parallel.

### T01 — Lint baseline · Haiku
- **Spec**
  - Run `ruff check . --fix`.
  - Fix the non-auto-fixable errors by hand:
    - F821: `calculate_insurance` in `tests/verification_fixtures.py:46` and `ResultExporter` in `tests/verification_scenarios.py:70`. Import the right symbol, or delete the dead helper.
    - F811: redefinitions of `StrategyFinder` in `tests/e2e/test_full_pipeline.py`. Keep a single import.
    - F841 in `tests/test_audit_remediation.py:93`.
  - No behaviour change.
- **Accept:** `ruff check .` returns 0 errors, and all 209 tests still pass.

### T02 — Repo hygiene · Haiku
- **Spec**
  - `git rm --cached -r results/ .env`. Un-comment `.env` in `.gitignore` and add `logs/`.
  - Add an autouse fixture in `tests/conftest.py` that redirects the log directory and the results directory to `tmp_path`, so the test run no longer writes `logs/app.log` into the repo. If `src/core/logging.py` doesn't support an override, add an env var `APP_LOG_DIR`.
  - Align `.env.example` `MAX_EXTRA_APPORT_PCT` with the code default (0.95), and document it.
- **Accept:** after `pytest`, `git status --ignored` shows no new files in `logs/` or `results/`.

### T03 — Packaging · Haiku
- **Spec**
  - Make `requirements.txt` consistent with `pyproject.toml`. Use `numpy>=1.26,<3`, then run the test suite on numpy 2.
  - Remove the unused `pydantic-settings` dependency.
  - Delete the broken console script `[project.scripts] app-immo`. The app is launched with `streamlit run`.
  - Delete the dead mypy override `src.models.*`.
  - In `.pre-commit-config.yaml`: bump ruff to the current version and replace `types-all` with the specific stub packages that are needed.
- **Accept:**
  - `pip install -e ".[dev]"` and `pip install -r requirements.txt` resolve the same majors.
  - `pre-commit run --all-files` passes.

### T04 — CI and Docker · Haiku
- **Spec**
  - **Dockerfile:**
    - Add a non-root `USER`.
    - Add `HEALTHCHECK --start-period=30s`.
    - Add a healthcheck and `env_file` to `docker-compose.yml`.
  - **CI:**
    - Python matrix 3.10–3.13.
    - Split the unit and integration/e2e jobs.
    - Bump `codecov-action` to the current major version.
    - Set `--cov-fail-under=60` as the current floor (raised in T52).
    - Add a `docker build` job.
  - `health_check.py`: either fix its stale paths (`src/services/...` and `src/core/simulation.py`) and make it `sys.exit(1)` on issues, or delete it together with its CI job. Recommendation: delete it, because ruff, mypy and coverage already cover it.
- **Accept:** CI is green on a PR, the container runs as non-root, and a deliberate lint error fails CI.

### T05 — Hide broken options (interim) · Haiku
- **Spec**
  - In the sidebar regime selectbox (`app.py:73-76`), remove "SCI IS" and add "Micro-BIC" (already supported by `calculate_tax`). T23 restores SCI IS.
  - Disable the "Rentier (Max Cash-Flow)" mode (`sidebar.py:~95-104`): show it as "bientôt disponible" or remove it. T34 restores it.
- **Accept:** only implemented regimes and modes can be selected.

### T06 — Disk side effects opt-in · Haiku
- **Spec**
  - Gate the autosave of `results/simulation_*.json` and `results/debug_context_*.json` (`app_controller.py:206-230, 309`) behind the env var `APP_AUTOSAVE=1` (default off).
  - Make the results directory configurable (`APP_RESULTS_DIR`).
- **Accept:** a default run writes nothing to disk except the logs.

---

## Phase 1 — Foundations (about 2–3 days)
Goal: one source of truth for parameters, and a frozen run context. Most of the later bugs come from scattered defaults and from the UI re-reading the sidebar.

### T10 — Central defaults · Sonnet
- **Problem:** defaults drift between files.
  - CFE: 150 in one place, 500 in another (`strategy_finder.py:46/64`, `simulation.py:190/501`).
  - Appreciation: 2.0 vs 2.5 (`app_controller.py:264`).
  - Loan rates: 3.2/3.4/3.6 vs 3.5/3.6/3.8, and notary fees 7.5 vs 8 (`BaseParams` vs sidebar).
  - Search weights: `BASE_WEIGHTS` vs the "Équilibré" preset (`strategy_finder.py:24-30` vs `sidebar.py:14-20`).
  - Evaluator `quality_weight` default: 0.5 vs 0.25 in `scoring.py`.
  - `MAX_EXTRA_APPORT_PCT` is read via `os.getenv` inside the allocator constructor.
- **Spec**
  - Create `src/core/defaults.py` with frozen dataclasses: `MarketDefaults`, `FinancingDefaults`, `TaxDefaults`, `ScoringWeights`, and `PRESETS: dict[PresetId, ScoringWeights]`.
  - Replace the substring matching on French preset labels (`strategy_finder.py:629-643`) with a `PresetId` enum.
  - Every module and widget imports its defaults from there. Env overrides are read once, in this module.
- **Files:** `core/defaults.py` (new), `strategy_finder.py`, `simulation.py`, `evaluator.py`, `scoring.py`, `allocator.py`, `sidebar.py`, `app.py`, `app_controller.py`, `state.py`
- **Accept:**
  - Grepping the numeric literals above finds them only in `defaults.py`.
  - A test asserts that the sidebar widget defaults equal `defaults.py`.

### T11 — RunContext snapshot · Sonnet (depends on T10)
- **Problem:**
  - Charts and details are re-simulated with the current sidebar values (`app.py:279-288`).
  - The stress test reads session keys that are never written (`state.py:111-115`).
  - Nothing tells the user when the results are stale.
- **Spec**
  - Create a frozen pydantic model `RunContext`: every search and simulation parameter, plus the `archetypes_version` hash.
  - Build it once from the sidebar when "Lancer l'analyse" is clicked, and store it in the session next to the results.
  - The search, the detail simulation, the charts and the sensitivity tab all take a `RunContext` argument. None of them read `st.session_state` directly.
  - If the current sidebar no longer matches the stored context, show a banner: "Paramètres modifiés — relancer l'analyse".
- **Accept:**
  - Changing a slider after a search doesn't change the KPIs or charts, and the banner appears.
  - A unit test checks that `RunContext` round-trips to and from a dict.

### T12 — Layering cleanup · Haiku
- **Spec**
  - Move `src/ui/progress.py` to `src/application/progress.py` and update its imports in `optimizer.py` and `strategy_finder.py`.
  - Remove the domain and application imports from `src/core/__init__.py`.
  - Fix `utils/debug.py`: it imports `ui.app_controller` (line 63) and reads the wrong keys (`prix_achat` and `loyer_CC` at lines 56-57; use the real archetype keys).
  - Remove the duplicate `_get_val` (`allocator.py:76-140`).
  - Merge the duplicated `format_euro`/`format_pct`/`get_taxonomy_badge` into `ui/helpers.py`.
- **Accept:** `src/domain` and `src/application` never import `src.ui` or `streamlit` (add a test that greps the imports).

---

## Phase 2 — Financial and tax correctness (about 1–1.5 weeks)
Goal: numbers you can trust.
- T20 is first: it defines the expected values.
- T21–T25 all touch `simulation.py`, so run them **sequentially** in this order: T22 → T21 → T24 → T23. They follow the golden values from T20.
- T26a and T26b can run in parallel with them.

### T20 — Tax design and golden reference cases · Opus
- **Spec**
  - **Write `docs/TAX_SPEC.md`**, citing sources (BOFiP, CGI article numbers). It covers:
    1. **LMNP réel:** amortization by component, land excluded, furniture over 5–10 years, building over 25–40 years with a stop when fully amortized. Amortization cannot create a deficit; the excess is carried forward (ARD). BIC deficits carry forward 10 years.
    2. **Micro-BIC (loi Le Meur):** 50% allowance up to the 77,700€ ceiling for long-term furnished lets; 30% allowance with a 15,000€ ceiling for non-classé tourist rentals.
    3. **Revenus fonciers (unfurnished):**
       - Micro-foncier: 30% allowance, 15,000€ ceiling.
       - Réel: déficit foncier imputable on global income up to 10,700€ (21,400€ for energy renovation works until 2025, to verify for 2026). The interest part of the deficit only offsets rental income. Carryforward 10 years.
    4. **SCI IS:**
       - 15% up to 42,500€, then 25%, with amortization.
       - Exit plus-value computed on the net book value.
       - Dividend flat tax: the spec must say whether it is modelled.
    5. **Plus-value on sale (individuals):**
       - IR (income tax) 19%: allowance of 6% a year for years 6–21, then 4% in year 22.
       - PS (social charges): allowance of 1.65% a year for years 6–21, 1.60% in year 22, then 9% a year up to year 30.
       - Surtax on net gains over 50k€.
       - Sale costs deducted.
       - Computed **per property**, with no netting across properties.
       - LMNP amortization added back for sales after 15/02/2025 (exception for résidences-services).
    6. **Prélèvements sociaux per income type:** verify the 2026 LFSS (social security finance law) CSG change. Which incomes go to 18.6% and which stay at 17.2%?
    7. **CFE:** applies to furnished lets only.
  - **Golden cases:** write `tests/golden/cases.yaml` with about 8 hand-computed cases: one property, explicit inputs, expected yearly tax, cash flow, exit net proceeds, IRR, wealth. Cover:
    - LMNP réel at 10, 20 and 25 years
    - micro-BIC
    - micro-foncier
    - foncier réel with a deficit
    - SCI IS
    - a land-ratio case
  - Show the arithmetic for each case.
- **Accept:** the user reviews and approves `TAX_SPEC.md` before T23 starts.

### T21 — Plus-value fixes · Sonnet
- **Spec**
  - Fix the IR allowance (`simulation.py:439`). Required values: 6%/yr for years 6–21 (96% at year 21) and 100% from year 22.
  - Deduct sale costs from the gain.
  - Compute the gain per property and clamp each one at ≥ 0.
  - Add the surtax above 50k€.
  - Add the LMNP amortization add-back. Use a per-archetype flag `residence_services: bool = False` for the exception.
- **Files:** `simulation.py`, `domain/models/archetype.py`, data JSON (add the flag)
- **Accept:**
  - Parametrized tests of the allowance at 5, 6, 17, 21, 22 and 30 years.
  - The relevant golden cases pass.

### T22 — Amortization fixes and land ratio · Sonnet
- **Spec**
  - Furniture amortization stops after its duration. Building amortization stops after its duration.
  - Add `ratio_terrain_pct` to the Archetype model. Default 15%; set Lyon and Annecy to 20% in the data.
  - Pass it through `brick_factory` → strategy → simulation. It is never 0 by default.
  - Add a UI override ("Part terrain %") in the advanced section.
- **Files:** `simulation.py`, `brick_factory.py`, `archetype.py`, `app_controller.py`, `strategy_finder.py`, sidebar, data JSON
- **Accept:**
  - A 25-year LMNP run deducts exactly the furniture cost once in total.
  - An e2e test through `StrategyFinder` shows that the land ratio changes the tax.

### T23 — Tax regime engine · Opus (depends on T20 being approved)
- **Spec**
  - Replace the `if regime == "microbic" else LMNP` branch (`simulation.py:43-105, 331`) with a `TaxRegime` protocol, one implementation per regime in `TAX_SPEC.md`:
    - `compute_year(ctx, state) -> TaxResult`
    - `compute_exit(...) -> ExitTax`
    - a carryforward state object
  - The regime is chosen **per property**:
    - unfurnished archetypes (`meuble=False`): user choice between micro-foncier and réel;
    - furnished archetypes: micro-BIC or LMNP réel;
    - a global "SCI IS" option overrides both.
  - The social-charges rate is a per-regime parameter taken from `defaults.py`.
  - Restore "SCI IS" in the UI (undoing T05).
- **Files:** `simulation.py`, new `src/domain/tax/` package, `defaults.py`, sidebar, `app.py`
- **Accept:**
  - All golden cases pass to within 1€.
  - Unfurnished archetypes can no longer claim amortization (regression test).

### T24 — Wealth metric, IRR and CFE · Sonnet
- **Spec**
  - Change the `enrichissement_net` definition (`simulation.py:412`, glossary `core/glossary.py:129`):
    - new definition: net exit proceeds + Σ after-tax yearly cash flow − down payment;
    - update the glossary text.
  - Failed IRR (`simulation.py:398-408`): return `None` plus a warning. The evaluator drops the strategy, or ranks it last with a visible flag. Never show 0%.
  - CFE only applies to furnished properties.
- **Accept:**
  - Two strategies with identical exits but different cash flows get different wealth figures.
  - A test shows a NaN IRR never displays as 0%.

### T25 — Scoring fixes · Sonnet
- **Spec**
  - The evaluator counts `s_enrich` twice (`evaluator.py:222-228`). Replace the `cap_eff` proxy with real capital efficiency (wealth / down payment), or drop it and renormalize the weights.
  - Align the `quality_weight` default via `defaults.py` (done in T10, verify here).
  - Rent-cap margin (`scoring.py:83`): a rent exactly at the cap scores neutral (0.6), not 0.
  - Remove the no-op expression at `financial.py:159`.
- **Accept:** unit tests for each change.

### T26a — Realistic assumptions · Sonnet
- **Spec**
  - New parameters in `defaults.py` and the UI:
    - maintenance/capex provision: €/m²/year, default 10;
    - unpaid-rent / rent-guarantee insurance (GLI): % of rent, default 3%;
    - works period: months without rent when works are planned, default 3.
  - Management fees default to 7%. Taxe foncière growth defaults to 3%.
  - The UI vacancy slider becomes an **override or floor** on each archetype's `vacance_pct`. It must no longer be ignored silently. Document the behaviour in the help text.
- **Files:** `simulation.py` (after T23), `defaults.py`, sidebar
- **Accept:** parameter-effect tests in `test_parameter_effects.py` for each new parameter.

### T26b — Archetype data recalibration · Sonnet with web search (parallel with T21–T25)
- **Spec**
  - Check each of the 18 archetypes' price/m², rent/m², taxe foncière and charges against 2025–2026 public data: DVF median per commune (data.gouv.fr), the Clameur/ANIL rent observatories, city rent-cap decrees.
  - Points to check:
    - Grenoble-suburb studio prices look 20–40% high.
    - The `soumis_encadrement` flag on Saint-Martin-d'Hères, Échirolles, Fontaine and Meylan.
    - Student vacancy should be ≥ 8%.
  - Write `docs/DATA_SOURCES.md` with the source and date for each value.
  - Bump the file to `archetypes_2026_v1.json`.
- **Accept:** every changed value has a cited source, and the schema validation passes.

### Phase 2 gate — Opus review
- Run the code-review skill on the Phase 2 diff at high effort.
- Opus compares 3 golden cases against an independent spreadsheet-style calculation.

---

## Phase 3 — Search and ranking (about 1 week)

### T30 — Mode-aware cash-flow acceptance and ranking · Sonnet
- **Spec**
  - **Mode "min":**
    - Acceptance: `cf >= target` with no tolerance below the target (`allocator.py:335`).
    - Ranking penalty: `max(0, target - cf)` (`strategy_finder.py:137-140`).
  - **Mode "precise":** keep `abs(cf - target)`.
  - The scorer receives `mode_cf`.
- **Accept:**
  - In "min" mode, a surplus is never penalised and a deficit is never accepted.
  - Closes `docs/feedbacks.md` 2.2.

### T31 — Search algorithm rewrite · Opus
- **Problem:**
  - Combinations grow as 3^k (3 durations per property).
  - The genetic algorithm returns over-budget portfolios and duplicate properties (`optimizer.py:105-131, 346-428`).
  - Up to 1M futures run on threads, which gives no speedup for CPU-bound work, and the order of tied results isn't deterministic.
- **Spec**
  1. Precompute each brick's monthly payment, costs and flat data once (no `model_dump()` per combination).
  2. Enumerate **property subsets** with k ≤ max_props, with budget pruning (sorted by minimum down payment, stop early). C(18,≤5) ≈ 12.6k subsets.
  3. For each subset, choose the durations:
     - exact when 3^k ≤ 243 and the subset count is small;
     - otherwise greedy, starting from the duration that best meets the CF target, then a single pass of local swaps.
     - Document the bound.
  4. Evaluate in chunks with a `ProcessPoolExecutor` (or keep it serial if it's fast enough; benchmark first).
  5. Sort deterministically with an explicit tie-breaker (the key built from the sorted property names).
  6. Report progress through a callback, so it works for every `max_props`.
  7. **Delete the genetic algorithm** and `THRESHOLD_EXHAUSTIVE`.
  8. Hard invariants checked on every result: `apport_total <= budget`, unique property names, `len <= max_props`.
- **Files:** `optimizer.py` (rewrite), `strategy_finder.py`, `allocator.py` (interface only), tests
- **Accept:**
  - Empire mode (5 properties, 54 bricks) finishes in under 10s on a laptop.
  - Two runs give identical results.
  - On small instances, the results match the old exhaustive search.
  - A benchmark is recorded in the PR.

### T32 — Absolute scoring and Pareto front · Sonnet (depends on T31)
- **Spec**
  - Replace the per-run min-max normalisation (`strategy_finder.py:~143`) with fixed benchmark ranges in `defaults.py`. For example: IRR 0–12%, DSCR 0.8–1.6, wealth/down payment 0–5×.
  - The Pareto filter runs **always**, not only when `len > top_n`. It uses the objectives (wealth↑, IRR↑, DSCR↑, down payment↓, quality↑, CF distance↓ according to the mode).
  - The preset weights only order strategies within and across fronts.
  - The UI shows which strategies are on the front.
- **Accept:**
  - The same strategy gets the same score in two different runs.
  - Under the "Patrimoine" preset, a high-wealth strategy is not discarded.

### T33 — Near misses and constraint diagnostics · Sonnet (depends on T31)
- **Spec**
  - Keep the best N failing combinations with their reason (budget, CF, max properties).
  - When no strategy passes, show which constraint eliminated the most candidates, and by how much it would need relaxing. For example: "CF cible -50€ → 12 stratégies".
  - Delete the unused tier-2/3 code, or replace it with this.
- **Accept:** an empty result always explains why, with a suggestion.

### T34 — Implement Rentier mode · Sonnet (depends on T31)
- **Spec**
  - Pass `use_full_capital` from `StrategyFinder` into the new optimizer and on to `allocate`.
  - Define the behaviour: put all the available down payment in, to maximise monthly cash flow.
  - Fix the phantom 1€ residual loan (`allocator.py:290`): a full cash payment gives a loan of 0.
  - Re-enable the mode in the UI (undoing T05).
- **Accept:** with the same inputs, Rentier mode uses more down payment and gives a higher CF than the balanced mode.

---

## Phase 4 — UI correctness and performance (about 4–5 days, mostly parallel)

### T40 — Results UI bug batch · Haiku
- **Spec** (each with the exact location):
  - `results.py:122`: read the CF target from the `RunContext` (from T11), not from `st.session_state["cf_cible"]`.
  - `results.py:189`: show `apport_final_bien` instead of `apport_min`.
  - `results.py:233`: fix the help text (before or after tax, depending on what is now shown).
  - `app.py:197-198`: `combos_evaluated` counts evaluations only.
  - `app.py:158,162`: use the real brick count and the mode label.
  - `state.py`: reset `selected_strategy_idx` and `show_details` on a new search. Write the URL query param when a strategy is selected.
  - `filters.py:113`: decide and document what an empty city selection means.
- **Accept:** each fix is verified manually and listed in the PR.

### T41 — Sensitivity tab rewrite · Sonnet (depends on T11)
- **Spec**
  - The baseline and the stressed case are computed by the same function with the same `RunContext`, differing only by the deltas.
  - The DSCR comes from the shared evaluator function.
  - Add a tornado chart: impact on IRR and CF of ±1 point on rate, rent, vacancy and appreciation.
  - Cache it on (strategy, context, delta).
- **Accept:** at zero delta, every impact is exactly 0.

### T42 — Caching · Sonnet
- **Spec**
  - `st.cache_data` on archetype loading, rent caps, filtering and `create_investment_bricks`, keyed by inputs plus the data hash.
  - Cache the detail simulation by (strategy, `RunContext`).
  - Selecting a row in the table must not trigger a re-search.
- **Accept:** a profiling log shows that selecting a row re-runs no simulation already computed.

### T43 — Input validation and UX · Sonnet
- **Spec**
  - Cross-field validation, run before the search:
    - down payment ≥ the cheapest brick's minimum down payment;
    - tolerance ≥ 0;
    - budget ≥ down payment.
  - Explain the "Ciblage Précis" toggle.
  - Separate "Nombre max de biens" (1–5) from the mode radio.
  - Allow custom loan durations (10–25 years).
  - A responsive KPI layout (no 6 columns on narrow screens), and reduce the 2.5em fonts.
- **Accept:** invalid inputs show inline errors, and no search is launched.

---

## Phase 5 — Test and typing hardening (about 3–4 days)

### T50 — Property-based tests · Sonnet (after T31)
- **Spec:** use `hypothesis` with random budgets, down payments, targets and archetype subsets. Assert:
  - the invariants from T31;
  - determinism;
  - monotonicity: higher rent leads to IRR ≥, a higher rate leads to CF ≤.

### T51 — Streamlit AppTest suite · Sonnet (after Phase 4)
- **Spec:** smoke tests with `streamlit.testing.v1.AppTest`:
  - launch, run the search, select a strategy, open each tab;
  - change a parameter and check the stale banner appears;
  - the empty-result diagnostics.

### T52 — Strict typing and coverage gates · Sonnet
- **Spec**
  - Fix the 61 mypy errors (worst files: `utils/debug.py`, `sidebar.py`, `strategy_finder.py`; property decorators in the `strategy` and `brick` models).
  - Enable `strict = true` for `src.domain.*` and `src.application.*`.
  - Remove `continue-on-error` on mypy in CI.
  - Raise `--cov-fail-under` to 80.

### T53 — README and docs · Haiku
- **Spec**
  - Update the README structure and env vars, and add a "how scoring works" section.
  - Archive the obsolete docs (`Audit_Report_Roadmap.md`, `Some insights to ponder.md`) to `docs/archive/`.

---

## Phase 6 — Features (prioritised; mostly independent once Phases 2–4 are done)

| # | Feature | Model | Depends on | Spec summary |
|---|---|---|---|---|
| T60 | **HCSF bankability** | Sonnet | T10 | Inputs: household net income and existing monthly debts. Debt ratio = (existing + new loan payments) / (income + 70% of gross rents). Strategies above 35% are flagged or filtered (user choice), and the ratio is shown per strategy. Loans longer than 25 years are blocked. |
| T61 | **DPE rental ban** | Sonnet | T23, T26a | Ban years: G 2025, F 2028, E 2034. Per property, either force a renovation budget (cost per DPE class jump, in `defaults.py`) before the ban year, or set rent to 0 from that year. The user chooses. Show a warning badge. |
| T62 | **Tax regime comparison** | Sonnet | T23 | For a selected strategy, run every eligible regime with the same `RunContext`. Show a table and chart of yearly tax, CF, IRR and wealth. Highlight the best regime and the break-even year. |
| T63 | **Report export (PDF + Excel)** | Sonnet | T11, T24 | `st.download_button`. Contents: assumptions (`RunContext`), loan schedules, yearly cash-flow and tax tables, exit calculation, risk flags. Excel via openpyxl; PDF via an HTML template and weasyprint (or reportlab). |
| T64 | **Risk scenarios** | Sonnet | T41 | Pessimistic / base / optimistic presets for appreciation, rent indexation, vacancy and rates. Optional Monte Carlo (N=500) with P10/P50/P90 bands for IRR and wealth. Show break-even vacancy and break-even rate. |
| T65 | **Saved scenarios + URL** | Sonnet | T11 | Serialize `RunContext` to a compact URL param, so a shared link reproduces the search. Save and load named scenarios as a JSON download/upload; later, a per-user store. |
| T66a | **Real-data pipeline design (offline, no API)** | Opus | T26b | **Strategic goal of the product.** No API access for now, so the design is **file-based**: (1) DVF bulk CSV files downloaded by hand (data.gouv.fr "DVF géolocalisées", per département) → price/m² percentiles per commune, type and surface; (2) rent references from hand-maintained CSV/YAML tables (observatoires, rent-cap decrees); (3) a listings CSV import with a documented schema, filled by hand or from a portal export. The pipeline is a pure function `files → archetypes JSON` with sources and dates. Keep a `DataSource` interface so an API adapter can be added later without touching the domain. |
| T66b | **Real-data pipeline build** | Sonnet | T66a | Implement the T66a design as `scripts/build_archetypes.py` with tests and small fixtures (no network). Add a "source" label per brick and an upload widget in the UI for the listings CSV. |
| T67 | **Hosting** | Sonnet | T06, T65 | Streamlit auth (OIDC), per-user storage, a rate limit on search, deployment docs. |

**Suggested Phase 6 order:**
1. T60 and T61 (they change which results are valid)
2. T62
3. T63
4. T64
5. T65
6. T66a, then T66b
7. T67

### Phase 6 gate — Opus review
After T60–T62, run the code-review skill at high effort.

---

## Summary by agent
| Model | Tasks | Share |
|---|---|---|
| Haiku | T01–T06, T12, T40, T53 | ~9 tasks, mechanical |
| Sonnet | T10, T11, T21, T22, T24–T26b, T30, T32–T34, T41–T43, T50–T52, T60–T65, T66b, T67 | ~26 tasks |
| Opus | T20, T23, T31, T66a, plus the phase-gate reviews | 4 tasks + reviews: the tax law, the tax engine, the search algorithm and the data design |

## Critical path
T01 → T10 → T11 → T20 → (T22 → T21 → T24 → T23) → T31 → T32 → T41 → T51 → T60/T61/T62

All Phase 0 tasks except T01, plus T12, T26b, T40 and T42, can run alongside the critical path when they don't share files with the task in progress.
