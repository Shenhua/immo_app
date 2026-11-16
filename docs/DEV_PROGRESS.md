# Development progress (conductor-owned)

Plan: `docs/DEV_PLAN.md` · Agent brief: `docs/AGENT_BRIEF.md`

## How to resume (read this first in a new session)
1. `git checkout develop`, then read this file. The **Log** at the bottom tells you the last action.
2. Any task marked `running` whose branch has no merge in the log was interrupted:
   - If `git branch --list 'worktree-*' 'task/*'` shows a branch with commits, review it (`git log develop..<branch>`). Merge it if it is complete, otherwise relaunch the task with a note to continue from that branch.
   - Clean up stale worktrees: `git worktree list`, then `git worktree remove <path>`.
3. Pick the next wave from "Next waves" below. Launch **at most 2 agents at a time**.

## Conductor protocol
- Shell hygiene: chain commands with `&&` (never `;`) after a `cd`, so that a failed `cd` cannot run git commands in the main repo.
- Integration branch: `develop` (created from `main`). `main` stays untouched until the user decides to merge.
- Each agent runs with `isolation: worktree` and gets `docs/AGENT_BRIEF.md` + its task ID(s) + conductor notes. The model is the one the plan names.
- When an agent finishes:
  1. Merge its branch into `develop` with `--no-ff`.
  2. Run the tests and ruff.
  3. Update this file.
  4. Commit `chore: progress Txx`.
  5. Remove the worktree.
- Carry anything later tasks need (renamed modules, new APIs, decisions) into **Conductor notes** below. It goes into the prompts of later agents.
- Phase gates: an Opus review after Phase 2 and Phase 6. **T20's TAX_SPEC.md needs the user's approval before T23 starts.**

## Status
Legend: todo · running · done · blocked · skipped

| Task | Model | Status | Branch / commit | Notes |
|---|---|---|---|---|
| T01 lint baseline | Haiku | done | 4454fb9 (merge 0e3ed0f) | ruff 124 to 0 errors |
| T02 repo hygiene | Haiku | done | 476a5cd | | batch with T03, T04 |
| T03 packaging | Haiku | done | 96bc84e | |
| T04 CI and Docker | Haiku | done | a61ee09 | coverage floor 59; Docker build not verified locally | |
| T05 hide broken options | Haiku | done | 24fdf9b (merge 1ba95b1) | batch with T06, T12 |
| T06 disk side effects | Haiku | done | fbed6a5 | |
| T12 layering cleanup | Haiku | done | 323b1c4 | also fixed `import app` crash | |
| T10 central defaults | Sonnet | done | d9e7440 | | |
| T11 RunContext | Sonnet | done | 8bd0deb | | |
| T20 tax spec + golden | Opus | done | 7e1b60c | approved by the user (TAX_SPEC §11b) | | needs user approval |
| T22 amortization + land | Sonnet | done | 583d109 | + golden harness | | |
| T21 plus-value | Sonnet | done | c1cb93f | | |
| T24 wealth/IRR/CFE | Sonnet | done | 0eaa978 | | |
| T23 tax regime engine | Opus | done | f44af5b..ef88a02 | golden C1–C9 all pass | | after T20 approved |
| T25 scoring fixes | Sonnet | done | b702d05 | | |
| T26a realistic assumptions | Sonnet | done | 83e4029 | | after T23 |
| T26b data recalibration | Sonnet | done | ee96023 | TF/charges/caps unverified | | web research |
| Gate P2 review | Opus | done | | docs/review/phase2/README.md: tax engine sound; C1 critical |
| T27 search/detail parity | Sonnet | done | 8ac2698 | C1 fixed; parity tests in 3 tax modes |
| T28 tax hygiene + multi-prop tests | Sonnet | done | 8760fd6 | | after T27 |
| T29 stabilised-year metrics | Sonnet | done | 762c5c6 (merge 23e1e3d) | stabilised = year 2 | | after T27, parallel with T28 | | |
| T30 CF mode ranking | Sonnet | done | 75e0e0f | | |
| T31 search rewrite | Opus | running | | |
| T32 absolute scoring + Pareto | Sonnet | todo | | |
| T33 near misses | Sonnet | todo | | |
| T34 Rentier mode | Sonnet | todo | | |
| T40 results UI batch | Haiku | done | 260dd2e | | |
| T41 sensitivity rewrite | Sonnet | done | 448cd08 | | |
| T42 caching | Sonnet | todo | | |
| T43 validation and UX | Sonnet | todo | | |
| T50 property tests | Sonnet | todo | | |
| T51 AppTest | Sonnet | todo | | |
| T52 typing + coverage | Sonnet | todo | | |
| T53 README/docs | Haiku | todo | | |
| T60–T67 features | mixed | todo | | see plan; T66 = offline files, no API |

## Next waves
1. **W1:** T01 (alone; touches many files)
2. **W2:** [T02+T03+T04] ‖ [T05+T06+T12]
3. **W3:** T10, then T11 (sequential). T20 (Opus) can run in parallel with T10.
4. **W4 (after the user approves T20):** T22 → T21 → T24 → T23 → T25 (sequential, all touch `simulation.py`); T26b in parallel
5. Then T26a, Gate P2, Phase 3 …

## Conductor notes (carry into later prompts)
- Project venv: `/Users/mperrier/immo_app/.venv` (Python 3.15; 209 tests green at the start).
- No API access for real data: T66 is file-based (see plan).
- **Worktrees are created from `main`, not `develop`.** Every agent prompt must begin with: run `git reset --hard develop` in your worktree (safe, no commits yet).
- After T01: `tests/verification_fixtures.py` and `tests/verification_scenarios.py` import modules that no longer exist (`src.core.simulation`, `src.services.*`). Assigned to T02: fix the paths, or delete them if they duplicate the golden tests.
- After T01: `src/core/__init__.py` imports from the application layer (isort moved it first). T12 must remove it.
- After T01: `test_audit_remediation.py::test_combinatorial_explosion_safety` doesn't test what its name says. Revisit in T50.
- After T05/T06/T12:
  - `progress.py` now lives in `src/application/progress.py`.
  - `build_financing_config` and `build_operating_config` moved to `brick_factory`.
  - The formatters live in `ui/helpers.py`.
  - T10 must move `APP_AUTOSAVE` / `APP_RESULTS_DIR` (now read in `app_controller.py`) into `defaults.py`.
  - T34 must restore Rentier (dead `is_rentier` branches remain in the sidebar).
  - The regime selectbox offers only LMNP and Micro-BIC.
- After T02–T04:
  - The venv runs numpy 2.5.4.
  - `health_check.py` is deleted.
  - The CI coverage floor is 59; T52 raises it.
  - `pre-commit run --all-files` doesn't pass yet (ruff-format, EOF and mypy hooks). Fix it in T52 or a format pass.
  - The CI and Docker build must be verified on the first PR.
  - `APP_LOG_DIR` is honoured.
  - `tests/conftest.py` does not yet redirect `APP_RESULTS_DIR`. Add it in T10.
- After T10:
  - All defaults live in `src/core/defaults.py`: MARKET, FINANCING, OPERATING, TAX, SEARCH, the `PresetId` enum, PRESETS and BASE_WEIGHTS. Env vars are read there too.
  - Defaults were aligned to the sidebar values: CFE 150, quality_weight 0.25, BASE_WEIGHTS = the Équilibré preset.
  - Insurance 0.36 still appears in `financial.py` and `brick.py`, and the evaluator has `w.get(..., 0.25)` fallbacks. Clean these up in T25.
  - Session keys `cf_cible` and `cash_flow_cible` are inconsistent. T11 resolves this.
- After T11:
  - `src/application/run_context.py`: frozen `RunContext` with `from_sidebar`, `to_eval_params`, `to_tax_params` and the other views.
  - app.py stores the context with the results, and the stale banner works.
  - `BaseParams` and `get_base_params` are removed.
  - `app_controller.run_strategy_search(archetypes, ctx, cb)` and `simulate_selected_strategy(strategy, ctx)`.
  - Leftovers for T40: the sidebar reads unused session keys `cf_cible`/`apport`; `SessionManager.set_horizon` and `is_ira_enabled` are unused; `render_debug_section` gets the raw dict.
- After T40:
  - For T41: `sensitivity.py` divides the already-monthly `cash_flow_final` by 12, and it mutates `brick["apport_min"]` in place (also a T42 concern).
  - For T43: the help text on the type filter should say that an empty selection means "all".
  - UX question for the user: rename "CF Mensuel" to "CF Mensuel (an 1, avant impôt)"?
  - `StrategyFinder.solver_mode` now exists (T31 should keep it accurate).
- After T26b:
  - The data file is now `data/archetypes_2026_v1.json` (the old one is kept); sources are in `docs/DATA_SOURCES.md`.
  - Taxe foncière, charges and most `loyer_m2_max` caps are still unverified.
  - T21 and T22 must add their new fields to the **2026** file.
- After T20:
  - `docs/TAX_SPEC.md` (§10 lists the code-vs-spec diffs per task) and `tests/golden/{cases.yaml,reference_calc.py}` (9 cases C1–C9; check them with `--check`).
  - The golden tests need `pyyaml`. It is installed in the venv, but T22 must add it to the dev deps.
  - Spec decisions:
    - one regime per category (furnished / unfurnished) per household, not per property;
    - exit value uses price × (1+g)^H, not ^(H−1);
    - PS 18.6% on LMNP income, 17.2% on revenus fonciers and plus-values;
    - micro-BIC ceiling 83,600 €;
    - 21,400 € déficit foncier cap until 2027.
- After T30:
  - `glossary.cf_accepts` and `glossary.cf_penalty` are the single source for CF acceptance.
  - In min mode, tolerance no longer applies to acceptance. T43 should hide the tolerance input in min mode.
  - `StrategyScorer` defaults `mode_cf` to "target", while `RunContext` defaults to "min". Align them in T32.
- User decisions on the tax spec are in TAX_SPEC §11b:
  - CFE default 300 € → T24;
  - `nb_owners` → T21;
  - SCI IS PFU at exit → T23;
  - Jeanbrun → T68.
- After T22:
  - The golden harness is `tests/golden/test_golden.py`. `KNOWN_GAPS` holds fnmatch patterns mapped to task ids, applied as `xfail(strict=True)`. **Each Phase 2 task must remove its entries.**
  - The adapter passes rent net of vacancy.
  - Amortization is by component: building over 30 y (notary fees capitalized, land excluded), works over 15 y, furniture over 10 y.
- After T21:
  - `src/domain/tax/plus_value.py` is the plus-value module. Its breakdown is in the output keys `plus_value` and `plus_value_par_bien`.
  - Exit value is price × (1+g)^H.
  - `nb_owners` and `residence_services` are plumbed through.
  - The C1/C9 exit gaps were retagged T24 (cause: CFE in year 1).
- After T24:
  - `enrichissement_net` now includes Σ after-tax CF.
  - A failed IRR is `None`: it formats as "n/a" and ranks last.
  - CFE: 300 €, furnished only, year-1 and ≤5k€ receipts exemptions, indexed.
  - `_is_furnished` is in `simulation.py`.
  - **For T41 / UI:** the "Enrichissement" KPI in results.py, sensitivity and charts shows `liquidation_nette`, not `enrichissement_net`, so the label doesn't match the metric. Charts and tables pass a None IRR into DataFrames.
  - For T25: `cap_eff` still double-counts wealth.
- After T23:
  - `src/domain/tax/` is the tax package: `base.py` (protocol + state), `individuals.py`, `sci_is.py`, `household.py`.
  - Rates and ceilings are in `defaults.TAX_RATES`.
  - The UI has separate furnished and unfurnished regime selectors, plus an SCI IS checkbox.
  - The bilan has `fiscal_annuel`, `exit_tax`, `regimes` and `tax_warnings`.
  - Deviations are in TAX_SPEC §11c.
  - Follow-ups:
    - `provision_pct` (vacancy) is treated as a deductible charge → T26a.
    - `SimulationParams.regime_fiscal` (domain/models/strategy.py) is stale → T52.
    - Strategy cards don't show regime warnings → T43.
- After T25:
  - The evaluator's `cap_eff` is now enrichment / down payment, against `CAP_EFF_BENCHMARK_MAX` = 5.
  - At the cap, the rent-cap margin scores neutral (0.6).
  - **For T32:** `StrategyScorer` in `strategy_finder.py` still has its own min-max `cap_eff` normalization.
  - `cap_eff` and `s_enrich` remain correlated. Revisit the weights in T32.
- After T26a:
  - Vacancy and works months reduce receipts; the vacancy slider is a floor.
  - The `REALISM` block in defaults.py holds maintenance, GLI, works months and TF growth.
  - The new widgets are in app.py (Frais & Charges expander).
  - Open items:
    - The search-stage CF estimate (`depenses_mensuelles_hors_credit_initial`) ignores works months and TF growth.
    - Default vacancy is inconsistent: archetype 2.0, `OperatingConfig.provision_pct` 5.0, UI 3.0.
- After T27:
  - `simulation.build_engine(market_hypotheses, tax_params)` is the single engine builder (`RunContext.build_engine()` wraps it).
  - `EvaluationParams.to_tax_params()` exists.
  - `MarketHypotheses.tf_growth_pct` defaults to `REALISM.tf_growth_pct`.
  - The amortization durations are in `defaults.TAX`.
  - Parity tests are in `tests/integration/test_search_detail_parity.py`.
  - The golden adapter passes `tf_growth_pct` = charges indexation.
  - `docs/review/` is excluded from ruff.
- After T28:
  - `archetype.is_furnished()` is the single furnished rule.
  - The yearly-table column was renamed to "Charges d'exploitation". The conductor fixed `glossary.py:98` after the merge. **T29 was branched before the rename: when merging T29, keep "Charges d'exploitation" in glossary.py.**
  - The plus-value flat rates (7.5 %, 15 %, surtax scale) are still module constants (low priority).
  - In docs/review/phase2/README.md, the `e.py` and `g.py` labels are swapped.
- After T29:
  - Stabilised year = simulation **year 2** (`glossary.STABILISED_YEAR`, `cf_stabilised_monthly`).
  - The allocator takes `market=`, and `stabilised_monthly_cf()` is its CF estimate.
  - The bilan has `dscr_y1` and `dscr_stab`; the evaluator exposes `cf_monthly_y1`, `dscr_y1`, `cf_monthly_stab`, `dscr_stab` and `warnings`.
  - `glossary.CHARGES_COL` is the column name.
  - Exceptions: only ValueError, ZeroDivisionError and SimulationError are caught.
  - Open items:
    - **T31:** the allocator CF is before tax, while the evaluator CF is after tax. In "target" mode a déficit-foncier refund can push the CF out of the window (−724 pre-tax, −622 after-tax, target −800 ±100).
    - **T31:** allocator step truncation in min mode (−1,501 vs a −1,500 target).
    - **T33/T43:** the evaluator's `warnings` are not shown in the UI.
    - **T41:** the sensitivity DSCR is computed ad hoc on year 1.
- After T41:
  - The pure sensitivity logic is in `src/application/services/sensitivity.py`; the UI uses `st.cache_data`.
  - Follow-ups:
    - The rate delta keeps `credit_final` fixed.
    - The works delta is paid by the down payment.
- `ruff format` not applied (45 files would be reformatted). Optional later; skip it to avoid merge churn.

## Log
- 2026-11-13: Plan written, `develop` branch created, venv set up. Starting W1.
- 2026-11-13: T01 merged (0e3ed0f). Starting W2.
- 2026-11-13: T05/T06/T12 merged (1ba95b1), 233 tests. T20 launched while T02-T04 still run.
- 2026-11-13: T02–T04 merged; Phase 0 complete. T10 launched (T20 still running).
- 2026-11-13: T10 merged, 246 tests. T11 launched.
- 2026-11-13: T11 merged, 247 tests. T40 launched (T20 still running).
- 2026-11-13: T40 merged, 254 tests. T26b launched.
- 2026-11-13: T26b merged. T30 launched (T20 still running).
- 2026-11-13: T20 merged. Waiting for the user to decide on the TAX_SPEC §11 points before T21–T25.
- 2026-11-13: The user approved the spec (§11b). T30 merged, 275 tests. T22 launched (Phase 2 sequence: T22 → T21 → T24 → T23 → T25).
- 2026-11-13: T22 merged (435 passed, 476 xfail). T21 launched.
- 2026-11-13: T21 merged (539 passed, 395 xfail). T24 launched.
- 2026-11-13: T24 merged (593 passed, 346 xfail). T23 launched (Opus).
- 2026-11-13: T23 merged (976 passed, 0 xfail). T25 and T26a launched in parallel.
- 2026-11-13: The usage limit interrupted T25 (uncommitted edits in worktree agent-a0b0f095f0fbb0865) and T26a (no changes).
- 2026-11-16: T25 and T26a resumed via SendMessage. If this session dies, inspect those worktrees: commit or keep the edits, then relaunch with a "continue from this worktree" note.
- 2026-11-16: T25 merged (984 passed). Waiting for T26a.
- 2026-11-16: T26a merged (993 passed). Phase 2 implementation complete. Gate P2 review launched (Opus, read-only).
- 2026-11-16: Gate P2 done (verdict: fix C1 first). T27/T28/T29 added to the plan. T27 launched.
- 2026-11-16: T27 merged; `docs/review` excluded from ruff. T28 and T29 launched in parallel.
- 2026-11-16: T28 merged (1020 passed); glossary column name fixed by the conductor. T29 relaunched after a classifier outage, still running.
- 2026-11-16: T29 merged (1171 passed; glossary conflict resolved via `CHARGES_COL`). **Phase 2 complete.** T31 launched (Opus).
- 2026-11-16: T41 merged (1176 passed). T31 still running; the second slot is kept free (T42 and T43 touch files T31 may change).
