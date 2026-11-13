# Development progress (conductor-owned)

Plan: `docs/DEV_PLAN.md` · Agent brief: `docs/AGENT_BRIEF.md`

## How to resume (read this first in a new session)
1. `git checkout develop`, then read this file. The **Log** at the bottom tells you the last action.
2. Any task marked `running` whose branch has no merge in the log was interrupted:
   - If `git branch --list 'worktree-*' 'task/*'` shows a branch with commits, review it (`git log develop..<branch>`). Merge it if it is complete, otherwise relaunch the task with a note to continue from that branch.
   - Clean up stale worktrees: `git worktree list`, then `git worktree remove <path>`.
3. Pick the next wave from "Next waves" below. Launch **at most 2 agents at a time**.

## Conductor protocol
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
| T22 amortization + land | Sonnet | running | | |
| T21 plus-value | Sonnet | todo | | |
| T24 wealth/IRR/CFE | Sonnet | todo | | |
| T23 tax regime engine | Opus | todo | | after T20 approved |
| T25 scoring fixes | Sonnet | todo | | |
| T26a realistic assumptions | Sonnet | todo | | after T23 |
| T26b data recalibration | Sonnet | done | ee96023 | TF/charges/caps unverified | | web research |
| Gate P2 review | Opus | todo | | |
| T30 CF mode ranking | Sonnet | done | 75e0e0f | | |
| T31 search rewrite | Opus | todo | | |
| T32 absolute scoring + Pareto | Sonnet | todo | | |
| T33 near misses | Sonnet | todo | | |
| T34 Rentier mode | Sonnet | todo | | |
| T40 results UI batch | Haiku | done | 260dd2e | | |
| T41 sensitivity rewrite | Sonnet | todo | | |
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
