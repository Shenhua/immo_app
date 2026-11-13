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
| T11 RunContext | Sonnet | running | | |
| T20 tax spec + golden | Opus | running | | needs user approval |
| T22 amortization + land | Sonnet | todo | | |
| T21 plus-value | Sonnet | todo | | |
| T24 wealth/IRR/CFE | Sonnet | todo | | |
| T23 tax regime engine | Opus | todo | | after T20 approved |
| T25 scoring fixes | Sonnet | todo | | |
| T26a realistic assumptions | Sonnet | todo | | after T23 |
| T26b data recalibration | Sonnet | todo | | web research |
| Gate P2 review | Opus | todo | | |
| T30 CF mode ranking | Sonnet | todo | | |
| T31 search rewrite | Opus | todo | | |
| T32 absolute scoring + Pareto | Sonnet | todo | | |
| T33 near misses | Sonnet | todo | | |
| T34 Rentier mode | Sonnet | todo | | |
| T40 results UI batch | Haiku | todo | | |
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
- `ruff format` not applied (45 files would be reformatted). Optional later; skip it to avoid merge churn.

## Log
- 2026-11-13: Plan written, `develop` branch created, venv set up. Starting W1.
- 2026-11-13: T01 merged (0e3ed0f). Starting W2.
- 2026-11-13: T05/T06/T12 merged (1ba95b1), 233 tests. T20 launched while T02-T04 still run.
- 2026-11-13: T02–T04 merged; Phase 0 complete. T10 launched (T20 still running).
- 2026-11-13: T10 merged, 246 tests. T11 launched.
