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
| T01 lint baseline | Haiku | todo | | |
| T02 repo hygiene | Haiku | todo | | batch with T03, T04 |
| T03 packaging | Haiku | todo | | |
| T04 CI and Docker | Haiku | todo | | |
| T05 hide broken options | Haiku | todo | | batch with T06, T12 |
| T06 disk side effects | Haiku | todo | | |
| T12 layering cleanup | Haiku | todo | | |
| T10 central defaults | Sonnet | todo | | |
| T11 RunContext | Sonnet | todo | | |
| T20 tax spec + golden | Opus | todo | | needs user approval |
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

## Log
- 2026-11-13: Plan written, `develop` branch created, venv set up. Starting W1.
