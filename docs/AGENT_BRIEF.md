# Brief for development agents

You are implementing **one task (or a small batch)** from `docs/DEV_PLAN.md` for App Immo, a Python/Streamlit simulator that ranks French real-estate investment strategies. A conductor session assigns the tasks, merges the branches and keeps the progress tracker.

## Read first (only these)
1. The section(s) of `docs/DEV_PLAN.md` for your task ID(s), plus "Rules for every task" at the top of that file. Don't read the rest of the plan.
2. The files listed in your task. Explore further only as far as the task needs.
3. Any notes the conductor added to your prompt (they come from earlier tasks and override the plan).

## Environment
- Python: `/Users/mperrier/immo_app/.venv/bin/python` (absolute path; works from a worktree). Run tools as `<python> -m pytest -q`, `<python> -m ruff check .`, `<python> -m mypy src/`.
- Do not `pip install -e .`. Install new packages into that venv only if your task requires it, and add them to `requirements.txt` and `pyproject.toml`.
- Never write outside your working directory except to the venv.

## Git
- You work in your own git worktree, on your own branch. Commit there. One or a few commits, with a message like `T21: fix plus-value allowance schedule`, ending with:
  `Co-Authored-By: Claude <noreply@anthropic.com>`
- Do **not** push, merge, rebase onto other branches, or touch `main`/`develop`.
- Do **not** edit `docs/DEV_PROGRESS.md` or `docs/DEV_PLAN.md` (the conductor owns them).

## Quality bar (before your final commit)
- `pytest -q`: all green. Bug fixes get a regression test.
- `ruff check .`: clean (from T01 onward).
- Stay in scope. If you find other problems, **list them in your report; don't fix them**.
- French tax or law rules: cite the source (BOFiP, CGI article, Légifrance, service-public.fr) in a code comment. If a rule is uncertain, implement what the spec says and flag it. If the spec itself looks wrong, stop and report rather than guess.

## Final report (your last message, at most 250 words)
```
TASK: Txx[, Tyy]
BRANCH: <branch name>   COMMIT: <short hash(es)>
STATUS: done | partial | blocked
CHANGED: <files, one line each, with a few words>
TESTS: <pytest summary line>; ruff: <clean/n errors>
DEVIATIONS: <where you departed from the spec and why, or "none">
FOLLOW-UPS: <out-of-scope issues found, open questions for the user>
```
