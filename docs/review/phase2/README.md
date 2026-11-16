# Phase 2 gate review (Opus, 2026-11-16, develop @ e4814b7)

**Verdict:** the tax engine (`src/domain/tax/`) is sound. It was checked by hand on households with several properties, on SCI IS with several properties, on horizons beyond the amortization durations, and on edge cases. One critical integration bug must be fixed first.

The scenario scripts are kept here (`h.py` is the shared helper). They were run with the project venv from the repo root.
- `a.py`: mixed household (furnished + unfurnished), category independence.
- `b.py`: micro ceilings across properties.
- `e.py`: déficit foncier and recapture.
- `g.py`: SCI IS with several properties.

T28 turns them into regression tests.

## Findings
- **C1, CRITICAL, CONFIRMED.**
  - `strategy_finder.py` builds `MarketHypotheses` without `tf_growth_pct`. The search therefore uses the charges inflation (2%), while detail and sensitivity use 3%.
  - Result: card IRR and wealth differ from the detail view, the TF slider does nothing in the search, and sensitivity shows an impact at zero delta.
  - → T27
- **M1, MAJOR, CONFIRMED.**
  - Works months (all 18 archetypes have works) cut year-1 CF.
  - The allocator accepts on cf0, then the evaluator rejects on year 1, and GA mode drops those strategies.
  - DSCR in year 1 falls by 40–50% for every strategy.
  - → T29: a stabilised-year metric policy.
- **m1:** `plus_value.py` hard-codes its rates, and `TAX_RATES.ps_plus_value_pct` is unused. → T28
- **m2:** three different "furnished" defaults (archetype model, brick_factory CFE, `simulation._is_furnished`). → T28
- **m3:** dead code (`ExitContext.tmi_pct`, `TaxHousehold.state()`, `PropertyExit` export); amortization durations 30/10 hard-coded. → T27 (durations), T28 (dead code)
- **m4:**
  - `evaluator.evaluate` and `simulate_selected_strategy` swallow every exception. → T29
  - The `sensitivity.py` `provision_pct` fallback. → T27
- **m5:** the golden harness bypasses the engine's vacancy path. → T28
- **m6:** no multi-property regression tests. → T28
- **m7:** the "Charges Déductibles" label is wrong under micro regimes. → T28
- **Outside Phase 2:** in min mode, allocator step truncation can stop at −1,501 against a −1,500 target. → T31
