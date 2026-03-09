# Rapport d'Étonnement — App Immo

**Date:** 2026-03-06  
**Scope:** Full audit of 8 specification documents against codebase  
**Author:** Audit AI Pass (Phases 1–8)

> This document lists observations, surprises, and concerns found during the audit pass. It is sorted by criticality and is intended as a decision-making aid, not a bug tracker.

---

## 🔴 P0 — Critical: Immediate Attention Required

| # | Observation | Why it's a problem |
|---|---|---|
| 1 | **The GA is described as concurrent/threaded everywhere — it's fully sequential.** `GeneticOptimizer.evolve()` uses a plain `for` loop. `ThreadPoolExecutor` exists only in `ExhaustiveOptimizer`. | Any future dev reading the specs will build the wrong mental model. Performance expectations for large datasets are incorrect. |
| 2 | **`health_check.py` is referenced in 3 documents as an executable gate — it does not exist.** | The most important pre-deployment safety check is completely absent. There is no automated way to verify the engine's structural health. |
| 3 | **The field naming pipeline creates a triple rename for the same data.** `JSON → archetype → brick factory → InvestmentBrick → dict`. Every layer renames fields (e.g. `tension_locative_score_norm → indice_tension`, `nom → nom_bien → nom`, `taux_pret ↔ taux_annuel_pct`). | This forced a `__getitem__` shim on `InvestmentBrick` — a domain model that now pretends to be a dict. It's a maintenance time-bomb. |
| 4 | **LMNP deficit carryforward has no 10-year expiry.** French tax law limits déficits LMNP to a 10-year carry window. The code carries them indefinitely. | This silently overstates tax optimization for long investment horizons (>10 years), which is when users are most likely to be in LMNP. |
| 5 | **`ratio_terrain_pct` defaults to 0%.** This means 100% of the purchase price is depreciable under LMNP. In reality, land (terrain) cannot be depreciated — in Paris, land can represent 15–25% of property value. | The amortization (and therefore tax shield) is systematically overstated for urban properties. |
| 6 | **`sensitivity.py` runs full financial simulations inside the Streamlit render loop.** This is the highest-severity architectural violation in the codebase. | Streamlit re-renders on any state change. Every slider interaction could trigger a full multi-year simulation, blocking the UI thread with no async protection. |

---

## 🟠 P1 — High: Should Be Addressed Before Next Feature Work

| # | Observation | Why it's a problem |
|---|---|---|
| 7 | **The spec documents described a different project than what was built.** 20 of 48 audit findings were "critical" — not minor wording issues, but fundamentally wrong facts (wrong thresholds, wrong field names, wrong formulas, non-existent classes). | Someone reading the specs to build on top of this project would make serious mistakes. This suggests specs were written from memory or aspirationally, not from the code. |
| 8 | **`allocator.py` uses exception-based field access** (`try/except TypeError/KeyError`) to read brick fields instead of typed model access. | This is the same symptom as the `__getitem__` shim — the consequence of having two field-naming systems in parallel. It hides bugs and makes debugging unpredictable. |
| 9 | **`app_controller.py` silently swallows engine exceptions** during `simulate_selected_strategy`. The call is wrapped in a bare `except Exception: pass`. | If the simulation engine crashes during result display, the user sees nothing — no error, no message, no fallback. Silent failures in financial software are uniquely dangerous. |
| 10 | **`PortfolioStrategy` is a Pydantic model that's never used as a typed object.** The pipeline immediately serializes it to `list[dict]` and passes dicts everywhere. The model's field validators and constraints never run on actual pipeline data. | You have a type system that provides false safety. Bugs that Pydantic would catch in a properly typed pipeline slip through undetected. |
| 11 | **`SCI IS` regime is selectable in the UI but not simulated.** The code falls back to LMNP logic when `SCI IS` is selected. | Users who select `SCI IS` are being shown LMNP results silently. This is arguably a product correctness issue, not just a tech debt item. |

---

## 🟡 P2 — Medium: Worth Planning For

| # | Observation | Why it's a problem |
|---|---|---|
| 12 | **The qualitative scoring model was completely redesigned without updating the specs.** Spec said 3 factors (DPE 40%, Tension 35%, Yield 25%). Code has 7 factors with entirely different weights. | This is the largest single divergence between spec and code. It suggests the scoring model evolved significantly mid-development. The old spec could mislead analysis of why certain properties score as they do. |
| 13 | **The `verification/` directory is empty.** It exists, suggesting intent, but has no content. | A planned test tier with no tests. If it was meant for something specific (e.g. mathematical verification scenarios), that intent is now lost. |
| 14 | **`test_audit_remediation.py` at the test root suggests prior bug-fix cycles, but these are undocumented.** | Unknown what was broken and how it was fixed. This file has no corresponding spec or tracking document. |
| 15 | **The GA uses `random.seed(42)` hardcoded.** The genetic algorithm produces the same sequence of "random" results on every run. | Good for reproducibility in tests. Bad if users run the simulation twice with the same inputs and expect the GA to explore differently. They will get identical results — likely without knowing why. |
| 16 | **`exporter.py` exists in `application/services/` but is not mentioned in any specification document.** | An entire export feature that's undocumented. It's unclear what it exports, in what format, and whether it's reachable from the UI. |

---

## 🔵 P3 — Low / Observations

| # | Observation |
|---|---|
| 17 | `StrategyResult` Pydantic model exists in `strategy.py` but the pipeline uses raw `list[dict]`. The model is effectively dead code — its validators never run on real data. |
| 18 | `debug.py` auto-serializes ALL active memory state to a JSON file on disk. In a production cloud deployment, this creates uncontrolled data persistence of user financial data. It should be gated by an environment flag. |
| 19 | `filters.py` and `pages/` in the UI layer are completely undocumented in all specifications. These features exist but their purpose and scope are unknown from reading the docs. |
| 20 | The app version is `v27.6.0` in `pyproject.toml` and hardcoded as `v27.6` in `app.py`. For a project at this stage, v27 implies either very aggressive versioning or a non-standard version numbering convention. The version is not driven by environment config. |
| 21 | `helpers.py` injects raw HTML into the Streamlit global scope. This is a known Streamlit anti-pattern as it can interact unpredictably with Streamlit's own DOM management. |
| 22 | The `40/100 qualitative floor pruning` mentioned in the original data schema spec (auto-rejecting low-scoring properties before optimization) was never implemented. This could be a meaningful performance optimization for large datasets — skipping clearly poor properties before they enter the combo generator. |

---

## Summary

| Priority | Count | Theme |
|---|---|---|
| 🔴 P0 | 6 | Architectural misrepresentation, tax modeling gaps, dangerous UI patterns |
| 🟠 P1 | 5 | Silent failures, type system bypassed, spec drift |
| 🟡 P2 | 5 | Undocumented features, GA determinism, empty test tiers |
| 🔵 P3 | 6 | Code hygiene, versioning, minor gaps |

**The single most impactful fix** would be the field naming standardization (see `HANDOFF_DataModel_Refactoring.md`). It resolves P0-3 (the `__getitem__` shim), unlocks P1-8 (proper typed pipeline), and reduces the maintenance surface for every future feature.
