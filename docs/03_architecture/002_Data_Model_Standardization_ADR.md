# Architecture Decision Record: Data Model Standardization & Typed Pipeline

## Status
Accepted

## Context
The application (`app_immo`) previously suffered from severe technical debt due to a "triple rename pipeline" where data variables mapped inconsistently across different layers: JSON (`ArchetypeV2`), `InvestmentBrick` dict instances in `brick_factory.py`, and Pydantic validation structures. This resulted in the usage of legacy field names (e.g. `nom_bien`, `taux_pret`, `duree_pret`, `credit_final`) interacting via exception-based dictionary lookups throughout the allocation, scoring, simulation, and frontend components, posing a serious threat to the system's typed integrity and maintainability. In addition, the Pydantic class `InvestmentBrick` implemented legacy fallback dictionary functions (`__getitem__`, `__contains__`, and `get`) mapping legacy names seamlessly, which acted as a shim that papered over architecture deficiencies instead of enforcing correct typing.

## Decision
We have elected to standardize the data model, enforcing a robust typed pipeline.
1. All property fields now adhere to canonical names mapped directly to the `ArchetypeV2` ingestion schema and standard mathematical variables (e.g., `duree_credit_mois`, `taux_annuel_pct`, `capital_emprunte`, `nom`, `assurance_annuelle_pct`, and `tension_locative_score_norm`).
2. Legacy field aliases in `src/application/services/brick_factory.py` have been eliminated, and `InvestmentBrick` initialization directly utilizes Pydantic validation with these canonical names.
3. The custom dictionary shim lookup methods in `InvestmentBrick` (`__getitem__`, `get`, etc.) were removed to restore strict typed constraints.
4. Throughout the calculation engines (`evaluator.py`, `simulation.py`, `allocator.py`, `strategy_finder.py`) and UI (`results.py`), all instances of index-based queries acting on `InvestmentBrick` models have been migrated to use direct class attributes (`hasattr` and literal properties).
5. Existing test regression suites spanning E2E, integration invariants, and unit golden verification checks were patched iteratively to use standardized fields and objects.

## Consequences
- **Positive:** Type safety allows IDE static analyzers to detect bugs early. Performance is marginally enhanced by avoiding exception-based loops on large datasets (as done previously via `try/except KeyError` access patterns). Maintainability drastically improved since fields retain singular, consistent naming from the database through to the final frontend render.
- **Negative:** Downstream modules or custom notebook scripts executing logic against the old pseudo-dictionary representations of `InvestmentBricks` will abruptly break and require an update path mapping to class attributes alongside the new nomenclature.

## References
- `RAPPORT_ETONNEMENT_Audit.md`
- `HANDOFF_DataModel_Refactoring.md` (Phase 1 plan)
- `02_design/03_Data_Schema_and_Taxonomy_Specification.md`
