# QA & Testing Strategy Document: App Immo Engine

**Document Status:** Approved
**Author:** AI QA Lead
**Date:** 2026-03-06
**Related PRD:** [App Immo PRD](../01_requirements/01_app_immo_PRD.md)

## 1. Executive Summary
The App Immo engine generates 25-year financial simulations based on localized French tax mechanics. Minor floating-point compound errors or incorrect loan scheduling will categorically invalidate the core product offering. Therefore, standard "code branching" unit tests are insufficient. The testing suite relies strictly on **Golden Value Regression Testing**, pure uncoupled mathematical validation, and distinct integration phases.

---

## 2. Core Testing Philosophies

### 2.1 Pure Function Isolation
*   All financial components within `src/domain/calculator/` (e.g., `financial.py`, `scoring.py`) and `src/core/glossary.py` must contain **ZERO I/O**. 
*   They must not interact with Streamlit, databases, or random generation logic. 
*   They must take standard Python primitives (`floats`, `ints`) and return equivalent standard types.

### 2.2 Anti-Degradation Tolerance
*   Testing is automated via `pytest`.
*   Coverage must exceed **95%** for all files within `domain/` and `application/services/`.
*   Floating-point equivalence must be evaluated using `pytest.approx()` with a tolerance strictness of `rel=1e-4` to bypass deep `numpy` binary variations without allowing actual numeric drift.

---

## 3. Test Suites & Taxonomy

### 3.1 Unit Testing (`tests/unit/`)
*   **Execution Time Constraint:** Entire suite must execute in `< 3.0 seconds`. It forms the basis of the `pre-commit` hook.
*   **Golden Value Integration:** Core financial functions are validated against hand-calculated or `numpy_financial`-verified expected values embedded directly in Python test cases (see `test_golden_validation.py`). Values are verified against known loan scenarios (e.g., 100k @ 3.6% over 25y = 506€/mo). **There are no external CSV/Excel files** — expected values are inline Python literals.
*   **Floating-point Tolerance:** Golden tests use `abs()` comparisons with monetary tolerances (e.g., `abs(result - expected) < 0.50`). The `pytest.approx(rel=1e-4)` convention described originally is **not used** in existing tests.
*   **Zero & Negative Boundaries:** `test_financial.py` validates that `calculate_monthly_payment()` and `generate_amortization_schedule()` return graceful **`0.0`** for zero/negative principal or duration — they do **not** raise `AppImmoError`. The exception classes are reserved for higher-level service errors (e.g., `SimulationError` for empty strategies).

### 3.2 Integration Testing (`tests/integration/`)
*   **Execution Time Constraint:** Runs on CI pipelines (`< 30.0 seconds`).
*   **Pipeline Verification:** Tests the combination of subsystems using two test files: `test_pipeline.py` and `test_invariants.py`. Tests build `FinancingConfig` + `OperatingConfig` instances, pass them through `brick_factory.py:create_investment_bricks()`, then run assertions across the abstractions directly. **`SimulationParams` is not used as the integration entry point** — financial configuration is passed via `FinancingConfig`/`OperatingConfig` data classes.

### 3.3 End-to-End Orchestration (`tests/e2e/`)
*   **Execution Time Constraint:** Runs nightly or manually prior to production releases (`< 5.0 minutes`).
*   **Files:** `test_full_pipeline.py` (full strategy search pipeline) and `test_qa_hardening.py` (edge cases and hardening scenarios).
*   **Full Matrix Tests:** Exercises `run_strategy_search()` via `app_controller.py` using a realistic dataset designed to trigger the `GeneticOptimizer`. Validates that the engine returns non-empty strategy lists. **There is no `SearchStats` block returned** — the function returns `list[dict]` with per-strategy metrics.
*   *(Note: Selenium/Playwright browser UI testing is currently deferred; E2E tests target the backend controller layer.)*

### 3.4 Additional Test Files (Not Tiered)
The following test files exist at the `tests/` root and are not part of any sub-tier:
*   `test_audit_remediation.py` — tests verifying fixes applied during audit-driven remediation cycles.
*   `verification_fixtures.py` — shared fixture data for verification scenarios.
*   `verification_scenarios.py` — scenario-based verification tests.
*   `tests/verification/` — reserved directory, currently **empty**.

---

## 4. Diagnostics & Static Health Checking

The App Immo suite employs a unique native CI verification mechanism encapsulated inside `src/health_check.py`.

### 4.1 Automated Health Check Gates
**Status: `src/health_check.py` is implemented and active.** The 4 automated gates described below are currently operational:
1.  **Syntax Checking:** Execute `ast.parse()` against all `.py` files to trap structural Python bugs prior to runtime.
2.  **Import Validation:** Guard against circular dependency failures.
3.  **Function Size Cap:** Flag any function strictly exceeding 50 lines (per the global `INSTRUCTIONS.md` rule: max 50 lines per function, not 100 as previously stated). Currently emits a warning without failing the build to support legacy code migration.
4.  **Configuration Verification:** Confirm `data/archetypes_recale_2025_v2.json` is valid, readable, and present.
