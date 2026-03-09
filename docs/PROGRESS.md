# Project Remediation Progress Plan

**Date:** 2026-03-07  
**Context:** Based on Audit pass (`RAPPORT_ETONNEMENT_Audit.md` & `HANDOFF_DataModel_Refactoring.md`)  
**Objective:** Remediate P0-P3 gaps, align codebase with theoretical capabilities, and ensure living specifications.

---

## Process Rule: Spec Updates
For **every** task in this plan, the final step is to **update the relevant spec documents** in `/docs`. Changes in logic, structures, or architecture must immediately be reflected in:
- `01_requirements/` (PRD, Feature Specs)
- `02_design/` (Data Schema, Taxonomies)
- `03_architecture/` (SAD, ADRs)

All spec updates should be detailed as commits referencing the respective fix.

---

## Phase 1: Data Model Standardization & Typed Pipeline (P0, P1)
*Focus: Eliminate Triple Rename & Re-establish Type Safety*

- [x] **1.1 Canonical Naming Pipeline**
  - Standardize `tension_locative_score_norm`, `budget_travaux`, etc., bridging the JSON -> Archetype -> Brick Dict gap.
  - Eliminate redundant aliases (`nom_bien`, `taux_pret`, `duree_pret`, `credit_final`).
  - *Spec Update: Update `02_design/03_Data_Schema_and_Taxonomy_Specification.md` to map the exact field names used.*
- [x] **1.2 Deprecate Brick Shim**
  - Delete `__getitem__`, `__contains__`, and `get()` from `InvestmentBrick`.
  - Migrate all exception-based accesses (`try/except KeyError`) in `allocator.py` to direct typed attribute access.
  - *Spec Update: Revise structural docs in `03_architecture/` that mention the Brick pipeline data structures.*
- [x] **1.3 Typed Strategy Pipeline**
  - Ensure `PortfolioStrategy` and `StrategyResult` validation is strictly enforced; replace raw `list[dict]` usages with domain models.
  - *Spec Update: Document typed interfaces.*

## Phase 2: Core Simulation & Tax Modeling Corrections (P0, P1)
*Focus: Financial Formula Accuracy*

- [x] **2.1 LMNP 10-Year Deficit Expiry**
  - Implement 10-year expiration constraint for `déficits LMNP`.
  - *Spec Update: Create or update mathematical appendices in System Architecture / Financial Glossaries.*
- [x] **2.2 Land Ratio Defaults (`ratio_terrain_pct`)**
  - Fix default from 0% to realistic figures (e.g., 15-25% based on location), preventing overstated tax shields.
  - *Spec Update: Document default assumptions in feature specifications.*
- [x] **2.3 `SCI IS` Regime Implementation**
  - Build actual branch simulation logic for `SCI IS` (currently falling back to LMNP silently).
  - *Spec Update: Update tax engine specifications with SCI IS rules.*

## Phase 3: Architectural Safety & Error Handling (P0, P1)
*Focus: Stop Silent Failures & UI Thrashing*

- [x] **3.1 Simulation Engine Error Propagation**
  - Remove bare `except Exception: pass` in `app_controller.py`. Provide user-facing error boundaries.
  - *Spec Update: Log error states and fallback rules in `02_design/04_QA_and_Testing_Strategy.md`.*
- [x] **3.2 UI Thread Protection (`sensitivity.py`)**
  - Refactor to prevent Streamlit from running full simulations inside the primary UI render loop. Add debouncing or caching.
  - *Spec Update: Note concurrency/rendering limits in performance architectural documents.*
- [x] **3.3 Implementation of `health_check.py`**
  - Create the required executable gate mentioned in specs but missing in code.
  - *Spec Update: Detail health check behaviors.*

## Phase 4: Algorithms, UI Hygiene, and Configuration (P2, P3)
*Focus: Documentation Drift & Edge Behaviours*

- [x] **4.1 Algorithm Specs vs Reality**
  - Correct the specs that claim GA is threaded/concurrent (it is sequential).
  - *Spec Update: Updated `03_architecture/`, `01_App_Immo_Engine_TDD.md`, and `04_as_built/01_System_Architecture_Document.md`. Also changed GA default seed from 42 to None (true random).*
- [x] **4.2 Scoring Redesign Reflection**
  - Update specifications to explain the 7 factors used in qualitative scoring rather than the original 3.
  - *Spec Update: `03_Data_Schema_and_Taxonomy_Specification.md` already accurately describes all 7 scoring factors. No further changes needed.*
- [x] **4.3 Config & Persistence Cleanup**
  - Remove deterministic GA seeding (`random.seed(42)`) from production paths.
  - Environment gate disk serialization in `debug.py`.
  - *Spec Update: Added `APP_DEBUG` variable to `.env.example`. Debug writes now only execute when `APP_DEBUG=true`.*

## Phase 5: Final Validation & Testing Alignment (P2)
*Focus: Testing Missing Links*

- [x] **5.1 Test Coverage Enhancements**
  - Populate the empty `verification/` folder with structural mathematical checks.
  - Document what `test_audit_remediation.py` accomplishes.
  - *Spec Update: Updated `QA_and_Testing_Strategy.md` to include the new `tests/verification/test_math_invariants.py` tier.*
  - Added 16 structural invariant tests: amortization math, DSCR bounds, golden PMT values, edge case boundaries.
