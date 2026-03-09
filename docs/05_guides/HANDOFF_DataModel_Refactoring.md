# Handoff: Data Model Refactoring — Field Naming Standardization

**Date:** 2026-03-06  
**Context:** Produced during Phase 4 of the Spec Audit (PRD → code verification pass)  
**Priority:** High — affects simulation correctness, spec accuracy, and developer onboarding

---

## 1. Problem Summary

The data pipeline has **three naming layers** for the same concepts:

```
JSON File (archetypes_recale_2025_v2.json)
    ↓  renamed by brick_factory.py
InvestmentBrick (in-memory Pydantic model)
    ↓  re-aliased via __getitem__ shim
Strategy details dict (passed to simulation)
```

Each transition renames several fields. The result is:
- Fields like `tension_locative_score_norm` become `indice_tension` in the brick
- `budget_travaux` becomes `travaux` in the brick dict
- `nom` becomes `nom_bien` mid-pipeline, then back to `nom`
- Duration exists as both `duree_pret` (years) and `duree_credit_mois` (months)

The `InvestmentBrick.__getitem__` compatibility shim (lines 90–118 of `brick.py`) exists **solely** to patch over these renames. It is a warning sign, not a feature.

---

## 2. Field-by-Field: What to Do

### 2.1 Legacy Aliases to Remove (Identical Concepts, Two Names)

| Canonical Name (keep) | Legacy Alias (remove) | Location |
|---|---|---|
| `taux_annuel_pct` | `taux_pret` | `brick_factory.py` brick_dict |
| `assurance_annuelle_pct` | `assurance_ann_pct` | `brick_factory.py` brick_dict |
| `duree_credit_mois` | `duree_pret` *(years)* | `brick_factory.py` brick_dict |
| `capital_emprunte` | `credit_final` | `brick_factory.py` brick_dict |
| `nom` | `nom_bien` | `brick_factory.py` (rename then rename back) |
| `pmt_total` | stored redundantly as literal + computed field | `brick_factory.py` brick_dict |

**How:** Remove the legacy key assignments in `brick_factory.py` (lines 161–165). Update any callers that use the old key names. Run `grep -r "taux_pret\|nom_bien\|credit_final\|duree_pret\|assurance_ann_pct"` to find all usages.

---

### 2.2 Renamed Mid-Pipeline (Source ≠ Brick Name)

| JSON / Archetype field | InvestmentBrick field | Recommended action |
|---|---|---|
| `tension_locative_score_norm` | `indice_tension` | **Standardize on `tension_locative_score_norm`** in both archetype and brick |
| `budget_travaux` | `budget_travaux` (archetype) → `travaux` (brick dict alias) | **Keep `budget_travaux` throughout**; remove `travaux` alias |
| `renovation_energetique_cout` | `renovation_energetique` (brick) | Acceptable rename — cost → amount after application; document this clearly |
| `valeur_mobilier` | `mobilier` (brick) | Acceptable rename — same logic as above |
| `charges_m2_an` | `charges_const_mth0` (brick, monthly) | Keep both: raw per-m² in archetype, computed monthly in brick — but **name the computed field clearly**: `charges_mensuelles_initiales` |
| `taxe_fonciere_m2_an` | `tf_const_mth0` (brick, monthly) | Same — rename `tf_const_mth0` → `taxe_fonciere_mensuelle_initiale` |

---

### 2.3 Fields to Keep As-Is (Core Financial Inputs)

These are clean, unambiguous, and used correctly everywhere:

```
surface, prix_m2, loyer_m2, loyer_m2_max
soumis_encadrement, meuble, mode_loyer
dpe_initial, vacance_pct, transport_score, liquidite_score
prix_achat_bien, frais_notaire, apport_min, capital_emprunte
pmt_principal_interet, pmt_assurance, taux_annuel_pct
loyer_mensuel_initial, qual_score_bien
```

---

### 2.4 Fields to Keep for Future Development

These aren't fully exploited yet but represent real data we have:

| Field | Future Use |
|---|---|
| `delai_vente_j_median` | Exit timing sensitivity (how long to liquidate) |
| `transport_modes` | Location intelligence / desirability scoring |
| `facteurs_qualitatifs` | Free-text override notes per property |
| `liquidite_score` | Already in scoring — extend to exit-risk weighting |

---

### 2.5 Fields to Evaluate for Removal

| Field | Reason |
|---|---|
| `mode_loyer_label` | UI display string derived from `mode_loyer` — compute it, don't store it |
| `facteurs_qualitatifs` | Currently unused in any calculation; keep only if notes are surfaced in the UI |

---

## 3. What to Eliminate: The `__getitem__` Shim

**File:** `src/domain/models/brick.py` — lines 90–131

This shim (`__getitem__`, `__contains__`, `get()`) lets the `simulation.py` and `allocator.py` access brick fields using legacy key names like `"nom_bien"`, `"taux_pret"`, `"duree_pret"`. 

**Once the field names are standardized (section 2.1 above), this shim can be deleted entirely.** That's ~42 lines of defensive code gone, and `InvestmentBrick` becomes a clean Pydantic model.

**Steps:**
1. Fix all field names in `brick_factory.py`
2. Run `grep -rn "__getitem__\|\.get(\|nom_bien\|taux_pret\|duree_pret\|credit_final"` in `src/`
3. Update each call site to use the canonical field name
4. Delete `__getitem__`, `__contains__`, `get()` from `InvestmentBrick`
5. Run the test suite

---

## 4. What to Standardize in the JSON Data File

`data/archetypes_recale_2025_v2.json` currently uses:
- `tension_locative_score_norm` → rename the brick field to match this (don't rename the JSON)
- All other JSON keys are clean — **do not change the JSON schema**

---

## 5. Modeling Limitations to Address Eventually

These are known simplifications flagged during the audit. They're not wrong per se, but future developers should be aware:

| Limitation | Impact | Notes |
|---|---|---|
| **No 10-year deficit carryforward expiry** | Tax optimization overstated for very long holds | The real LMNP rule limits deficit carryforward to 10 years |
| **Land (`terrain`) ratio defaults to 0%** | 100% of purchase price is depreciable — overestimates amortization | A real Paris property has ~15–25% land value that cannot be depreciated |
| **No `SCI IS` simulation path** | The regime is selectable in the UI but simulation falls back to LMNP logic | `SCI IS` has fundamentally different depreciation and tax rules |
| **`vacance_pct` used as both qualitative score input AND operating cost** | A property with 5% vacancy shows in two unrelated places | Consider splitting into `vacance_scoring_pct` vs `provision_pct` |

---

## 6. Recommended Refactoring Order

1. **Phase A — Naming cleanup** *(low risk, high reward)*
   - Standardize `tension_locative_score_norm` → `indice_tension` (or vice versa) everywhere
   - Remove `nom_bien` shim; `nom` only
   - Remove `taux_pret`, `credit_final`, `assurance_ann_pct`, `duree_pret` aliases from brick_dict
   - Rename `charges_const_mth0` → `charges_mensuelles_initiales`
   - Rename `tf_const_mth0` → `taxe_fonciere_mensuelle_initiale`

2. **Phase B — Remove shim** *(after Phase A is done and tests green)*
   - Delete `__getitem__`, `__contains__`, `get()` from `InvestmentBrick`

3. **Phase C — Modeling improvements** *(higher complexity, requires tax expertise)*
   - Add 10-year deficit expiry counter
   - Add `ratio_terrain_pct` archetype field (defaults to 15% for cities, 5% rural)
   - Implement `SCI IS` simulation branch

4. **Phase D — Spec alignment** *(documentation, low risk)*
   - Update all spec documents to reflect the standardized field names
   - The audit in `docs/05_guides/audit_results.md` tracks which docs already corrected

---

*This document was produced during the Spec Audit, Phase 4 (Data Schema & Taxonomy). See `audit_results.md` in the brain artifacts for the full audit trail.*
