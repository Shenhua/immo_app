# Data Schema & Taxonomy Specification: App Immo

**Document Status:** Approved
**Author:** AI Technical Lead
**Date:** 2026-03-06
**Related TDD:** [App Immo Engine TDD](01_App_Immo_Engine_TDD.md)

## 1. Executive Summary
This document defines the strict data taxonomy, field validations, and enumerations expected by the App Immo mathematical engine. It covers the exact structure of the `ArchetypeV2` template (used for ingestion) and the `InvestmentBrick` (the dynamically generated mathematical object). 

All external JSON ingestion structures (e.g., `data/archetypes_recale_2025_v2.json`) MUST adhere to these schema definitions.

---

## 2. Global Enumerations (Taxonomy)

### 2.1 Market Tension
Rental demand heuristic used for qualitative scoring. **Note:** The system does not use a discrete zone enum. The actual field in `ArchetypeV2` is `tension_locative_score_norm` — a normalized float `[0.0, 1.0]`. The zone descriptions below are informational only:
*   **`~1.0`**: Exceptionally high tension (Paris equivalent). Standard yield compression.
*   **`~0.75`**: High tension (Major metropolises like Lyon, Bordeaux).
*   **`~0.5`**: Moderate tension (Large regional cities) — used as neutral default.
*   **`~0.25`**: Low tension (Medium regional cities).
*   **`~0.0`**: No tension (Rural). Highest statistical risk.

### 2.2 Energy Performance Rating (`DPE_Rating`)
Diagnostique de Performance Énergétique. Stored as string field `dpe_initial` in `ArchetypeV2`. Used by `scoring.py:DPE_SCORES` for qualitative scoring and by `brick_factory.py` to conditionally include `renovation_energetique_cout`.
*   **`A`, `B`, `C`**: Premium. No renovation cost applied.
*   **`D`**: Acceptable. Baseline configuration. Qualitative score: 0.6.
*   **`E`**: Degraded. Qualitative score: 0.3. `renovation_energetique_cout` included in brick cost.
*   **`F`, `G`**: Thermal "passoires". Qualitative score: 0.0. `renovation_energetique_cout` included.
*   **`ND`**: Not determined. Treated as `D` (score: 0.6).
* Note: The renovation cost is read from the archetype JSON field `renovation_energetique_cout` (pre-calculated), NOT computed as `Surface × Rate`.

---

## 3. Data Entities

### 3.1 `ArchetypeV2` (Raw Ingestion Schema)
The root Pydantic template. Instances of this model represent theoretical properties.

| Field Name | Type | Validations / Constraints | Description |
| :--- | :--- | :--- | :--- |
| `nom` | `str` | Required | Unique property identifier/name. |
| `ville` | `str` | Required | City name. |
| `surface` | `Float` | `> 0` | Habitable square meters. |
| `prix_m2` | `Float` | `>= 0` | Purchase price per m². |
| `loyer_m2` | `Float` | `>= 0` | Monthly rent per m². |
| `loyer_m2_max` | `Float\|None` | Optional | Rent cap per m² (regulated zone). |
| `mode_loyer` | `str` | Required | Rental mode: `meuble_classique`, `meuble_boost`, `nu`. |
| `mode_loyer_label` | `str` | Optional | Human-readable rental mode label. |
| `meuble` | `bool` | Default `False` | Whether furnished. |
| `soumis_encadrement` | `bool` | Default `False` | Subject to rent control. |
| `dpe_initial` | `str` | `A`–`G` or `ND` | DPE energy rating (validated via `@field_validator`). |
| `charges_m2_an` | `Float` | `>= 0` | Annual co-ownership charges per m². |
| `taxe_fonciere_m2_an` | `Float` | `>= 0` | Annual property tax per m². |
| `budget_travaux` | `Float` | `>= 0` | Renovation budget in €. |
| `renovation_energetique_cout` | `Float` | `>= 0` | Energy renovation cost in € (pre-calculated, applied only if DPE ∈ E/F/G). |
| `valeur_mobilier` | `Float` | `>= 0` | Furniture value in €. |
| `tension_locative_score_norm` | `Float` | `[0, 1]` | Market tension normalized score. |
| `vacance_pct` | `Float` | `[0, 100]` | Vacancy rate percentage. |
| `transport_score` | `Float` | `[0, 1]` | Transport accessibility score. |
| `liquidite_score` | `Float` | `[0, 1]` | Resale liquidity score. |
| `transport_modes` | `List[str]` | Optional | Available transport modes. |
| `delai_vente_j_median` | `int` | `>= 0` | Median days to sell. |
| `facteurs_qualitatifs` | `Dict[str, str]` | Optional | Qualitative factor key-value pairs. |
| `frais_gestion_pct` | `Float\|None` | Optional | Management fee % override (colocation). |

> **Note:** `ArchetypeV2` has **no** `id` (uuid), `type_bien` enum, or zone-based `tension_marche` enum field. The model uses `extra="forbid"` to reject unexpected fields.

### 3.2 `InvestmentBrick` (Generated Financial Entity)
The computational core. An `InvestmentBrick` is an `ArchetypeV2` that has been explicitly financed by the `brick_factory.py`.

| Computed Field Name | Type | Formula / Origin | Purpose |
| :--- | :--- | :--- | :--- |
| `cout_total` | `Float` | `prix_achat_bien + frais_notaire + budget_travaux + renovation_energetique + mobilier` | Total capital to acquire and equip the property. |
| `frais_notaire` | `Float` | `prix_achat * frais_notaire_pct / 100` (default 7.5%) | Notary fees at acquisition. |
| `budget_travaux` | `Float` | From archetype `budget_travaux`; 0 if `inclure_travaux=False` | Renovation cost. |
| `renovation_energetique` | `Float` | From `renovation_energetique_cout`; applied only if DPE ∈ E/F/G and `inclure_reno_ener=True` | Energy renovation cost. |
| `mobilier` | `Float` | From `valeur_mobilier`; 0 if `inclure_mobilier=False` | Furniture cost. |
| `apport_min` | `Float` | `frais_notaire + (prix_achat * apport_min_pct / 100)` | Minimum required down payment. |
| `capital_emprunte` | `Float` | `cout_total_projet - apport_min + frais_pret` | Borrowed amount including loan setup fees. |
| `duree_credit_mois` | `Int` | From `FinancingConfig.credit_rates` keys × 12 | Loan duration in **months** (not years). One brick variant is generated per configured duration. |
| `taux_annuel_pct` | `Float` | From `FinancingConfig.credit_rates[duration]` | Annual interest rate for this specific duration. |
| `pmt_principal_interet` | `Float` | `numpy_financial.pmt()` | Monthly P&I payment. |
| `pmt_assurance` | `Float` | `capital_emprunte * assurance_ann_pct / 100 / 12` | Monthly insurance. |
| `pmt_total` | `Float` | `pmt_principal_interet + pmt_assurance` | Total monthly debt service (computed field). |
| `loyer_mensuel_initial` | `Float` | `surface * loyer_m2` | Initial monthly rent. |
| `qual_score_bien` | `Float` | Computed by `scoring.py:calculate_property_qualitative_score()` | Qualitative score 0–100. |

> **Note:** `brick_factory.py` generates **one `InvestmentBrick` per (archetype × loan duration)** combination based on `FinancingConfig.credit_rates`. The loan durations (e.g. 15, 20, 25 years) are user-configurable in the UI, not hardcoded.

### 3.3 `PortfolioStrategy` (Aggregated Simulation Model)
The ultimate wrapper containing arrays of `InvestmentBrick`s and single-source metrics.

| Field Name | Type | Validations / Constraints | Description |
| :--- | :--- | :--- | :--- |
| `details` | `List[Dict]` | `length >= 1` | Serialized `InvestmentBrick` data dicts (via `model_dump()`). |
| `apport_total` | `Float` | `>= 0` | Final down payment after allocator optimization. |
| `cash_flow_final` | `Float` | Derived | Net monthly cash flow after allocation. |
| `balanced_score` | `Float` | Derived from scorer | Combined finance + qualitative score (0–100). |
| `qual_score` | `Float` | Default `50.0` | Qualitative score 0–100. |
| `taxonomy` | `String\|None` | Optional | Strategy type label: `Optimisé`, `Patrimonial`, `Mix`. |
| `nombre_biens` | `Int` | `@computed_field` | Count of properties: `len(details)`. |

> **Note:** Deduplication in `dedupe_strategies()` uses `tuple(sorted(brick.nom for brick in details))` as signature — **not** a hashed UUID field. There is no `identifiant_unique` or `synthese_metrics` field in the model.

---

The system scores properties using a **7-factor weighted vector** defined in `src/domain/calculator/scoring.py:DEFAULT_QUAL_WEIGHTS`.

**`DEFAULT_QUAL_WEIGHTS` Vector (actual code):**
| Factor | Weight | Notes |
|:---|:---:|:---|
| `tension` | **25%** | Market tension score (`tension_locative_score_norm`) |
| `transport` | **15%** | Transport accessibility (`transport_score`) |
| `dpe` | **15%** | DPE score via `DPE_SCORES` lookup (A=1.0, B=0.9, C=0.8, D=0.6, E=0.3, F/G=0.0) |
| `encadrement` | **15%** | Rent cap margin: `(cap - loyer) / cap` if regulated, else 0.6 |
| `vacance` | **10%** | Vacancy score: `1 - vacance_pct/10` (or tension as fallback) |
| `travaux` | **10%** | Inverse renovation ratio: `1 - (travaux / prix_achat)` |
| `liquidite` | **10%** | Resale liquidity (`liquidite_score`) |

Final score is `weighted_sum × 100` (range 0–100). Weights are **user-overridable** via `calculate_property_qualitative_score(weights=...)` parameter.

> **Note:** There is **no 40/100 floor pruning** implemented in the code. The qualitative score is used as a weighted component in the fitness function, not as a hard cutoff gate.
