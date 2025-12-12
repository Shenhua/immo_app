# 🧩 Spécifications Fonctionnelles & Techniques — Super-Simulateur de Stratégies Immobilières (v2.1)

> Version : **2.1 — Intégration complète du schéma `archetypes_recale_2025_v2`**
> Date : 2025-10-10
> Objectif : version unifiée complète incluant le modèle d’archétype enrichi, la logique métier, le moteur financier, et les aspects UI/UX.

---

## 1. Objectifs Produit

1. Composer une **stratégie d’investissement** à partir d’**archétypes enrichis (v2)**.
2. **Simuler** cash-flows, endettement, et patrimoine sur un **horizon dynamique** (par défaut 25 ans, ajustable).
3. **Évaluer et classer** les stratégies selon des **KPIs financiers et qualitatifs**.
4. **Comparer, visualiser et exporter** les meilleures stratégies (JSON, UI Streamlit).

**Non-objectifs (v2)** : fiscalité exhaustive, solveur global, gestion multi-utilisateur.

---

## 2. Architecture & Contrats

### 2.1 Modules Fonctionnels

- `app` : orchestration + interface Streamlit (UI).
- `strategy_finder` : génération, simulation et classement des stratégies.
- `financial_calculations` : moteur de calcul (mensualités, échéancier, IRR, etc.).
- `utils_core` : I/O, validation, helpers d’encadrement et typologie.
- `utils_ui` : helpers d’affichage, color coding, formatage dynamique (Ha).
- `config` : paramètres et hypothèses par défaut.
- `models` : schémas Pydantic des archétypes, briques et stratégies.

> Organisation libre tant que les **signatures publiques** (§10) restent stables.

### 2.2 Dépendances minimales
Python 3.10+, `streamlit`, `pandas`, `numpy`, `numpy_financial`, `pydantic`, `plotly`.

### 2.3 Contrats d’Architecture

- Séparation stricte **UI / logique**.  
- Pureté : fonctions métier sans I/O sauf dans `utils_core` et `app`.  
- Tests : unitaires, intégration, property-based.  
- Exceptions typées, pas de `except Exception:` générique.  
- CI : `pytest`, `black`, `ruff`, `mypy`, hook anti-ellipses.

---

## 3. Données & Source des Archétypes (v2.1)

### 3.1. Sources et Priorités

- **Upload JSON utilisateur**
- **Source intégrée `v2 enrichie`**
- **Priorité** : `Upload > Source intégrée`.

> Au premier lancement : sélection par défaut de `v2 enrichie`.  
> Si aucun fichier local trouvé → message : “Aucune donnée locale. Utilisez la source v2 enrichie ou importez un JSON.”

### 3.2. Schéma d’Archétype — v2.1 (étendu)

| Champ | Type | Description |
|:--|:--|:--|
| `nom` | str | Nom complet de l’actif (ville + quartier). |
| `ville` | str | Ville principale. |
| `surface` | float | Surface en m² (> 0). |
| `prix_m2` | float | Prix du m². |
| `loyer_m2` | float | Loyer mensuel au m². |
| `mode_loyer` | enum | `meuble_classique`, `meuble_etudiant`, `colocation_meublee`, `nu_classique`, `saisonnier`, etc. |
| `meuble` | bool | Bien meublé ? |
| `soumis_encadrement` | bool | Bien soumis à encadrement des loyers ? |
| `loyer_m2_max` | float \| null | Plafond applicable si encadrement actif. |
| `charges_m2_an` | float | Charges annuelles au m². |
| `taxe_fonciere_m2_an` | float | Taxe foncière annuelle au m². |
| `valeur_mobilier` | float | Valeur du mobilier (€). |
| `budget_travaux` | float | Travaux prévus (€). |
| `dpe_initial` | str | Classe énergétique initiale (A–G ou ND). |
| `dpe_objectif` | str | Objectif de rénovation énergétique (A–G). |
| `renovation_energetique_cout` | float | Coût de rénovation énergétique (€). |
| `facteurs_qualitatifs` | dict | `{tension_locative, potentiel_valorisation, qualite_emplacement}` (valeurs textuelles). |
| `tension_locative_score_abs` | float \| null | Indice absolu de tension locative (ex. LocService). |
| `tension_locative_score_norm` | float | Score normalisé (0–1). |
| `tension_locative_category` | str \| null | Libellé qualitatif ("Très tendu", "Équilibré…"). |
| `transport_score` | float | Accessibilité transport (0–1). |
| `transport_modes` | list[str] | Liste de modes : `["metro", "tram", "bus", …]`. |
| `delai_vente_j_median` | int | Délai médian de vente (jours). |
| `liquidite_score` | float | Score de liquidité (0–1). |
| `data_sources` | dict | Provenances des indicateurs (`tension`, `transport`, `delai_vente`). |

### 3.3. Règles de Validation

- Champs absents → valeurs par défaut (`None` ou `0.0`).  
- `loyer_effectif = min(loyer_m2, loyer_m2_max)` si encadrement actif.  
- Scores bornés entre 0 et 1.  
- Fallback automatique pour DPE.

### 3.4. Pondérations Qualitatives

| Domaine | Champ | Poids | Rôle |
|:--|:--|:--|:--|
| Tension locative | `tension_locative_score_norm` | 0.4 | Sécurité locative |
| Transport | `transport_score` | 0.3 | Attractivité |
| Liquidité | `liquidite_score` | 0.2 | Revente |
| Délai vente | `delai_vente_j_median` | 0.1 | Frein revente |

---

## 4. Filtres & Recherche de Biens

- Multiselect dynamiques pour villes et types.  
- Checkbox “Appliquer encadrement des loyers” activée par défaut.  
- Si aucun filtre → tous les biens affichés.

---

## 5. Paramètres de Simulation & Fiscalité

- **Horizon dynamique** : slider `horizon_ans`.  
- **IRA** : option activable (min(6 mois intérêts, 3% CRD)).  
- **Régime fiscal** : LMNP réel / Micro-BIC (abattement 50%, désactive TMI+PS).

---

## 6. Calculs Financiers

- Mensualité, assurance et échéancier inchangés.  
- Simulation longue : prend en compte IRA et horizon variable.  
- Fonction `calculate_ira()` conforme norme FR.

---

## 7. Scoring & Sélection des Stratégies

- Signature unique avec `eval_params`.  
- Catégorisation : *Optimisé*, *Patrimonial*, *Mix*.  
- Comparatif coloré : vert = mieux, rouge = pire.

---

## 8. UX / UI (Streamlit)

- Interface regroupée par sections : Source, Filtres, Paramètres.  
- Bouton “Lancer l’Analyse Stratégique 🚀”.  
- Étapes de progression textuelles.  
- Affichage version (v2.1.x) et sources de données.

---

## 9. Fonctions Publiques Stables

```python
# utils_core
def load_archetypes_from_json(path_or_file) -> list[Archetype]: ...

# financial_calculations
def mensualite_et_assurance(...): ...
def echeancier_mensuel(...): ...
def simuler_strategie_long_terme(...): ...
def calculate_ira(crd: float, taux_annuel_pct: float) -> float: ...

# strategy_finder
def creer_briques_investissement(...): ...
def trouver_top_strategies(..., eval_params: dict | None = None) -> list[dict]: ...
```

---

## 10. Validation & Tests

- Unitaires : mensualités, IRR, encadrement, IRA, typologie.  
- Intégration : pipeline complet (archétypes → briques → simulation).  
- Acceptation : 6 cas fonctionnels (sources, filtres, horizon, fiscalité, comparatif, typologie).

---

## 11. Non-Fonctionnels

- Performance stable.  
- Robustesse et valeurs par défaut sûres.  
- Pas de doublons UI, indentation 4 espaces.  
- Logging structuré niveau DEBUG.

---

## 12. Extensibilité

- Fiscalité : plugins LMNP / Micro-BIC / futur “Nu foncier”.  
- Scoring : presets Rentabilité / Sécurité / Équilibré.  
- Optimisation : solveur ou recuit simulé via `Optimizer`.

---

## 13. Glossaire

- **CF** : Cash-flow net annuel.  
- **DSCR** : `NOI / DebtService`.  
- **IRR (TRI)** : Taux de rendement interne annuel.  
- **Liquidation nette** : Valeur nette après revente.  
- **Cap_eff** : `(liquidation_nette - apport_total) / apport_total`.  
- **Enrich_net** : `liquidation_nette - apport_total`.  
- **IRA** : Indemnités de remboursement anticipé.

---

## 14. Arborescence Indicative

```
app.py
strategy_finder.py
financial_calculations.py
utils_core.py
utils_ui.py
config.py
models.py
tests/
data/
```

---

## 15. Versioning

- SemVer : rupture de signature publique = version majeure.  
- Version affichée dans UI et entête code.

---
