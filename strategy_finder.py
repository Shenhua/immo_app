import inspect
import itertools
import math
import os
from financial_calculations import simuler_strategie_long_terme, mensualite_et_assurance, _accept_cf, k_factor
from utils import calculate_qualitative_score
from typing import Any, Dict, List, Tuple, Callable, Optional

# Module-level default eval params (used by local_optimality_check adapter)
_default_eval_params: Dict[str, Any] = {}
BASE_WEIGHTS = {
    'enrich_net': 0.30,
    'irr': 0.25,
    'cf_proximity': 0.20,  # was 'cf'
    'dscr': 0.15,          # was 'dscr_y1'
    'cap_eff': 0.10,
}



# Fichier: strategy_finder.py

def _call_sim_with_horizon(sim_fn, *args, horizon_years=None, **kwargs):
    """Call sim_fn with filtered kwargs; add horizon_years only if supported."""
    if not callable(sim_fn):
        raise TypeError("sim_fn is not callable")
    sig = inspect.signature(sim_fn)
    params = sig.parameters
    allowed_kwargs = {k: v for k, v in kwargs.items() if k in params}
    if horizon_years is not None and 'horizon_years' in params:
        allowed_kwargs['horizon_years'] = horizon_years
    return sim_fn(*args, **allowed_kwargs)

def trouver_top_strategies(apport_disponible: float, cash_flow_cible: float, tolerance: float, briques: list, mode_cf: str, qualite_weight: float, eval_params: dict | None = None, horizon_years: int = 25):
    # --- helpers ---
    from math import isfinite

    def _renorm_weights(w: dict) -> dict:
        if not isinstance(w, dict):
            return BASE_WEIGHTS
        s = sum(max(0.0, float(v)) for v in w.values() if v is not None)
        if s <= 0:
            return BASE_WEIGHTS
        return {k: max(0.0, float(v)) / s for k, v in w.items()}

    def _minmax(seq, lo=None, hi=None):
        vals = [v for v in seq if isfinite(v)]
        if not vals:
            return [0.0]*len(seq)
        if lo is None: lo = min(vals)
        if hi is None: hi = max(vals)
        if hi <= lo:
            return [0.5]*len(seq)
        out = []
        for v in seq:
            if not isfinite(v):
                out.append(0.0)
            else:
                vv = v
                if lo is not None: vv = max(lo, vv)
                if hi is not None: vv = min(hi, vv)
                out.append((vv - lo) / (hi - lo))
        return out

    def _dedupe_strategies(strategies):
        """
        Supprime les doublons quasi-identiques :
        signature = tuples triés (nom_bien, durée, apport_final_bien arrondi à 100€).
        """
        seen = set()
        out = []
        for s in strategies:
            details = s.get('details', [])
            sig = tuple(sorted(
                (
                    d.get('nom_bien'),
                    int(d.get('duree_pret')),
                    int(round(float(d.get('apport_final_bien', 0.0))/100.0)*100),
                )
                for d in details
            ))
            if sig in seen:
                continue
            seen.add(sig)
            out.append(s)
        return out
    
    scenarios = []
    max_biens = 3
    for k in range(1, max_biens + 1):
        for combo in itertools.combinations(briques, k):
            # interdire deux fois le même bien
            noms_des_biens = {c["nom_bien"] for c in combo if "nom_bien" in c}
            if len(noms_des_biens) != len(combo):
                continue
            apport_min_combo = sum(c.get("apport_min", 0.0) for c in combo)
            if apport_min_combo > apport_disponible:
                continue
            apport_suppl_max = apport_disponible - apport_min_combo
            ok, details, cf_mth, apport_total = allouer_apport_pour_cf(list(combo), cash_flow_cible, tolerance, apport_suppl_max, mode_cf)
            if ok:
                scenarios.append({
                    "details": details,
                    "apport_total": apport_total,
                    "patrimoine_acquis": sum(b.get("cout_total", 0.0) for b in combo),
                    "cash_flow_final": cf_mth
                })
    if not scenarios:
        return []

    # ---- Finance metrics via simulation (IRR, liquidation, DSCR, CF proximity) ----
    defaults = {
        "duree_simulation_ans": 25,
        "hypotheses_marche": {"appreciation_bien_pct": 2.0, "revalo_loyer_pct": 1.5, "inflation_charges_pct": 2.0},
        "regime_fiscal": "lmnp",
        "tmi_pct": 30.0,
        "frais_vente_pct": 6.0,
        "cfe_par_bien_ann": 150.0,
        "apply_ira": True,
        "ira_cap_pct": 3.0,
    }
    ep = {**defaults, **(eval_params or {})}

    def _compute_dscr_y1(df):
        try:
            import pandas as pd  # noqa
            if df is None or df.empty:
                return 0.0
            row = df.iloc[0].to_dict()
            cap = float(row.get("Capital Remboursé", row.get("Capital Rembourse", 0.0)))
            ia  = float(row.get("Intérêts & Assurance", row.get("Interets & Assurance", 0.0)))
            ds  = cap + ia
            loy = float(row.get("Loyers Bruts", 0.0))
            # charges sont souvent négatives (déduites)
            charges_key = None
            for k in row.keys():
                if "Charges" in k and ("Dédu" in k or "Dedu" in k or "Dédu" in k):
                    charges_key = k; break
            chg = float(row.get(charges_key, 0.0)) if charges_key else 0.0
            noi = loy + chg
            if ds <= 1e-9:
                return None  # or float('inf')
            return (noi / ds)
        except Exception:
            return 0.0

    for s in scenarios:
        try:
            df_sim, bilan = _call_sim_with_horizon(simuler_strategie_long_terme, 
                s,
                horizon_years=horizon_years,
                duree_simulation_ans=int(ep.get("duree_simulation_ans", 25)),
                hypotheses_marche=ep.get("hypotheses_marche"),
                regime_fiscal=ep.get("regime_fiscal", "lmnp"),
                tmi_pct=float(ep.get("tmi_pct", 30.0)),
                frais_vente_pct=float(ep.get("frais_vente_pct", 6.0)),
                cfe_par_bien_ann=float(ep.get("cfe_par_bien_ann", 150.0))
            )
            s['tri_annuel'] = float(bilan.get('tri_annuel', 0.0))
            s['liquidation_nette'] = float(bilan.get('liquidation_nette', 0.0))
            s['dscr_y1'] = _compute_dscr_y1(df_sim)
        except Exception:
            s['tri_annuel'] = 0.0
            s['liquidation_nette'] = 0.0
            s['dscr_y1'] = 0.0

        # Qualitatif & distance CF
        try:
            s['qual_score'] = calculate_qualitative_score(s)  # 0..100
        except Exception:
            s['qual_score'] = 50.0
        s['cf_distance'] = abs(float(s.get('cash_flow_final', 0.0)) - float(cash_flow_cible))
        ap = float(s.get('apport_total', 0.0)) or 1.0
        s['cap_eff'] = (s['liquidation_nette'] - ap) / ap
        s['enrich_net'] = s['liquidation_nette'] - float(s.get('apport_total', 0.0))

    # ---- Score financier avec vrais poids (override UI si dispo) ----
    weights = ep.get("finance_weights_override") or BASE_WEIGHTS
    weights = _renorm_weights(weights)

    enrich = [s['enrich_net'] for s in scenarios]
    irr    = [s['tri_annuel'] for s in scenarios]
    capef  = [s['cap_eff'] for s in scenarios]
    # DSCR None (pas de dette) → top de l’échelle: 1.5 (cap)
    dscr   = [1.5 if s.get('dscr_y1', None) is None else max(0.0, float(s.get('dscr_y1', 0.0))) for s in scenarios]
    cfdist = [abs(s.get('cf_distance', 0.0)) for s in scenarios]
    cfprox = [max(0.0, 1.0 - (d/300.0)) for d in cfdist]  # 0..1 (0€ d'écart => 1.0)

    # Normalisations robustes (clamp)
    enr_n = _minmax(enrich)                                   # data-driven
    irr_n = _minmax(irr)                                      # data-driven
    cap_n = _minmax([min(v, 6.0) for v in capef], 0.0, 6.0)   # cap 6x
    dscr_n = _minmax([min(v, 1.5) for v in dscr], 0.0, 1.5)   # cap 1.5
    cf_n  = cfprox                                            # déjà 0..1

    for s, a,b,c,d,e in zip(scenarios, enr_n, irr_n, cap_n, dscr_n, cf_n):
        s['enrich_norm'] = a
        s['tri_norm'] = b
        s['cap_eff_norm'] = c
        s['dscr_norm'] = d
        s['cf_proximity'] = e
        fin = (weights['enrich_net']*a + weights['irr']*b + weights['cap_eff']*c
               + weights['dscr']*d + weights['cf_proximity']*e)
        s['finance_score'] = fin

        # mix avec qualité
        qw = qualite_weight if isinstance(qualite_weight, (int, float)) else 0.0
        qs = s.get('qual_score', 0.0) / 100.0
        s['balanced_score'] = (1.0 - qw)*fin + qw*qs

    # ---- Tri piloté par le preset ----
    preset_name = str((ep.get("finance_preset_name") if ep else "") or "Équilibré (défaut)")
    name = preset_name.lower()

    def _sort_key(x):
        # priorité selon le preset
        if "dscr" in name:
            return (x['dscr_norm'], x['finance_score'], -abs(x.get('cf_distance', 0.0)))
        if "irr" in name or "rendement" in name:
            return (x['tri_norm'], x['finance_score'], -abs(x.get('cf_distance', 0.0)))
        if "cash" in name:
            return (x['cf_proximity'], x['finance_score'], x['dscr_norm'])
        if "patrimoine" in name:
            return (x['balanced_score'], x['enrich_norm'], x['tri_norm'])
        # équilibré
        return (x['balanced_score'], x['dscr_norm'], x['tri_norm'])

    # scenarios.sort(key=_sort_key, reverse=True)
    # return scenarios[:6]
    # Déduplication avant tri/retour
    scenarios = _dedupe_strategies(scenarios)
    scenarios.sort(key=_sort_key, reverse=True)
    return scenarios[:6]
def creer_briques_investissement(archetypes: list, _taux_credits, frais_gestion_pct: float, provision_pct: float, frais_notaire_pct: float, apport_min_pct_prix: float, inclure_travaux: bool, inclure_reno_ener: bool, inclure_mobilier: bool, financer_mobilier: bool, assurance_ann_pct: float, frais_pret_pct: float, cfe_par_bien_ann: float):
    briques = []; 
    for a in archetypes:
        surf = a["surface"]; prix_achat = surf * a["prix_m2"]; loyer_mth0 = surf * (float(a["loyer_m2"]) or 0.0); charges_mth0 = (surf * a["charges_m2_an"]) / 12.0; tf_mth0 = (surf * a["taxe_fonciere_m2_an"]) / 12.0; travaux = a.get("budget_travaux", 0.0) if inclure_travaux else 0.0; renoEner = float(a.get("renovation_energetique_cout", 0.0)) if inclure_reno_ener and a.get("dpe_initial", "D").upper() == "E" else 0.0; mobilier = a.get("valeur_mobilier", 0.0) if inclure_mobilier else 0.0; frais_notaire = prix_achat * (frais_notaire_pct / 100.0); cout_total = prix_achat + frais_notaire + travaux + renoEner + mobilier; apport_min = frais_notaire + prix_achat * (apport_min_pct_prix / 100.0)
        if inclure_mobilier and not financer_mobilier: apport_min += mobilier
        credit_base = max(0.0, cout_total - apport_min); frais_pret = credit_base * (frais_pret_pct / 100.0); capital_emprunte = credit_base + frais_pret; gest_mth0 = loyer_mth0 * (frais_gestion_pct / 100.0); prov_mth0 = loyer_mth0 * (provision_pct / 100.0); depenses_mth0 = charges_mth0 + tf_mth0 + gest_mth0 + prov_mth0 + (cfe_par_bien_ann / 12.0); base_data = a.copy(); base_data['nom_bien'] = base_data.pop('nom')
        for duree, taux in _taux_credits.items():
            _, _, p_tot = mensualite_et_assurance(capital_emprunte, taux, int(duree), assurance_ann_pct)
            calculated_data = {"prix_achat_bien": prix_achat, "frais_notaire": frais_notaire, "budget_travaux": travaux, "renovation_energetique": renoEner, "mobilier": mobilier, "cout_total": cout_total, "apport_min": apport_min, "capital_emprunte": capital_emprunte, "duree_pret": int(duree), "taux_pret": float(taux), "assurance_ann_pct": float(assurance_ann_pct), "pmt_total": p_tot, "loyer_mensuel_initial": loyer_mth0, "charges_const_mth0": charges_mth0, "tf_const_mth0": tf_mth0, "frais_gestion_pct": frais_gestion_pct, "provision_pct": provision_pct, "depenses_mensuelles_hors_credit_initial": depenses_mth0, "credit_final": capital_emprunte}
            final_brick = {**base_data, **calculated_data}; briques.append(final_brick)
    return briques
def allouer_apport_pour_cf(combo, cf_cible, tol, apport_suppl_max: float, mode_cf: str):
    biens = [{**b, "capital_restant": b["capital_emprunte"]} for b in combo]; cf0 = sum(b["loyer_mensuel_initial"] - b["depenses_mensuelles_hors_credit_initial"] - b["pmt_total"] for b in biens)
    if _accept_cf(cf0, cf_cible, tol, mode_cf):
        details = [{**b, "apport_add_bien": 0.0, "apport_final_bien": b["apport_min"], "credit_final": b["capital_restant"]} for b in biens]; return True, details, cf0, sum(d["apport_final_bien"] for d in details)
    apport_rest = max(0.0, apport_suppl_max); ordre = sorted(biens, key=lambda x: k_factor(x["taux_pret"], x["duree_pret"], x["assurance_ann_pct"]), reverse=True)
    def besoin(cf_obs): return max(0.0, (cf_cible - cf_obs)) if mode_cf == "min" else (cf_cible - cf_obs if abs(cf_cible - cf_obs) > tol else 0.0)
    manque_cf = besoin(cf0)
    for b in ordre:
        if apport_rest <= 1e-9 or manque_cf <= 1e-9: break
        k = k_factor(b["taux_pret"], b["duree_pret"], b["assurance_ann_pct"])
        if k <= 0: continue
        apport_necessaire = manque_cf / k; delta = min(apport_rest, apport_necessaire, b["capital_restant"])
        # --- CAP EXTRA APPORT (pour éviter le remboursement total au t=0) ---
        # Limite par bien: MAX_EXTRA_APPORT_PCT * capital_emprunte initial (par défaut 75%)
        max_extra_pct = float(os.getenv("MAX_EXTRA_APPORT_PCT", "0.75"))
        cap_extra = max_extra_pct * float(b.get("capital_emprunte", b.get("capital_restant", 0.0)))
        deja = float(b.get("apport_add_bien", 0.0))
        reste_cap = max(0.0, cap_extra - deja)
        if delta > reste_cap:
            delta = reste_cap
        # Toujours laisser un reliquat minimal pour éviter cap/assurance à zéro strict.
        b["capital_restant"] = max(1e-2, b["capital_restant"] - delta)
        _, _, p_tot = mensualite_et_assurance(b["capital_restant"], b["taux_pret"], b["duree_pret"], b["assurance_ann_pct"]) 
        b["pmt_total"] = p_tot 
        b["apport_add_bien"] = b.get("apport_add_bien", 0.0) + delta 
        apport_rest -= delta; cf0 = sum(x["loyer_mensuel_initial"] - x["depenses_mensuelles_hors_credit_initial"] - x["pmt_total"] for x in biens); manque_cf = besoin(cf0)
    ok = _accept_cf(cf0, cf_cible, tol, mode_cf); details = [{**b, "apport_add_bien": b.get("apport_add_bien", 0.0), "apport_final_bien": b["apport_min"] + b.get("apport_add_bien", 0.0), "credit_final": b["capital_restant"]} for b in biens]; apport_total = sum(d["apport_final_bien"] for d in details); return ok, details, cf0, apport_total

# ==== Integrated from overlay_patch_v2 (Round 2) ====
def _norm01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def clamp_norm_metrics(m: Dict[str, float]) -> Dict[str, float]:
    dscr_raw = float(m.get('dscr_y1', m.get('dscr', 0.0)))
    irr_raw = float(m.get('irr', m.get('tri_annuel', 0.0)))
    cap_raw = float(m.get('cap_eff', 0.0))
    enrich_raw = float(m.get('enrich_net', m.get('enrichissement_net', 0.0)))

    # CF proximity
    if 'cf_proximity' in m:
        cf_prox = float(m['cf_proximity'])
    else:
        d = abs(float(m.get('cf_distance', 0.0)))
        scale = max(50.0, float(m.get('cf_scale', 300.0)))
        cf_prox = max(0.0, 1.0 - (d / scale))

    dscr_norm = min(dscr_raw, 1.5) / 1.5
    cap_eff_norm = min(cap_raw, 6.0) / 6.0
    irr_norm = _norm01(irr_raw, 0.0, 0.20)  # cap IRR at 20%

    # Soft logistic for enrichment normalization
    k = 1.0 / 75000.0
    enrich_norm = 1.0 - math.exp(-k * max(0.0, enrich_raw))

    cf_prox_norm = max(0.0, min(1.0, cf_prox))

    return dict(
        dscr_norm=dscr_norm,
        irr_norm=irr_norm,
        cap_eff_norm=cap_eff_norm,
        cf_prox_norm=cf_prox_norm,
        enrich_norm=enrich_norm,
    )


def compute_finance_score(norms: Dict[str, float], weights: Dict[str, float]) -> float:
    w = {k: float(v) for k, v in weights.items()}
    total = sum(w.get(k, 0.0) for k in ['dscr', 'irr', 'cap_eff', 'cf_proximity', 'enrich_net'])
    if total <= 0:
        return 0.0
    for k in w:
        w[k] = w[k] / total
    return (
        w.get('dscr', 0.0) * norms.get('dscr_norm', 0.0) +
        w.get('irr', 0.0) * norms.get('irr_norm', 0.0) +
        w.get('cap_eff', 0.0) * norms.get('cap_eff_norm', 0.0) +
        w.get('cf_proximity', 0.0) * norms.get('cf_prox_norm', 0.0) +
        w.get('enrich_net', 0.0) * norms.get('enrich_norm', 0.0)
    )


def balanced_score(finance_score: float, qual_score_pct: float, w_qual: float) -> float:
    wq = max(0.0, min(1.0, float(w_qual)))
    return (1.0 - wq) * float(finance_score) + wq * (float(qual_score_pct) / 100.0)


def sort_key_by_preset(preset: str, s: Dict[str, float]) -> Tuple:
    p = (preset or '').strip().lower()
    if p.startswith('dscr'):
        return (s.get('dscr_norm', 0.0), s.get('finance_score', 0.0), s.get('cf_prox_norm', 0.0))
    if p.startswith('rend') or 'irr' in p:
        return (s.get('irr_norm', 0.0), s.get('finance_score', 0.0))
    if 'cash' in p:
        return (s.get('cf_prox_norm', 0.0), s.get('finance_score', 0.0), s.get('dscr_norm', 0.0))
    return (s.get('balanced_score', 0.0), s.get('finance_score', 0.0), s.get('dscr_norm', 0.0))


def pareto_filter(strats: List[Dict[str, Any]], tol_apport: int = 100) -> List[Dict[str, Any]]:
    if not strats:
        return []
    buckets = {}
    for s in strats:
        used = int(round(float(s.get('apport_utilise', s.get('apport_total', 0))), -2))
        key = int(used / max(1, tol_apport))
        buckets.setdefault(key, []).append(s)
    result = []
    for key, items in buckets.items():
        kept = []
        for i, a in enumerate(items):
            dominated = False
            for j, b in enumerate(items):
                if i == j:
                    continue
                if float(b.get('apport_utilise', b.get('apport_total', 0))) <= float(a.get('apport_utilise', a.get('apport_total', 0))):
                    if (
                        float(b.get('finance_score', 0)) >= float(a.get('finance_score', 0))
                        and (
                            (b.get('balanced_score', 0) > a.get('balanced_score', 0)) or
                            (b.get('dscr_norm', 0) > a.get('dscr_norm', 0)) or
                            (b.get('irr_norm', 0) > a.get('irr_norm', 0)) or
                            (b.get('cf_prox_norm', 0) > a.get('cf_prox_norm', 0))
                        )
                    ):
                        dominated = True
                        break
            if not dominated:
                kept.append(a)
        result.extend(kept)
    return result

# ---------------- Horizon helpers ----------------


def resimulate_with_horizon(
    strategies: List[Dict[str, Any]],
    horizon_years: int,
    simulate_fn: Callable[[Dict[str, Any], int], Dict[str, float]],
    weights: Dict[str, float],
    preset_name: str,
    w_quality: float = 0.25,
    apply_pareto: bool = True,
) -> List[Dict[str, Any]]:
    """Re-simulate each strategy at the given horizon, recompute scores, sort & pareto-filter."""
    out = []
    for s in strategies:
        metrics = simulate_fn(s, horizon_years)  # adapter provided by host app
        norms = clamp_norm_metrics(metrics)
        fs = compute_finance_score(norms, weights)
        bs = balanced_score(fs, s.get('qualite_score', s.get('qualite', 0.0)), w_quality)
        enriched = dict(s)
        enriched.update(norms)
        enriched['finance_score'] = fs
        enriched['balanced_score'] = bs
        enriched['duree_simulation_ans'] = horizon_years
        out.append(enriched)
    if apply_pareto:
        out = pareto_filter(out)
    out.sort(key=lambda x: sort_key_by_preset(preset_name, x), reverse=True)
    return out

# ---------------- Optimiser (adapter-based) ----------------


def optimise_apport_greedy(
    strategy: Dict[str, Any],
    horizon_years: int,
    simulate_proxy_fn: Callable[[Dict[str, Any], List[int], int], Dict[str, float]],
    simulate_final_fn: Callable[[Dict[str, Any], List[int], int], Dict[str, float]],
    initial_apports: List[int],
    apport_budget: int,
    weights: Dict[str, float],
    optim_cfg: Optional[Dict[str, Any]] = None,
    cf_target: float = 0.0,
    cf_scale: float = 300.0,
    loan_terms: Optional[List[Tuple[float, int, float]]] = None,
) -> Tuple[List[int], Dict[str, float]]:
    """
    Greedy multi-pass optimiser:
    - simulate_proxy_fn(strategy, apports, horizon) -> metrics 'cf', 'dscr', 'cap_eff', 'irr', 'enrich_net'
    - simulate_final_fn(...) same signature, but performs full simulation.
    - loan_terms: list of (annual_rate, months, insurance_annual) per bien. Only used if proxy needs it.
    Returns (best_apports, final_metrics).
    """
    n = len(initial_apports)
    apports = list(initial_apports)
    total_apport = sum(apports)
    if total_apport > apport_budget:
        # normalise down proportionally
        scale = apport_budget / float(total_apport)
        apports = [int(round(a*scale/100.0)*100) for a in apports]

    def eval_proxy(apports_vec: List[int]) -> float:
        m = simulate_proxy_fn(strategy, apports_vec, horizon_years)
        return proxy_finance_score_from_cf_dscr(
            cf=m.get('cf', 0.0), dscr=m.get('dscr_y1', m.get('dscr', 0.0)),
            cap_eff=m.get('cap_eff', 0.0), irr=m.get('irr', 0.0),
            enrich_net=m.get('enrich_net', 0.0),
            weights=weights, cf_target=cf_target, cf_scale=cf_scale
        )

    best_score = eval_proxy(apports)

    # Get config values with defaults (optim_cfg is a dict, not an object)
    cfg = optim_cfg or {}
    step_coarse = int(cfg.get('step_coarse', 5000))
    step_fine = int(cfg.get('step_fine', 1000))
    max_passes = int(cfg.get('max_passes', 10))
    eps_gain = float(cfg.get('eps_gain', 1e-6))

    # Coarse then fine passes
    for step in (step_coarse, step_fine):
        for _ in range(max_passes):
            improved = False
            # try moving 'step' from i -> j
            for i in range(n):
                if apports[i] < step:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    # move
                    apports[i] -= step
                    apports[j] += step
                    # round to 100€
                    apports[i] = int(round(apports[i] / 100.0) * 100)
                    apports[j] = int(round(apports[j] / 100.0) * 100)
                    score = eval_proxy(apports)
                    if score > best_score + eps_gain:
                        best_score = score
                        improved = True
                    else:
                        # revert
                        apports[j] -= step
                        apports[i] += step
            if not improved:
                break

    # Final heavy simulation once
    final_metrics = simulate_final_fn(strategy, apports, horizon_years)
    return apports, final_metrics


def local_optimality_check(
    strategy: Dict[str, Any],
    horizon_years: int,
    simulate_final_fn: Callable[[Dict[str, Any], List[int], int], Dict[str, float]],
    apports: List[int],
    weights: Dict[str, float],
    delta: int = 1000,
    cf_target: float = 0.0,
    cf_scale: float = 300.0,
) -> Dict[str, Any]:
    """Return a dict with 'is_local_optimal' and optional 'better_variant' if found by ±delta moves."""
    n = len(apports)
    base = simulate_final_fn(strategy, apports, horizon_years)
    base_score = proxy_finance_score_from_cf_dscr(
        cf=base.get('cf', 0.0), dscr=base.get('dscr_y1', base.get('dscr', 0.0)),
        cap_eff=base.get('cap_eff', 0.0), irr=base.get('irr', 0.0),
        enrich_net=base.get('enrich_net', 0.0),
        weights=weights, cf_target=cf_target, cf_scale=cf_scale
    )
    best = (apports[:], base, base_score)
    improved = False
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if apports[i] < delta:
                continue
            trial = apports[:]
            trial[i] -= delta
            trial[j] += delta
            trial[i] = int(round(trial[i]/100.0)*100)
            trial[j] = int(round(trial[j]/100.0)*100)
            m = simulate_final_fn(strategy, trial, horizon_years)
            s = proxy_finance_score_from_cf_dscr(
                cf=m.get('cf', 0.0), dscr=m.get('dscr_y1', m.get('dscr', 0.0)),
                cap_eff=m.get('cap_eff', 0.0), irr=m.get('irr', 0.0),
                enrich_net=m.get('enrich_net', 0.0),
                weights=weights, cf_target=cf_target, cf_scale=cf_scale
            )
            if s > best[2] + 1e-6:
                best = (trial, m, s)
                improved = True
    if improved:
        return {"is_local_optimal": False, "better_variant": {"apports": best[0], "metrics": best[1]}}
    return {"is_local_optimal": True}

# ---------------- High-level pipeline helpers ----------------


def optimise_and_resimulate_pipeline(
    strategies: List[Dict[str, Any]],
    horizon_years: int,
    preset_name: str,
    weights: Dict[str, float],
    w_quality: float,
    optimiser_enabled: bool,
    simulate_proxy_fn: Callable[[Dict[str, Any], List[int], int], Dict[str, float]],
    simulate_final_fn: Callable[[Dict[str, Any], List[int], int], Dict[str, float]],
    initial_apports_key: str = "apports_par_bien",
    apport_budget_key: str = "apport_budget",
    add_local_certificate: bool = True,
    pareto: bool = True,
) -> List[Dict[str, Any]]:
    """
    For each strategy:
      - if optimiser_enabled → run greedy + final sim; else → just final sim at current horizon.
      - recompute finance/balanced scores.
      - optional Pareto + sort by preset.
    Minimal UX changes: you can call this before rendering your cards.
    """
    out = []
    for s in strategies:
        apports_init = list(s.get(initial_apports_key, []))
        budget = int(s.get(apport_budget_key, sum(apports_init)))

        if optimiser_enabled and apports_init:
            apports_opt, metrics = optimise_apport_greedy(
                strategy=s, horizon_years=horizon_years,
                simulate_proxy_fn=simulate_proxy_fn,
                simulate_final_fn=simulate_final_fn,
                initial_apports=apports_init,
                apport_budget=budget,
                weights=weights,
            )
        else:
            apports_opt = apports_init
            metrics = simulate_final_fn(s, apports_opt, horizon_years)

        norms = clamp_norm_metrics(metrics)
        fs = compute_finance_score(norms, weights)
        bs = balanced_score(fs, s.get('qualite_score', s.get('qualite', 0.0)), w_quality)

        enriched = dict(s)
        enriched.update(norms)
        enriched['finance_score'] = fs
        enriched['balanced_score'] = bs
        enriched['duree_simulation_ans'] = horizon_years
        enriched[initial_apports_key] = apports_opt

        if add_local_certificate and apports_opt:
            cert = local_optimality_check(
                strategy=s, horizon_years=horizon_years,
                simulate_final_fn=simulate_final_fn,
                apports=apports_opt, weights=weights
            )
            enriched['local_optimal'] = cert.get('is_local_optimal', False)
            if not enriched['local_optimal']:
                enriched['better_variant'] = cert.get('better_variant')

        out.append(enriched)

    if pareto:
        out = pareto_filter(out)
    out.sort(key=lambda x: sort_key_by_preset(preset_name, x), reverse=True)
    return out

# ==== Proxy/Adapter Helpers ====

def proxy_finance_score_from_cf_dscr(
    cf: float,
    dscr: float,
    cap_eff: float,
    irr: float,
    enrich_net: float,
    weights: Dict[str, float],
    cf_target: float = 0.0,
    cf_scale: float = 300.0,
) -> float:
    """Compute a finance score from raw metrics using proxy normalization."""
    cf_prox = max(0.0, 1.0 - abs(cf - cf_target) / max(50.0, cf_scale))
    norms = {
        'dscr_norm': min(max(dscr, 0.0), 1.5) / 1.5,
        'irr_norm': _norm01(max(0.0, irr), 0.0, 0.20),
        'cap_eff_norm': min(max(cap_eff, 0.0), 6.0) / 6.0,
        'cf_prox_norm': cf_prox,
        'enrich_norm': 1.0 - math.exp(-(max(0.0, enrich_net)) / 75000.0),
    }
    return compute_finance_score(norms, weights)


def _simulate_final_adapter(
    strategy: dict,
    apports: list,
    horizon_years: int,
    eval_params: dict,
) -> dict:
    """Canonical adapter: merges apports and calls engine with keyword args only."""
    s = dict(strategy)
    if apports:
        s["apports_par_bien"] = list(apports)

    df, bilan = _call_sim_with_horizon(
        simuler_strategie_long_terme,
        s,
        duree_simulation_ans=int(horizon_years),
        hypotheses_marche=eval_params["hypotheses_marche"],
        tmi_pct=float(eval_params["tmi_pct"]),
        frais_vente_pct=float(eval_params["frais_vente_pct"]),
        cfe_par_bien_ann=float(eval_params["cfe_par_bien_ann"]),
        micro_bic_abatt_pct=float(eval_params.get("micro_bic_abatt_pct", 50.0)),
        regime_fiscal=eval_params["regime_fiscal"],
        amort_mobilier=bool(eval_params.get("amort_mobilier", True)),
        lmnp_amort_mob=bool(eval_params.get("lmnp_amort_mob", True)),
        horizon_years=int(horizon_years),
        apply_ira=bool(eval_params.get("apply_ira", True)),
        ira_cap_pct=float(eval_params.get("ira_cap_pct", 3.0)),
    )

    # Extract metrics for scoring
    try:
        cf = float(df["cf_net_annuel"].iloc[-1]) if hasattr(df, "empty") and not df.empty else 0.0
    except Exception:
        cf = 0.0
    try:
        dscr = float(df.get("dscr_y1", df.get("dscr", 0.0)).iloc[0]) if hasattr(df, "empty") and not df.empty else 0.0
    except Exception:
        dscr = 0.0
    try:
        cap_eff = float(df.get("cap_eff", 0.0).iloc[-1]) if hasattr(df, "empty") and not df.empty else 0.0
    except Exception:
        cap_eff = 0.0
    irr = float(bilan.get("tri_annuel", 0.0)) / 100.0 if isinstance(bilan, dict) else 0.0
    enrich = float(bilan.get("liquidation_nette", bilan.get("enrichissement_net", 0.0))) if isinstance(bilan, dict) else 0.0

    return {"cf": cf, "dscr_y1": dscr, "cap_eff": cap_eff, "irr": irr, "enrich_net": enrich}


def local_optimality_check_with_adapter(
    strategy: Dict[str, Any],
    horizon_years: int,
    apports: List[int],
    weights: Dict[str, float],
    eval_params: Dict[str, Any],
    delta: int = 1000,
) -> Dict[str, Any]:
    """
    Alternative local optimality check using _simulate_final_adapter.
    
    Use this when you have eval_params dict and don't want to pass a callback.
    For callback-based usage, use local_optimality_check() instead.
    """
    base = _simulate_final_adapter(strategy, apports, horizon_years, eval_params)
    base_s = proxy_finance_score_from_cf_dscr(
        base['cf'], base['dscr_y1'], base['cap_eff'], base['irr'], base['enrich_net'], weights
    )
    n = len(apports)
    improved = False
    best = (apports[:], base_s, base)
    
    for i in range(n):
        for j in range(n):
            if i == j or apports[i] < delta:
                continue
            trial = apports[:]
            trial[i] -= delta
            trial[j] += delta
            trial[i] = int(round(trial[i] / 100.0) * 100)
            trial[j] = int(round(trial[j] / 100.0) * 100)
            m = _simulate_final_adapter(strategy, trial, horizon_years, eval_params)
            s = proxy_finance_score_from_cf_dscr(
                m['cf'], m['dscr_y1'], m['cap_eff'], m['irr'], m['enrich_net'], weights
            )
            if s > best[1] + 1e-6:
                best = (trial, s, m)
                improved = True
    
    if improved:
        return {"is_local_optimal": False, "better_variant": {"apports": best[0], "metrics": best[2]}}
    return {"is_local_optimal": True}


def _get_finance_weights(eval_params: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Get finance weights from eval_params or return BASE_WEIGHTS as default."""
    w = (eval_params or {}).get('finance_weights_override')
    if isinstance(w, dict) and w:
        return w
    return BASE_WEIGHTS

