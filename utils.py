# Fichier: utils.py
import re
import json
import numpy as np
import streamlit as st
from typing import List, Dict, Any
from typing import Dict, Any
from config import TAXONOMY_THRESHOLDS, OPTIMIZED_MODES

def classify_strategy(strategy: Dict[str, Any], bilan: Dict[str, Any]) -> str:
    """
    Returns: 'Optimisé' | 'Patrimonial' | 'Mix'
    Uses simple signals: yield/CF/part de modes optimisés + qualitatif.
    """
    th = TAXONOMY_THRESHOLDS

    # 1) Signals from bilan / metrics already computed
    yield_pct = float(bilan.get("yield_net_H_pct") or bilan.get("coc_H_pct") or 0.0)
    cf_m = float(bilan.get("cf_mensuel_net_H") or bilan.get("cf_mensuel_net") or 0.0)
    qual = float(bilan.get("qual_score_norm") or bilan.get("qual_score") or 0.0)

    # 2) Look at mix of biens (details list exists on your strategies)
    details = strategy.get("details") or []
    total = max(1, len(details))
    share_optimized = 0.0
    if details:
        modes = [str(d.get("mode_loyer","")).strip() for d in details]
        share_optimized = sum(1 for m in modes if m in OPTIMIZED_MODES) / total

    # 3) Rules
    if (yield_pct >= th["min_yield_pct"] or cf_m >= th["min_cf_month"] or
        share_optimized >= th["min_share_optimized_modes"]):
        return "Optimisé"

    if (qual >= th["min_qual_score"] and yield_pct <= th["max_yield_for_patrimonial"]):
        return "Patrimonial"

    return "Mix"

def load_archetypes_from_json(file_obj_or_path):
    try:
        if hasattr(file_obj_or_path, "read"): data = json.load(file_obj_or_path)
        else:
            with open(file_obj_or_path, "r", encoding="utf-8") as f: data = json.load(f)
        assert isinstance(data, list)
        return data
    except Exception as e:
        st.error(f"Erreur de lecture du JSON : {e}")
        return []

def enforce_compliance(archetypes: List[Dict[str, Any]], apply_rent_cap: bool = True) -> List[Dict[str, Any]]:
    cleaned = []
    for a in archetypes:
        if (a.get("dpe_initial") or "").upper() in ("F", "G"): continue
        if apply_rent_cap and a.get("soumis_encadrement") and a.get("loyer_m2_max") is not None:
            if a.get("loyer_m2") and a["loyer_m2"] > a["loyer_m2_max"]:
                a = {**a, "loyer_m2": float(a["loyer_m2_max"])}
        cleaned.append(a)
    return cleaned

def colorize_kpi(value, kpi_name, size="h3"):
    color_map = {"green": "#28a745", "orange": "#fd7e14", "red": "#dc3545"}
    color = "red" 
    if kpi_name == "tri":
        if value > 7: color = "green"
        elif value > 4: color = "orange"
        formatted_value = fmt_pct(value, 2)
    elif kpi_name == "coc":
        if value > 5: color = "green"
        elif value >= -1e-9: color = "orange" 
        formatted_value = fmt_pct(value, 1)
    elif kpi_name == "dscr":
        if value > 1.2: color = "green"
        elif value >= 1.0: color = "orange"
        formatted_value = f"{value:.2f}"
    elif kpi_name == "cf":
        if value >= -1e-9: color = "green"
        formatted_value = f"{fmt_e(value)} €"
    else: formatted_value = str(value)
    hex_color = color_map.get(color, "#FFFFFF")
    style = f"color: {hex_color}; margin-top: -10px; margin-bottom: -10px; padding: 0; font-weight: bold;"
    if size == "h3": return f"<h3 style='{style}'>{formatted_value}</h3>"
    elif size == "h4": return f"<h4 style='{style}'>{formatted_value}</h4>"
    return f"<span style='{style}'>{formatted_value}</span>"



def extract_city(nom: str) -> str:
    m = re.search(r"\(([^)]+)\)", nom);
    if not m: return "N/A"
    inside = m.group(1); city = inside.split("-")[0].strip()
    return city or "N/A"
def fmt_e(x):
    try: return f"{int(round(float(x))):,}".replace(",", " ")
    except (ValueError, TypeError): return "—"
def fmt_pct(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    return f"{round(float(x), d)}%"
def badges_strategie(strat: dict, apport_disponible: float) -> list:
    labels = []
    if strat["cash_flow_final"] >= 0: labels.append("✅ CF ≥ 0 an1")
    if apport_disponible > 0 and strat["apport_total"] >= 0.8 * apport_disponible: labels.append("💰 Utilise ≥ 80% de l’apport")
    nb = len(strat["details"]); labels.append("🧩 3 biens" if nb >= 3 else ("🧩 2 biens" if nb == 2 else "🏢 1 bien"))
    villes = {d.get("ville", "N/A") for d in strat["details"]};
    if len(villes) >= 2: labels.append("🌍 Multi‑villes")
    modes = {d.get("mode_loyer", "") for d in strat["details"]};
    if any("colocation" in m for m in modes): labels.append("👥 Colocation")
    if any("etudiant" in m for m in modes): labels.append("🎓 Étudiant")
    return labels


# ===== Qualitative scoring helpers =====
QUAL_WEIGHTS = {
    "tension": 0.25,       # marché tendu => vacance moindre
    "transport": 0.15,     # accès/temps transport
    "dpe": 0.15,           # DPE initial (A..G)
    "encadrement": 0.15,   # marge vs loyer max si encadrement
    "vacance": 0.10,       # proxie de vacance = tension (positif si élevé)
    "travaux": 0.10,       # ratio travaux/prix (inverse)
    "liquidite": 0.10      # facilité de revente
}

def _dpe_score(letter: str) -> float:
    m = {"A":1.0,"B":0.9,"C":0.8,"D":0.6,"E":0.3,"F":0.0,"G":0.0}
    return m.get((letter or "").upper(), 0.6)

def compute_qual_score_bien(source: dict, *, loyer_m2: float|None=None, loyer_m2_max: float|None=None,
                            prix_achat: float=0.0, travaux: float=0.0) -> tuple[float, dict]:
    """Retourne (score_0_100, details_dict_0_1) pour un bien."""
    tension   = float(source.get("indice_tension", 0.5))
    transport = float(source.get("transport_score", 0.5))
    dpe       = _dpe_score(source.get("dpe_initial", source.get("dpe", "D")))

    # marge encadrement si soumise (loyer_m2_max > loyer_m2 => mieux)
    if source.get("soumis_encadrement") and loyer_m2_max:
        try:
            enc_marge = max(0.0, min(1.0, (float(loyer_m2_max) - float(loyer_m2 or 0.0)) / max(1e-6, float(loyer_m2_max))))
        except Exception:
            enc_marge = 0.6
    else:
        enc_marge = 0.6  # neutre si hors-encadrement / inconnu

    vac_pos = max(0.0, min(1.0, tension))  # tension haute => vacance basse (positif)
    ratio_trav = min(1.0, float(travaux or 0.0) / max(1e-6, float(prix_achat or 0.0)))
    trav_pos = 1.0 - ratio_trav
    liq = float(source.get("liquidite_score", 0.5))

    feats = {
        "tension": vac_pos,
        "transport": transport,
        "dpe": dpe,
        "encadrement": enc_marge,
        "vacance": vac_pos,
        "travaux": trav_pos,
        "liquidite": liq
    }
    score01 = sum(QUAL_WEIGHTS[k] * max(0.0, min(1.0, feats.get(k, 0.5))) for k in QUAL_WEIGHTS)
    return round(score01 * 100.0, 1), feats

def calculate_qualitative_score(strategy: dict) -> float:
    """Score 0..100 d'une stratégie. Agrège par bien pondéré par cout_total."""
    details = strategy.get("details") or []
    if details:
        total = sum(d.get("cout_total", 0.0) for d in details) or 1.0
        agg = 0.0
        for d in details:
            qs = d.get("qual_score_bien")
            if qs is None:
                loyer_m2 = d.get("loyer_m2")
                loyer_m2_max = d.get("loyer_m2_max")
                prix_achat = d.get("prix_achat_bien", d.get("cout_total", 0.0))
                travaux = d.get("travaux", 0.0)
                qs, _ = compute_qual_score_bien(d, loyer_m2=loyer_m2, loyer_m2_max=loyer_m2_max,
                                                prix_achat=prix_achat, travaux=travaux)
            agg += qs * (d.get("cout_total", 0.0) / total)
        return round(agg, 1)
    return 50.0

def display_score_stars(score: float, help_text: str):
    """Affiche un score en étoiles 0..5. Accepte aussi 0..100 et convertit automatiquement."""
    import streamlit as st
    try:
        s = float(score or 0.0)
    except Exception:
        s = 0.0
    if s > 5.0 and s <= 100.0:
        s = s / 20.0
    s = max(0.0, min(5.0, s))
    stars = "⭐" * int(round(s)) + "☆" * (5 - int(round(s)))
    st.markdown("**Score Qualitatif**", help=help_text)
    st.markdown(f"<div style='font-size:1.1rem'>{s:.1f}/5 {stars}</div>", unsafe_allow_html=True)

def to_json_safe(obj):
    """Recursively convert common Python/numpy/datetime etc. to JSON-safe types."""
    import datetime as _dt
    try:
        import numpy as _np
    except Exception:
        _np = None

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if _np is not None:
        if isinstance(obj, (_np.integer,)):  # type: ignore[attr-defined]
            return int(obj)
        if isinstance(obj, (_np.floating,)):  # type: ignore[attr-defined]
            return float(obj)
        if isinstance(obj, (_np.bool_,)):    # type: ignore[attr-defined]
            return bool(obj)

    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()

    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(x) for x in obj]

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                key = str(k)
            except Exception:
                key = repr(k)
            out[key] = to_json_safe(v)
        return out

    if callable(obj):
        return f"<callable:{getattr(obj, '__name__', 'func')}>"
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"

def _coerce_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def apply_rent_cap(loyer_m2: float, loyer_m2_max, apply_cap: bool = True) -> float:
    """Apply encadrement cap to base loyer_m2 if toggle is ON and cap exists."""
    if not apply_cap:
        return _coerce_float(loyer_m2)
    cap = None if (loyer_m2_max is None or loyer_m2_max == "") else _coerce_float(loyer_m2_max, None)
    base = _coerce_float(loyer_m2)
    return min(base, cap) if cap is not None else base

REQUIRED_FIELDS_V2 = [
    "nom","ville","surface","prix_m2","loyer_m2",
    "mode_loyer","meuble","soumis_encadrement",
    "charges_m2_an","taxe_fonciere_m2_an",
]

V2_DEFAULTS = {
    "dpe_initial":"ND","dpe_objectif":"ND",
    "loyer_m2_max": None,
    "valeur_mobilier":0.0,"budget_travaux":0.0,"renovation_energetique_cout":0.0,
    "tension_locative_score_norm":0.5,"transport_score":0.5,"liquidite_score":0.5,
    "delai_vente_j_median":0,"transport_modes":[]
}

def validate_archetypes_v2(items: list) -> list:
    if not isinstance(items, list):
        raise ValueError("Archetypes must be a list[dict]")
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        missing = [k for k in REQUIRED_FIELDS_V2 if k not in it]
        if missing:
            it = it.copy()
            for k in missing:
                if k in ("nom","ville","mode_loyer"):
                    it[k] = it.get(k, "N/A")
                elif k in ("meuble","soumis_encadrement"):
                    it[k] = bool(it.get(k, False))
                else:
                    it[k] = _coerce_float(it.get(k, 0.0))
        it2 = {**V2_DEFAULTS, **it}
        for k in ("tension_locative_score_norm","transport_score","liquidite_score"):
            v = _coerce_float(it2.get(k, 0.5), 0.5)
            it2[k] = max(0.0, min(1.0, v))
        try:
            it2["delai_vente_j_median"] = max(0, int(it2.get("delai_vente_j_median", 0) or 0))
        except Exception:
            it2["delai_vente_j_median"] = 0
        out.append(it2)
    return out

def _dpe_score_v2(letter: str) -> float:
    """DPE score mapping for v2 archetypes (includes ND handling)."""
    m = {"A": 1.0, "B": 0.9, "C": 0.8, "D": 0.6, "E": 0.3, "F": 0.0, "G": 0.0, "ND": 0.6}
    return m.get((letter or "ND").upper(), 0.6)


def compute_qual_score_bien_v2(bien: dict, *, weights: dict | None = None) -> float:
    """Compute qualitative score for a single v2 archetype bien."""
    default_weights = {
        "tension": 0.40,
        "transport": 0.30,
        "liquidite": 0.20,
        "dpe": 0.10,
        "encadrement_margin": 0.00,
    }
    W = (weights or default_weights).copy()
    t = float(bien.get("tension_locative_score_norm", 0.5))
    tr = float(bien.get("transport_score", 0.5))
    liq = float(bien.get("liquidite_score", 0.5))
    dpe = _dpe_score_v2(bien.get("dpe_initial", "ND"))
    loyer = _coerce_float(bien.get("loyer_m2", 0.0))
    cap = bien.get("loyer_m2_max", None)
    enc_margin = 0.5
    if cap is not None:
        capf = _coerce_float(cap, None)
        if capf:
            enc_margin = max(0.0, min(1.0, (capf - loyer) / max(capf, 1e-6) + 0.5))
    score01 = (
        W["tension"] * t
        + W["transport"] * tr
        + W["liquidite"] * liq
        + W["dpe"] * dpe
        + W["encadrement_margin"] * enc_margin
    )
    wsum = sum(abs(v) for v in W.values()) or 1.0
    return max(0.0, min(100.0, (score01 / wsum) * 100.0))


def calculate_qualitative_score_for_bien(bien: dict, *, weights: dict | None = None) -> float:
    """Calculate qualitative score for a single bien (v2 archetype format).
    
    Use calculate_qualitative_score() for scoring a full strategy with details.
    """
    return compute_qual_score_bien_v2(bien, weights=weights)

