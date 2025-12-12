# Fichier: app.py

import streamlit as st
import pandas as pd
import json
import inspect
import os
import plotly.express as px
import datetime
from typing import Dict, Any
# Note: base_params and eval_params are stored in st.session_state, not module globals

# Import from config - explicit imports only
from config import (
    TAUX_PRELEVEMENTS_SOCIAUX,
    TAXO_ICON,
    TAXO_TIP,
)

# Import from utils - explicit imports only
from utils import (
    classify_strategy,
    colorize_kpi,
    fmt_e,
    fmt_pct,
    badges_strategie,
    display_score_stars,
    to_json_safe,
    load_archetypes_from_json,
    enforce_compliance,
    apply_rent_cap,
)
import utils as u

from financial_calculations import simuler_strategie_long_terme, echeancier_mensuel
from strategy_finder import (
    creer_briques_investissement,
    trouver_top_strategies as trouver_top_strategies_core,
)

# --- Horizon-aware compute wrappers (signature-compatible) ---
def _call_simulation_with_horizon(sim_fn, *args, horizon_years=None, **kwargs):
    """Call a simulation function with horizon_years if it supports it; filter unknown kwargs."""
    if not callable(sim_fn):
        raise TypeError("sim_fn is not callable")
    sig = inspect.signature(sim_fn)
    params = sig.parameters
    allowed_kwargs = {k: v for k, v in kwargs.items() if k in params}
    if horizon_years is not None and 'horizon_years' in params:
        allowed_kwargs['horizon_years'] = horizon_years
    # Single, deterministic call using filtered kwargs only
    return sim_fn(*args, **allowed_kwargs)

def _kpi_get(kpis: dict, key_ha: str, key_25a: str):
    """Prefer (Ha) KPIs, fallback to 25a for backward compatibility."""
    if not isinstance(kpis, dict):
        return None
    return kpis.get(key_ha, kpis.get(key_25a))


# --- Compat caller for enforce_compliance (handles varying signatures) ---
def _call_enforce_compliance(archetypes, apply_cap_ui: bool):
    fn = getattr(u, 'enforce_compliance', None)
    if callable(fn):
        try:
            sig = inspect.signature(fn)
            params = sig.parameters
            if 'apply_cap' in params:
                return fn(archetypes, apply_cap=apply_cap_ui)
            if 'apply_rent_cap' in params:
                return fn(archetypes, apply_rent_cap=apply_cap_ui)
            # no kw arg supported -> try positional
            try:
                return fn(archetypes, apply_cap_ui)
            except TypeError:
                return fn(archetypes)
        except Exception:
            pass
    # Fallback: minimal inline cap using utils.apply_rent_cap if present
    out = []
    cap_fn = getattr(u, 'apply_rent_cap', None)
    for it in archetypes or []:
        if not isinstance(it, dict):
            continue
        bien = it.copy()
        if bool(bien.get("soumis_encadrement", False)):
            if callable(cap_fn):
                bien["loyer_m2"] = cap_fn(bien.get("loyer_m2", 0.0), bien.get("loyer_m2_max", None), apply_cap=apply_cap_ui)
            else:
                try:
                    l = float(bien.get("loyer_m2", 0.0) or 0.0)
                except Exception:
                    l = 0.0
                m = bien.get("loyer_m2_max", None)
                try:
                    capv = None if m is None else float(m)
                except Exception:
                    capv = None
                if capv is not None:
                    l = min(l, capv)
                bien["loyer_m2"] = l
        out.append(bien)
    return out

from validation import validate_archetypes

st.write(f"Version de Streamlit exécutée : {st.__version__}")

# --- FONCTION POUR LA FENÊTRE DE COMPARAISON ---
def display_comparison_content(processed_strategies, qualite_weight):
    """
    Affiche le contenu comparatif (tableau et graphiques) des stratégies.
    """
     # H local: lit le slider depuis la session
    H = int(st.session_state.get("horizon_ans", 25))
    st.info("Utilisez ce tableau et ces graphiques pour comparer rapidement les performances des stratégies.")
    
    # 1. Préparer les données pour le tableau
    comparison_data = []
    for i, strat_info in enumerate(processed_strategies, start=1):
        s = strat_info["strategy_data"]
        df = strat_info["df"]
        bilan = strat_info["bilan"]
        
        enrichissement_net = bilan.get("liquidation_nette", 0) + (df["Cash-Flow Net d'Impôt"].sum() if not df.empty else 0) - s["apport_total"]
        cf_annuel_avant_impot = s['cash_flow_final'] * 12
        cash_on_cash = (cf_annuel_avant_impot / s['apport_total']) * 100 if s['apport_total'] > 0 else 0
        noi_annuel = sum((b['loyer_mensuel_initial'] * 12) - (b['depenses_mensuelles_hors_credit_initial'] * 12) for b in s['details'])
        dette_annuelle = sum(b['pmt_total'] * 12 for b in s['details'])
        dscr = noi_annuel / dette_annuelle if dette_annuelle > 0 else 0

        comparison_data.append({
            "Stratégie": f"N°{i}",
            "Score Équilibré": s.get('balanced_score', 0),
            f"Enrichissement Net (H{H}a)": enrichissement_net,
            f"TRI (H{H}a) %": bilan.get("tri_annuel"),
            "CF Mensuel Net (€)": s['cash_flow_final'],
            "Cash-on-Cash %": cash_on_cash,
            "DSCR": dscr,
            "Apport Utilisé (€)": s['apport_total'],
            "Patrimoine Acquis (€)": s['patrimoine_acquis'],
            "Nb. Biens": len(s['details'])
        })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # 2. Afficher le tableau
    # ---- Add IRA (H{H}a) € column from each strategy's bilan ----
    try:
        src = processed_strategies  # preferred if it's in scope here
    except NameError:
        # fallbacks in case the list is stored on session_state
        src = st.session_state.get("processed_strategies") or st.session_state.get("top") or []

    try:
        ira_col = [(s.get("bilan", {}) or {}).get("ira_total", 0.0) for s in src]
    except Exception:
        ira_col = [0.0] * len(comparison_data)

    # make sure the column length matches comp_df rows
    if len(ira_col) < len(comp_df):
        ira_col = ira_col + [0.0] * (len(comp_df) - len(ira_col))
    comp_df[f"IRA (H{H}a) €"] = ira_col
    # Get base_params from session state, not global
    session_base_params = st.session_state.get("base_params", {})
    if (not session_base_params.get("apply_ira", False)) or all(abs(x or 0.0) < 1e-9 for x in ira_col):
        comp_df.drop(columns=[f"IRA (H{H}a) €"], inplace=True)

    # 2. Afficher le tableau
    st.dataframe(
        comp_df.style
        .format({
            "Score Équilibré": "{:.2f}",
            f"Enrichissement Net (H{H}a)": "{:,.0f} €",
            f"TRI (H{H}a) %": "{:.2f}%",
            "CF Mensuel Net (€)": "{:,.0f} €",
            "Cash-on-Cash %": "{:.1f}%",
            "DSCR": "{:.2f}",
            "Apport Utilisé (€)": "{:,.0f} €",
            "Patrimoine Acquis (€)": "{:,.0f} €",
            f"IRA (H{H}a) €": "{:,.0f} €",        # <— add a formatter for the new column
        })
        .background_gradient(
            cmap="viridis",
            subset=["Score Équilibré", f"Enrichissement Net (H{H}a)", f"TRI (H{H}a) %"]
        )
    )

    # 3. Afficher les graphiques comparatifs
    st.markdown("---")
    g1, g2 = st.columns(2)
    with g1:
        fig1 = px.bar(comp_df, x="Stratégie", y=f"Enrichissement Net (H{H}a)", title=f"Comparatif: Enrichissement Net à {H} ans", text_auto='.2s')
        fig1.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(comp_df, x="Stratégie", y="CF Mensuel Net (€)", title="Comparatif: Cash-Flow Mensuel Net (An 1)", text_auto=True)
        st.plotly_chart(fig2, use_container_width=True)

    with g2:
        fig3 = px.bar(comp_df, x="Stratégie", y=f"TRI (H{H}a) %", title=f"Comparatif: TRI à {H} ans", text_auto='.2f')
        st.plotly_chart(fig3, use_container_width=True)
        
        fig4 = px.bar(comp_df, x="Stratégie", y=["Apport Utilisé (€)", "Patrimoine Acquis (€)"], title="Comparatif: Apport vs. Patrimoine", barmode='group')
        st.plotly_chart(fig4, use_container_width=True)



# --- Ensure top-strategy search uses the same eval_params dict ---
def trouver_top_strategies_with_eval(*args, **kwargs):
    try:
        kwargs.setdefault("eval_params", eval_params)
    except NameError:
        pass
    return trouver_top_strategies_core(*args, **kwargs)


def main():
     # === Source des archétypes (v2 only) ===
    arch_v2_path = os.path.join(os.path.dirname(__file__), 'archetypes_recale_2025_v2.json')
    if not os.path.exists(arch_v2_path):
        st.error("Le dataset d'archétypes v2 est introuvable."); st.stop()
    arch_path = arch_v2_path
    # keep a clean label in session for UI
    st.session_state["archetypes_source_label"] = f"Fichier: {os.path.basename(arch_path)}"
    try:
        with open(arch_path, 'r', encoding='utf-8') as f:
            archetypes = json.load(f)
        # === Filtres: villes & types (s'appliquent au dataset courant) ===
        villes = sorted({a.get('ville') for a in archetypes if a.get('ville')})
        types_bien = sorted({str(a.get('mode_loyer','')).strip() for a in archetypes})


    except Exception:
        st.error("Lecture du dataset d’archétypes v2 impossible.")
        st.stop()
    # Configuration de la page Streamlit
    st.set_page_config(layout="wide", page_title="Super-Simulateur Stratégique v27.6")

# Derive H (horizon) safely
    H = st.session_state.get('horizon_ans') if 'horizon_ans' in st.session_state else 25
    st.title("🛰️ Super-Simulateur de Stratégie Immobilière (v27.6)")
    st.markdown("_Analyse, diversification et notation qualitative de vos stratégies d'investissement._")

    # GUIDE UTILISATEUR
    with st.expander("❓ Comment ça marche ?"):
        st.markdown("""
        **Bienvenue sur le Super-Simulateur ! Cet outil vous aide à trouver les meilleures stratégies d'investissement immobilier en 3 étapes simples :**
        
        1.  **🎯 Définissez vos objectifs et paramètres** dans la barre latérale à gauche. Indiquez votre apport, votre cible de cash-flow, vos hypothèses de crédit, de fiscalité et de marché.
        2.  **🚀 Lancez l'analyse.** L'algorithme va tester des milliers de combinaisons de biens pour trouver celles qui correspondent le mieux à vos critères et à vos priorités (financières vs. qualitatives).
        3.  **📊 Analysez les résultats.** Explorez les stratégies recommandées, comparez leurs indicateurs de performance, consultez les détails de chaque bien et visualisez les projections financières sur 25 ans.
        
        Utilisez le bouton **"Exporter en JSON"** pour sauvegarder une simulation complète ou **"Afficher le comparatif"** pour une vue d'ensemble rapide.
        """)

    # BARRE LATÉRALE (SIDEBAR)
    with st.sidebar:
        with st.expander("Mes Objectifs", expanded=True, icon="🎯"):
            apport_disponible = st.number_input("Apport total disponible (€)", min_value=0, value=100000, step=5000, help="Votre capacité d'apport personnel pour tous les projets.")
            cash_flow_cible = st.number_input("CF mensuel cible (€)", value=-100, step=10, help="Le cash-flow net que vous visez chaque mois.")
            c1, c2 = st.columns([0.6, 0.4])
            with c1:
                tolerance = st.number_input("Tolérance (€/mois)", value=50, step=10, help="Marge d'erreur acceptable autour de votre cible de cash-flow.")
            with c2:
                is_ciblage_mode = st.toggle("Ciblage Précis (±)", value=True, help="Désactivé : le CF sera au MINIMUM la cible.\n\nActivé : le CF visera à être le plus PROCHE possible de la cible.")
                mode_cf_key = "target" if is_ciblage_mode else "min"
            
            st.markdown("---")
            st.markdown("#### **Priorité de recherche**", help="Définissez votre priorité entre la performance financière pure (0%) et la qualité intrinsèque des biens (100%).")
            col1, col2, col3 = st.columns([0.15, 0.7, 0.15]); col1.markdown("<div style='font-size: 1.5em; text-align: center;'>💰</div>", unsafe_allow_html=True)
            priorite_qualite_pct = col2.slider("Priorité", 0, 100, 50, 5, format="%d%%", label_visibility="collapsed")
            col3.markdown("<div style='font-size: 1.5em; text-align: center;'>🛡️</div>", unsafe_allow_html=True)
            qualite_weight = priorite_qualite_pct / 100.0

            # Aide : récap pondération actuelle
            finance_pct = int(round((1.0 - qualite_weight) * 100))
            qual_pct = int(round(qualite_weight * 100))
            preset_name = st.session_state.get("finance_preset", "Équilibré (défaut)")
            st.caption(f"Tri = **Financier {finance_pct}%** / **Qualité {qual_pct}%** — Profil financier : *{preset_name}*.")
            horizon_ans = st.slider("Horizon d'investissement (ans)", 10, 30, int(st.session_state.get("horizon_ans", 25)), 1, key="horizon_ans")
            st.caption(f"Horizon de simulation : **{horizon_ans} ans**")

            
            H = int(horizon_ans)
            # Avancé — Profil de tri financier
            with st.expander("Avancé — Profil de tri financier", expanded=False):
                st.caption("Le tri financier combine **Enrichissement net**, **IRR**, **Efficacité d’apport**, **DSCR** et **Proximité du CF** selon le profil choisi.")
                PRESETS = {
                    "Équilibré (défaut)": {"enrich_net": 0.30, "irr": 0.25, "cap_eff": 0.20, "dscr": 0.15, "cf_proximity": 0.10},
                    "Cash-flow d’abord":  {"enrich_net": 0.10, "irr": 0.10, "cap_eff": 0.10, "dscr": 0.20, "cf_proximity": 0.50},
                    "Rendement / IRR":    {"enrich_net": 0.15, "irr": 0.50, "cap_eff": 0.30, "dscr": 0.05, "cf_proximity": 0.00},
                    "Sécurité (DSCR)":    {"enrich_net": 0.05, "irr": 0.15, "cap_eff": 0.10, "dscr": 0.50, "cf_proximity": 0.20},
                    "Patrimoine LT":      {"enrich_net": 0.60, "irr": 0.15, "cap_eff": 0.20, "dscr": 0.03, "cf_proximity": 0.02},
                }
                if "finance_preset" not in st.session_state:
                    st.session_state.finance_preset = "Équilibré (défaut)"
                if "finance_custom" not in st.session_state:
                    st.session_state.finance_custom = PRESETS[st.session_state.finance_preset].copy()

                col_p, col_c = st.columns([2,1])
                with col_p:
                    preset = st.selectbox("Profil de tri financier", list(PRESETS.keys()), index=list(PRESETS.keys()).index(st.session_state.finance_preset))
                    if preset != st.session_state.finance_preset:
                        st.session_state.finance_preset = preset
                        st.session_state.finance_custom = PRESETS[preset].copy()
                with col_c:
                    custom_on = st.toggle("Personnaliser", value=False, help="Ajuster finement les poids (somme renormalisée).")

                def _norm(d):
                    s = sum(d.values())
                    if s <= 0:
                        return d
                    return {k: (v/s) for k, v in d.items()}

                weights = PRESETS[st.session_state.finance_preset].copy()
                if custom_on:
                    st.write("Réglage fin des poids (somme renormalisée = 1)")
                    c1, c2, c3 = st.columns(3)
                    c4, c5 = st.columns(2)
                    w1 = c1.slider("Enrich. net", 0, 100, int(round(st.session_state.finance_custom["enrich_net"]*100)), 1)
                    w2 = c2.slider("IRR",          0, 100, int(round(st.session_state.finance_custom["irr"]*100)), 1)
                    w3 = c3.slider("Efficacité",    0, 100, int(round(st.session_state.finance_custom["cap_eff"]*100)), 1)
                    w4 = c4.slider("DSCR",          0, 100, int(round(st.session_state.finance_custom["dscr"]*100)), 1)
                    w5 = c5.slider("Proximité CF",  0, 100, int(round(st.session_state.finance_custom["cf_proximity"]*100)), 1)
                    st.session_state.finance_custom = {
                        "enrich_net": float(w1)/100.0,
                        "irr": float(w2)/100.0,
                        "cap_eff": float(w3)/100.0,
                        "dscr": float(w4)/100.0,
                        "cf_proximity": float(w5)/100.0,
                    }
                    weights = _norm(st.session_state.finance_custom)
                st.session_state.finance_weights_override = weights

        with st.expander("Ma Recherche", icon="🏠"):
            uploaded = st.file_uploader("Charger un fichier d’archétypes", type=["json"])
            # Dataset label (v2 only)
            st.markdown("**Dataset archétypes (v2)**")
            st.caption(st.session_state.get("archetypes_source_label", "Fichier: (indisponible)"))
            try:
                sidebar_arch = list(archetypes)
            except FileNotFoundError:
                st.error("Fichier d’archétypes introuvable."); st.stop()

            
            mode_loyer_labels = {"meuble_etudiant": "Étudiant", "meuble_classique": "Meublé", "nu_classique": "Nu", "colocation_meublee": "Colocation"}

            # Nouveau périmètre (déplacé ici depuis la vue principale)
            st.markdown("### Sélection du périmètre")
            villes = sorted({a.get('ville') for a in archetypes if a.get('ville')})

            sel_villes = st.multiselect("Filtrer les villes", villes, default=villes if len(villes) <= 12 else [])
            # Build list of codes and a per-code label map from the dataset
            types_bien = sorted({str(a.get('mode_loyer','')).strip() for a in archetypes if a.get('mode_loyer')})
            code_to_label = {}
            for a in archetypes:
                code = str(a.get('mode_loyer','')).strip()
                if not code:
                    continue
                # prefer per-row label if present
                lbl = a.get('mode_loyer_label')
                if lbl:
                    code_to_label[code] = lbl
            # fallback prettifier (used only if some code has no per-row label)
            def _pretty_code(code: str) -> str:
                if not code:
                    return "—"
                text = code.replace("_"," ").replace("-"," ").strip()
                tokens = []
                for t in text.split():
                    tokens.append(t.upper() if t.lower() in {"lmnp","bic","lcd"} else t.capitalize())
                return " ".join(tokens)
            def _label_for(code: str) -> str:
                return code_to_label.get(code, _pretty_code(code))
            sel_types  = st.multiselect(
                "Filtrer les types de bien",
                options=types_bien,
                default=types_bien,
                format_func=_label_for,   # ← show labels, keep codes as values
                help="Sélectionnez les familles de stratégies à considérer."
            )
            if sel_villes:
                archetypes = [a for a in archetypes if a.get('ville') in sel_villes]
            if sel_types:
                archetypes = [a for a in archetypes if str(a.get('mode_loyer','')).strip() in sel_types]

            apply_cap_ui = st.checkbox("Appliquer l'encadrement des loyers si configuré", value=True)

        with st.expander("Paramètres de Simulation", icon="⚙️"):
            tab_credit, tab_fiscal, tab_marche = st.tabs(["🏦 Crédit & Coûts", "🧾 Gestion & Fiscalité", "📈 Marché (LT)"])
            with tab_credit:
                c1,c2,c3 = st.columns(3); taux_15=c1.number_input("15 ans (%)",value=3.2,format="%.2f"); taux_20=c2.number_input("20 ans (%)",value=3.4,format="%.2f"); taux_25=c3.number_input("25 ans (%)",value=3.6,format="%.2f")
                taux_credits_dict = {15: taux_15, 20: taux_20, 25: taux_25}
                st.markdown("---")
                frais_notaire_pct = st.slider("Frais de notaire (%)", 0.0, 12.0, 7.5, 0.1); assurance_ann_pct = st.slider("Assurance emprunteur (% cap.)", 0.00, 1.00, 0.35, 0.01); frais_pret_pct = st.slider("Frais de prêt (% cap.)", 0.0, 3.0, 1.0, 0.1)
                st.markdown("**Inclusion des coûts dans le financement**")
                c1,c2,c3,c4 = st.columns(4); inclure_travaux=c1.checkbox("Travaux",value=True); inclure_reno_ener=c2.checkbox("Reno E→D",value=True); inclure_mobilier=c3.checkbox("Mobilier",value=True); financer_mobilier=c4.checkbox("Financé",value=True)
                # --- IRA (déplacé depuis '🧾 Gestion & Fiscalité') ---
                st.markdown("**Frais de remboursement anticipés**")
                apply_ira = st.toggle(
                    "Appliquer les frais (IRA)",
                    value=st.session_state.get("apply_ira", True),
                    help="Indemnité si le prêt est remboursé avant son terme (H < durée)."
                )
                ira_cap_pct = st.number_input(
                    "Plafond % CRD",
                    0.0, 10.0, st.session_state.get("ira_cap_pct", 3.0), 0.1,
                    help="Plafond légal sur le capital restant dû."
                )
                st.session_state["apply_ira"] = apply_ira
                st.session_state["ira_cap_pct"] = ira_cap_pct

            with tab_fiscal:
                frais_gestion = st.slider("Frais de gestion locative (% loyer)", 0.0, 15.0, 0.0, 0.5); provision = st.slider("Provision vacance/entretien (% loyer)", 0.0, 15.0, 5.0, 0.5)
                cfe_par_bien_ann = st.number_input("CFE annuelle par bien (€)", 0, 1000, 150, 10)
                regime_key = "lmnp" if st.radio("Régime fiscal",["LMNP réel","Micro-BIC"],index=0,horizontal=True).startswith("LMNP") else "microbic"
                # --- Sidebar: TMI slider behavior depends on regime ---
                if regime_key == "microbic":
                    # Keep the same slider *in place*, but disabled/greyed out (UI signal only)
                    tmi_pct_seul = st.slider(
                        "TMI (désactivé en Micro-BIC)",
                        0.0, 55.0, 30.0, 0.5,
                        disabled=True,
                        key="tmi_slider"
                    )
                    tmi_pct_effective = 0.0
                    st.caption(f"TMI ignoré en Micro-BIC → TMI effectif = 0.0% | PS (ajoutés par le moteur) : {TAUX_PRELEVEMENTS_SOCIAUX}%")
                else:
                    tmi_pct_seul = st.slider(
                        "TMI (LMNP réel)",
                        0.0, 55.0, 30.0, 0.5,
                        key="tmi_slider"
                    )
                    tmi_pct_effective = max(0.0, tmi_pct_seul)
                    st.caption(f"TMI (LMNP réel) = {tmi_pct_effective:.1f}% | PS ajoutés par le moteur : {TAUX_PRELEVEMENTS_SOCIAUX}%")


                
                frais_vente_pct = st.slider("Frais de vente en fin d'horizon (%)", 0.0, 10.0, 6.0, 0.5)
            with tab_marche:
                if 'appreciation' not in st.session_state: st.session_state.appreciation = 2.0
                if 'revalo' not in st.session_state: st.session_state.revalo = 1.5
                if 'inflation' not in st.session_state: st.session_state.inflation = 2.0
                
                def set_scenario(app, rev, inf):
                    st.session_state.appreciation = app
                    st.session_state.revalo = rev
                    st.session_state.inflation = inf

                st.markdown("**Scénarios pré-remplis**")
                c1, c2, c3 = st.columns(3)
                c1.button("Prudent", on_click=set_scenario, args=(1.0, 1.0, 2.5), use_container_width=True)
                c2.button("Équilibré", on_click=set_scenario, args=(2.0, 1.5, 2.0), use_container_width=True, type="primary")
                c3.button("Optimiste", on_click=set_scenario, args=(3.5, 2.5, 1.5), use_container_width=True)

                appreciation_bien_pct = st.slider("Appréciation annuelle biens (%)", -2.0, 8.0, st.session_state.appreciation, 0.1, key="slider_app")
                revalo_loyer_pct = st.slider("Revalorisation annuelle loyers (%)", -2.0, 8.0, st.session_state.revalo, 0.1, key="slider_rev")
                inflation_charges_pct = st.slider("Inflation annuelle charges (%)", -2.0, 8.0, st.session_state.inflation, 0.1, key="slider_inf")


    hypoth = {"appreciation_bien_pct": appreciation_bien_pct, "revalo_loyer_pct": revalo_loyer_pct, "inflation_charges_pct": inflation_charges_pct}
    # 'archetypes' is already filtered by villes/types in “Ma Recherche”
    arch_filtered = list(archetypes)
    arch = _call_enforce_compliance(arch_filtered, apply_cap_ui)


    if st.button(" Lancer l'Analyse Stratégique 🚀 ", type="primary", use_container_width=True):
        progress_text = "Analyse, diversification et notation des stratégies..."
        my_bar = st.progress(0, text=progress_text)
        briques = creer_briques_investissement(arch, taux_credits_dict, frais_gestion, provision, frais_notaire_pct, 0.0, inclure_travaux, inclure_reno_ener, inclure_mobilier, financer_mobilier, assurance_ann_pct, frais_pret_pct, cfe_par_bien_ann)
        st.session_state["briques"] = briques
        my_bar.progress(33, text=progress_text)
        # Build the shared eval_params dict just above the call
        eval_params = {
            "hypotheses_marche": hypoth,
            "regime_fiscal": regime_key,
            "tmi_pct": tmi_pct_effective,  # pass TMI only; engine adds PS
            "frais_vente_pct": frais_vente_pct,
            "cfe_par_bien_ann": cfe_par_bien_ann,
            "duree_simulation_ans": int(st.session_state.get("horizon_ans", 25)),
            "finance_weights_override": st.session_state.get("finance_weights_override"),
            "finance_preset_name": st.session_state.get("finance_preset", "Équilibré (défaut)"),
            "apply_ira": apply_ira,
            "ira_cap_pct": ira_cap_pct,
        }

        top = trouver_top_strategies_core(
            apport_disponible,
            cash_flow_cible,
            tolerance,
            briques,
            mode_cf_key,
            qualite_weight,
            eval_params=eval_params,
            horizon_years=H,
        )

        my_bar.progress(66, text=progress_text)
        st.session_state["top"] = top
        st.session_state["base_params"] = eval_params
        st.session_state["apport_disponible_lancement"] = apport_disponible
        my_bar.progress(100, text="Analyse terminée !")
        st.rerun()

    if "top" in st.session_state:
        if "show_comparison" not in st.session_state:
            st.session_state.show_comparison = False
        
        top = st.session_state.get("top", [])
        base_params = st.session_state.get("base_params", {})
        
        if not top:
            st.warning("Aucune stratégie ne respecte vos critères avec les filtres actuels.")
        else:
            st.header(f"🏆 Top {len(top)} Stratégies (selon votre priorité et vos filtres)")

            def _taxo_badge_html(label: str) -> str:
                icon = TAXO_ICON.get(label, "🔀")
                tip  = TAXO_TIP.get(label, "Typologie de la stratégie")
                return (
                    f"<span title='{tip}' "
                    f"style='display:inline-block;padding:2px 8px;border-radius:999px;"
                    f"background:#eef1f6;border:1px solid #d6dbe6;font-size:13px;"
                    f"vertical-align:middle;'>"
                    f"{icon}&nbsp;{label}</span>"
                )
            processed_strategies = []
            for s in top:
                df, bilan = _call_simulation_with_horizon(simuler_strategie_long_terme, s, **base_params, horizon_years=H)
                taxo = classify_strategy(s, bilan)
                badge_html = _taxo_badge_html(taxo)
                processed_strategies.append({"strategy_data": s, "df": df, "bilan": bilan})
            
            search_parameters_for_export = {}
            export_data_list = []
            for strat_info in processed_strategies:
                df_as_records = strat_info["df"].to_dict('records')
                s = strat_info["strategy_data"]
                bilan = strat_info["bilan"]
                # compute per-bien IRA list for export
                ira_list = []
                try:
                    H_local = int(st.session_state.get('horizon_ans', 25))
                    if base_params.get('apply_ira', False):
                        for b in s['details']:
                            duree_pret_i = int(b.get('duree_pret', 0))
                            if H_local < duree_pret_i:
                                sch_i = echeancier_mensuel(b['credit_final'], b['taux_pret'], b['duree_pret'], b['assurance_ann_pct'])
                                nmois_i = int(sch_i.get('nmois', 0))
                                mH_i = min(H_local * 12, nmois_i)
                                crd_H_i = float(sch_i['balances'][mH_i - 1]) if mH_i > 0 else float(b.get('credit_final', 0.0))
                                monthly_rate_i = (float(b.get('taux_pret', 0.0)) / 100.0) / 12.0
                                cap_pct_fee_i = (float(base_params.get('ira_cap_pct', 3.0)) / 100.0) * crd_H_i
                                six_months_interest_fee_i = 6.0 * monthly_rate_i * crd_H_i
                                ira_i = min(cap_pct_fee_i, six_months_interest_fee_i)
                            else:
                                ira_i = 0.0
                            ira_list.append(ira_i)
                    else:
                        ira_list = [0.0 for _ in s['details']]
                except Exception:
                    ira_list = [0.0 for _ in s['details']]

                export_data_list.append({
                    "donnees_strategie": strat_info["strategy_data"],
                    "projection_annuelle": df_as_records,
                    "bilan_final_25_ans": strat_info["bilan"],
                    "ira_total_H": bilan.get('ira_total', 0.0),
                    "ira_per_bien_H": ira_list
                })

            briques_for_export = st.session_state.get("briques")
            search_parameters_for_export = {
                "objectifs": {
                    "apport_disponible": apport_disponible,
                    "cash_flow_cible": cash_flow_cible,
                    "tolerance": tolerance,
                },
                "briques": st.session_state.get("briques", []),
                "regime_fiscal": regime_key,
                "tmi_pct_effective": tmi_pct_effective,
                "frais_vente_pct": frais_vente_pct,
                "cfe_par_bien_ann": cfe_par_bien_ann,
                "horizon_ans": H,
                "apply_ira": apply_ira,
                "ira_cap_pct": ira_cap_pct,
                "recherche": {
                    "source_archetypes": uploaded.name if uploaded else None,
                    "finance_weights_override": st.session_state.get("finance_weights_override"),
                    "finance_preset_name": st.session_state.get("finance_preset", "Équilibré (défaut)"),
                },
                "hypotheses_marche_long_terme": hypoth,
            }

            full_export_package = {
                "date_export": datetime.datetime.now().isoformat(),
                "search_parameters": search_parameters_for_export,   # already defined above
                "hypotheses": hypoth,                                # appreciation, revalo, etc.
                "encadrement_applique": bool(apply_cap_ui),
                "archetypes_filtered": arch,                         # after enforce compliance
                # include results only if they exist at this point
                "kpis": kpis if 'kpis' in locals() else None,
                "top": top if 'top' in locals() else None,
            }

            safe_export_package = to_json_safe(full_export_package)
            json_string = json.dumps(safe_export_package, indent=4, ensure_ascii=False).encode('utf-8')

            button_container = st.container()
            with button_container:
                c1, c2 = st.columns(2)

                if c1.button("📊 Afficher le comparatif", use_container_width=True, help="Affiche/Masque le tableau de comparaison des KPIs."):
                    st.session_state.show_comparison = not st.session_state.show_comparison
                    st.rerun()

                c2.download_button(label="📥 Exporter tous les résultats en JSON", data=json_string, file_name=f"export_strategies_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.json", mime="application/json", use_container_width=True, help="Télécharge un fichier JSON avec les paramètres et résultats détaillés.")

            if st.session_state.show_comparison:
                with st.container(border=True):
                    display_comparison_content(processed_strategies, qualite_weight)
            
            st.markdown("---")
            
            for i, strat_info in enumerate(processed_strategies, start=1):
                s, df, bilan = strat_info["strategy_data"], strat_info["df"], strat_info["bilan"]
                kpis = {"enrichissement_net_25a": bilan.get("liquidation_nette", 0) + (df["Cash-Flow Net d'Impôt"].sum() if not df.empty else 0) - s["apport_total"], "tri_25a_pct": bilan.get("tri_annuel"), "cash_flow_mensuel_net_an1": s['cash_flow_final'], "cash_on_cash_an1_pct": (s['cash_flow_final'] * 12 / s['apport_total']) * 100 if s['apport_total'] > 0 else 0, "dscr_an1": (sum((b['loyer_mensuel_initial']-b['depenses_mensuelles_hors_credit_initial'])*12 for b in s['details']))/(sum(b['pmt_total']*12 for b in s['details'])) if sum(b['pmt_total'] for b in s['details'])>0 else 0, "apport_total": s['apport_total'], "patrimoine_acquis": s['patrimoine_acquis']}
                
                with st.container(border=True):
                    if i == 1:
                        st.success("🥇 **Meilleure recommandation**", icon="🏆")
                    
                    col1_title, col2_popover = st.columns([0.9, 0.1])
                    with col1_title:
                        st.subheader(f"N°{i} : Stratégie (Score Équilibré : {s.get('balanced_score', 0):.2f})")
                        # st.metric(
                        #         label=f"{TAXO_ICON.get(taxo, '🔀')} {taxo}",
                        #         value="",
                        #         help=TAXO_TIP.get(taxo, "Typologie de la stratégie")
                        #     )
                            
                    with col2_popover:
                        with st.popover("ℹ️", help="Détail du calcul du score"):
                            st.markdown(f"""
                            Le Score Équilibré est une moyenne pondérée de deux notes :
                            - **Score Financier :** `{s.get('finance_score', 0):,.0f}` (basé sur le patrimoine acquis)
                            - **Score Qualitatif :** `{s.get('qual_score', 0):.2f}/5` (basé sur les facteurs des biens)
                            
                            Votre priorité actuelle est de **{priorite_qualite_pct}%** pour la qualité.
                            """)

                    tab_synthese, tab_details, tab_projection = st.tabs(["📊 Synthèse & KPIs", "🏠 Détail des Biens", "📈 Projection & Graphes"])
                    
                    with tab_synthese:
                        c1, c2, c3, c4 = st.columns([0.25, 0.25, 0.25, 0.25])
                        with c1:
                            st.markdown(f"**Enrichissement net (H{H}a)**", help="Gain total (CF cumulés + plus-value nette - apport).")
                            st.markdown(f"<h2>{fmt_e(_kpi_get(kpis, 'enrichissement_net_Ha', 'enrichissement_net_25a'))} €</h2>", unsafe_allow_html=True)
                        with c2:
                            st.markdown(f"**TRI (H{H}a)**", help="Taux de Rentabilité Interne annualisé.")
                            st.markdown(colorize_kpi(_kpi_get(kpis, 'tri_Ha_pct', 'tri_25a_pct'), 'tri', size='h2'), unsafe_allow_html=True)
                        with c3:
                            st.metric("Patrimoine Acquis", f"{fmt_e(kpis['patrimoine_acquis'])} €", help="Coût total de tous les biens.")
                        with c4:
                            st.metric("Apport Utilisé", f"{fmt_e(kpis['apport_total'])} €", help="Le montant total de votre capital personnel engagé.")

                        st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
                        # IRA summary line
                        ira_total_display = _kpi_get(kpis, "ira_total_Ha", "ira_total_25a") if "kpis" in locals() else None
                        if ira_total_display is None:
                            ira_total_display = bilan.get("ira_total") if "bilan" in locals() else None
                        if ira_total_display is not None and ira_total_display > 0:
                            st.caption(f"**IRA (H{H}a)** : - {fmt_e(ira_total_display)} €")

                        c1, c2, c3, c4 = st.columns(4)
                        with c1: 
                            st.markdown("**CF Mensuel Net (An 1)**", help="Flux de trésorerie net d'impôt estimé la première année.")
                            st.markdown(colorize_kpi(kpis['cash_flow_mensuel_net_an1'], 'cf'), unsafe_allow_html=True)
                        with c2:
                            st.markdown("**Cash-on-Cash (An 1)**", help="(Cash-Flow Annuel Avant Impôt / Apport Total) - Rentabilité du capital investi la 1ère année.")
                            st.markdown(colorize_kpi(kpis['cash_on_cash_an1_pct'], 'coc'), unsafe_allow_html=True)
                        with c3:
                            st.markdown("**DSCR (An 1)**", help="(Revenus Locatifs Nets / Service Annuel de la Dette) - Ratio de couverture de la dette. > 1.2 est généralement exigé par les banques.")
                            st.markdown(colorize_kpi(kpis['dscr_an1'], 'dscr'), unsafe_allow_html=True)
                        with c4: 
                            display_score_stars(s.get('qual_score', 0), help_text="Score qualitatif des biens.")

                        st.markdown("---")
                        badges = badges_strategie(s, st.session_state.get("apport_disponible_lancement", 0)); st.caption(" | ".join(badges) if badges else "")
                    
                    with tab_details:
                        # Dictionnaires pour mapper les notes qualitatives (inchangés)
                        score_mapping = {"Excellent": 5, "Premium": 5, "Très Élevée": 5, "Très Bon": 4, "Élevé": 4, "Bon": 3, "Moyen": 2, "Correct": 1, "Faible": 0}
                        qual_rating_to_color_icon = {"Excellent": "🟢", "Premium": "🟢", "Très Élevée": "🟢", "Très Bon": "🟢", "Élevé": "🟢", "Bon": "🟡", "Moyen": "🟠", "Correct": "🔴", "Faible": "🔴"}

                        for b_idx, b in enumerate(s["details"]):
                            with st.container(border=True):
                                header_cols = st.columns([0.6, 0.4])
                                with header_cols[0]:
                                    st.markdown(f"#### 🏢 **{b['nom_bien']}**")
                                dpe_val = b.get('dpe_initial', b.get('dpe', 'N/A'))
                                enc_str = 'Oui' if bool(b.get('soumis_encadrement', False)) else 'Non'
                                surf = float(b.get('surface', 0) or 0)
                                loyer_m2 = float(b.get('loyer_m2', 0) or 0)
                                st.caption(f"DPE : {dpe_val} | Surface : {surf:.0f} m² | Loyer cible : {fmt_e(loyer_m2*surf)} €/mois | Soumis encadrement : {enc_str}")
                                
                                with header_cols[1]:
                                    total_score, num_factors = 0, 0
                                    if "facteurs_qualitatifs" in b and b["facteurs_qualitatifs"]:
                                        for value in b["facteurs_qualitatifs"].values():
                                            total_score += score_mapping.get(value, 0)
                                            num_factors += 1
                                    avg_score = total_score / num_factors if num_factors > 0 else 0
                                    
                                    if avg_score > 0:
                                        rounded_score = round(avg_score)
                                        stars = "⭐" * rounded_score + "☆" * (5 - rounded_score)
                                        st.markdown(f"<div style='text-align: right; font-size: 1.2rem;'>{stars}</div>", unsafe_allow_html=True)
                                        st.markdown(f"<div style='text-align: right; font-weight: bold;'>Score Qualitatif : {avg_score:.1f}/5</div>", unsafe_allow_html=True)

                                st.divider()

                                c1, c2 = st.columns(2)

                                with c1:
                                    st.subheader("💰 Financement & Coûts")
                                    
                                    st.metric(
                                        label="Coût total de l'opération",
                                        value=f"{fmt_e(b['cout_total'])} €",
                                        help="Prix d'achat + frais de notaire + budget travaux + valeur du mobilier."
                                    )
                                    
                                    st.metric(
                                        label="Apport personnel alloué",
                                        value=fmt_e(b["apport_final_bien"]) + " €",
                                        help="Part de votre apport imputée à ce bien."
                                    )
                                    if base_params.get("apply_ira", False):
                                        try:
                                            H_local = int(st.session_state.get("horizon_ans", 25))
                                            duree_pret_i = int(b.get("duree_pret", 0))
                                            if H_local < duree_pret_i:
                                                sch_i = echeancier_mensuel(b["credit_final"], b["taux_pret"], b["duree_pret"], b["assurance_ann_pct"])
                                                nmois_i = int(sch_i.get("nmois", 0))
                                                mH_i = min(H_local * 12, nmois_i)
                                                crd_H_i = float(sch_i["balances"][mH_i - 1]) if mH_i > 0 else float(b.get("credit_final", 0.0))
                                                monthly_rate_i = (float(b.get("taux_pret", 0.0)) / 100.0) / 12.0
                                                cap_pct_fee_i = (float(base_params.get("ira_cap_pct", 3.0)) / 100.0) * crd_H_i
                                                six_months_interest_fee_i = 6.0 * monthly_rate_i * crd_H_i
                                                ira_i = min(cap_pct_fee_i, six_months_interest_fee_i)
                                            else:
                                                ira_i = 0.0
                                        except Exception:
                                            ira_i = 0.0

                                        ira_label = f"IRA (H{H}a)"
                                        # Only show when toggle is ON and ira_i > 0
                                        # Render ONLY when toggle is ON and the computed IRA is strictly positive
                                        if base_params.get("apply_ira", False) and (ira_i > 0):
                                            st.metric(
                                                label=ira_label,
                                                value="- " + fmt_e(ira_i) + " €",
                                                help="Indemnité de remboursement anticipé (calculée à l'horizon si le prêt n'est pas échu).",
                                            )

                                    if b.get('credit_final', 0) > 0:
                                        st.metric(
                                            label="Crédit immobilier",
                                            value=f"{fmt_e(b['credit_final'])} €",
                                        )
                                        
                                        badge_cols = st.columns(2)
                                        with badge_cols[0]:
                                            st.markdown(
                                                f"""
                                                <div style="background-color: #f0f2f6; border-radius: 10px; padding: 8px; text-align: center;">
                                                    <span style="font-size: 0.9em; color: #555;">⏳ Durée</span><br>
                                                    <strong style="font-size: 1.1em;">{b['duree_pret']} ans</strong>
                                                </div>
                                                """,
                                                unsafe_allow_html=True
                                            )
                                        with badge_cols[1]:
                                            st.markdown(
                                                f"""
                                                <div style="background-color: #f0f2f6; border-radius: 10px; padding: 8px; text-align: center;">
                                                    <span style="font-size: 0.9em; color: #555;">📊 Taux</span><br>
                                                    <strong style="font-size: 1.1em;">{b['taux_pret']}%</strong>
                                                </div>
                                                """,
                                                unsafe_allow_html=True
                                            )

                                with c2:
                                    st.subheader("📊 Flux Mensuels (An 1)")
                                    
                                    charges_hors_credit = b['depenses_mensuelles_hors_credit_initial'] - (base_params.get('cfe_par_bien_ann', 0) / 12.0)
                                    charges_copro_tf = b['charges_const_mth0'] + b['tf_const_mth0']
                                    frais_gestion_local = b['loyer_mensuel_initial'] * (b['frais_gestion_pct'] / 100.0)
                                    provision_local = b['loyer_mensuel_initial'] * (b['provision_pct'] / 100.0)
                                    charges_help_text = f"Détail : Copro/TF ({fmt_e(charges_copro_tf)}€) + Gestion ({fmt_e(frais_gestion_local)}€) + Provision ({fmt_e(provision_local)}€)."

                                    st.metric(
                                        label="✅ Loyer mensuel HC",
                                        value=f"+ {fmt_e(b['loyer_mensuel_initial'])} €",
                                        help="Loyer Hors Charges (HC) estimé.",
                                        delta_color="off"
                                    )
                                    
                                    st.metric(
                                        label="❌ Mensualité du crédit",
                                        value=f"- {fmt_e(b['pmt_total'])} €",
                                        help="Inclut capital, intérêts et assurance.",
                                        delta_color="inverse"
                                    )
                                    
                                    st.metric(
                                        label="❌ Charges et provisions",
                                        value=f"- {fmt_e(charges_hors_credit)} €",
                                        help=charges_help_text,
                                        delta_color="inverse"
                                    )
                                    
                                    st.divider()
                                    
                                    cf_bien_brut = b['loyer_mensuel_initial'] - b['pmt_total'] - charges_hors_credit
                                    st.markdown("**Résultat Brut Mensuel**", help="Résultat du bien (Loyer - Mensualité - Charges) avant impôts et CFE.")
                                    st.markdown(colorize_kpi(cf_bien_brut, 'cf', size='h2'), unsafe_allow_html=True)

                                with st.expander("🔍 Voir le détail du score qualitatif"):
                                    if "facteurs_qualitatifs" in b and b["facteurs_qualitatifs"]:
                                        qual_html_details = ""
                                        for key, value in b["facteurs_qualitatifs"].items():
                                            icon = qual_rating_to_color_icon.get(value, "⚪")
                                            qual_html_details += f"<li>{icon} {key.replace('_', ' ').title()}: <strong>{value}</strong></li>"
                                        st.markdown(f"<ul>{qual_html_details}</ul>", unsafe_allow_html=True)
                                    else:
                                        st.info("Aucun facteur qualitatif n'a été renseigné pour ce bien.")

                    with tab_projection:
                        if df.empty:
                            st.warning("La projection n'a pas pu être calculée.")
                        else:
                            # --- Add IRA column only when positive and toggle ON ---
                            df2 = df.copy()
                            ira_total = float(bilan.get("ira_total", 0.0))
                            if base_params.get("apply_ira", False) and ira_total > 1e-9:
                                ira_col_name = f"IRA (H{H}a) €"
                                df2[ira_col_name] = 0.0
                                # Your 'Année' column uses capital A throughout your charts:
                                df2.loc[df2["Année"] == H, ira_col_name] = -ira_total
                                formatter_map = {col: '{:,.0f} €' for col in df2.columns if col != 'Année'}
                            else:
                                formatter_map = {col: '{:,.0f} €' for col in df2.columns if col != 'Année'}

                            st.dataframe(df2.style.format(formatter=formatter_map))

                            st.markdown("---"); st.write("**Graphiques d'analyse (Interactifs)**")

                            g1, g2 = st.columns(2)
                            with g1:
                                fig1 = px.line(df, x="Année", y=["Patrimoine Net", "Dette"], title="Évolution Patrimoine/Dette", markers=True)
                                st.plotly_chart(fig1, use_container_width=True, key=f"patrimoine_dette_{i}")
                                
                                fig2 = px.area(df, x="Année", y=["Capital Remboursé", "Intérêts & Assurance"], title="Répartition des Paiements du Crédit")
                                st.plotly_chart(fig2, use_container_width=True, key=f"repartition_credit_{i}")
                            with g2:
                                fig3 = px.bar(df, x="Année", y="Cash-Flow Net d'Impôt", title="Évolution du Cash-Flow Annuel")
                                st.plotly_chart(fig3, use_container_width=True, key=f"evolution_cf_{i}")
                                
                                if base_params.get("regime_fiscal") == 'lmnp':
                                    fig4 = px.bar(df, x='Année', y=['Loyers Bruts', 'Charges Déductibles', 'Amortissements'], title="Composition du Résultat Fiscal (LMNP)", barmode='relative')
                                    st.plotly_chart(fig4, use_container_width=True, key=f"composition_fiscale_{i}")

if __name__ == "__main__":
    main()