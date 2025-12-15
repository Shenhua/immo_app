"""Sidebar components for the main app.

Reusable sidebar sections that can be composed in the main page.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import streamlit as st


# Preset financial profiles
FINANCIAL_PRESETS = {
    "Équilibré (défaut)": {
        "enrich_net": 0.30,
        "irr": 0.25,
        "cap_eff": 0.20,
        "dscr": 0.15,
        "cf_proximity": 0.10,
    },
    "Cash-flow d'abord": {
        "enrich_net": 0.10,
        "irr": 0.10,
        "cap_eff": 0.10,
        "dscr": 0.20,
        "cf_proximity": 0.50,
    },
    "Rendement / IRR": {
        "enrich_net": 0.15,
        "irr": 0.50,
        "cap_eff": 0.30,
        "dscr": 0.05,
        "cf_proximity": 0.00,
    },
    "Sécurité (DSCR)": {
        "enrich_net": 0.05,
        "irr": 0.15,
        "cap_eff": 0.10,
        "dscr": 0.50,
        "cf_proximity": 0.20,
    },
    "Patrimoine LT": {
        "enrich_net": 0.60,
        "irr": 0.15,
        "cap_eff": 0.20,
        "dscr": 0.03,
        "cf_proximity": 0.02,
    },
}


def render_objectives_section() -> Tuple[float, float, float, str, float, int]:
    """Render the objectives section of the sidebar.
    
    Returns:
        Tuple of (apport, cf_cible, tolerance, mode_cf, qualite_weight, horizon)
    """
    with st.expander("Mes Objectifs", expanded=True, icon="🎯"):
        apport = st.number_input(
            "Apport total disponible (€)",
            min_value=0,
            value=100000,
            step=5000,
            help="Votre capacité d'apport personnel pour tous les projets.",
        )
        
        cf_cible = st.number_input(
            "CF mensuel cible (€)",
            value=-100,
            step=10,
            help="Le cash-flow net que vous visez chaque mois.",
        )
        
        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            tolerance = st.number_input(
                "Tolérance (€/mois)",
                value=50,
                step=10,
                help="Marge d'erreur acceptable autour de votre cible de cash-flow.",
            )
        with col2:
            is_precise = st.toggle(
                "Ciblage Précis (±)",
                value=True,
                help="Désactivé : le CF sera au MINIMUM la cible.\n\nActivé : le CF visera à être le plus PROCHE possible de la cible.",
            )
            mode_cf = "target" if is_precise else "min"
        
        st.markdown("---")
        st.markdown(
            "#### **Priorité de recherche**",
            help="Définissez votre priorité entre la performance financière pure (0%) et la qualité intrinsèque des biens (100%).",
        )
        
        col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
        col1.markdown('<div style="font-size: 1.5em; text-align: center;">💰</div>', unsafe_allow_html=True)
        priorite_pct = col2.slider(
            "Priorité",
            0, 100, 50, 5,
            format="%d%%",
            label_visibility="collapsed",
        )
        col3.markdown('<div style="font-size: 1.5em; text-align: center;">🛡️</div>', unsafe_allow_html=True)
        qualite_weight = priorite_pct / 100.0
        
        # Initialize horizon_ans if not set
        if "horizon_ans" not in st.session_state:
            st.session_state.horizon_ans = 25
        
        horizon = st.slider(
            "Horizon d'investissement (ans)",
            10, 30,
            key="horizon_ans",
        )
        st.caption(f"Horizon de simulation : **{horizon} ans**")
        
    return apport, cf_cible, tolerance, mode_cf, qualite_weight, horizon


def render_credit_params_tab() -> Dict[str, Any]:
    """Render credit parameters tab.
    
    Returns:
        Dictionary of credit parameters.
    """
    c1, c2, c3 = st.columns(3)
    taux_15 = c1.number_input("15 ans (%)", value=3.2, format="%.2f")
    taux_20 = c2.number_input("20 ans (%)", value=3.4, format="%.2f")
    taux_25 = c3.number_input("25 ans (%)", value=3.6, format="%.2f")
    
    st.markdown("---")
    
    frais_notaire = st.slider("Frais de notaire (%)", 0.0, 12.0, 7.5, 0.1)
    assurance = st.slider("Assurance emprunteur (% cap.)", 0.00, 1.00, 0.35, 0.01)
    frais_pret = st.slider("Frais de prêt (% cap.)", 0.0, 3.0, 1.0, 0.1)
    
    st.markdown("**Inclusion des coûts dans le financement**")
    c1, c2, c3, c4 = st.columns(4)
    inclure_travaux = c1.checkbox("Travaux", value=True)
    inclure_reno = c2.checkbox("Reno E→D", value=True)
    inclure_mobilier = c3.checkbox("Mobilier", value=True)
    financer_mobilier = c4.checkbox("Financé", value=True)
    
    st.markdown("**Frais de remboursement anticipés**")
    apply_ira = st.toggle(
        "Appliquer les frais (IRA)",
        value=st.session_state.get("apply_ira", True),
        help="Indemnité si le prêt est remboursé avant son terme.",
    )
    ira_cap = st.number_input(
        "Plafond % CRD",
        0.0, 10.0,
        st.session_state.get("ira_cap_pct", 3.0),
        0.1,
        help="Plafond légal sur le capital restant dû.",
    )
    
    st.session_state["apply_ira"] = apply_ira
    st.session_state["ira_cap_pct"] = ira_cap
    
    return {
        "taux_credits": {15: taux_15, 20: taux_20, 25: taux_25},
        "frais_notaire_pct": frais_notaire,
        "assurance_ann_pct": assurance,
        "frais_pret_pct": frais_pret,
        "inclure_travaux": inclure_travaux,
        "inclure_reno_ener": inclure_reno,
        "inclure_mobilier": inclure_mobilier,
        "financer_mobilier": financer_mobilier,
        "apply_ira": apply_ira,
        "ira_cap_pct": ira_cap,
    }


def render_market_hypotheses() -> Dict[str, float]:
    """Render market hypotheses section.
    
    Returns:
        Dictionary with appreciation_bien_pct, revalo_loyer_pct, inflation_charges_pct
    """
    with st.expander("📈 Hypothèses de Marché", expanded=False):
        st.caption("Projections annuelles utilisées pour la simulation long terme.")
        
        # Initialize session state defaults
        if "appreciation_bien_pct" not in st.session_state:
            st.session_state.appreciation_bien_pct = 2.5
        if "revalo_loyer_pct" not in st.session_state:
            st.session_state.revalo_loyer_pct = 1.5
        if "inflation_charges_pct" not in st.session_state:
            st.session_state.inflation_charges_pct = 2.0
        
        appreciation = st.slider(
            "Appréciation annuelle biens (%)",
            -2.0, 8.0,
            step=0.1,
            key="appreciation_bien_pct",
            help="Croissance annuelle estimée de la valeur des biens immobiliers.",
        )
        
        revalo = st.slider(
            "Revalorisation annuelle loyers (%)",
            -2.0, 5.0,
            step=0.1,
            key="revalo_loyer_pct",
            help="Augmentation annuelle estimée des loyers (généralement basée sur l'IRL).",
        )
        
        inflation = st.slider(
            "Inflation annuelle charges (%)",
            -2.0, 5.0,
            step=0.1,
            key="inflation_charges_pct",
            help="Augmentation annuelle estimée des charges (copropriété, taxe foncière).",
        )
        
    return {
        "appreciation_bien_pct": appreciation,
        "revalo_loyer_pct": revalo,
        "inflation_charges_pct": inflation,
    }


def render_scoring_preset() -> Tuple[str, Dict[str, float]]:
    """Render scoring preset selector.
    
    Returns:
        Tuple of (preset_name, weights_dict)
    """
    with st.expander("🎲 Profil de Tri Financier", expanded=False):
        st.caption(
            "Le tri financier combine **Enrichissement**, **IRR**, **Efficacité**, "
            "**DSCR** et **Proximité du CF** selon le profil choisi."
        )
        
        # Initialize session state with robust checks
        if "finance_preset" not in st.session_state or st.session_state.finance_preset is None:
            st.session_state.finance_preset = "Équilibré (défaut)"
        if "finance_custom" not in st.session_state or st.session_state.finance_custom is None:
            st.session_state.finance_custom = FINANCIAL_PRESETS["Équilibré (défaut)"].copy()
        
        col_p, col_c = st.columns([2, 1])
        with col_p:
            preset = st.selectbox(
                "Profil de tri",
                list(FINANCIAL_PRESETS.keys()),
                index=list(FINANCIAL_PRESETS.keys()).index(st.session_state.finance_preset),
            )
            if preset != st.session_state.finance_preset:
                st.session_state.finance_preset = preset
                st.session_state.finance_custom = FINANCIAL_PRESETS[preset].copy()
        
        with col_c:
            custom_on = st.toggle(
                "Personnaliser",
                value=False,
                help="Ajuster finement les poids (somme renormalisée).",
            )
        
        weights = FINANCIAL_PRESETS[st.session_state.finance_preset].copy()
        
        if custom_on:
            st.write("Réglage fin des poids :")
            c1, c2, c3 = st.columns(3)
            c4, c5 = st.columns(2)
            
            w1 = c1.slider("Enrich. net", 0, 100, int(st.session_state.finance_custom["enrich_net"] * 100), 1)
            w2 = c2.slider("IRR", 0, 100, int(st.session_state.finance_custom["irr"] * 100), 1)
            w3 = c3.slider("Efficacité", 0, 100, int(st.session_state.finance_custom["cap_eff"] * 100), 1)
            w4 = c4.slider("DSCR", 0, 100, int(st.session_state.finance_custom["dscr"] * 100), 1)
            w5 = c5.slider("Proximité CF", 0, 100, int(st.session_state.finance_custom["cf_proximity"] * 100), 1)
            
            raw_weights = {
                "enrich_net": float(w1) / 100.0,
                "irr": float(w2) / 100.0,
                "cap_eff": float(w3) / 100.0,
                "dscr": float(w4) / 100.0,
                "cf_proximity": float(w5) / 100.0,
            }
            
            # Normalize
            total = sum(raw_weights.values())
            if total > 0:
                weights = {k: v / total for k, v in raw_weights.items()}
            else:
                weights = raw_weights
            
            st.session_state.finance_custom = raw_weights
        
        st.session_state.finance_weights_override = weights
        
    return st.session_state.finance_preset, weights

