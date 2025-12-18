"""Sidebar components for the main app.

Reusable sidebar sections that can be composed in the main page.
"""

from __future__ import annotations

from typing import Any

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


def render_objectives_section() -> dict[str, Any]:
    """Render the objectives section of the sidebar.

    Returns:
        Tuple of (apport, cf_cible, tolerance, mode_cf, qualite_weight, horizon)
    """
    # --- Investment Strategy ---
    st.sidebar.markdown("### 🎯 Stratégie d'Investissement")
    strategy_mode = st.sidebar.radio(
        "Mode d'allocation",
        ["Classique (Levier)", "Empire (Max Volume)", "Rentier (Max Cash-Flow)"],
        index=0,
        help=(
            "**Classique (Levier)** : Cherche l'équilibre. Utilise l'apport minimum pour maximiser le retour sur investissement (TRI). Limité à 3 biens.\n\n"
            "**Empire (Max Volume)** : Cherche à acquérir le plus grand parc immobilier possible (jusqu'à 5 biens) avec votre apport.\n\n"
            "**Rentier (Max Cash-Flow)** : Objectif de rente. Utilise TOUT votre apport pour réduire les mensualités de crédit et maximiser le cash-flow mensuel net."
        )
    )

    # --- Financial Profile Auto-Switch Logic ---
    # To help the user, we switch the scoring profile when the strategy changes.
    last_strat = st.session_state.get("last_strategy_mode", None)

    if last_strat != strategy_mode:
        st.session_state.last_strategy_mode = strategy_mode
        # Only switch if we are strictly changing (avoid override on first load if user had custom)
        # But here we want to guide them.

        target_preset = None
        if "Rentier" in strategy_mode:
            target_preset = "Cash-flow d'abord" # or Sécurité
        elif "Empire" in strategy_mode:
            target_preset = "Patrimoine LT"
        elif "Classique" in strategy_mode:
            target_preset = "Équilibré (défaut)"

        if target_preset and target_preset in FINANCIAL_PRESETS:
             st.session_state.finance_preset = target_preset
             st.session_state.finance_custom = FINANCIAL_PRESETS[target_preset].copy()
             # Notify user via toast (Streamlit 1.30+) or just let them see the change
             # st.toast(f"Profil financier ajusté : {target_preset}")

    max_props = 3
    use_full_capital = False

    # Logic flags
    is_rentier = "Rentier" in strategy_mode
    is_empire = "Empire" in strategy_mode

    if is_empire:
        max_props = 5
    elif is_rentier:
        use_full_capital = True

    # --- Financial Inputs ---
    with st.sidebar.expander("💰 Configuration Financière", expanded=True):
        apport = st.number_input(
            "Apport Total Disponible (€)",
            min_value=0,
            value=st.session_state.get("apport", 50000),
            step=5000,
            format="%d",
            help="Capital total que vous êtes prêt à investir (frais de notaire + apport bancaire)."
        )

        # In Rentier mode, CF Target is irrelevant because we maximize it.
        # In other modes, we let user set a minimum target.
        c_target, c_tol = st.columns(2)
        with c_target:
             if not is_rentier:
                cf_cible = st.number_input(
                    "Cash-Flow Cible (/mois)",
                    value=st.session_state.get("cf_cible", 0.0),
                    step=50.0,
                    help="Le cash-flow net minimum que vous visez."
                )
             else:
                cf_cible = 0.0
                st.info("Rentier:\nMaximisation Auto")

        with c_tol:
            # Tolerance is only meaningful if we have a target
            if not is_rentier:
                tolerance = st.number_input(
                    "Tolérance (€)",
                    value=100,
                    step=10,
                    help="Marge d'erreur acceptable autour de votre cible.",
                )
            else:
                tolerance = 100.0 # Default value, unused

        # Precise Targeting:
        # - Rentier: Forced to False (Min Mode / Maximize)
        # - Empire/Classique: User choice
        if is_rentier:
            is_precise = False
            # Visual feedback that it's disabled/forced
            st.caption("🔒 Ciblage Précis désactivé (Mode Rentier)")
        else:
            is_precise = st.toggle(
                "Ciblage Précis (±)",
                value=False,
                help="Désactivé : le CF sera au MINIMUM la cible (peut être supérieur).\n\nActivé : le CF visera à être le plus PROCHE possible de la cible.",
            )
        mode_cf = "target" if is_precise else "min"

        # Initialize horizon_ans if not set
        if "horizon_ans" not in st.session_state:
            st.session_state.horizon_ans = 25

        curr_h = st.session_state.get("horizon_ans", 25)
        horizon = st.slider(
            f"Horizon d'investissement ({curr_h} ans)",
            10, 30,
            key="horizon_ans",
        )

        st.markdown("---")
        st.markdown(
            "#### **Priorité de recherche**",
            help="Définissez votre priorité entre la performance financière pure (0%) et la qualité intrinsèque des biens (100%).",
        )

        col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
        col1.markdown('<div style="font-size: 1.5em; text-align: center;">💰</div>', unsafe_allow_html=True)
        priorite_pct = col2.slider(
            "Priorité",
            0, 100, 30, 5,
            format="%d%%",
            label_visibility="collapsed",
        )
        col3.markdown('<div style="font-size: 1.5em; text-align: center;">🛡️</div>', unsafe_allow_html=True)
        qualite_weight = priorite_pct / 100.0



    return {
        "apport": apport,
        "cf_cible": cf_cible,
        "tolerance": tolerance,
        "mode_cf": mode_cf,
        "qualite_weight": qualite_weight,
        "horizon": horizon,
        "max_properties": max_props,
        "use_full_capital": use_full_capital,
    }


def render_credit_params_tab() -> dict[str, Any]:
    """Render credit parameters tab.

    Returns:
        Dictionary of credit parameters.
    """
    c1, c2, c3 = st.columns(3)
    taux_15 = c1.number_input("15 ans (%)", value=3.5, format="%.2f")
    taux_20 = c2.number_input("20 ans (%)", value=3.6, format="%.2f")
    taux_25 = c3.number_input("25 ans (%)", value=3.8, format="%.2f")

    st.markdown("---")

    frais_notaire = st.slider("Frais de notaire (%)", 0.0, 12.0, 8.0, 0.1)
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


def render_market_hypotheses() -> dict[str, float]:
    """Render market hypotheses section.

    Returns:
        Dictionary with appreciation_bien_pct, revalo_loyer_pct, inflation_charges_pct
    """
    with st.expander("📈 Hypothèses de Marché", expanded=False):
        st.caption("Projections annuelles utilisées pour la simulation long terme.")

        # Initialize session state defaults
        if "appreciation_bien_pct" not in st.session_state:
            st.session_state.appreciation_bien_pct = 2.0
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


def render_scoring_preset() -> tuple[str, dict[str, float]]:
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


def render_debug_section(params: dict[str, Any]) -> None:
    """Render debug tools section."""
    st.markdown("---")
    with st.expander("🛠️ Debug AI Agent", expanded=False):
        st.write("Générer un contexte complet pour le débogage par IA.")

        if st.button("📸 Capturer le Contexte"):
            import json

            from src.ui.state import SessionManager
            from src.utils.debug import collect_debug_context

            # Gather state
            strategies = SessionManager.get_strategies()
            # Note: We don't have direct access to last simulation df here unless passed or stored in session
            # For now, we'll rely on what we can gather.

            ctx = collect_debug_context(
                session_state=dict(st.session_state),
                params=params,
                strategies=strategies,
                last_simulation=None # Optional for now
            )

            # Serialize for download
            json_str = json.dumps(ctx, default=str, indent=2, ensure_ascii=False)

            st.download_button(
                label="📥 Télécharger context.json",
                data=json_str,
                file_name="ai_debug_context.json",
                mime="application/json",
            )
            st.success("Contexte capturé !")

