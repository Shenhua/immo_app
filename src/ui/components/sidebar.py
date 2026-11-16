"""Sidebar components for the main app.

Reusable sidebar sections that can be composed in the main page.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.application.validation import MAX_RATE_PCT
from src.core.defaults import (
    DEFAULT_PRESET,
    FINANCING,
    MARKET,
    PRESETS,
    SEARCH,
    TAX,
)

# Preset financial profiles, keyed by display label. Weights come from src/core/defaults.py.
FINANCIAL_PRESETS: dict[str, dict[str, float]] = {
    pid.label: weights.as_dict() for pid, weights in PRESETS.items()
}
DEFAULT_PRESET_LABEL = DEFAULT_PRESET.label


PRECISE_TARGETING_HELP = (
    "**Désactivé (défaut)** : le cash-flow doit atteindre AU MINIMUM la cible ; un portefeuille plus "
    "rentable est accepté. La tolérance n'intervient pas.\n\n"
    "**Activé** : on ne garde que les portefeuilles dont le cash-flow est le plus PROCHE possible de la "
    "cible, à ± la tolérance près (ni nettement au-dessus, ni en dessous)."
)


def render_objectives_section() -> dict[str, Any]:
    """Render the objectives section of the sidebar.

    Returns:
        Tuple of (apport, cf_cible, tolerance, mode_cf, qualite_weight, horizon)
    """
    # --- Investment Strategy ---
    st.sidebar.markdown("### 🎯 Stratégie d'Investissement")
    strategy_mode = st.sidebar.radio(
        "Mode d'allocation",
        # Rentier mode is hidden until T34 restores it (its backend flag stays in place).
        ["Classique (Levier)", "Empire (Max Volume)"],
        index=0,
        help=(
            "**Classique (Levier)** : Cherche l'équilibre. Utilise l'apport minimum pour maximiser le retour sur investissement (TRI).\n\n"
            "**Empire (Max Volume)** : Cherche à acquérir le plus grand parc immobilier possible avec votre apport.\n\n"
            "Le nombre maximum de biens se règle séparément ci-dessous (le changement de mode le repositionne "
            "à sa valeur habituelle : 3 en Classique, 5 en Empire)."
        )
    )

    # --- Financial Profile Auto-Switch Logic ---
    # To help the user, we switch the scoring profile when the strategy changes.
    last_strat = st.session_state.get("last_strategy_mode", None)

    mode_changed = last_strat is not None and last_strat != strategy_mode
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
            target_preset = DEFAULT_PRESET_LABEL

        if target_preset and target_preset in FINANCIAL_PRESETS:
             st.session_state.finance_preset = target_preset
             st.session_state.finance_custom = FINANCIAL_PRESETS[target_preset].copy()
             # Notify user via toast (Streamlit 1.30+) or just let them see the change
             # st.toast(f"Profil financier ajusté : {target_preset}")

    # Logic flags
    is_rentier = "Rentier" in strategy_mode
    is_empire = "Empire" in strategy_mode
    use_full_capital = is_rentier

    # Max number of properties: its own control (T43), independent of the mode radio. A mode change
    # resets it to the mode's usual value; the user can then adjust it.
    if "max_properties" not in st.session_state:
        st.session_state.max_properties = SEARCH.max_properties_classique
    if mode_changed:
        st.session_state.max_properties = (
            SEARCH.max_properties_empire if is_empire else SEARCH.max_properties_classique
        )
    max_props = st.sidebar.slider(
        "Nombre max de biens",
        SEARCH.max_properties_min,
        SEARCH.max_properties_max,
        key="max_properties",
        help="Taille maximale du portefeuille. Plus il est élevé, plus la recherche est longue.",
    )

    # --- Financial Inputs ---
    with st.sidebar.expander("💰 Configuration Financière", expanded=True):
        apport = st.number_input(
            "Apport Total Disponible (€)",
            min_value=0,
            value=SEARCH.apport,
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
                    min_value=-2000.0,
                    max_value=2000.0,
                    value=float(SEARCH.cf_cible),
                    step=50.0,
                    help="Le cash-flow net minimum que vous visez (négatif = effort d'épargne)."
                )
             else:
                cf_cible = 0.0
                st.info("Rentier:\nMaximisation Auto")

        # Precise Targeting:
        # - Rentier: Forced to False (Min Mode / Maximize)
        # - Empire/Classique: User choice
        # Rendered before the tolerance because the tolerance only applies when it is on.
        if is_rentier:
            is_precise = False
            # Visual feedback that it's disabled/forced
            st.caption("🔒 Ciblage Précis désactivé (Mode Rentier)")
        else:
            is_precise = st.toggle(
                "Ciblage Précis (±)",
                value=False,
                help=PRECISE_TARGETING_HELP,
            )
        mode_cf = "target" if is_precise else "min"

        with c_tol:
            # The tolerance only has an effect in "target" mode (T30): a portfolio is kept when its
            # cash flow is within cible ± tolérance. In "min" mode the cash flow only has to reach the
            # target, so the input is hidden and the default is passed on (it is ignored by the search).
            if is_precise:
                tolerance = st.number_input(
                    "Tolérance (±€/mois)",
                    min_value=0,
                    value=SEARCH.tolerance,
                    step=10,
                    help="Écart accepté autour de la cible, en mode Ciblage Précis uniquement.",
                )
            else:
                tolerance = SEARCH.tolerance
                st.caption("Tolérance : sans effet hors Ciblage Précis.")

        # Initialize horizon_ans if not set
        if "horizon_ans" not in st.session_state:
            st.session_state.horizon_ans = SEARCH.horizon_ans

        curr_h = st.session_state.get("horizon_ans", SEARCH.horizon_ans)
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
            0, 100,
            value=SEARCH.priorite_pct,
            step=5,
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
    durees = st.multiselect(
        "Durées de prêt (ans)",
        options=list(range(FINANCING.duree_min_ans, FINANCING.duree_max_ans + 1)),
        default=list(FINANCING.durees_defaut),
        max_selections=FINANCING.max_durations,
        help=(
            f"De {FINANCING.duree_min_ans} à {FINANCING.duree_max_ans} ans, {FINANCING.max_durations} durées au "
            "plus. La recherche compare ces durées pour chaque bien ; chacune a son taux."
        ),
    )
    taux_credits: dict[int, float] = {}
    for years in sorted(durees):
        taux_credits[years] = st.number_input(
            f"Taux {years} ans (%)",
            min_value=0.0,
            max_value=MAX_RATE_PCT,
            value=FINANCING.default_rate(years),
            step=0.05,
            format="%.2f",
            key=f"taux_duree_{years}",
        )

    st.markdown("---")

    with st.expander("⚙️ Options Avancées & Coûts", expanded=False):
        frais_notaire = st.slider("Frais de notaire (%)", 0.0, 12.0, value=FINANCING.frais_notaire_pct, step=0.1)
        assurance = st.slider("Assurance emprunteur (% cap.)", 0.00, 1.00, value=FINANCING.assurance_ann_pct, step=0.01)
        frais_pret = st.slider("Frais de prêt (% cap.)", 0.0, 3.0, value=FINANCING.frais_pret_pct, step=0.1)

        st.markdown("**Inclusion des coûts dans le financement**")
        c1, c2, c3, c4 = st.columns(4)
        inclure_travaux = c1.checkbox("Travaux", value=FINANCING.inclure_travaux)
        inclure_reno = c2.checkbox("Reno E→D", value=FINANCING.inclure_reno_ener)
        inclure_mobilier = c3.checkbox("Mobilier", value=FINANCING.inclure_mobilier)
        financer_mobilier = c4.checkbox("Financé", value=FINANCING.financer_mobilier)

        st.markdown("**Part terrain (non amortissable)**")
        override_terrain = st.toggle(
            "Imposer la part terrain",
            value=False,
            help="Désactivé : la part de chaque archétype est utilisée (15 % par défaut, 20 % à Lyon et Annecy).",
        )
        ratio_terrain = st.number_input(
            "Part terrain %",
            min_value=0.0,
            max_value=60.0,
            value=TAX.ratio_terrain_pct,
            step=1.0,
            disabled=not override_terrain,
            help="Part du prix (frais de notaire inclus) qui représente le terrain : elle n'est pas amortie en LMNP réel.",
        )

        st.markdown("**Plus-value à la revente**")
        nb_owners = st.selectbox(
            "Nombre de propriétaires",
            options=[1, 2],
            index=TAX.nb_owners - 1,
            help="Chaque bien est détenu et vendu par 1 ou 2 propriétaires (couple en indivision). "
            "Le seuil de la surtaxe sur les plus-values élevées (50 000 €) s'applique à la part de chacun.",
        )

        st.markdown("**Frais de remboursement anticipés**")
        apply_ira = st.toggle(
            "Appliquer les frais (IRA)",
            value=st.session_state.get("apply_ira", FINANCING.apply_ira),
            help="Indemnité si le prêt est remboursé avant son terme.",
        )
        ira_cap = st.number_input(
            "Plafond % CRD",
            0.0, 10.0,
            value=st.session_state.get("ira_cap_pct", FINANCING.ira_cap_pct),
            step=0.1,
            help="Plafond légal sur le capital restant dû.",
        )

        st.session_state["apply_ira"] = apply_ira
        st.session_state["ira_cap_pct"] = ira_cap

    return {
        "taux_credits": taux_credits,
        "frais_notaire_pct": frais_notaire,
        "assurance_ann_pct": assurance,
        "frais_pret_pct": frais_pret,
        "inclure_travaux": inclure_travaux,
        "inclure_reno_ener": inclure_reno,
        "inclure_mobilier": inclure_mobilier,
        "financer_mobilier": financer_mobilier,
        "apply_ira": apply_ira,
        "ira_cap_pct": ira_cap,
        "ratio_terrain_pct": float(ratio_terrain) if override_terrain else None,
        "nb_owners": int(nb_owners),
    }


# Tax regimes offered per income category (TAX_SPEC 8): label -> engine value.
REGIMES_MEUBLE: dict[str, str] = {"LMNP réel": "lmnp", "Micro-BIC": "microbic"}
REGIMES_NU: dict[str, str] = {"Réel (déficit foncier)": "reel", "Micro-foncier": "micro"}


def _label_for(options: dict[str, str], value: str) -> int:
    values = list(options.values())
    return values.index(value) if value in values else 0


def render_tax_section() -> dict[str, Any]:
    """Tax choices: TMI, one regime per category (furnished / unfurnished), or a global SCI at IS.

    Returns:
        Dict with tmi, regime (furnished), regime_foncier (unfurnished) and sci_is.
    """
    tmi = st.slider("TMI (%)", 0, 45, value=int(TAX.tmi_pct), step=1)
    sci_is = st.checkbox(
        "SCI à l'IS (tous les biens)",
        value=TAX.sci_is,
        key="tax_sci_is",
        help=(
            "Tous les biens dans une SCI à l'IS : IS 15 % jusqu'à 42 500 € puis 25 %, amortissements, "
            "plus-value sur la valeur nette comptable. Les bénéfices restent dans la SCI ; "
            "flat tax 31,4 % sur le boni de liquidation à la revente. Remplace les deux régimes ci-dessous."
        ),
    )
    meuble_label = st.selectbox(
        "Régime des biens meublés",
        list(REGIMES_MEUBLE),
        index=_label_for(REGIMES_MEUBLE, TAX.regime_fiscal),
        disabled=sci_is,
        key="tax_regime_meuble",
        help=(
            "Un seul régime pour tous les biens meublés du foyer. Micro-BIC : abattement 50 %, "
            "plafond 83 600 € de recettes ; au-delà, le LMNP réel est appliqué (avertissement affiché)."
        ),
    )
    nu_label = st.selectbox(
        "Régime des biens loués nus",
        list(REGIMES_NU),
        index=_label_for(REGIMES_NU, TAX.regime_foncier),
        disabled=sci_is,
        key="tax_regime_nu",
        help=(
            "Un seul régime pour tous les biens nus du foyer. Réel : charges, intérêts et travaux déductibles, "
            "déficit foncier imputable sur le revenu global jusqu'à 10 700 €. Micro-foncier : abattement 30 %, "
            "plafond 15 000 € de loyers ; au-delà, le réel est appliqué (avertissement affiché)."
        ),
    )
    return {
        "tmi": tmi,
        "regime": REGIMES_MEUBLE[meuble_label],
        "regime_foncier": REGIMES_NU[nu_label],
        "sci_is": bool(sci_is),
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
            st.session_state.appreciation_bien_pct = MARKET.appreciation_bien_pct
        if "revalo_loyer_pct" not in st.session_state:
            st.session_state.revalo_loyer_pct = MARKET.revalo_loyer_pct
        if "inflation_charges_pct" not in st.session_state:
            st.session_state.inflation_charges_pct = MARKET.inflation_charges_pct

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
            st.session_state.finance_preset = DEFAULT_PRESET_LABEL
        if "finance_custom" not in st.session_state or st.session_state.finance_custom is None:
            st.session_state.finance_custom = FINANCIAL_PRESETS[DEFAULT_PRESET_LABEL].copy()

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

