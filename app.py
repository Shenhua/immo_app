"""Main Application Entry Point.

Refactored to use modular layered architecture.
Orchestrates UI components and services via app_controller.
"""

import os
import sys
from typing import Any

import streamlit as st

# Add src to path if not present (for running from root)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.logging import get_logger
from src.ui.app_controller import (
    apply_rent_caps,
    autosave_results,
    build_financing_config,
    build_operating_config,
    load_archetypes,
    run_strategy_search,
    simulate_selected_strategy,
)
from src.ui.components.filters import filter_archetypes, render_property_filters
from src.ui.components.sidebar import (
    render_credit_params_tab,
    render_market_hypotheses,
    render_objectives_section,
    render_scoring_preset,
)
from src.ui.pages.main import render_main_page
from src.ui.state import SessionManager

# --- Configuration & Setup ---

st.set_page_config(
    page_title="Simulateur Immo v27.6",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    """Load custom CSS styles."""
    css_file = os.path.join(os.path.dirname(__file__), "src/ui/assets/style.css")
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()

log = get_logger(__name__)


def render_sidebar() -> dict[str, Any]:
    """Render sidebar and collect all parameters.

    Returns:
        Dictionary containing all sidebar parameters
    """
    with st.sidebar:
        st.title("⚙️ Paramètres")

        # Objectives
        apport, cf_cible, tolerance, mode_cf, qual_weight, horizon = render_objectives_section()

        # Credit
        with st.expander("🏦 Financement", expanded=False):
            credit_params = render_credit_params_tab()

        # Operating Costs (Exploitation)
        with st.expander("📉 Frais & Charges", expanded=False):
            cfe = st.number_input("CFE (€/an)", 100, 2000, 500, 50)
            gestion = st.slider("Gestion (%)", 0.0, 15.0, 5.0, 0.5)
            vacance = st.slider("Vacance/Imp. (%)", 0.0, 10.0, 3.0, 0.5)
            frais_vente = st.slider("Frais Revente (%)", 0.0, 10.0, 6.0, 0.5)

        # Tax
        with st.expander("⚖️ Fiscalité", expanded=False):
            tmi = st.slider("TMI (%)", 0, 45, 30, 1)
            regime = st.selectbox("Régime", ["LMNP", "SCI IS"], index=0).lower().replace(" ", "")
            if "lmnp" in regime:
                regime = "lmnp"

        # Property Filters
        raw_archetypes = load_archetypes()
        # User requested folded (expanded=False)
        with st.expander("🏠 Sélection des Biens", expanded=False):
            if raw_archetypes:
                selected_villes, selected_types, apply_cap = render_property_filters(raw_archetypes)
            else:
                selected_villes, selected_types, apply_cap = [], [], True

        # Market Hypotheses
        market_hypo = render_market_hypotheses()

        # Scoring Preset
        finance_preset_name, finance_weights = render_scoring_preset()

    return {
        "apport": apport,
        "cf_cible": cf_cible,
        "tolerance": tolerance,
        "mode_cf": mode_cf,
        "qual_weight": qual_weight,
        "horizon": horizon,
        "credit_params": credit_params,
        "cfe": cfe,
        "gestion": gestion,
        "vacance": vacance,
        "frais_vente": frais_vente,
        "tmi": tmi,
        "regime": regime,
        "raw_archetypes": raw_archetypes,
        "selected_villes": selected_villes,
        "selected_types": selected_types,
        "apply_cap": apply_cap,
        "market_hypo": market_hypo,
        "finance_preset_name": finance_preset_name,
        "finance_weights": finance_weights,
    }


def handle_analysis(params: dict[str, Any], archetypes: list[dict[str, Any]]) -> None:
    """Handle the analysis button click.

    Args:
        params: Sidebar parameters
        archetypes: Filtered and compliant archetypes
    """
    with st.spinner("Analyse des stratégies en cours..."):
        # Build configs
        fin_config = build_financing_config(params["credit_params"])
        op_config = build_operating_config(
            params["gestion"],
            params["vacance"],
            params["cfe"]
        )

        # Build eval params
        eval_params = {
            "tmi_pct": params["tmi"],
            "regime_fiscal": params["regime"],
            "frais_vente_pct": params["frais_vente"],
            "apply_ira": params["credit_params"]["apply_ira"],
            "ira_cap_pct": params["credit_params"]["ira_cap_pct"],
            "cfe_par_bien_ann": params["cfe"],
            "hypotheses_marche": params["market_hypo"],
            "finance_preset_name": params["finance_preset_name"],
            "finance_weights_override": params["finance_weights"],
        }

        # Run search
        strategies = run_strategy_search(
            archetypes=archetypes,
            fin_config=fin_config,
            op_config=op_config,
            apport=params["apport"],
            cf_cible=params["cf_cible"],
            tolerance=params["tolerance"],
            qual_weight=params["qual_weight"],
            mode_cf=params["mode_cf"],
            eval_params=eval_params,
            horizon_years=params["horizon"],
        )

        # Save results
        SessionManager.set_strategies(strategies)
        autosave_results(strategies, {"horizon": params["horizon"], "compliance": params["apply_cap"]})

        st.rerun()


def main() -> None:
    """Main application entry point."""
    # 1. Initialize session
    SessionManager.initialize()
    log.info("app_started")

    # 2. Render sidebar and collect parameters
    params = render_sidebar()

    # 3. Exit early if no data
    if not params["raw_archetypes"]:
        return

    # 4. Apply filters & compliance
    filtered = filter_archetypes(
        params["raw_archetypes"],
        params["selected_villes"],
        params["selected_types"],
    )
    compliant = apply_rent_caps(filtered, params["apply_cap"])

    # 5. Analysis button
    if st.sidebar.button("🚀 Lancer l'analyse", type="primary"):
        handle_analysis(params, compliant)

    # 6. Get current results
    strategies = SessionManager.get_strategies()

    # 7. Simulate selected strategy for charts
    df_sim = None
    idx = SessionManager.get_selected_idx()
    if strategies and 0 <= idx < len(strategies):
        df_sim = simulate_selected_strategy(
            strategy=strategies[idx],
            horizon=params["horizon"],
            credit_params=params["credit_params"],
            tmi=params["tmi"],
            regime=params["regime"],
            cfe=params["cfe"],
            frais_vente=params["frais_vente"],
            market_hypo=params["market_hypo"],
        )

    # 8. Render main page
    render_main_page(compliant, strategies, df_sim)


if __name__ == "__main__":
    main()
