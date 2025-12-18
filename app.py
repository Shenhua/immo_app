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
# Note: Streamlit calls moved to main() to prevent blocking on import


def load_css() -> None:
    """Load custom CSS styles."""
    css_file = os.path.join(os.path.dirname(__file__), "src/ui/assets/style.css")
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_sidebar() -> dict[str, Any]:
    """Render sidebar and collect all parameters.

    Returns:
        Dictionary containing all sidebar parameters
    """
    with st.sidebar:
        st.title("⚙️ Paramètres")

        # Objectives
        objectives = render_objectives_section()
        
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
        
        # Advanced Settings (Phase 17.1)
        with st.expander("⚙️ Avancé", expanded=False):
            top_n = st.slider(
                "Nombre max de résultats",
                min_value=10, max_value=500, value=50, step=10,
                help="Limite le nombre de stratégies affichées après optimisation."
            )

    return {
        "apport": objectives["apport"],
        "cf_cible": objectives["cf_cible"],
        "tolerance": objectives["tolerance"],
        "mode_cf": objectives["mode_cf"],
        "qual_weight": objectives["qualite_weight"],
        "horizon": objectives["horizon"],
        "max_properties": objectives.get("max_properties", 3),
        "use_full_capital": objectives.get("use_full_capital", False),
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
        "top_n": top_n,
    }


def handle_analysis(params: dict[str, Any], archetypes: list[dict[str, Any]]) -> None:
    """Handle the analysis button click with progress display.

    Args:
        params: Sidebar parameters
        archetypes: Filtered and compliant archetypes
    """
    import time
    from src.ui.components.progress_display import render_progress_display
    from src.ui.progress import SearchProgress, SearchStats
    from src.ui.state import set_state
    
    # Track timing
    start_time = time.time()
    
    # Track stats during search
    search_stats_tracker = {
        "bricks_count": 0,
        "combos_evaluated": 0,
        "valid_strategies": 0,
    }
    
    # Create a placeholder for progress updates
    progress_placeholder = st.empty()
    
    def on_progress(progress: SearchProgress):
        """Callback to update progress display and track stats."""
        with progress_placeholder.container():
            render_progress_display(progress)
        
        # Track stats
        if progress.items_total > 0:
            search_stats_tracker["combos_evaluated"] = progress.items_total
        if progress.valid_count > 0:
            search_stats_tracker["valid_strategies"] = progress.valid_count
    
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
        "max_properties": params.get("max_properties", 3),
        "use_full_capital": params.get("use_full_capital", False),
    }

    # Run search with progress callback
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
        top_n=params.get("top_n", 50),
        progress_callback=on_progress,
    )
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Store search stats for persistent display
    search_stats = SearchStats(
        duration_seconds=duration,
        bricks_count=len(archetypes) * 3,  # Approx bricks per archetype
        combos_evaluated=search_stats_tracker["combos_evaluated"],
        valid_strategies=search_stats_tracker["valid_strategies"],
        strategies_after_dedupe=len(strategies),
        mode="EXHAUSTIVE",
        max_properties=params.get("max_properties", 3),
    )
    # Store as dict for session state serialization
    set_state("search_stats", {
        "timestamp": search_stats.timestamp,
        "duration_seconds": search_stats.duration_seconds,
        "bricks_count": search_stats.bricks_count,
        "combos_generated": search_stats.combos_generated,
        "combos_evaluated": search_stats.combos_evaluated,
        "valid_strategies": search_stats.valid_strategies,
        "strategies_after_dedupe": search_stats.strategies_after_dedupe,
        "mode": search_stats.mode,
        "max_properties": search_stats.max_properties,
    })
    
    # Clear progress display
    progress_placeholder.empty()

    # Save results
    SessionManager.set_strategies(strategies)
    autosave_results(strategies, {"horizon": params["horizon"], "compliance": params["apply_cap"]})

    st.rerun()


def main() -> None:
    """Main application entry point."""
    # Streamlit configuration (must be first Streamlit call)
    st.set_page_config(
        page_title="Simulateur Immo v27.6",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_css()
    log = get_logger(__name__)
    
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

    # 9. Render Debug Tools (at bottom of sidebar)
    from src.ui.components.sidebar import render_debug_section
    render_debug_section(params)


if __name__ == "__main__":
    main()
