"""Main page rendering.

Composes all UI components into the main application page.
This module can be used standalone or imported by app.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import streamlit as st

from src.ui.state import SessionManager, get_state, set_state
from src.ui.components.filters import render_property_filters, filter_archetypes
from src.ui.components.results import (
    render_strategy_card,
    render_strategy_details,
    render_kpi_summary,
)
from src.ui.components.charts import (
    render_simulation_chart,
    render_cashflow_chart,
    render_comparison_charts,
    render_strategy_radar,
)


def render_header() -> None:
    """Render page header."""
    st.markdown(
        """
        <h1 style="text-align: center;">
            🏢 Simulateur Multi-Stratégies Immo v27.6
        </h1>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Analyse et comparaison de stratégies d'investissement locatif")


def render_no_results() -> None:
    """Render empty state when no strategies found."""
    st.info(
        "🔍 Aucune stratégie trouvée.\n\n"
        "Essayez d'ajuster vos critères:\n"
        "- Augmenter l'apport disponible\n"
        "- Élargir la tolérance du cash-flow\n"
        "- Réduire le cash-flow cible\n"
        "- Sélectionner plus de types de biens"
    )


def render_strategy_list(
    strategies: List[Dict[str, Any]], 
    horizon: int = 25,
    df_sim: Optional[Any] = None,
) -> int:
    """Render list of strategy cards.
    
    Args:
        strategies: List of strategies
        horizon: Simulation horizon
        df_sim: Simulation data for selected strategy (needed for expansion)
        
    Returns:
        Selected strategy index
    """
    selected_idx = get_state("selected_strategy_idx", 0)
    
    for i, strategy in enumerate(strategies):
        # Render Card with embedded button
        is_selected = (i == selected_idx)
        
        # Capture click from inside the card component
        if render_strategy_card(
            strategy,
            index=i + 1,
            horizon=horizon,
            is_selected=is_selected,
        ):
            set_state("selected_strategy_idx", i)
            st.rerun()
            
        # Render Details Panel immediately if selected
        if is_selected:
            with st.container():
                render_selected_strategy_panel(
                    strategy,
                    df_sim,
                    horizon,
                )
    
    return selected_idx


def render_selected_strategy_panel(
    strategy: Dict[str, Any],
    df_sim: Any,
    horizon: int = 25,
) -> None:
    """Render detailed panel for selected strategy.
    
    Args:
        strategy: Selected strategy
        df_sim: Simulation DataFrame
        horizon: Simulation horizon
    """
    # Removed separate header "Analyse Détaillée" as it's implied by expansion
    # render_kpi_summary(strategy, horizon) # Redundant with card? Maybe keep for detailed view
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["🏠 Biens & KPIs", "📈 Projections", "🎯 Analyse Radar"])
    
    with tab1:
        # Show KPI summary here inside tab
        render_kpi_summary(strategy, horizon)
        st.divider()
        render_strategy_details(strategy, horizon)
    
    with tab2:
        if df_sim is not None and not df_sim.empty:
            col1, col2 = st.columns(2)
            with col1:
                render_simulation_chart(df_sim)
            with col2:
                render_cashflow_chart(df_sim)
        else:
            st.info("Simuler la stratégie pour voir les projections.")
    
    with tab3:
        render_strategy_radar(strategy)


def render_comparison_panel(strategies: List[Dict[str, Any]], horizon: int = 25) -> None:
    """Render comparison view for multiple strategies.
    
    Args:
        strategies: Strategies to compare
        horizon: Simulation horizon
    """
    st.markdown("---")
    st.markdown("## ⚖️ Comparaison des Stratégies")
    
    render_comparison_charts(strategies[:6], horizon)


def render_main_content(
    strategies: List[Dict[str, Any]],
    df_sim: Optional[Any] = None,
    horizon: int = 25,
    show_comparison: bool = False,
) -> None:
    """Render the main content area.
    
    Args:
        strategies: Available strategies
        df_sim: Simulation data for selected strategy
        horizon: Simulation horizon
        show_comparison: Whether to show comparison view
    """
    if not strategies:
        render_no_results()
        return
    
    st.success(f"✅ {len(strategies)} stratégie(s) trouvée(s)")
    
    if show_comparison:
        render_comparison_panel(strategies, horizon)
    else:
        # Render list AND details in one go
        render_strategy_list(strategies, horizon, df_sim)


def render_main_page(
    archetypes: List[Dict[str, Any]],
    strategies: Optional[List[Dict[str, Any]]] = None,
    df_sim: Optional[Any] = None,
) -> None:
    """Render the complete main page.
    
    This is the top-level entry point for the main content.
    Sidebar rendering should be handled separately by app.py.
    
    Args:
        archetypes: Available archetypes
        strategies: Found strategies (or None if not yet searched)
        df_sim: Simulation data
    """
    render_header()
    
    # Get current state
    horizon = SessionManager.get_horizon()
    show_comparison = get_state("show_comparison", False)
    
    if strategies is None:
        st.info("👈 Configurez vos paramètres dans la barre latérale puis lancez l'analyse.")
    else:
        render_main_content(strategies, df_sim, horizon, show_comparison)
