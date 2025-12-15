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
from src.ui.components.sensitivity import render_sensitivity_analysis


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
    """Render list of strategy cards using expander pattern.
    
    Args:
        strategies: List of strategies
        horizon: Simulation horizon
        df_sim: Simulation data for selected strategy (needed for expansion)
        
    Returns:
        Selected strategy index
    """
    selected_idx = get_state("selected_strategy_idx", 0)
    show_details = get_state("show_details", False)
    
    # --- HELPER: Details Renderer ---
    def render_details_content():
        """Render details for the selected strategy."""
        if 0 <= selected_idx < len(strategies):
             selected_strat = strategies[selected_idx]
             render_kpi_summary(selected_strat, horizon)
             tab1, tab2, tab3, tab4 = st.tabs(["🏠 Biens", "📈 Projections", "🎯 Radar", "⚡ Stress Test"])
             with tab1: render_strategy_details(selected_strat, horizon)
             with tab2:
                if df_sim is not None and not df_sim.empty:
                    col1, col2 = st.columns(2)
                    with col1: render_simulation_chart(df_sim, key=f"strat_chart_{selected_idx}")
                    with col2: render_cashflow_chart(df_sim, key=f"strat_cf_{selected_idx}")
                else: st.info("Projections disponibles après sélection.")
             with tab3: render_strategy_radar(selected_strat, key=f"strat_radar_{selected_idx}")
             with tab4: render_sensitivity_analysis(selected_strat, horizon, key=f"strat_sens_{selected_idx}")

    # --- HERO SECTION (Strategy #1) ---
    if strategies:
        hero = strategies[0]
        st.markdown("### 🏆 Meilleure Stratégie")
        
        # Determine selection state
        is_hero_selected = (selected_idx == 0)
        
        # Use lambda for deferred rendering
        content_fn = render_details_content if (is_hero_selected and show_details) else None
        
        if render_strategy_card(
            hero, 
            index=1, 
            horizon=horizon, 
            is_selected=is_hero_selected,
            show_details=is_hero_selected and show_details,
            expanded_content=content_fn
        ):
            # Toggle logic
            if is_hero_selected and show_details:
                set_state("show_details", False) # Close if already open
            else:
                set_state("selected_strategy_idx", 0)
                set_state("show_details", True)  # Open if closed or different
            st.rerun()
            
    # --- CONTENDERS (All Strategies including #1) ---
    if len(strategies) > 0:
        st.markdown("---")
        st.subheader(f"📋 Comparatif ({len(strategies)})")
        
        # Legend
        c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 2])
        c1.markdown("**Stratégie**")
        c2.markdown("**Cash Flow**")
        c3.markdown("**TRI**")
        c4.markdown("**Score**")
        c5.markdown("**Action**")
        st.markdown("---")
        
        for i, strategy in enumerate(strategies, 0): # Start at 0 to include Hero
            real_idx = i
            is_selected = (selected_idx == real_idx)
            is_hero = (real_idx == 0)
            
            with st.container():
                c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 2])
                
                # Visuals
                hero_badge = "🏆 " if is_hero else ""
                prefix = "👉 **" if is_selected else ""
                suffix = "**" if is_selected else ""
                
                taxonomy = strategy.get("taxonomy", "Mix")
                icon = {"Optimisé": "🚀", "Patrimonial": "🏛️", "Mix": "⚖️"}.get(taxonomy, "🔀")
                
                c1.markdown(f"{prefix}{hero_badge}{icon} Stratégie #{real_idx+1}{suffix}")
                
                cf = strategy.get("cash_flow_final", 0)
                cf_color = "green" if cf >= 0 else "red"
                c2.markdown(f":{cf_color}[{cf:+.0f} €]")
                c3.markdown(f"{strategy.get('tri_annuel', 0):.1f} %")
                c4.markdown(f"{strategy.get('balanced_score', 0)*100:.0f}/100")
                
                # Dynamic Action Button
                btn_label = "🔍 Voir"
                if is_selected and show_details:
                     btn_label = "🔼 Masquer"
                     
                if c5.button(btn_label, key=f"btn_inspect_{real_idx}", use_container_width=True):
                     if is_selected and show_details:
                         set_state("show_details", False) # Toggle OFF
                     else:
                         set_state("selected_strategy_idx", real_idx)
                         set_state("show_details", True)  # Toggle ON
                     st.rerun()
                
                # Render content INSIDE row container if selected
                if is_selected and show_details:
                     st.markdown("---")
                     render_details_content()
                
                st.markdown("<div style='margin-bottom: 5px;'></div>", unsafe_allow_html=True)
    
    return selected_idx


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
