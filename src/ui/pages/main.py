"""Main page rendering.

Composes all UI components into the main application page.
This module can be used standalone or imported by app.py.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.ui.components.charts import (
    render_cashflow_chart,
    render_comparison_charts,
    render_comparison_heatmap,
    render_credit_breakdown_chart,
    render_fiscal_chart,
    render_risk_reward_scatter,
    render_simulation_chart,
    render_strategy_radar,
)
from src.ui.components.results import (
    render_kpi_summary,
    render_strategy_card,
    render_strategy_details,
)
from src.ui.components.sensitivity import render_sensitivity_analysis
from src.ui.state import SessionManager, get_state, set_state


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
    strategies: list[dict[str, Any]],
    horizon: int = 25,
    df_sim: Any | None = None,
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

    # --- MASTER-DETAIL LAYOUT ---

    # 1. TOP SECTION: The Stage (Selected Strategy)
    if strategies and 0 <= selected_idx < len(strategies):
        current_strat = strategies[selected_idx]
        is_hero = (selected_idx == 0)

        # Dynamic Header
        if is_hero:
            st.markdown("### 🏆 Meilleure Stratégie")
        else:
            st.markdown(f"### 📊 Stratégie #{selected_idx + 1}")

        # Helper for details content
        def render_details_content():
            """Render details for the selected strategy."""
            render_kpi_summary(current_strat, horizon)
            tab1, tab2, tab3, tab4 = st.tabs(["🏠 Biens", "📈 Projections", "🎯 Radar", "⚡ Stress Test"])
            with tab1:
                render_strategy_details(current_strat, horizon)
            with tab2:
               if df_sim is not None and not df_sim.empty:
                   # Row 1: Patrimoine & CashFlow
                   col1, col2 = st.columns(2)
                   with col1:
                       render_simulation_chart(df_sim, key=f"strat_chart_{selected_idx}")
                   with col2:
                       render_cashflow_chart(df_sim, key=f"strat_cf_{selected_idx}")
                   
                   # Row 2: Credit breakdown & Fiscal (if LMNP)
                   col3, col4 = st.columns(2)
                   with col3:
                       render_credit_breakdown_chart(df_sim, key=f"strat_cred_{selected_idx}")
                   with col4:
                       render_fiscal_chart(df_sim, key=f"strat_fisc_{selected_idx}")
               else:
                   st.info("Projections disponibles après sélection.")
            with tab3:
                render_strategy_radar(current_strat, key=f"strat_radar_{selected_idx}")
            with tab4:
                render_sensitivity_analysis(current_strat, horizon, key=f"strat_sens_{selected_idx}")

        # Render the Card for the Selected Strategy
        # We force 'show_details' to be True if it was triggered by a table click?
        # User said "load itself in the Hero section". Usually implies expanded view.
        # But we respect the 'show_details' toggle state to allow collapsing.

        # If selection changed recently, we might want to auto-expand.
        # But 'show_details' state is our truth.

        expanded_fn = render_details_content if show_details else None

        if render_strategy_card(
            current_strat,
            index=selected_idx + 1,
            horizon=horizon,
            is_selected=True, # Always selected in the Stage
            show_details=show_details,
            expanded_content=expanded_fn
        ):
            # Toggle logic for the Stage Card button
            set_state("show_details", not show_details)
            st.rerun()

    # 2. BOTTOM SECTION: The Playlist (Table)
    if len(strategies) > 0:
        st.markdown("---")
        st.subheader(f"📋 Comparatif ({len(strategies)})")

        import pandas as pd

        data = []
        max_enrich = 0
        min_enrich = 0
        
        for i, s in enumerate(strategies):
            taxonomy = s.get("taxonomy", "Mix")
            icon = {"Optimisé": "🚀", "Patrimonial": "🏛️", "Mix": "⚖️"}.get(taxonomy, "🔀")
            is_hero = (i == 0)
            hero_mark = "🏆 " if is_hero else ""
            
            enrich = int(s.get("liquidation_nette", 0) / 1000)
            max_enrich = max(max_enrich, enrich)
            min_enrich = min(min_enrich, enrich)

            data.append({
                "Stratégie": f"{hero_mark}{icon} Stratégie #{i+1}",
                "Cash-Flow": s.get("cash_flow_final", 0),
                "TRI (%)": s.get("tri_annuel", 0),
                "Enrich. (k€)": enrich,
                "Score": int(s.get("balanced_score", 0) * 100)
            })

        df_display = pd.DataFrame(data)
        
        # Apply color gradients using Pandas Styler
        # Note: We must be careful not to break st.dataframe interactivity.
        # Simple text coloring is robust.
        def color_score(val):
            # Red (0) -> Green (100)
            # Simple threshold logic for readability
            if val >= 80: return 'color: #28a745; font-weight: bold'
            if val >= 50: return 'color: #ffc107'
            return 'color: #dc3545'
            
        def color_tri(val):
            if val >= 8.0: return 'color: #28a745; font-weight: bold'
            if val >= 4.0: return 'color: #ffc107'
            return 'color: #dc3545'
            
        # Create Styler
        styler = df_display.style.map(color_score, subset=["Score"])\
                                 .map(color_tri, subset=["TRI (%)"])

        column_config = {
            "Stratégie": st.column_config.TextColumn("Stratégie", width="medium"),
            "Cash-Flow": st.column_config.NumberColumn("Cash-Flow", format="%.0f €"),
            "TRI (%)": st.column_config.NumberColumn("TRI", format="%.1f %%"),
            "Enrich. (k€)": st.column_config.ProgressColumn(
                "Enrich.", 
                format="%d k€",
                min_value=min(0, min_enrich),
                max_value=max(100, max_enrich),
            ),
            "Score": st.column_config.NumberColumn("Score", format="%d"),
        }

        # Interactive Table
        selection = st.dataframe(
            styler, # Pass styler instead of raw df
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
            on_select="rerun",
            selection_mode="single-row",
            key="strategy_table"
        )

        # Handle Selection
        if selection.selection.rows:
            new_idx = selection.selection.rows[0]
            if new_idx != selected_idx:
                set_state("selected_strategy_idx", new_idx)
                # State persistence: We DO NOT force show_details=True here.
                # We keep the user's current fold/unfold state.
                st.rerun()

    return selected_idx


def render_comparison_panel(strategies: list[dict[str, Any]], horizon: int = 25) -> None:
    """Render comparison view for multiple strategies.

    Args:
        strategies: Strategies to compare
        horizon: Simulation horizon
    """
    st.markdown("---")
    st.markdown("## ⚖️ Comparaison des Stratégies")

    render_risk_reward_scatter(strategies)
    st.markdown("#### Performance Relative")
    render_comparison_heatmap(strategies)
    
    st.markdown("#### Détails côte-à-côte")
    render_comparison_charts(strategies[:6], horizon)


def render_main_content(
    strategies: list[dict[str, Any]],
    df_sim: Any | None = None,
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
    archetypes: list[dict[str, Any]],
    strategies: list[dict[str, Any]] | None = None,
    df_sim: Any | None = None,
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
    
    # View Mode Toggle
    mode = st.radio(
        "Mode d'affichage",
        ["📝 Liste & Détails", "⚖️ Comparaison Globale"],
        index=1 if show_comparison else 0,
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # Update state only if changed
    new_show_comp = (mode == "⚖️ Comparaison Globale")
    if new_show_comp != show_comparison:
        set_state("show_comparison", new_show_comp)
        st.rerun()

    if strategies is None:
        st.info("👈 Configurez vos paramètres dans la barre latérale puis lancez l'analyse.")
    else:
        render_main_content(strategies, df_sim, horizon, show_comparison)
