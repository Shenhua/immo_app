"""Chart components for visualization."""

from __future__ import annotations

from typing import Any, Dict, List
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def render_simulation_chart(df: pd.DataFrame, key: str = "sim") -> None:
    """Render simulation timeline chart.
    
    Args:
        df: Simulation DataFrame with yearly data
        key: Unique key for the chart element
    """
    if df is None or df.empty:
        st.warning("Pas de données de simulation disponibles.")
        return
    
    # Patrimoine net over time
    fig = px.area(
        df,
        x="Année",
        y="Patrimoine Net",
        title="Évolution du Patrimoine Net",
        labels={"Patrimoine Net": "Patrimoine (€)"},
    )
    fig.update_layout(
        hovermode="x unified",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"patrimoine_{key}")


def render_cashflow_chart(df: pd.DataFrame, key: str = "cf") -> None:
    """Render cash flow timeline chart.
    
    Args:
        df: Simulation DataFrame
        key: Unique key for the chart element
    """
    if df is None or df.empty:
        return
    
    fig = px.bar(
        df,
        x="Année",
        y="Cash-Flow Net d'Impôt",
        title="Cash-Flow Net Annuel",
        color="Cash-Flow Net d'Impôt",
        color_continuous_scale=["#dc3545", "#ffc107", "#28a745"],
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key=f"cashflow_{key}")


def render_comparison_charts(
    strategies: List[Dict[str, Any]],
    horizon: int = 25,
) -> None:
    """Render comparison charts for multiple strategies.
    
    Args:
        strategies: List of strategy dicts
        horizon: Simulation horizon
    """
    if not strategies:
        st.warning("Aucune stratégie à comparer.")
        return
    
    # Build comparison DataFrame
    data = []
    for i, s in enumerate(strategies, 1):
        data.append({
            "Stratégie": f"#{i}",
            "TRI (%)": s.get("tri_annuel", 0),
            "Cash-Flow (€)": s.get("cash_flow_final", 0),
            "Patrimoine (€)": s.get("patrimoine_acquis", 0),
            "Apport (€)": s.get("apport_total", 0),
            "Score": s.get("balanced_score", 0) * 100,
        })
    
    comp_df = pd.DataFrame(data)
    
    # Layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Score comparison
        fig1 = px.bar(
            comp_df,
            x="Stratégie",
            y="Score",
            title="Comparaison des Scores",
            color="Score",
            color_continuous_scale="Viridis",
        )
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Cash flow comparison
        fig2 = px.bar(
            comp_df,
            x="Stratégie",
            y="Cash-Flow (€)",
            title="Comparaison Cash-Flow",
            color="Cash-Flow (€)",
            color_continuous_scale=["#dc3545", "#28a745"],
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        # TRI comparison
        fig3 = px.bar(
            comp_df,
            x="Stratégie",
            y="TRI (%)",
            title=f"Comparaison TRI ({horizon}a)",
            text_auto=".2f",
        )
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Apport vs Patrimoine
        fig4 = px.bar(
            comp_df,
            x="Stratégie",
            y=["Apport (€)", "Patrimoine (€)"],
            title="Apport vs Patrimoine Acquis",
            barmode="group",
        )
        st.plotly_chart(fig4, use_container_width=True)


def render_strategy_radar(strategy: Dict[str, Any], key: str = "radar") -> None:
    """Render radar chart for strategy scores.
    
    Args:
        strategy: Strategy dictionary
        key: Unique key for the chart element
    """
    categories = [
        "TRI",
        "Cash-Flow",
        "Sécurité (DSCR)",
        "Qualité",
        "Efficacité",
    ]
    
    values = [
        strategy.get("tri_norm", 0) * 100,
        strategy.get("cf_proximity", 0) * 100,
        strategy.get("dscr_norm", 0) * 100,
        strategy.get("qual_score", 50),
        strategy.get("cap_eff_norm", 0) * 100,
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        name="Performance",
        line_color="#4CAF50",
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
            ),
        ),
        showlegend=False,
        title="Profil de la Stratégie",
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"radar_{key}")
