"""Chart components for visualization."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


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
    if "Dette" in df.columns:
        # Create figure with secondary y-axis style if needed, 
        # but here we just want a line on the same plot or similar.
        # Plotly Express is great for simple, but Graph Objects gives more control for mixed types.
        fig = go.Figure()
        
        # Area for Net Worth
        fig.add_trace(go.Scatter(
            x=df["Année"], 
            y=df["Patrimoine Net"], 
            name="Patrimoine Net",
            fill='tozeroy',
            line=dict(color='#28a745'),
            mode='lines'
        ))
        
        # Line for Debt
        fig.add_trace(go.Scatter(
            x=df["Année"], 
            y=df["Dette"], 
            name="Dette Restante",
            line=dict(color='#dc3545', width=3),
            mode='lines'
        ))
        
        fig.update_layout(
            title="Évolution du Patrimoine Net et Dette",
            xaxis_title="Année",
            yaxis_title="Montant (€)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    else:
        # Fallback if debt not in columns
        fig = px.area(
            df,
            x="Année",
            y="Patrimoine Net",
            title="Évolution du Patrimoine Net"
        )

    st.plotly_chart(fig, use_container_width=True, key=f"patrimoine_{key}")


def render_credit_breakdown_chart(df: pd.DataFrame, key: str = "cred") -> None:
    """Render Stacked Area chart for Credit Payment breakdown.
    
    Args:
        df: Simulation DataFrame
        key: Unique key
    """
    if df is None or df.empty:
        return
        
    required = ["Capital Remboursé", "Intérêts & Assurance"]
    if not all(c in df.columns for c in required):
        return

    fig = px.area(
        df, 
        x="Année", 
        y=required,
        title="Répartition des Paiements du Crédit",
        labels={"value": "Montant (€)", "variable": "Type"},
        color_discrete_map={
            "Capital Remboursé": "#17a2b8", 
            "Intérêts & Assurance": "#ffc107"
        }
    )
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True, key=f"credit_breakdown_{key}")


def render_fiscal_chart(df: pd.DataFrame, key: str = "fisc") -> None:
    """Render LMNP Fiscal composition chart.
    
    Args:
        df: Simulation DataFrame
        key: Unique key
    """
    if df is None or df.empty:
        return

    # Check for LMNP specific columns
    required = ['Loyers Bruts', 'Charges Déductibles', 'Amortissements']
    if not all(c in df.columns for c in required):
        return
        
    fig = px.bar(
        df, 
        x='Année', 
        y=required, 
        title="Composition du Résultat Fiscal (LMNP)", 
        barmode='relative',
        labels={"value": "Montant (€)", "variable": "Poste"},
        color_discrete_map={
            "Loyers Bruts": "#28a745",
            "Charges Déductibles": "#dc3545",
            "Amortissements": "#6c757d"
        }
    )
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True, key=f"fiscal_comp_{key}")


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


def render_risk_reward_scatter(strategies: list[dict[str, Any]], key: str = "scatter") -> None:
    """Render Risk/Reward Scatter Plot (DSCR vs TRI).
    
    Args:
        strategies: List of strategies
        key: Unique key for chart
    """
    if not strategies:
        return
        
    data = []
    for i, s in enumerate(strategies, 1):
        data.append({
            "Label": f"Strat #{i}",
            "TRI (%)": s.get("tri_annuel", 0),
            "DSCR": s.get("dscr_y1", 0),
            "Cash-Flow (€)": s.get("cash_flow_final", 0),
            "Score": s.get("balanced_score", 0) * 100,
            "Taxonomy": s.get("taxonomy", "Autre")
        })
    
    df_chart = pd.DataFrame(data)
    
    fig = px.scatter(
        df_chart,
        x="DSCR",
        y="TRI (%)",
        size="Score", # Bubble size based on Score
        color="Cash-Flow (€)", # Color based on CF
        hover_name="Label",
        hover_data=["Cash-Flow (€)", "Score", "Taxonomy"],
        title="Matrice Risque (DSCR) / Rendement (TRI)",
        color_continuous_scale="RdYlGn",
        size_max=40
    )
    
    # Add quadrants lines if possible, or just layout
    fig.add_hline(y=5.0, line_dash="dash", line_color="gray", annotation_text="Objectif Rendement")
    fig.add_vline(x=1.3, line_dash="dash", line_color="gray", annotation_text="Objectif Banque")
    
    st.plotly_chart(fig, use_container_width=True, key=f"risk_reward_scatter_{key}")


def render_comparison_heatmap(strategies: list[dict[str, Any]], key: str = "heat") -> None:
    """Render heatmap of normalized scores for comparison.
    
    Args:
        strategies: List of strategies
        key: Unique key
    """
    if not strategies:
        return

    # Extract normalized metrics (0-1 scale)
    # We want to show Strengths & Weaknesses
    data = []
    labels = []
    
    for i, s in enumerate(strategies, 1):
        labels.append(f"Strat #{i}")
        # Ensure we handle missing keys gracefully with default 0
        row = [
            s.get("tri_norm", 0) * 100,
            s.get("cf_proximity", 0) * 100, # This is usually the score component for CF
            s.get("dscr_norm", 0) * 100,
            s.get("qual_score", 50),        # Already 0-100
            s.get("cap_eff_norm", 0) * 100,
            s.get("balanced_score", 0) * 100
        ]
        data.append(row)
        
    metrics = ["Rentabilité (TRI)", "Cash-Flow", "Sécurité (DSCR)", "Qualité", "Efficacité", "<b>GLOBAL</b>"]
    
    # Transpose if many strategies? Std is Strategies on Y, Metrics on X
    
    import plotly.graph_objects as go
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=metrics,
        y=labels,
        colorscale="RdYlGn",
        zmin=0, zmax=100,
        hoverongaps=False,
        texttemplate="%{z:.0f}",
        showscale=True
    ))
    
    fig.update_layout(
        title="Carte de Chaleur - Performance Relative (0-100)",
        xaxis_side="top",
        height=300 + (len(strategies) * 30), # Dynamic height
        margin=dict(t=100) # Space for x-axis labels
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"comp_heatmap_{key}")


def render_comparison_charts(
    strategies: list[dict[str, Any]],
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


def render_strategy_radar(strategy: dict[str, Any], key: str = "radar") -> None:
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
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
            },
        },
        showlegend=False,
        title="Profil de la Stratégie",
    )

    st.plotly_chart(fig, use_container_width=True, key=f"radar_{key}")
