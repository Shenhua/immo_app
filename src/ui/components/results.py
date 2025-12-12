"""Strategy display components for results rendering."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import streamlit as st


def format_euro(value: float, decimals: int = 0) -> str:
    """Format a number as Euro currency."""
    if value is None:
        return "—"
    if decimals == 0:
        return f"{int(round(value)):,}".replace(",", " ") + " €"
    return f"{value:,.{decimals}f}".replace(",", " ") + " €"


def format_pct(value: float, decimals: int = 1) -> str:
    """Format a number as percentage."""
    if value is None:
        return "—"
    return f"{value:.{decimals}f} %"


def get_taxonomy_badge(taxonomy: str) -> tuple:
    """Get badge info for strategy taxonomy."""
    badges = {
        "Optimisé": ("🚀", "Optimisé", "#28a745"),
        "Patrimonial": ("🏛️", "Patrimonial", "#17a2b8"),
        "Mix": ("⚖️", "Mix", "#ffc107"),
    }
    return badges.get(taxonomy, ("🔀", taxonomy or "—", "#6c757d"))


def render_strategy_card(
    strategy: Dict[str, Any],
    index: int,
    horizon: int = 25,
    is_selected: bool = False,
) -> None:
    """Render a single strategy card.
    
    Args:
        strategy: Strategy dictionary
        index: Strategy index (1-based for display)
        horizon: Simulation horizon in years
        is_selected: Whether this card is currently selected
    """
    taxonomy = strategy.get("taxonomy", "Mix")
    icon, label, color = get_taxonomy_badge(taxonomy)
    
    # Card styling
    border = "3px solid #4CAF50" if is_selected else "1px solid #ddd"
    
    with st.container():
        st.markdown(
            f"""
            <div style="
                border: {border};
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 10px;
                background: linear-gradient(135deg, #f8f9fa, #ffffff);
            ">
            """,
            unsafe_allow_html=True,
        )
        
        # Header with badge and button
        col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
        with col1:
            st.markdown(f"### {icon} Stratégie #{index}")
        with col2:
            st.markdown(
                f'<div style="text-align:right"><span style="background:{color};color:white;padding:4px 8px;'
                f'border-radius:4px;font-size:0.9em">{label}</span></div>',
                unsafe_allow_html=True,
            )
        with col3:
            # Button with key based on index
            clicked = st.button("📊 Détails", key=f"btn_details_{index}", use_container_width=True)
        
        # Key metrics
        cols = st.columns(4)
        with cols[0]:
            cf = strategy.get("cash_flow_final", 0)
            cf_color = "#28a745" if cf >= 0 else "#dc3545"
            st.metric("CF Mensuel", format_euro(cf), delta_color="off")
        
        with cols[1]:
            tri = strategy.get("tri_annuel", 0)
            st.metric(f"TRI ({horizon}a)", format_pct(tri))
        
        with cols[2]:
            patrimoine = strategy.get("patrimoine_acquis", 0)
            st.metric("Patrimoine", format_euro(patrimoine))
        
        with cols[3]:
            score = strategy.get("balanced_score", 0) * 100
            st.metric("Score", f"{score:.0f}/100")
        
        # Property count
        details = strategy.get("details", [])
        st.caption(f"{len(details)} bien(s) • Apport: {format_euro(strategy.get('apport_total', 0))}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        return clicked


def render_strategy_details(strategy: Dict[str, Any], horizon: int = 25) -> None:
    """Render detailed view of a strategy.
    
    Args:
        strategy: Strategy dictionary
        horizon: Simulation horizon
    """
    details = strategy.get("details", [])
    
    st.markdown("### 🏠 Détails des biens")
    
    for i, bien in enumerate(details, 1):
        with st.expander(f"**{bien.get('nom_bien', f'Bien {i}')}** - {bien.get('ville', '?')}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Acquisition**")
                st.write(f"Prix: {format_euro(bien.get('prix_achat_bien', 0))}")
                st.write(f"Frais notaire: {format_euro(bien.get('frais_notaire', 0))}")
                st.write(f"Travaux: {format_euro(bien.get('budget_travaux', 0))}")
                st.write(f"**Coût total: {format_euro(bien.get('cout_total', 0))}**")
            
            with col2:
                st.markdown("**Financement**")
                st.write(f"Apport: {format_euro(bien.get('apport_min', 0))}")
                st.write(f"Crédit: {format_euro(bien.get('credit_final', 0))}")
                st.write(f"Durée: {bien.get('duree_pret', 0)} ans")
                st.write(f"Taux: {bien.get('taux_pret', 0):.2f}%")
            
            with col3:
                st.markdown("**Revenus**")
                st.write(f"Loyer: {format_euro(bien.get('loyer_mensuel_initial', 0))}/mois")
                st.write(f"Surface: {bien.get('surface', 0):.0f} m²")
                st.write(f"DPE: {bien.get('dpe_initial', '?')}")
                qs = bien.get('qual_score_bien', 50)
                st.write(f"Score qualité: {qs:.0f}/100")


def render_kpi_summary(strategy: Dict[str, Any], horizon: int = 25) -> None:
    """Render KPI summary row.
    
    Args:
        strategy: Strategy dictionary
        horizon: Simulation horizon
    """
    cols = st.columns(6)
    
    metrics = [
        ("CF Mensuel", strategy.get("cash_flow_final", 0), format_euro),
        (f"TRI ({horizon}a)", strategy.get("tri_annuel", 0), format_pct),
        ("Enrichissement", strategy.get("liquidation_nette", 0), format_euro),
        ("DSCR Y1", strategy.get("dscr_y1", 0), lambda x: f"{x:.2f}" if x else "—"),
        ("Score Qualité", strategy.get("qual_score", 50), lambda x: f"{x:.0f}/100"),
        ("Score Global", strategy.get("balanced_score", 0) * 100, lambda x: f"{x:.0f}/100"),
    ]
    
    for col, (label, value, formatter) in zip(cols, metrics):
        with col:
            st.metric(label, formatter(value))
