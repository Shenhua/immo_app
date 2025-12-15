"""Strategy display components for results rendering."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import streamlit as st

def get_cf_color(value: float, target: float) -> str:
    """Calculate color based on deviation from target."""
    diff = value - target
    if abs(diff) < 5: return "#ffffff"
    if diff > 0: return "#28a745"
    dist = abs(diff)
    if dist < 50: return "#ffc107"
    elif dist < 150: return "#fd7e14"
    else: return "#dc3545"

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
    show_details: bool = False,
    expanded_content: Optional[Callable[[], None]] = None,
) -> bool:
    """Render a single strategy card.
    
    Args:
        strategy: Strategy dictionary
        index: Strategy index (1-based for display)
        horizon: Simulation horizon in years
        is_selected: Whether this card is currently selected
        show_details: Whether details are currently shown
        expanded_content: Optional function to render content inside the card when expanded
    """
    taxonomy = strategy.get("taxonomy", "Mix")
    icon, label, color = get_taxonomy_badge(taxonomy)
    
    # Card styling - Minimalist container
    border_color = "#FFD700" if is_selected else None # Gold border if selected
    # Note: st.container(border=True) doesn't support custom color yet, 
    # so we use a visual indicator inside or markdown
    
    with st.container(border=True):        
        # Header with Feature Title and Badge
        col_header, col_badge = st.columns([0.7, 0.3])
        with col_header:
            if is_selected:
                 st.markdown(f"#### {icon} {taxonomy} <span style='color:#FFD700;font-size:0.8em'>● Sélectionné</span>", unsafe_allow_html=True)
            else:
                 st.markdown(f"#### {icon} {taxonomy}")
            st.caption(f"Stratégie #{index}")
            
        # Separator removed as requested
        
        # Key metrics row - Horizontal Layout (Score, CF, TRI, Enrich)
        # Using 4 cols to fit all on same horizontal line
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
             score = strategy.get("balanced_score", 0) * 100
             st.markdown(f"<div style='font-size: 0.9em; color: gray;'>Score</div>", unsafe_allow_html=True)
             st.markdown(f"<div style='font-size: 2.5em; font-weight: bold;'>{score:.0f}/100</div>", unsafe_allow_html=True)




# ... inside render_strategy_card ...

        with c2:
            cf = strategy.get("cash_flow_final", 0)
            
            # Retrieve target from session state (or default to -100 if missing)
            target = st.session_state.get("cf_cible", -100)
            color_cf = get_cf_color(cf, target)
            
            st.markdown(f"<div style='font-size: 0.9em; color: gray;'>Cash-flow</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 2.5em; font-weight: bold; color: {color_cf};'>{format_euro(cf)}</div>", unsafe_allow_html=True)

        
        with c3:
            tri = strategy.get("tri_annuel", 0)
            st.markdown(f"<div style='font-size: 0.9em; color: gray;'>Rentabilité (TRI)</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 2.5em;'>{format_pct(tri)}</div>", unsafe_allow_html=True)
        
        with c4:
            enrich = strategy.get("liquidation_nette", 0)
            st.markdown(f"<div style='font-size: 0.9em; color: gray;'>Enrichissement</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 2.5em;'>{format_euro(enrich)}</div>", unsafe_allow_html=True)

        # Footer Actions
        st.markdown("") # Spacer
        
        # Dynamic label based on state
        btn_label = "🔽 Voir le détail"
        if is_selected and show_details:
             btn_label = "🔼 Masquer"
             
        if st.button(btn_label, key=f"btn_details_{index}", use_container_width=True):
            clicked = True
        else:
            clicked = False
            
        # Render expanded content if active
        if show_details and expanded_content:
            st.markdown("---")
            expanded_content()
            
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
