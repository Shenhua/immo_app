"""Strategy display components for results rendering."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st


def get_cf_color(value: float, target: float) -> str:
    """Calculate color based on deviation from target."""
    diff = value - target
    if abs(diff) < 5:
        return "inherit"
    if diff > 0:
        return "#28a745"
    dist = abs(diff)
    if dist < 50:
        return "#ffc107"
    elif dist < 150:
        return "#fd7e14"
    else:
        return "#dc3545"

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


def get_star_rating(score: float, max_stars: int = 5) -> str:
    """Convert 0-100 score to star rating string."""
    if score is None: 
        return "—"
    
    # Calculate filled stars
    normalized = (score / 100) * max_stars
    filled = int(round(normalized))
    
    # Build string
    stars = "★" * filled + "☆" * (max_stars - filled)
    return stars


def render_strategy_card(
    strategy: dict[str, Any],
    index: int,
    horizon: int = 25,
    is_selected: bool = False,
    show_details: bool = False,
    expanded_content: Callable[[], None] | None = None,
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
             stars = get_star_rating(score)
             st.markdown("<div style='font-size: 0.9em; color: gray;'>Score</div>", unsafe_allow_html=True)
             st.markdown(f"<div style='font-size: 2.5em; font-weight: bold; line-height: 1.0;'>{stars}<div style='font-size:0.4em; color:gray; font-weight:normal; margin-top:5px'>({score:.0f}/100)</div></div>", unsafe_allow_html=True)




# ... inside render_strategy_card ...

        with c2:
            cf = strategy.get("cash_flow_final", 0)

            # Retrieve target from session state (or default to -100 if missing)
            target = st.session_state.get("cf_cible", -100)
            color_cf = get_cf_color(cf, target)

            st.markdown("<div style='font-size: 0.9em; color: gray;'>Cash-flow</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 2.5em; font-weight: bold; color: {color_cf};'>{format_euro(cf)}</div>", unsafe_allow_html=True)


        with c3:
            tri = strategy.get("tri_annuel", 0)
            st.markdown("<div style='font-size: 0.9em; color: gray;'>Rentabilité (TRI)</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 2.5em;'>{format_pct(tri)}</div>", unsafe_allow_html=True)

        with c4:
            enrich = strategy.get("liquidation_nette", 0)
            st.markdown("<div style='font-size: 0.9em; color: gray;'>Enrichissement</div>", unsafe_allow_html=True)
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


def render_strategy_details(strategy: dict[str, Any], horizon: int = 25) -> None:
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
                # Restore Property Score Display with Stars
                qs = bien.get('qual_score_bien', 50)
                stars = get_star_rating(qs, 5)
                st.markdown(f"**Qualité**: {stars} <span style='color:grey;font-size:0.8em'>({qs:.0f}/100)</span>", unsafe_allow_html=True)
                
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
                st.markdown("**Flux Mensuels (Moy)**")
                st.metric(
                    label="Loyer brut",
                    value=f"{format_euro(bien.get('loyer_mensuel_initial', 0))}",
                    delta="+ Revenu",
                    delta_color="normal"
                )
                
                charges_tot = bien.get("charges_non_recuperables_mensuel", 0) + bien.get("taxe_fonciere_mensuel", 0)
                st.metric(
                    label="Mensualité Crédit",
                    value=f"{format_euro(bien.get('pmt_total', 0))}",
                    delta="- Crédit",
                    delta_color="inverse"
                )
 
            # Qualitative Factors Restoration
            if "facteurs_qualitatifs" in bien:
                with st.expander("🔍 Voir le détail du score qualitatif"):
                    factors = bien["facteurs_qualitatifs"]
                    # If factors is dict
                    if isinstance(factors, dict):
                         for k, v in factors.items():
                             st.write(f"- **{k.replace('_', ' ').title()}**: {v}")
                    else:
                         st.write("Détails disponibles.")


def render_kpi_summary(strategy: dict[str, Any], horizon: int = 25) -> None:
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
        ("Score Qualité", strategy.get("qual_score", 50), lambda x: f"{get_star_rating(x)}"), 
        ("Score Global", strategy.get("balanced_score", 0) * 100, lambda x: f"{get_star_rating(x)}"), # Star Rating for Global too
    ]

    for col, (label, value, formatter) in zip(cols, metrics):
        with col:
            # Special handling for Scores to show numeric value as secondary (delta)
            if label in ["Score Qualité", "Score Global"]:
                 st.metric(
                     label, 
                     formatter(value),
                     delta=f"{value:.0f}/100",
                     delta_color="off"
                 )
            else:
                 st.metric(label, formatter(value))
