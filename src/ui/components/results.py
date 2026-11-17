"""Strategy display components for results rendering."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

from src.core.defaults import HCSF
from src.domain.dpe import property_ban_warning
from src.ui.helpers import format_euro, format_pct, get_taxonomy_badge


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

def get_hcsf_color(taux: float) -> str:
    """Debt ratio colour (T60): green <= 33 %, orange up to 35 %, red above."""
    if taux <= HCSF.taux_alerte_pct:
        return "#28a745"
    return "#fd7e14" if taux <= HCSF.taux_max_pct else "#dc3545"


def hcsf_badge(strategy: dict[str, Any]) -> str:
    """HTML for the debt ratio, or "" when the run did not compute it."""
    taux = strategy.get("taux_endettement")
    if taux is None:
        return ""
    flag = "" if strategy.get("hcsf_ok", True) else " · non finançable (HCSF)"
    return (f"<span title='Taux d\'endettement HCSF (max 35 %), assurance incluse' "
            f"style='color:{get_hcsf_color(float(taux))};font-weight:600'>Endettement {float(taux):.1f} %{flag}</span>")


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
    cf_cible: float = 0.0,
    is_selected: bool = False,
    show_details: bool = False,
    expanded_content: Callable[[], None] | None = None,
) -> bool:
    """Render a single strategy card.

    Args:
        strategy: Strategy dictionary
        index: Strategy index (1-based for display)
        horizon: Simulation horizon in years
        cf_cible: Cash-flow target of the run (RunContext), used to colour the cash-flow
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
        with col_badge:
            if strategy.get("pareto_rank") == 1:
                st.markdown(
                    "<span title='Non dominée sur patrimoine, TRI, DSCR, apport, qualité et cash-flow'"
                    " style='background:#e8f5e9;color:#1b5e20;border-radius:6px;padding:2px 8px;"
                    "font-size:0.8em;font-weight:600'>Pareto</span>",
                    unsafe_allow_html=True,
                )

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

            color_cf = get_cf_color(cf, cf_cible)

            st.markdown("<div style='font-size: 0.9em; color: gray;'>Cash-flow (année stabilisée)</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 2.5em; font-weight: bold; color: {color_cf};'>{format_euro(cf)}</div>", unsafe_allow_html=True)
            cf_y1 = strategy.get("cf_monthly_y1")
            if cf_y1 is not None:
                st.caption(f"Année 1 (avec travaux) : {format_euro(cf_y1)}/mois, après impôt")


        with c3:
            tri = strategy.get("tri_annuel", 0)
            st.markdown("<div style='font-size: 0.9em; color: gray;'>Rentabilité (TRI)</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 2.5em;'>{format_pct(tri)}</div>", unsafe_allow_html=True)

        with c4:
            enrich = strategy.get("liquidation_nette", 0)
            st.markdown("<div style='font-size: 0.9em; color: gray;'>Enrichissement</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 2.5em;'>{format_euro(enrich)}</div>", unsafe_allow_html=True)

        if badge := hcsf_badge(strategy):
            st.markdown(badge, unsafe_allow_html=True)

        for bien in strategy.get("details", []):
            if (warn := property_ban_warning(bien, horizon)):
                st.warning(f"{bien.get('nom_bien', 'Bien')} : {warn}", icon="⚠️")

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
            if (warn := property_ban_warning(bien, horizon)):
                st.warning(warn, icon="⚠️")
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
                st.write(f"Apport: {format_euro(bien.get('apport_final_bien', 0))}")
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

                bien.get("charges_non_recuperables_mensuel", 0) + bien.get("taxe_fonciere_mensuel", 0)
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
        ("CF Mensuel", strategy.get("cash_flow_final", 0), format_euro, "Cash-flow mensuel de l'année stabilisée (année 2, loyers de 12 mois), avant impôt sur le revenu (loyers moins charges et mensualités de crédit, tous biens)."),
        (f"TRI ({horizon}a)", strategy.get("tri_annuel", 0), format_pct, "Taux de Rentabilité Interne annualisé (inclut la revente)."),
        ("Enrichissement", strategy.get("liquidation_nette", 0), format_euro, f"Capital net disponible à la revente au bout de {horizon} ans (après impôts et remboursement dette)."),
        ("DSCR stabilisé", strategy.get("dscr_stab", strategy.get("dscr_y1", 0)), lambda x: f"{x:.2f}" if x else "—", "Ratio de couverture de la dette (Revenus / Charges crédit) de l'année stabilisée (année 2). >1.0 signifie autofinancement. Année 1 (avec travaux) : voir le détail."),
        ("Score Qualité", strategy.get("qual_score", 50), lambda x: f"{get_star_rating(x)}", "Qualité intrinsèque des biens (Emplacement, Tension, Liquidité)."),
        ("Score Global", strategy.get("balanced_score", 0) * 100, lambda x: f"{get_star_rating(x)}", "Note pondérée combinant performance financière et qualité."),
    ]

    for col, (label, value, formatter, help_text) in zip(cols, metrics):
        with col:
            # Special handling for Scores to show numeric value as secondary (delta)
            if label in ["Score Qualité", "Score Global"]:
                 st.metric(
                     label,
                     formatter(value),
                     delta=f"{value:.0f}/100",
                     delta_color="off",
                     help=help_text
                 )
            else:
                 st.metric(label, formatter(value), help=help_text)

    if badge := hcsf_badge(strategy):
        st.markdown(badge, unsafe_allow_html=True)

    cf_y1 = strategy.get("cf_monthly_y1")
    dscr_y1 = strategy.get("dscr_y1")
    if cf_y1 is not None or dscr_y1 is not None:
        parts = []
        if cf_y1 is not None:
            parts.append(f"CF année 1 (avec travaux) : {format_euro(cf_y1)}/mois après impôt")
        if dscr_y1 is not None:
            parts.append(f"DSCR année 1 (avec travaux) : {dscr_y1:.2f}")
        st.caption(" · ".join(parts) + ". Les travaux réduisent les loyers de la première année seulement.")
