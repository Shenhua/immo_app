"""UI helper functions for Streamlit.

Common formatting and display utilities.
"""

from __future__ import annotations

import streamlit as st


def format_euro(value: float, decimals: int = 0) -> str:
    """Format a number as Euro currency.

    Args:
        value: Amount to format
        decimals: Number of decimal places

    Returns:
        Formatted string like "1 234 567 €"
    """
    if value is None:
        return "—"
    if decimals == 0:
        return f"{int(round(value)):,}".replace(",", " ") + " €"
    return f"{value:,.{decimals}f}".replace(",", " ") + " €"


def format_pct(value: float, decimals: int = 1) -> str:
    """Format a number as percentage.

    Args:
        value: Value to format (as percentage, not decimal)
        decimals: Number of decimal places

    Returns:
        Formatted string like "3.5 %"
    """
    if value is None:
        return "—"
    return f"{value:.{decimals}f} %"


def colorize_kpi(
    value: float,
    kpi_type: str,
    size: str = "normal",
) -> str:
    """Return colored HTML for a KPI value.

    Args:
        value: The KPI value
        kpi_type: Type of KPI (cf, dscr, tri, coc)
        size: Font size (normal, h2, h3)

    Returns:
        HTML string with appropriate coloring
    """
    if value is None:
        return "<span>—</span>"

    # Determine color based on KPI type and value
    color = "#333333"  # default

    if kpi_type == "cf":
        # Cash flow: green if positive, red if negative
        color = "#28a745" if value >= 0 else "#dc3545"
        formatted = format_euro(value)
    elif kpi_type == "dscr":
        # DSCR: green if > 1.3, yellow if > 1.0, red if < 1.0
        if value >= 1.3:
            color = "#28a745"
        elif value >= 1.0:
            color = "#ffc107"
        else:
            color = "#dc3545"
        formatted = f"{value:.2f}"
    elif kpi_type == "tri":
        # IRR: gradient based on value
        if value >= 8:
            color = "#28a745"
        elif value >= 5:
            color = "#17a2b8"
        else:
            color = "#6c757d"
        formatted = format_pct(value)
    elif kpi_type == "coc":
        # Cash-on-cash: similar to TRI
        if value >= 10:
            color = "#28a745"
        elif value >= 5:
            color = "#17a2b8"
        else:
            color = "#6c757d"
        formatted = format_pct(value)
    else:
        formatted = str(value)

    # Size styling
    if size == "h2":
        return f'<h2 style="color:{color};margin:0">{formatted}</h2>'
    elif size == "h3":
        return f'<h3 style="color:{color};margin:0">{formatted}</h3>'
    else:
        return f'<span style="color:{color};font-weight:600">{formatted}</span>'


def display_score_stars(score: float, help_text: str = "") -> None:
    """Display a score as stars (0-5).

    Accepts scores from 0-5 or 0-100 and auto-converts.

    Args:
        score: Score value
        help_text: Optional tooltip text
    """
    if score is None:
        score = 0

    # Auto-convert 0-100 to 0-5
    if score > 5:
        score = score / 20.0

    # Clamp to 0-5
    score = max(0, min(5, score))

    # Build star string
    full_stars = int(score)
    half_star = 1 if (score - full_stars) >= 0.5 else 0
    empty_stars = 5 - full_stars - half_star

    stars = "★" * full_stars + "☆" * (half_star + empty_stars)

    if help_text:
        st.caption(f"{stars} ({score:.1f}/5)", help=help_text)
    else:
        st.caption(f"{stars} ({score:.1f}/5)")


def get_taxonomy_badge(taxonomy: str) -> tuple[str, str, str]:
    """Get badge info for a strategy taxonomy.

    Args:
        taxonomy: Strategy type (Optimisé, Patrimonial, Mix)

    Returns:
        Tuple of (icon, label, tooltip)
    """
    taxonomy_info = {
        "Optimisé": ("🚀", "Optimisé", "Stratégie axée sur le rendement et le cash-flow"),
        "Patrimonial": ("🏛️", "Patrimonial", "Stratégie axée sur la qualité et la sécurité"),
        "Mix": ("⚖️", "Mix", "Stratégie équilibrée entre rendement et qualité"),
    }
    return taxonomy_info.get(taxonomy, ("🔀", taxonomy or "—", "Typologie non déterminée"))
