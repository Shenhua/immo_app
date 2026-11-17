"""Search diagnostics display (T33): why nothing passed, relaxation suggestions, near misses.

The data comes from ``StrategyFinder.last_diagnostics``; ``app_controller.run_strategy_search`` stores it in
``st.session_state[DIAGNOSTICS_KEY]``. Each suggestion has an "Appliquer cette suggestion" button (T65): it patches the matching sidebar input through
``sidebar_state.request_patch`` and the next run applies it before the sidebar is drawn.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.application.run_context import RunContext
from src.application.services.diagnostics import REASON_LABELS, SearchDiagnostics, Suggestion
from src.ui.components.sidebar_state import request_patch

DIAGNOSTICS_KEY = "search_diagnostics"


def diagnostics_match(diag: SearchDiagnostics, ctx: RunContext) -> bool:
    """True when ``diag`` was produced by the search ``ctx`` describes (guards against stale session state)."""
    return (
        abs(diag.budget - ctx.apport) < 1e-6
        and abs(diag.target_cf - ctx.cf_cible) < 1e-6
        and abs(diag.tolerance - ctx.tolerance) < 1e-6
        and diag.max_props == ctx.max_properties
        and diag.mode_cf == ctx.mode_cf
    )


def no_result_lines(diag: SearchDiagnostics, top_n: int = 10, with_suggestions: bool = True) -> list[str]:
    """Markdown lines explaining an empty result: the top constraint, the counts, the suggestions."""
    top = diag.top_reason()
    if top is None:
        return ["Aucun portefeuille n'a pu être construit avec ces biens."]
    lines = [
        f"**Contrainte la plus éliminatoire : {REASON_LABELS[top]}** "
        f"({diag.counts[top]} portefeuille(s) sur {diag.total_rejected} écartés)."
    ]
    others = [f"{REASON_LABELS[r]} : {n}" for r, n in diag.counts.items() if n and r != top]
    if others:
        lines.append("Autres motifs : " + " ; ".join(others) + ".")
    sugg = diag.suggestions(top_n)
    if sugg and with_suggestions:
        lines.append("**Pour obtenir des résultats :**")
        lines.extend(f"- {s.text}" for s in sugg)
    return lines


def render_suggestions(suggestions: list[Suggestion], scope: str) -> None:
    """One row per suggestion: its text and a button that sets the matching sidebar value and reruns."""
    for i, sugg in enumerate(suggestions):
        text_col, button_col = st.columns([4, 1], vertical_alignment="center")
        text_col.markdown(f"- {sugg.text}")
        button_col.button(
            "Appliquer cette suggestion",
            key=f"apply_sugg_{scope}_{i}_{sugg.param}",
            on_click=request_patch,
            args=(sugg.param, sugg.suggested),
        )


def near_miss_lines(diag: SearchDiagnostics) -> list[str]:
    """One line per kept near miss, grouped by reason."""
    out = []
    for reason, items in diag.near_misses.items():
        if items:
            out.append(f"**{REASON_LABELS[reason].capitalize()}**")
            out.extend(f"- {n.describe()}" for n in items)
    return out


def render_search_diagnostics(strategies: list[dict[str, Any]], ctx: RunContext | None) -> None:
    """Explain an empty result, or list the near misses when fewer than ``top_n`` strategies passed."""
    diag = st.session_state.get(DIAGNOSTICS_KEY)
    if ctx is None or not isinstance(diag, SearchDiagnostics) or not diagnostics_match(diag, ctx):
        return
    if not strategies:
        st.warning("\n\n".join(no_result_lines(diag, ctx.top_n, with_suggestions=False)))
        sugg = diag.suggestions(ctx.top_n)
        if sugg:
            st.markdown("**Pour obtenir des résultats :**")
            render_suggestions(sugg, "empty")
        if diag.all_near_misses():
            with st.expander("Quasi-solutions"):
                st.markdown("\n".join(near_miss_lines(diag)))
        return
    near = diag.all_near_misses()
    if near and len(strategies) < ctx.top_n:
        with st.expander(f"{len(near)} quasi-solutions"):
            st.markdown("\n".join(near_miss_lines(diag)))
            sugg = diag.suggestions(ctx.top_n)
            render_suggestions(sugg, "near")
