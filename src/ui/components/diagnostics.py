"""Search diagnostics display (T33): why nothing passed, relaxation suggestions, near misses.

The data comes from ``StrategyFinder.last_diagnostics``; ``app_controller.run_strategy_search`` stores it in
``st.session_state[DIAGNOSTICS_KEY]``. Text only: the sidebar inputs have no widget keys, so a button that applies a
suggestion to them is not possible without touching the sidebar.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.application.run_context import RunContext
from src.application.services.diagnostics import REASON_LABELS, SearchDiagnostics

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


def no_result_lines(diag: SearchDiagnostics, top_n: int = 10) -> list[str]:
    """Markdown lines explaining an empty result: the top constraint, the counts, the suggestions."""
    top = diag.top_reason()
    if top is None:
        return ["Aucun portefeuille n'a pu être construit avec ces biens."]
    lines = [f"**Contrainte la plus éliminatoire : {REASON_LABELS[top]}** "
             f"({diag.counts[top]} portefeuille(s) sur {diag.total_rejected} écartés)."]
    others = [f"{REASON_LABELS[r]} : {n}" for r, n in diag.counts.items() if n and r != top]
    if others:
        lines.append("Autres motifs : " + " ; ".join(others) + ".")
    sugg = diag.suggestions(top_n)
    if sugg:
        lines.append("**Pour obtenir des résultats :**")
        lines.extend(f"- {s.text}" for s in sugg)
    return lines


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
        st.warning("\n\n".join(no_result_lines(diag, ctx.top_n)))
        if diag.all_near_misses():
            with st.expander("Quasi-solutions"):
                st.markdown("\n".join(near_miss_lines(diag)))
        return
    near = diag.all_near_misses()
    if near and len(strategies) < ctx.top_n:
        with st.expander(f"{len(near)} quasi-solutions"):
            st.markdown("\n".join(near_miss_lines(diag)))
            sugg = diag.suggestions(ctx.top_n)
            if sugg:
                st.markdown("\n".join(f"- {s.text}" for s in sugg))
