"""Sensitivity analysis component (thin streamlit rendering; the computation is in services.sensitivity)."""

import json
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from src.application.run_context import RunContext
from src.application.services.sensitivity import Deltas, run_sensitivity, tornado_data
from src.ui.helpers import format_euro, format_pct


def run_sensitivity_simulation(
    strategy: dict[str, Any],
    delta_loyer_pct: float,
    delta_vacance_pct: float,
    delta_travaux_pct: float,
    ctx: RunContext,
    delta_taux_pts: float = 0.0,
    delta_appreciation_pts: float = 0.0,
) -> dict[str, Any]:
    """Run the simulation with modified parameters (thin wrapper over the pure function)."""
    deltas = Deltas(delta_loyer_pct, delta_vacance_pct, delta_travaux_pct, delta_taux_pts, delta_appreciation_pts)
    return run_sensitivity(strategy, deltas, ctx)


def _strategy_key(strategy: dict[str, Any]) -> str:
    return json.dumps(strategy, sort_keys=True, default=str)


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_run(strategy_json: str, ctx_json: str, deltas: tuple[float, ...]) -> dict[str, Any]:
    return run_sensitivity(json.loads(strategy_json), Deltas(*deltas), RunContext.model_validate_json(ctx_json))


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_tornado(strategy_json: str, ctx_json: str) -> list[dict[str, Any]]:
    return tornado_data(json.loads(strategy_json), RunContext.model_validate_json(ctx_json))


def _tornado_figure(rows: list[dict[str, Any]], prefix: str, title: str, unit: str) -> go.Figure:
    fig = go.Figure()
    labels = [r["driver"] for r in rows]
    for tag, name, color in (("down", "-1", "#d9534f"), ("up", "+1", "#2e7d32")):
        fig.add_bar(y=labels, x=[r[f"{prefix}_{tag}"] or 0.0 for r in rows], name=name, orientation="h", marker_color=color)
    fig.update_layout(barmode="relative", title=title, xaxis_title=unit, height=300, margin={"t": 40, "b": 30})
    return fig


def _show_impact(label: str, original: Any, new: Any, fmt_func: Any) -> None:
    cols = st.columns([0.4, 0.3, 0.3])
    cols[0].write(label)
    cols[1].write(fmt_func(original))
    if original is None or new is None:  # failed IRR: nothing to compare
        cols[2].write(fmt_func(new))
        return
    delta = new - original
    if abs(delta) > 0.01:
        cols[2].markdown(f"**:{'green' if delta >= 0 else 'red'}[{fmt_func(new)}]**")
    else:
        cols[2].write("—")


def render_sensitivity_analysis(strategy: dict[str, Any], ctx: RunContext, key: str = "0") -> None:
    """Render interactive sensitivity analysis.

    Args:
        strategy: Strategy dictionary
        ctx: Parameters of the run that produced the strategy
        key: Unique key for widget elements
    """
    st.markdown("### ⚡ Stress Test & Sensibilité")
    st.caption("Simulez des variations de marché pour tester la robustesse de cette stratégie.")

    col_input, col_res = st.columns([0.4, 0.6])

    with col_input:
        st.markdown("**Paramètres**")
        d_loyer = st.slider("Loyer", -30, 30, 0, 5, format="%+d%%", key=f"sens_rent_{key}")
        d_vacance = st.slider("Vacance (pts)", -5, 15, 0, 1, format="%+d pt", key=f"sens_vac_{key}")
        d_travaux = st.slider("Budget Travaux", -20, 50, 0, 10, format="%+d%%", key=f"sens_works_{key}")
        d_taux = st.slider("Taux de crédit (pts)", -2.0, 4.0, 0.0, 0.25, format="%+.2f pt", key=f"sens_rate_{key}")
        d_appr = st.slider("Appréciation (pts)", -3.0, 3.0, 0.0, 0.5, format="%+.1f pt", key=f"sens_appr_{key}")

    s_json, c_json = _strategy_key(strategy), ctx.model_dump_json()
    base = _cached_run(s_json, c_json, (0.0,) * 5)

    with col_res:
        if d_loyer or d_vacance or d_travaux or d_taux or d_appr:
            res = _cached_run(s_json, c_json, (float(d_loyer), float(d_vacance), float(d_travaux), float(d_taux), float(d_appr)))
            st.markdown("**Impact Projeté**")
            _show_impact("CF Mensuel (stabilisé)", base.get("cf_stab_mensuel"), res.get("cf_stab_mensuel"), format_euro)
            _show_impact("TRI", base.get("tri_annuel"), res.get("tri_annuel"), format_pct)
            _show_impact("Enrichissement", base.get("enrichissement_net"), res.get("enrichissement_net"), format_euro)
            _show_impact("DSCR (stabilisé)", base.get("dscr_stab"), res.get("dscr_stab"), lambda x: f"{x:.2f}")
        else:
            st.info("👈 Modifiez les curseurs pour voir l'impact.")

    with st.expander("Tornade : ±1 point sur chaque hypothèse"):
        rows = _cached_tornado(s_json, c_json)
        c1, c2 = st.columns(2)
        c1.plotly_chart(_tornado_figure(rows, "tri", "Impact sur le TRI", "pts de TRI"), use_container_width=True)
        c2.plotly_chart(_tornado_figure(rows, "cf", "Impact sur le CF mensuel stabilisé", "EUR/mois"), use_container_width=True)
