"""Tab "Comparaison fiscale": thin rendering of services.regime_comparison."""

from __future__ import annotations

import json
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from src.application.run_context import RunContext
from src.ui.cached import cached_regime_comparison
from src.ui.helpers import format_euro, format_pct


def comparison_sentence(cmp: dict[str, Any]) -> str:
    best, user = cmp["best"], cmp["user"]
    if best["label"] == user["label"]:
        return f"Votre choix ({user['label']}) est déjà le régime le plus rentable."
    gain = f"{cmp['gain']:+,.0f}".replace(",", " ")
    text = f"Le régime {best['label']} rapporte {gain} € d'enrichissement vs votre choix ({user['label']})"
    year = cmp["break_even_year"]
    return text + (f"; rentable à partir de l'année {year}." if year else "; jamais devant en trésorerie cumulée.")


def render_regime_comparison(strategy: dict[str, Any], ctx: RunContext, key: str = "0") -> None:
    st.markdown("### ⚖️ Comparaison fiscale")
    st.caption("Même stratégie, mêmes hypothèses : seul le régime fiscal change.")
    cmp = cached_regime_comparison(json.dumps(strategy, sort_keys=True, default=str), ctx.model_dump_json())
    rows = cmp["rows"]
    st.info(comparison_sentence(cmp))
    table = [
        {
            "Régime": ("🏆 " if r["is_best"] else "") + r["label"] + (" (votre choix)" if r["is_user"] else ""),
            "Enrichissement net": format_euro(r["enrichissement_net"]),
            "TRI": format_pct(r["tri_annuel"]),
            "CF cumulé après impôt": format_euro(r["cumulative_cf"][-1] if r["cumulative_cf"] else 0.0),
            "Impôt cumulé": format_euro(sum(r["tax"])),
            "Impôt de sortie": format_euro(r["exit_tax"]),
        }
        for r in rows
    ]
    st.dataframe(table, hide_index=True, use_container_width=True)
    fig = go.Figure()
    for r in rows:
        fig.add_trace(
            go.Scatter(
                x=r["years"],
                y=r["cumulative_cf"],
                mode="lines",
                name=r["label"],
                line={
                    "width": 4 if r["is_best"] else 2,
                    "dash": "dot" if r["is_user"] and not r["is_best"] else "solid",
                },
            )
        )
    fig.update_layout(
        title="Cash-flow cumulé après impôt",
        xaxis_title="Année",
        yaxis_title="€",
        height=380,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    st.plotly_chart(fig, use_container_width=True, key=f"regime_cmp_{key}")
    for r in rows:
        for w in r["warnings"]:
            st.warning(f"{r['label']} : {w}")
