"""Saved scenarios and shareable link (T65).

* ``apply_url_params_once``: on the first run of a session, apply ``?p=`` to the sidebar.
* ``sync_url``: keep ``?p=`` equal to the current sidebar, so the address bar is always a shareable link.
* ``render_scenario_panel``: "Copier le lien", JSON download / upload and the session list of scenarios.

The encoding lives in ``src/application/scenario_io.py`` (no streamlit). Nothing is persisted server side (T67).
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

import streamlit as st

from src.application.run_context import RunContext
from src.application.scenario_io import (
    ScenarioError,
    build_query_params,
    decode_context,
    encode_context,
    scenario_from_json,
    scenario_to_json,
)
from src.application.services.strategy_finder import strategy_key
from src.ui.components.sidebar_state import (
    SAVED_SCENARIOS_KEY,
    URL_APPLIED_KEY,
    apply_context_to_session,
    request_apply,
)

URL_ERROR_KEY = "_url_params_error"
LAST_UPLOAD_KEY = "_scenario_last_upload"
TOP_KEYS = 5


def apply_url_params_once() -> None:
    """Apply ``?p=`` to the sidebar the first time a session loads. Must run before the sidebar is drawn."""
    if st.session_state.get(URL_APPLIED_KEY):
        return
    st.session_state[URL_APPLIED_KEY] = True
    try:
        payload = st.query_params.get("p")
    except Exception:
        return
    if not payload:
        return
    try:
        apply_context_to_session(decode_context(payload))
    except ScenarioError as exc:
        st.session_state[URL_ERROR_KEY] = f"Le lien de scénario n'a pas pu être appliqué : {exc}"


def sync_url(ctx: RunContext) -> None:
    """Write the current sidebar to ``?p=`` (removed when everything is at its default)."""
    try:
        payload = encode_context(ctx)
        current = st.query_params.get("p")
        if payload and payload != current:
            st.query_params["p"] = payload
        elif not payload and current is not None:
            del st.query_params["p"]
    except Exception:
        pass  # the URL is a convenience, never a reason to fail


def share_link(ctx: RunContext, strategy_idx: int | None = None) -> str:
    """Link reproducing ``ctx`` (absolute when the server URL is known, else just the query string)."""
    query = "?" + urlencode(build_query_params(ctx, strategy_idx)) if build_query_params(ctx, strategy_idx) else ""
    try:
        base = str(st.context.url).split("?")[0]
    except Exception:
        base = ""
    return base + query if base else (query or "?")


def _keys_of(strategies: list[dict[str, Any]]) -> list[str]:
    keys = []
    for s in strategies[:TOP_KEYS]:
        try:
            keys.append("+".join(f"{name}x{qty}" for name, qty in strategy_key(s)))
        except Exception:
            continue
    return keys


def _remember(name: str, ctx: RunContext) -> None:
    saved: dict[str, dict[str, Any]] = st.session_state.setdefault(SAVED_SCENARIOS_KEY, {})
    saved[name] = ctx.model_dump(mode="json")


def _handle_upload() -> None:
    uploaded = st.file_uploader("Charger un scénario", type=["json"], key="sc_upload")
    if uploaded is None:
        return
    if st.session_state.get(LAST_UPLOAD_KEY) == uploaded.file_id:
        return  # already applied; the file stays in the widget until the user removes it
    st.session_state[LAST_UPLOAD_KEY] = uploaded.file_id
    try:
        name, ctx, _keys = scenario_from_json(uploaded.getvalue())
    except ScenarioError as exc:
        st.error(f"Scénario illisible : {exc}")
        return
    _remember(name, ctx)
    request_apply(ctx)
    st.rerun()


def render_scenario_panel(ctx: RunContext, strategies: list[dict[str, Any]], strategy_idx: int | None) -> None:
    """Sidebar expander: share link, save / load a scenario file, session scenarios."""
    error = st.session_state.pop(URL_ERROR_KEY, None)
    if error:
        st.sidebar.warning(error, icon="⚠️")
    with st.sidebar.expander("💾 Scénarios et partage", expanded=False):
        st.markdown("**Copier le lien**")
        st.code(share_link(ctx, strategy_idx), language=None)
        st.caption("Ce lien reproduit les paramètres de la barre latérale (seulement ceux modifiés).")

        name = st.text_input("Nom du scénario", key="sc_name", placeholder="Mon scénario")
        label = name.strip() or "Scénario"
        st.download_button(
            "Sauvegarder le scénario",
            data=scenario_to_json(label, ctx, _keys_of(strategies)),
            file_name=f"scenario_{''.join(c if c.isalnum() else '_' for c in label.lower())}.json",
            mime="application/json",
            on_click=_remember,
            args=(label, ctx),
        )
        _handle_upload()

        saved: dict[str, dict[str, Any]] = st.session_state.get(SAVED_SCENARIOS_KEY, {})
        if saved:
            choice = st.selectbox("Scénarios de la session", list(saved), key="sc_choice")
            st.button(
                "Appliquer ce scénario",
                key="sc_apply",
                on_click=lambda: request_apply(RunContext.model_validate(saved[st.session_state["sc_choice"]])),
            )
            st.caption(f"{len(saved)} scénario(s) gardé(s) dans cette session ({choice}).")
