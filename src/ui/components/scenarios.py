"""Saved scenarios and shareable link (T65).

* ``apply_url_params_once``: on the first run of a session, apply ``?p=`` to the sidebar.
* ``sync_url``: keep ``?p=`` equal to the current sidebar, so the address bar is always a shareable link.
* ``render_scenario_panel``: "Copier le lien", JSON download / upload and the saved scenarios of the current user
  (save, load, rename, delete), stored by ``src/application/scenario_store.py`` (T67a).
* The user's last used context is stored too and restored on a fresh session without ``?p=`` (``?p=`` wins).

The encoding lives in ``src/application/scenario_io.py`` (no streamlit).
"""

from __future__ import annotations

from datetime import datetime
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
from src.application.scenario_store import ScenarioStoreError
from src.application.services.strategy_finder import strategy_key
from src.ui.components.profile import STORE_ERROR, current_user_id, get_store
from src.ui.components.sidebar_state import (
    URL_APPLIED_KEY,
    apply_context_to_session,
    request_apply,
)

URL_ERROR_KEY = "_url_params_error"
LAST_UPLOAD_KEY = "_scenario_last_upload"
LAST_SAVED_KEY = "_last_context_saved"
FLASH_KEY = "_scenario_flash"
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
        _restore_last_context()
        return
    try:
        apply_context_to_session(decode_context(payload))
    except ScenarioError as exc:
        st.session_state[URL_ERROR_KEY] = f"Le lien de scénario n'a pas pu être appliqué : {exc}"


def _restore_last_context() -> None:
    store = get_store()
    last = store.load_last_context(current_user_id()) if store else None
    if last is not None:
        apply_context_to_session(last)
        st.session_state[LAST_SAVED_KEY] = last.model_dump_json()


def _remember_last_context(ctx: RunContext) -> None:
    """Keep the user's last used context (only written when it changed)."""
    dumped = ctx.model_dump_json()
    if st.session_state.get(LAST_SAVED_KEY) == dumped:
        return
    store = get_store()
    if store is not None:
        store.save_last_context(current_user_id(), ctx)
        st.session_state[LAST_SAVED_KEY] = dumped


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
    try:
        _remember_last_context(ctx)
    except Exception:
        pass  # so is the last context


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


def _flash(kind: str, message: str) -> None:
    st.session_state[FLASH_KEY] = (kind, message)


def _save_scenario(name: str, ctx: RunContext, strategies: list[dict[str, Any]]) -> None:
    store = get_store()
    if store is None:
        _flash("error", STORE_ERROR)
        return
    try:
        store.save(current_user_id(), name, ctx, _keys_of(strategies))
        _flash("success", f"Scénario « {name} » enregistré.")
    except ScenarioStoreError as exc:
        _flash("error", str(exc))


def _load_scenario(scenario_id: int) -> None:
    store = get_store()
    if store is None:
        _flash("error", STORE_ERROR)
        return
    try:
        summary, ctx = store.load(current_user_id(), scenario_id)
    except (ScenarioStoreError, ScenarioError) as exc:
        _flash("error", f"Scénario illisible : {exc}")
        return
    request_apply(ctx)
    _flash("success", f"Scénario « {summary.name} » chargé.")


def _delete_scenario(scenario_id: int) -> None:
    store = get_store()
    if store is not None and store.delete(current_user_id(), scenario_id):
        _flash("success", "Scénario supprimé.")


def _rename_scenario(scenario_id: int, new_name: str) -> None:
    store = get_store()
    if store is None:
        _flash("error", STORE_ERROR)
        return
    try:
        store.rename(current_user_id(), scenario_id, new_name)
        _flash("success", "Scénario renommé.")
    except ScenarioStoreError as exc:
        _flash("error", str(exc))


def _handle_upload() -> None:
    uploaded = st.file_uploader("Charger un fichier de scénario", type=["json"], key="sc_upload")
    if uploaded is None:
        return
    if st.session_state.get(LAST_UPLOAD_KEY) == uploaded.file_id:
        return  # already applied; the file stays in the widget until the user removes it
    st.session_state[LAST_UPLOAD_KEY] = uploaded.file_id
    try:
        name, ctx, keys = scenario_from_json(uploaded.getvalue())
    except ScenarioError as exc:
        st.error(f"Scénario illisible : {exc}")
        return
    store = get_store()
    if store is not None:
        try:
            store.save(current_user_id(), name, ctx, keys)
        except ScenarioStoreError:
            pass  # the file is still applied
    request_apply(ctx)
    st.rerun()


def _format_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%d/%m/%Y %H:%M")
    except ValueError:
        return iso


def render_scenario_panel(ctx: RunContext, strategies: list[dict[str, Any]], strategy_idx: int | None) -> None:
    """Sidebar expander: share link, JSON file, and the current user's saved scenarios."""
    error = st.session_state.pop(URL_ERROR_KEY, None)
    if error:
        st.sidebar.warning(error, icon="⚠️")
    with st.sidebar.expander("💾 Scénarios et partage", expanded=False):
        flash = st.session_state.pop(FLASH_KEY, None)
        if flash:
            getattr(st, flash[0])(flash[1])

        st.markdown("**Copier le lien**")
        st.code(share_link(ctx, strategy_idx), language=None)
        st.caption("Ce lien reproduit les paramètres de la barre latérale (seulement ceux modifiés).")

        name = st.text_input("Nom du scénario", key="sc_name", placeholder="Mon scénario")
        label = name.strip() or "Scénario"
        st.button("Enregistrer dans mon profil", key="sc_save", on_click=_save_scenario, args=(label, ctx, strategies))
        st.download_button(
            "Télécharger le fichier JSON",
            data=scenario_to_json(label, ctx, _keys_of(strategies)),
            file_name=f"scenario_{''.join(c if c.isalnum() else '_' for c in label.lower())}.json",
            mime="application/json",
        )
        _handle_upload()

        store = get_store()
        if store is None:
            st.caption(STORE_ERROR)
            return
        saved = store.list_scenarios(current_user_id())
        if not saved:
            st.caption("Aucun scénario enregistré pour ce profil.")
            return
        by_id = {s.id: s for s in saved}
        choice = st.selectbox(
            "Mes scénarios",
            list(by_id),
            key="sc_choice",
            format_func=lambda i: f"{by_id[i].name} ({_format_date(by_id[i].updated_at)})",
        )
        if choice is None or choice not in by_id:
            return
        c1, c2 = st.columns(2)
        c1.button("Charger", key="sc_load", on_click=_load_scenario, args=(choice,))
        c2.button("Supprimer", key="sc_delete", on_click=_delete_scenario, args=(choice,))
        new_name = st.text_input("Nouveau nom", key="sc_rename_input", placeholder=by_id[choice].name)
        st.button(
            "Renommer",
            key="sc_rename",
            on_click=_rename_scenario,
            args=(choice, new_name),
            disabled=not new_name.strip(),
        )
        st.caption(f"{len(saved)} scénario(s) enregistré(s) ; créé le {_format_date(by_id[choice].created_at)}.")
