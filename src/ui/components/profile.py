"""Who is using the app (T67a): the user_id that scopes saved scenarios.

* Now (no auth): a local profile chosen in the sidebar, kept in ``st.session_state``; default "local".
* Later: when Streamlit OIDC is configured and the visitor is logged in, ``st.user.email`` is the user_id and the
  selector is hidden. ``oidc_identity`` is defensive: with no auth configured it simply returns None.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

import streamlit as st

from src.application.scenario_store import DEFAULT_USER_ID, ScenarioStore, default_store

PROFILE_KEY = "profile_id"
STORE_ERROR = "Le stockage des scénarios est indisponible sur ce serveur."


def oidc_identity() -> tuple[str, str] | None:
    """``(user_id, display_name)`` of the logged-in OIDC user, else None. Never raises."""
    try:
        user: Any = st.user
        if not getattr(user, "is_logged_in", False):
            return None
        email = str(getattr(user, "email", "") or "").strip().lower()
        if not email:
            return None
        return email, str(getattr(user, "name", "") or email)
    except Exception:
        return None


def current_user_id() -> str:
    identity = oidc_identity()
    if identity:
        return identity[0]
    return str(st.session_state.get(PROFILE_KEY) or DEFAULT_USER_ID)


def get_store() -> ScenarioStore | None:
    """The scenario store, or None when the data directory cannot be used (read-only disk, bad migration...)."""
    try:
        return default_store()
    except (OSError, sqlite3.Error, ValueError):
        return None


def profile_id_from_name(display_name: str) -> str:
    """Stable id for a typed profile name: lower case, letters / digits / ``-`` / ``_``."""
    slug = re.sub(r"[^a-z0-9_-]+", "-", display_name.strip().lower()).strip("-")
    return slug[:40]
