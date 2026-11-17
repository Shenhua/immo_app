"""Shared helpers for the end-to-end UI suite (T51): a fresh AppTest on app.py and small widget helpers.

Widgets are addressed by their stable ``key=`` (T65, see ``src/ui/components/sidebar_state.py``). Every test in
this directory gets the ``ui`` marker, so the suite runs alone with ``pytest -m ui`` and in the normal run too.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[2] / "app.py")
SEARCH_LABEL = "Lancer"
TIMEOUT = 120
_WIDGET_KINDS = ("number_input", "toggle", "checkbox", "radio", "selectbox", "multiselect", "slider")


def pytest_collection_modifyitems(items):
    for item in items:
        if "tests/ui/" in str(item.fspath).replace("\\", "/"):
            item.add_marker(pytest.mark.ui)


@pytest.fixture(autouse=True)
def _clean_cache():
    """Cached bricks / simulations are keyed on their inputs, but start every test cold and independent."""
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def new_app(query_params: dict[str, str] | None = None) -> AppTest:
    """A fresh app session, already run once (first load)."""
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    for k, v in (query_params or {}).items():
        at.query_params[k] = v
    at.run()
    return at


def search_button(at: AppTest):
    return next(b for b in at.sidebar.button if SEARCH_LABEL in b.label)


def run_search(at: AppTest) -> AppTest:
    """Click "Lancer l'analyse" and rerun."""
    search_button(at).click()
    at.run()
    return at


def set_inputs(at: AppTest, **values) -> AppTest:
    """Set widgets by key (numbers, toggles, checkboxes, radios, selectboxes, multiselects, sliders), rerun once."""
    for key, value in values.items():
        for kind in _WIDGET_KINDS:
            matches = [w for w in getattr(at, kind) if w.key == key]
            if matches:
                matches[0].set_value(value)
                break
        else:
            raise KeyError(f"no widget with key {key!r}")
    at.run()
    return at


def strategies(at: AppTest) -> list[dict]:
    return at.session_state["strategies"]


def open_details(at: AppTest, index: int = 1) -> AppTest:
    """Expand the detail view of the selected strategy (the card's "Voir le détail" button)."""
    at.button(key=f"btn_details_{index}").click()
    at.run()
    return at


def all_text(at: AppTest) -> str:
    """Every markdown / warning / error / info / success / caption text on the page, for substring checks."""
    out: list[str] = []
    for kind in ("markdown", "warning", "error", "info", "success", "caption"):
        out.extend(str(el.value) for el in getattr(at, kind))
    return "\n".join(out)


def page_warnings(at: AppTest) -> list[str]:
    return [str(w.value) for w in at.warning]
