"""Session state management for Streamlit app.

Provides a centralized interface for managing Streamlit session state,
with type-safe accessors and default values.
"""

from __future__ import annotations

from typing import Any, TypeVar, cast

import streamlit as st

from src.application.run_context import RunContext
from src.core.defaults import DEFAULT_PRESET, FINANCING, MARKET, SEARCH

T = TypeVar("T")


def get_state(key: str, default: T) -> T:
    """Get a value from session state with a default.

    Args:
        key: Session state key
        default: Default value if key not present

    Returns:
        Value from session state or default
    """
    if key not in st.session_state:
        st.session_state[key] = default
    return cast(T, st.session_state[key])


def set_state(key: str, value: Any) -> None:
    """Set a value in session state.

    Args:
        key: Session state key
        value: Value to set
    """
    st.session_state[key] = value


def init_state(defaults: dict[str, Any]) -> None:
    """Initialize multiple session state values with defaults.

    Only sets values that don't already exist.

    Args:
        defaults: Dictionary of key-value defaults
    """
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


class SessionManager:
    """Manages all session state for the app."""

    # Default state values
    DEFAULTS: dict[str, Any] = {
        "strategies": [],
        "run_context": None,
        "selected_strategy_idx": 0,
        "show_comparison": False,
        "analysis_run": False,
        "archetypes_source_label": "",
        "finance_preset": DEFAULT_PRESET.label,
        "finance_custom": None,
        "horizon_ans": SEARCH.horizon_ans,
        "appreciation": MARKET.appreciation_bien_pct,
        "revalo": MARKET.revalo_loyer_pct,
        "inflation": MARKET.inflation_charges_pct,
        "apply_ira": FINANCING.apply_ira,
        "ira_cap_pct": FINANCING.ira_cap_pct,
        "show_details": False,  # New state for Details Panel visibility
    }

    @classmethod
    def initialize(cls) -> None:
        """Initialize all session state with defaults."""
        init_state(cls.DEFAULTS)

        # Hydrate from URL if present
        try:
            params = st.query_params
            if "strategy" in params:
                idx = int(params["strategy"])
                set_state("selected_strategy_idx", idx)
        except Exception:
            pass

    @classmethod
    def set_selected_idx(cls, idx: int) -> None:
        """Select a strategy and write its index to the URL (`?strategy=N`) so the link can be shared."""
        set_state("selected_strategy_idx", idx)
        st.query_params["strategy"] = str(idx)

    @classmethod
    def reset_selection(cls) -> None:
        """Clear the selected strategy and the details panel. Called when a new search runs."""
        set_state("selected_strategy_idx", 0)
        set_state("show_details", False)
        try:
            st.query_params.pop("strategy", None)
        except Exception:
            pass

    @classmethod
    def get_strategies(cls) -> list[dict[str, Any]]:
        """Get current strategies list."""
        return get_state("strategies", [])

    @classmethod
    def get_archetypes(cls) -> list[dict[str, Any]]:
        """Get loaded archetypes."""
        return get_state("archetypes", [])

    @classmethod
    def set_archetypes(cls, archetypes: list[dict[str, Any]]) -> None:
        """Set archetypes data."""
        set_state("archetypes", archetypes)

    @classmethod
    def set_strategies(cls, strategies: list[dict[str, Any]]) -> None:
        """Set strategies list."""
        set_state("strategies", strategies)
        set_state("analysis_run", True)

    @classmethod
    def get_selected_strategy(cls) -> dict[str, Any] | None:
        """Get currently selected strategy."""
        strategies = cls.get_strategies()
        idx = get_state("selected_strategy_idx", 0)
        if strategies and 0 <= idx < len(strategies):
            return strategies[idx]
        return None

    @classmethod
    def get_selected_idx(cls) -> int:
        """Get selected strategy index."""
        return get_state("selected_strategy_idx", 0)

    @classmethod
    def get_run_context(cls) -> RunContext | None:
        """RunContext of the last analysis (None before the first run)."""
        return st.session_state.get("run_context")

    @classmethod
    def set_run_context(cls, ctx: RunContext | None) -> None:
        """Store the RunContext next to the results it produced."""
        st.session_state["run_context"] = ctx
