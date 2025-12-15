"""Session state management for Streamlit app.

Provides a centralized interface for managing Streamlit session state,
with type-safe accessors and default values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar

import streamlit as st

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
    return st.session_state[key]


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


@dataclass
class BaseParams:
    """Base simulation parameters from sidebar."""

    apport_disponible: float = 100000.0
    cash_flow_cible: float = -100.0
    tolerance: float = 50.0
    mode_cf: str = "target"
    qualite_weight: float = 0.5
    horizon_ans: int = 25

    # Credit
    taux_credits: dict[int, float] = field(default_factory=lambda: {15: 3.2, 20: 3.4, 25: 3.6})
    frais_notaire_pct: float = 7.5
    assurance_ann_pct: float = 0.35
    frais_pret_pct: float = 1.0

    # Costs inclusion
    inclure_travaux: bool = True
    inclure_reno_ener: bool = True
    inclure_mobilier: bool = True
    financer_mobilier: bool = True

    # IRA
    apply_ira: bool = True
    ira_cap_pct: float = 3.0

    # Fiscal
    frais_gestion_pct: float = 0.0
    provision_pct: float = 5.0
    cfe_par_bien_ann: float = 150.0
    regime_fiscal: str = "lmnp"
    tmi_pct: float = 30.0
    micro_bic_abatt_pct: float = 50.0
    frais_vente_pct: float = 6.0

    # Market
    appreciation: float = 2.0
    revalo: float = 1.5
    inflation: float = 2.0

    @classmethod
    def from_session_state(cls) -> BaseParams:
        """Create BaseParams from current session state."""
        return cls(
            apport_disponible=get_state("apport_disponible", 100000.0),
            cash_flow_cible=get_state("cash_flow_cible", -100.0),
            tolerance=get_state("tolerance", 50.0),
            mode_cf=get_state("mode_cf", "target"),
            qualite_weight=get_state("qualite_weight", 0.5),
            horizon_ans=get_state("horizon_ans", 25),
            taux_credits=get_state("taux_credits", {15: 3.2, 20: 3.4, 25: 3.6}),
            frais_notaire_pct=get_state("frais_notaire_pct", 7.5),
            assurance_ann_pct=get_state("assurance_ann_pct", 0.35),
            apply_ira=get_state("apply_ira", True),
            ira_cap_pct=get_state("ira_cap_pct", 3.0),
            regime_fiscal=get_state("regime_fiscal", "lmnp"),
            tmi_pct=get_state("tmi_pct", 30.0),
            appreciation=get_state("appreciation", 2.0),
            revalo=get_state("revalo", 1.5),
            inflation=get_state("inflation", 2.0),
        )

    def to_eval_params(self) -> dict[str, Any]:
        """Convert to eval_params dict for strategy finder."""
        return {
            "duree_simulation_ans": self.horizon_ans,
            "hypotheses_marche": {
                "appreciation_bien_pct": self.appreciation,
                "revalo_loyer_pct": self.revalo,
                "inflation_charges_pct": self.inflation,
            },
            "regime_fiscal": self.regime_fiscal,
            "tmi_pct": self.tmi_pct,
            "frais_vente_pct": self.frais_vente_pct,
            "cfe_par_bien_ann": self.cfe_par_bien_ann,
            "apply_ira": self.apply_ira,
            "ira_cap_pct": self.ira_cap_pct,
        }


class SessionManager:
    """Manages all session state for the app."""

    # Default state values
    DEFAULTS = {
        "strategies": [],
        "selected_strategy_idx": 0,
        "show_comparison": False,
        "analysis_run": False,
        "archetypes_source_label": "",
        "finance_preset": "Équilibré (défaut)",
        "finance_custom": None,
        "horizon_ans": 25,
        "appreciation": 2.0,
        "revalo": 1.5,
        "inflation": 2.0,
        "apply_ira": True,
        "ira_cap_pct": 3.0,
        "show_details": False, # New state for Details Panel visibility
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
        """Set selected strategy index."""
        set_state("selected_strategy_idx", idx)
        st.query_params["strategy"] = str(idx)

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
    def get_base_params(cls) -> BaseParams:
        """Get current base parameters."""
        return BaseParams.from_session_state()

    @classmethod
    def get_horizon(cls) -> int:
        """Get current horizon in years."""
        return get_state("horizon_ans", 25)

    @classmethod
    def set_horizon(cls, horizon: int) -> None:
        """Set current horizon."""
        set_state("horizon_ans", horizon)

    @classmethod
    def is_ira_enabled(cls) -> bool:
        """Check if IRA is enabled."""
        return get_state("apply_ira", True)
