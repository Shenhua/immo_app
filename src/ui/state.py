"""Session state management for Streamlit app.

Provides a centralized interface for managing Streamlit session state,
with type-safe accessors and default values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar

import streamlit as st

from src.core.defaults import DEFAULT_PRESET, FINANCING, MARKET, OPERATING, SEARCH, TAX

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

    apport_disponible: float = SEARCH.apport
    cash_flow_cible: float = SEARCH.cf_cible
    tolerance: float = SEARCH.tolerance
    mode_cf: str = "target"
    qualite_weight: float = SEARCH.qualite_weight
    horizon_ans: int = SEARCH.horizon_ans

    # Credit
    taux_credits: dict[int, float] = field(default_factory=lambda: FINANCING.taux_credits)
    frais_notaire_pct: float = FINANCING.frais_notaire_pct
    assurance_ann_pct: float = FINANCING.assurance_ann_pct
    frais_pret_pct: float = FINANCING.frais_pret_pct

    # Costs inclusion
    inclure_travaux: bool = FINANCING.inclure_travaux
    inclure_reno_ener: bool = FINANCING.inclure_reno_ener
    inclure_mobilier: bool = FINANCING.inclure_mobilier
    financer_mobilier: bool = FINANCING.financer_mobilier

    # IRA
    apply_ira: bool = FINANCING.apply_ira
    ira_cap_pct: float = FINANCING.ira_cap_pct

    # Fiscal
    frais_gestion_pct: float = OPERATING.gestion_pct
    provision_pct: float = OPERATING.provision_pct
    cfe_par_bien_ann: float = OPERATING.cfe_par_bien_ann
    regime_fiscal: str = TAX.regime_fiscal
    tmi_pct: float = TAX.tmi_pct
    micro_bic_abatt_pct: float = TAX.micro_bic_abatt_pct
    frais_vente_pct: float = TAX.frais_vente_pct

    # Market
    appreciation: float = MARKET.appreciation_bien_pct
    revalo: float = MARKET.revalo_loyer_pct
    inflation: float = MARKET.inflation_charges_pct

    @classmethod
    def from_session_state(cls) -> BaseParams:
        """Create BaseParams from current session state."""
        return cls(
            apport_disponible=get_state("apport_disponible", SEARCH.apport),
            cash_flow_cible=get_state("cash_flow_cible", SEARCH.cf_cible),
            tolerance=get_state("tolerance", SEARCH.tolerance),
            mode_cf=get_state("mode_cf", "target"),
            qualite_weight=get_state("qualite_weight", SEARCH.qualite_weight),
            horizon_ans=get_state("horizon_ans", SEARCH.horizon_ans),
            taux_credits=get_state("taux_credits", FINANCING.taux_credits),
            frais_notaire_pct=get_state("frais_notaire_pct", FINANCING.frais_notaire_pct),
            assurance_ann_pct=get_state("assurance_ann_pct", FINANCING.assurance_ann_pct),
            apply_ira=get_state("apply_ira", FINANCING.apply_ira),
            ira_cap_pct=get_state("ira_cap_pct", FINANCING.ira_cap_pct),
            regime_fiscal=get_state("regime_fiscal", TAX.regime_fiscal),
            tmi_pct=get_state("tmi_pct", TAX.tmi_pct),
            appreciation=get_state("appreciation", MARKET.appreciation_bien_pct),
            revalo=get_state("revalo", MARKET.revalo_loyer_pct),
            inflation=get_state("inflation", MARKET.inflation_charges_pct),
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
        "finance_preset": DEFAULT_PRESET.label,
        "finance_custom": None,
        "horizon_ans": SEARCH.horizon_ans,
        "appreciation": MARKET.appreciation_bien_pct,
        "revalo": MARKET.revalo_loyer_pct,
        "inflation": MARKET.inflation_charges_pct,
        "apply_ira": FINANCING.apply_ira,
        "ira_cap_pct": FINANCING.ira_cap_pct,
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
        return get_state("horizon_ans", SEARCH.horizon_ans)

    @classmethod
    def set_horizon(cls, horizon: int) -> None:
        """Set current horizon."""
        set_state("horizon_ans", horizon)

    @classmethod
    def is_ira_enabled(cls) -> bool:
        """Check if IRA is enabled."""
        return get_state("apply_ira", FINANCING.apply_ira)
