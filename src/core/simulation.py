"""Long-term strategy simulation.

Stub module - full implementation coming in later refactoring.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd


def simulate_long_term_strategy(
    strategy: Dict[str, Any],
    duration_years: int = 25,
    market_hypotheses: Dict[str, float] | None = None,
    tax_params: Dict[str, Any] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Simulate strategy performance over time.
    
    This is a stub - delegates to the legacy implementation for now.
    Will be fully refactored in a future phase.
    
    Args:
        strategy: Strategy dictionary with details
        duration_years: Simulation horizon
        market_hypotheses: Market growth assumptions
        tax_params: Tax calculation parameters
        
    Returns:
        Tuple of (yearly DataFrame, summary bilan dict)
    """
    # Import legacy implementation
    from financial_calculations import simuler_strategie_long_terme as legacy_simulate
    
    # Merge params
    defaults = {
        "duree_simulation_ans": duration_years,
        "hypotheses_marche": market_hypotheses or {},
        **(tax_params or {}),
    }
    
    return legacy_simulate(strategy, **defaults)
