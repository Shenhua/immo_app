"""Strategy finder service.

Stub module - delegates to legacy strategy_finder for now.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class StrategyFinder:
    """Service for finding optimal investment strategies.
    
    Stub class that wraps legacy strategy_finder functions.
    Will be fully refactored in a future phase.
    """
    
    def __init__(
        self,
        bricks: List[Dict[str, Any]],
        apport_disponible: float,
        cash_flow_cible: float,
        tolerance: float = 100.0,
        qualite_weight: float = 0.25,
    ):
        self.bricks = bricks
        self.apport_disponible = apport_disponible
        self.cash_flow_cible = cash_flow_cible
        self.tolerance = tolerance
        self.qualite_weight = qualite_weight
    
    def find_strategies(
        self,
        mode_cf: str = "≥",
        eval_params: Optional[Dict[str, Any]] = None,
        horizon_years: int = 25,
    ) -> List[Dict[str, Any]]:
        """Find top strategies matching criteria.
        
        Delegates to legacy implementation.
        """
        from strategy_finder import trouver_top_strategies
        
        return trouver_top_strategies(
            apport_disponible=self.apport_disponible,
            cash_flow_cible=self.cash_flow_cible,
            tolerance=self.tolerance,
            briques=self.bricks,
            mode_cf=mode_cf,
            qualite_weight=self.qualite_weight,
            eval_params=eval_params,
            horizon_years=horizon_years,
        )
