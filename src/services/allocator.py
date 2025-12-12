"""Portfolio allocation service.

Stub module for capital allocation across properties.
"""

from __future__ import annotations

from typing import Any, Dict, List


class PortfolioAllocator:
    """Service for allocating capital across properties.
    
    Stub class - will be expanded in future refactoring.
    """
    
    def __init__(self, budget: float):
        self.budget = budget
    
    def allocate(
        self,
        bricks: List[Dict[str, Any]],
        target_cf: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Allocate capital to maximize cash flow.
        
        Delegates to legacy implementation.
        """
        from allocators import allocate_apport_max_cf
        
        return allocate_apport_max_cf(bricks, self.budget, target_cf)
