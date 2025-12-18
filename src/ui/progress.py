"""
Progress tracking for strategy search.

Provides data structures and types for reporting search progress
from business logic to UI without coupling to Streamlit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable


class SearchPhase(Enum):
    """Phases of the strategy search process."""
    BRICK_GENERATION = "brick_generation"
    COMBO_GENERATION = "combo_generation"
    EVALUATION = "evaluation"
    SCORING = "scoring"
    DEDUPLICATION = "deduplication"


# French labels for each phase
PHASE_LABELS: dict[SearchPhase, str] = {
    SearchPhase.BRICK_GENERATION: "Génération des briques d'investissement",
    SearchPhase.COMBO_GENERATION: "Génération des combinaisons",
    SearchPhase.EVALUATION: "Évaluation des stratégies",
    SearchPhase.SCORING: "Scoring et classement",
    SearchPhase.DEDUPLICATION: "Déduplication et filtrage",
}

# List of phases in order
PHASES_ORDER: list[SearchPhase] = [
    SearchPhase.BRICK_GENERATION,
    SearchPhase.COMBO_GENERATION,
    SearchPhase.EVALUATION,
    SearchPhase.SCORING,
    SearchPhase.DEDUPLICATION,
]


@dataclass
class SearchProgress:
    """
    Tracks progress of strategy search for UI display.
    
    Attributes:
        phase: Current phase of the search
        phase_index: 1-based index of current phase (1-5)
        items_processed: Number of items processed in current phase
        items_total: Total items to process in current phase
        valid_count: Number of valid strategies found so far
        message: Optional message to display
    """
    phase: SearchPhase
    phase_index: int = 1
    total_phases: int = 5
    items_processed: int = 0
    items_total: int = 0
    valid_count: int = 0
    message: str = ""
    
    @property
    def percentage(self) -> float:
        """Calculate completion percentage (0-100)."""
        if self.items_total == 0:
            return 0.0
        return min(100.0, (self.items_processed / self.items_total) * 100)
    
    @property
    def label(self) -> str:
        """Get French label for current phase."""
        return PHASE_LABELS.get(self.phase, str(self.phase.value))
    
    @property
    def is_complete(self) -> bool:
        """Check if current phase is complete."""
        return self.items_total > 0 and self.items_processed >= self.items_total


@dataclass
class SearchStats:
    """
    Persistent search statistics for display after search completes.
    
    Stored in session state and shown in collapsible zone.
    """
    timestamp: str = ""
    duration_seconds: float = 0.0
    bricks_count: int = 0
    combos_generated: int = 0
    combos_evaluated: int = 0
    valid_strategies: int = 0
    strategies_after_dedupe: int = 0
    mode: str = "EXHAUSTIVE"
    max_properties: int = 3
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%H:%M:%S")


# Type alias for progress callbacks
ProgressCallback = Callable[[SearchProgress], None]
