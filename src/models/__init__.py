"""Data models for app_immo."""

from .archetype import Archetype, ArchetypeV2
from .brick import InvestmentBrick
from .strategy import Strategy, StrategyResult

__all__ = [
    "Archetype",
    "ArchetypeV2",
    "InvestmentBrick",
    "Strategy",
    "StrategyResult",
]
