"""Data models for app_immo."""

from .archetype import Archetype, ArchetypeV2
from .brick import InvestmentBrick
from .strategy import PortfolioStrategy, StrategyResult

__all__ = [
    "Archetype",
    "ArchetypeV2",
    "InvestmentBrick",
    "PortfolioStrategy",
    "StrategyResult",
]
