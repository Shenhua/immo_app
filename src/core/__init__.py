"""Core shared utilities (exceptions, logging, glossary).

Domain and application code must not be re-exported from here: importing
``src.core`` must not pull in ``src.application`` or ``src.domain``.
Import engines from their own modules instead.
"""

from .exceptions import (
    AppImmoError,
    DataLoadError,
    InvalidParameterError,
    NoStrategiesFoundError,
    SimulationError,
    StrategyError,
)

__all__ = [
    "AppImmoError",
    "DataLoadError",
    "SimulationError",
    "StrategyError",
    "NoStrategiesFoundError",
    "InvalidParameterError",
]
