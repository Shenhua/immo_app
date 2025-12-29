"""Core financial and scoring engines."""

from .exceptions import (
    AppImmoError,
    DataLoadError,
    InvalidParameterError,
    NoStrategiesFoundError,
    SimulationError,
    StrategyError,
)
from src.domain.calculator.financial import (
    calculate_insurance,
    calculate_monthly_payment,
    generate_amortization_schedule,
)
from src.domain.calculator.scoring import (
    calculate_balanced_score,
    calculate_dpe_score,
    calculate_qualitative_score,
)
from src.application.services.simulation import simulate_long_term_strategy

__all__ = [
    "calculate_monthly_payment",
    "calculate_insurance",
    "generate_amortization_schedule",
    "simulate_long_term_strategy",
    "calculate_dpe_score",
    "calculate_qualitative_score",
    "calculate_balanced_score",
    # Exceptions
    "AppImmoError",
    "DataLoadError",
    "SimulationError",
    "StrategyError",
    "NoStrategiesFoundError",
    "InvalidParameterError",
]
