"""Custom exceptions for app_immo.

Domain-specific exception types for better error handling and debugging.
"""

from __future__ import annotations

from typing import Any



class AppImmoError(Exception):
    """Base exception for all app_immo errors."""
    pass


# --- Data Errors ---

class DataLoadError(AppImmoError):
    """Failed to load or parse data files (e.g., archetypes JSON)."""
    pass


class ArchetypeError(AppImmoError):
    """Invalid or missing archetype data."""
    pass


# --- Calculation Errors ---

class SimulationError(AppImmoError):
    """Error during financial simulation."""
    pass


class AmortizationError(AppImmoError):
    """Error generating amortization schedule."""
    pass


class AllocationError(AppImmoError):
    """Error in capital allocation."""
    pass


# --- Strategy Errors ---

class StrategyError(AppImmoError):
    """General strategy-related error."""
    pass


class NoStrategiesFoundError(StrategyError):
    """No valid strategies match the given criteria."""
    pass


class InvalidParameterError(AppImmoError):
    """Invalid parameter value provided."""
    
    def __init__(self, param_name: str, value: Any, reason: str = ""):
        self.param_name = param_name
        self.value = value
        self.reason = reason
        msg = f"Invalid parameter '{param_name}': {value}"
        if reason:
            msg += f" - {reason}"
        super().__init__(msg)


# --- Configuration Errors ---

class ConfigurationError(AppImmoError):
    """Error in application configuration."""
    pass
