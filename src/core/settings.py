"""Application settings with environment variable support.

Uses pydantic-settings for typed configuration validation.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Feature flags
    debug_mode: bool = Field(default=False, description="Enable debug features")
    enable_export: bool = Field(default=True, description="Enable result export")
    
    # Defaults for simulation
    default_horizon_years: int = Field(default=25, ge=1, le=50)
    default_tmi_pct: float = Field(default=30.0, ge=0, le=45)
    max_properties_per_strategy: int = Field(default=3, ge=1, le=10)
    
    # Performance
    max_combinations: int = Field(default=10000, description="Max combinations to evaluate")
    
    model_config = {
        "env_prefix": "APPIMMO_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> AppSettings:
    """Get cached application settings."""
    return AppSettings()
