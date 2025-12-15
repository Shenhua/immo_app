"""Logging configuration for app_immo.

Provides structured logging using structlog with JSON output for production
and colored console output for development.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog


def configure_logging(
    level: str | None = None,
    json_output: bool = False,
) -> structlog.BoundLogger:
    """Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to env LOGLEVEL or INFO.
        json_output: If True, output JSON format (for production). Otherwise, colored console.

    Returns:
        Configured logger instance.
    """
    log_level = level or os.environ.get("LOGLEVEL", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    # Choose processors based on output format
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_output:
        # Production: JSON output
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: colored console output
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger()


# Default logger instance
log = configure_logging()


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a logger instance, optionally bound to a specific name.

    Args:
        name: Optional logger name (usually module name).

    Returns:
        Bound logger instance.
    """
    logger = structlog.get_logger()
    if name:
        return logger.bind(logger_name=name)
    return logger
