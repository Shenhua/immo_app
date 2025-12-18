"""Logging configuration for app_immo.

Provides structured logging using structlog with JSON output for production
and colored console output for development.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Any

import structlog

# Log file location
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_FILE = LOG_DIR / "app.log"

# Module-level state for lazy initialization
_configured: bool = False
_default_logger: structlog.BoundLogger | None = None


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
    global _configured, _default_logger

    # Skip if already configured (idempotent)
    if _configured:
        return structlog.get_logger()

    log_level = level or os.environ.get("LOGLEVEL", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    # 1. Configure Standard Library Logging (Handlers)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]

    # Only add file handler if not in test mode
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            LOG_DIR.mkdir(exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                str(LOG_FILE), maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
            handlers.append(file_handler)
        except (OSError, PermissionError):
            # Fail silently if we can't create log file
            pass

    logging.basicConfig(
        format="%(message)s",
        level=numeric_level,
        handlers=handlers,
        force=True,  # Overwrite any existing config
    )

    # 2. Configure Structlog Processors
    processors: list[Any] = [
        structlog.stdlib.filter_by_level,
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))

    # 3. Configure Structlog to wrap Stdlib
    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    _configured = True
    _default_logger = structlog.get_logger()
    return _default_logger


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a logger instance, optionally bound to a specific name.

    This function lazily initializes logging on first call.

    Args:
        name: Optional logger name (usually module name).

    Returns:
        Bound logger instance.
    """
    global _configured

    # Lazy initialization
    if not _configured:
        configure_logging()

    logger = structlog.get_logger()
    if name:
        return logger.bind(logger_name=name)
    return logger

