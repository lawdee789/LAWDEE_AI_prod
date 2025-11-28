"""Logging helpers for the service."""

from __future__ import annotations

import logging
from typing import Optional

from .config import get_settings

_configured = False


def _ensure_configured(level: Optional[int] = None) -> None:
    global _configured
    if _configured:
        return
    settings = get_settings()
    resolved_level = level or (logging.DEBUG if settings.enable_debug_logging else logging.INFO)
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    _configured = True


def get_logger(name: str = "lawyer-matching", level: Optional[int] = None) -> logging.Logger:
    """Return a module-level logger configured once."""

    _ensure_configured(level)
    return logging.getLogger(name)
