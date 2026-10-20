"""Logging setup for Module 1."""

from __future__ import annotations

import logging
from typing import Any


def configure_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_kv(logger: logging.Logger, event: str, **fields: Any) -> None:
    extras = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info("%s %s", event, extras)
