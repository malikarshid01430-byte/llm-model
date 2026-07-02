from __future__ import annotations

import logging
import os
from pathlib import Path


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Create and return a configured logger for the platform."""
    logger = logging.getLogger(name)
    logger.setLevel(
        getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    )
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
