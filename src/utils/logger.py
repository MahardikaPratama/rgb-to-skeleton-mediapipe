import logging
from typing import Optional


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured logger for the given module name.

    Handler and formatter are idempotent-safe (won't duplicate handlers
    on repeated calls for the same logger name).
    """
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


__all__ = ["get_logger"]
