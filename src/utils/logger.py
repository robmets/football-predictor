"""
Centralized logger using loguru.
Import this everywhere instead of using print().
"""

import sys
from pathlib import Path
from loguru import logger

# Remove default handler
logger.remove()

# Console output
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
    level="INFO",
    colorize=True,
)

# File output (rotated daily, kept 7 days)
Path("logs").mkdir(exist_ok=True)
logger.add(
    "logs/football_predictor_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
)


def get_logger(name: str):
    """Return a named logger for a module."""
    return logger.bind(name=name)
