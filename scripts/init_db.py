"""
Script: Initialize the database.
Run once before starting the project.

Usage:
    python scripts/init_db.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import init_db
from src.utils.logger import get_logger

log = get_logger("init_db")

if __name__ == "__main__":
    log.info("Initializing Football Predictor database...")
    init_db()
    log.success("Database ready. Next step: run scripts/fetch_data.py")
