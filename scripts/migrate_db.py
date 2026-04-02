"""
DB Migration — fügt fehlende Spalten zur bestehenden DB hinzu.
Löscht KEINE Daten. Sicher auf bestehender DB auszuführen.

Usage:
    python scripts/migrate_db.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
from src.utils.logger import get_logger
from src.utils.database import init_db

log = get_logger("migrate_db")

DB_PATH = "data/football.db"

def column_exists(cursor, table, column):
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())

def migrate():
    log.info("Starte DB-Migration...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # ── Team: transfermarkt_id hinzufügen ────────────────────────────────────
    if not column_exists(cur, "teams", "transfermarkt_id"):
        cur.execute("ALTER TABLE teams ADD COLUMN transfermarkt_id TEXT")
        log.success("teams.transfermarkt_id hinzugefügt")
    else:
        log.info("teams.transfermarkt_id bereits vorhanden")

    # ── Predictions: alte Tabelle löschen & neu erstellen ────────────────────
    # Prüfe ob die neue Struktur schon existiert
    if not column_exists(cur, "predictions", "home_team"):
        log.info("Predictions-Tabelle wird neu erstellt (alte Daten gehen verloren)...")
        cur.execute("DROP TABLE IF EXISTS predictions")
        log.info("Alte predictions Tabelle gelöscht")
    else:
        log.info("predictions Tabelle hat bereits neue Struktur")

    conn.commit()
    conn.close()

    # Neue Tabellen via SQLAlchemy erstellen
    init_db(DB_PATH)
    log.success("Migration abgeschlossen!")
    log.info("Alle Teams und Spiele sind erhalten.")

if __name__ == "__main__":
    migrate()