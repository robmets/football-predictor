"""
Korrigiert falsche Transfermarkt-IDs für Athletic Club und AS Monaco.
Run: python scripts/fix_tm_mappings.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import get_session, Team
from src.utils.logger import get_logger

log = get_logger("fix_tm_mappings")

# Falsche ID → (korrekter Name in DB, korrekte TM-ID)
FIXES = {
    "64918":  ("Athletic Club",  "621"),   # Bilbao B → Bilbao A
    "116229": ("AS Monaco FC",   "162"),   # Monaco Reserve → Monaco A
}

def fix():
    session = get_session()
    fixed = 0

    for wrong_id, (team_name, correct_id) in FIXES.items():
        team = session.query(Team).filter(Team.transfermarkt_id == wrong_id).first()
        if team:
            log.warning(f"Fix: {team.name} | ID {wrong_id} → {correct_id}")
            team.transfermarkt_id = correct_id
            fixed += 1
        else:
            # Fallback: suche nach Namen
            team = session.query(Team).filter(Team.name == team_name).first()
            if team:
                log.warning(f"Fix by name: {team.name} | {team.transfermarkt_id} → {correct_id}")
                team.transfermarkt_id = correct_id
                fixed += 1
            else:
                log.error(f"Team nicht gefunden: {team_name} (ID {wrong_id})")

    session.commit()
    session.close()
    log.info(f"✅ {fixed} Mappings korrigiert.")

if __name__ == "__main__":
    fix()