"""
Repariert falsche B-Team Mappings in der Datenbank.
Setzt nur die bekannten falschen IDs zurück und mappt neu.

Usage:
    python scripts/fix_mappings.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import get_session, Team
from src.collectors.transfermarkt_collector import TransfermarktCollector
from src.utils.logger import get_logger

log = get_logger("fix_mappings")

# Bekannte falsche IDs (B-Teams, Reserven, falsche Vereine)
WRONG_IDS = {
    "8512",   # Paris Saint-Germain B statt PSG
    "8524",   # Olympique de Marseille B
    "9371",   # Toulouse FC B
    "8154",   # Stade Rennais FC B
    "15080",  # Rayo Vallecano B
    "71646",  # Kairat-Zhas statt FK Kairat
}

# Manuelle Korrekturen für Teams die nicht automatisch gemappt werden
MANUAL_FIXES = {
    "Paris Saint-Germain FC":   "583",    # PSG Hauptteam
    "Olympique de Marseille":   "244",    # OM Hauptteam
    "Toulouse FC":              "415",    # Toulouse Hauptteam
    "Stade Rennais FC 1901":    "273",    # Rennes Hauptteam
    "Rayo Vallecano de Madrid": "366",    # Rayo Vallecano Hauptteam
    "Lille OSC":                "1082",   # LOSC Lille
    "US Sassuolo Calcio":       "6574",   # Sassuolo
    "Stade Brestois 29":        "3911",   # Brest
}


def fix():
    session = get_session()
    collector = TransfermarktCollector()

    # 1. Falsche IDs zurücksetzen
    log.info("Setze falsche B-Team-Mappings zurück...")
    fixed = 0
    for team in session.query(Team).all():
        if team.transfermarkt_id in WRONG_IDS:
            log.warning(f"Reset: {team.name} (ID {team.transfermarkt_id} → None)")
            team.transfermarkt_id = None
            fixed += 1
    session.commit()
    log.success(f"{fixed} falsche IDs zurückgesetzt")

    # 2. Manuelle Korrekturen anwenden
    log.info("Wende manuelle Korrekturen an...")
    for team_name, correct_id in MANUAL_FIXES.items():
        team = session.query(Team).filter_by(name=team_name).first()
        if team:
            team.transfermarkt_id = correct_id
            log.success(f"Manuell: {team_name} → {correct_id}")
        else:
            log.warning(f"Team nicht in DB: {team_name}")
    session.commit()

    # 3. Noch nicht gemappte Teams neu versuchen (mit verbesserter Blacklist)
    unmapped = session.query(Team).filter(Team.transfermarkt_id == None).all()
    log.info(f"\n{len(unmapped)} Teams ohne Mapping — versuche erneut...")

    print("\n" + "="*80)
    print(f"{'Team':<30} | {'TM-ID'}")
    print("="*80)

    for team in unmapped:
        # Überspringe exotische CL-Teams die sowieso nicht funktionieren
        skip_teams = ["Qarabağ", "Olympiakos", "Bodø", "Paphos", "Kairat"]
        if any(s in team.name for s in skip_teams):
            print(f"{team.name:<30} | ÜBERSPRUNGEN (exotisch)")
            continue

        from scripts.map_teams import get_search_terms
        terms = get_search_terms(team.name)
        tm_id = None
        for term in terms:
            tm_id = collector.search_club(search_term=term, original_name=team.name)
            if tm_id:
                break

        if tm_id:
            team.transfermarkt_id = str(tm_id)
            session.commit()
            print(f"{team.name:<30} | {tm_id} ✅")
        else:
            print(f"{team.name:<30} | NICHT GEFUNDEN")

    session.close()
    print("="*80)
    log.success("Fix abgeschlossen!")


if __name__ == "__main__":
    fix()