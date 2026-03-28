import sys
import re
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import get_session, Team
from src.collectors.transfermarkt_collector import TransfermarktCollector
from src.utils.logger import get_logger

log = get_logger("injuries")

def parse_tm_date(date_str: str):
    """Versteht alle Datumsformate und fixt den US/EU Mixup."""
    if not date_str or str(date_str).strip() in ["", "?", "-"]:
        return None
        
    date_str = str(date_str).strip()
    today = datetime.now().date()
    
    # 1. API Standard (YYYY-MM-DD) -> Kommt aus dem Verletzungs-Endpunkt
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
            # Der Harry-Kane-Fix: Wenn das Datum > 2 Monate in der Zukunft liegt
            # und der Tag <= 12 ist, hat Pydantic höchstwahrscheinlich Tag und Monat vertauscht!
            if (d - today).days > 60 and d.day <= 12:
                try:
                    return d.replace(month=d.day, day=d.month) # Zurücktauschen!
                except ValueError:
                    pass
            return d
        except ValueError:
            pass

    # 2. Raw Transfermarkt Formate (Sperren & Absences) -> Hier wird Jackson gefangen!
    for fmt in ("%d/%m/%Y", "%d.%m.%Y", "%b %d, %Y", "%d %b %Y"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
            
    return None

def check_current_injuries(team_name: str):
    session = get_session()
    team = session.query(Team).filter_by(name=team_name).first()
    session.close()

    if not team or not team.transfermarkt_id:
        log.error(f"Team '{team_name}' nicht gefunden oder keine TM-ID vorhanden.")
        return

    collector = TransfermarktCollector()
    players = collector.get_club_players(team.transfermarkt_id)
    
    print("\n" + "="*95)
    print(f"AKTUELLE AUSFÄLLE / SPERREN: {team.name.upper()} (Kadergröße: {len(players)})")
    print("="*95)

    today = datetime.now().date()
    currently_out = []
    
    for player in players:
        player_id = player.get("id")
        player_name = player.get("name")
        market_value = player.get("marketValue", "0")
        
        all_absences = collector.get_player_injuries(player_id)
        
        if all_absences:
            for absence in all_absences:
                until_str = absence.get("untilDate") or absence.get("until_date")
                from_str = absence.get("fromDate") or absence.get("from_date")
                injury_reason = absence.get("injury", "Unbekannt")
                
                is_out_now = False
                
                # Wir nutzen jetzt unseren smarten Parser!
                until_date = parse_tm_date(until_str)
                from_date = parse_tm_date(from_str)
                
                if until_date:
                    if until_date >= today:
                        is_out_now = True
                else:
                    if from_date:
                        if (today - from_date).days < 250:
                            is_out_now = True
                            
                if is_out_now:
                    currently_out.append({
                        "name": player_name,
                        "reason": injury_reason,
                        "until": until_date.strftime("%d.%m.%Y") if until_date else "Unbekannt",
                        "value": market_value
                    })
                    break 

    if not currently_out:
        print("Keine aktuellen Verletzungen oder Sperren gefunden. Alle fit!")
    else:
        print(f"{'Spieler':<25} | {'Grund (Medizinisch/Sperre)':<30} | {'Rückkehr':<12} | {'Marktwert'}")
        print("-" * 95)
        for p in currently_out:
            print(f"{p['name']:<25} | {p['reason']:<30} | {p['until']:<12} | {p['value']}")
            
    print("="*95)

if __name__ == "__main__":
    check_current_injuries("FC Bayern München")