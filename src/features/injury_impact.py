import sys
import re
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.database import get_session, Team
from src.collectors.transfermarkt_collector import TransfermarktCollector
from src.utils.logger import get_logger

log = get_logger("injury_impact")

def parse_tm_date(date_str: str):
    """Versteht alle Datumsformate und fixt den US/EU Mixup."""
    if not date_str or str(date_str).strip() in ["", "?", "-"]:
        return None
        
    date_str = str(date_str).strip()
    today = datetime.now().date()
    
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
            if (d - today).days > 60 and d.day <= 12:
                try: return d.replace(month=d.day, day=d.month)
                except ValueError: pass
            return d
        except ValueError: pass

    for fmt in ("%d/%m/%Y", "%d.%m.%Y", "%b %d, %Y", "%d %b %Y"):
        try: return datetime.strptime(date_str, fmt).date()
        except ValueError: continue
    return None

def parse_market_value(value_str) -> int:
    """Macht aus Transfermarkt-Strings ('€1.50m', '€500k') echte Zahlen."""
    if not value_str or str(value_str).strip() in ["-", "Unbekannt", "0", "None", ""]:
        return 0
        
    val_str = str(value_str).lower().replace(",", ".")
    
    # Isoliert die reine Zahl (z.B. "1.50" oder "500")
    match = re.search(r'[\d.]+', val_str)
    if not match:
        return 0
        
    try:
        number = float(match.group())
    except ValueError:
        return 0
        
    # Die richtigen Nullen dranhängen!
    if 'm' in val_str:
        return int(number * 1000000)
    elif 'k' in val_str:
        return int(number * 1000)
    else:
        return int(number)

def _wc_national_team_impact(team_name: str):
    """
    WC national team injury impact via Transfermarkt.
    Nutzt gespeicherte TM-ID aus DB (wenn vorhanden via map_wc_teams.py),
    sonst dynamische Suche.
    Returns (impact_pct, missing_names) — graceful 0.0 fallback on any error.
    """
    try:
        collector = TransfermarktCollector()

        # Use stored TM ID if available (set via map_teams.py --league WC)
        session = get_session()
        team_obj = session.query(Team).filter(
            Team.name == team_name, Team.league == "WC",
            Team.transfermarkt_id.isnot(None)
        ).first()
        session.close()
        tm_id = team_obj.transfermarkt_id if team_obj else None

        if tm_id:
            players = collector.get_club_players(tm_id)
        else:
            players = collector.get_national_team_players(team_name)

        if not players:
            log.warning(f"WC: Keine Spielerdaten für {team_name} → kein Verletzungsimpact")
            return 0.0, []

        today = datetime.now().date()
        total_value = 0
        missing_value = 0
        missing_players = []

        for player in players:
            val = parse_market_value(player.get("marketValue", "0"))
            total_value += val
            player_id = player.get("id")
            if not player_id:
                continue
            absences = collector.get_player_injuries(player_id)
            for absence in (absences or []):
                until_date = parse_tm_date(absence.get("untilDate") or absence.get("until_date"))
                from_date  = parse_tm_date(absence.get("fromDate") or absence.get("from_date"))
                is_out = (until_date and until_date >= today) or (
                    not until_date and from_date and (today - from_date).days < 250
                )
                if is_out:
                    missing_value += val
                    missing_players.append({
                        "name": player.get("name"), "value": val,
                        "reason": absence.get("injury", "Unbekannt"),
                    })
                    break

        impact = (missing_value / total_value * 100) if total_value > 0 else 0.0
        return impact, [p["name"] for p in missing_players]

    except Exception as e:
        log.warning(f"WC national team impact fallback für {team_name}: {e}")
        return 0.0, []


def calculate_missing_impact(team_name: str, league: str = None):
    """Berechnet den prozentualen Marktwertverlust durch Ausfälle."""

    # WC/EC: Nationalteams haben keine Club-TM-ID → graceful fallback
    if league in ("WC", "EC"):
        return _wc_national_team_impact(team_name)

    session = get_session()
    team = session.query(Team).filter_by(name=team_name).first()
    session.close()

    if not team or not team.transfermarkt_id:
        log.error(f"Team '{team_name}' nicht in der Datenbank oder keine TM-ID.")
        return None

    collector = TransfermarktCollector()
    log.info(f"Analysiere Kader-Werte für {team.name}...")
    players = collector.get_club_players(team.transfermarkt_id)
    
    today = datetime.now().date()
    total_squad_value = 0
    missing_squad_value = 0
    missing_players = []

    for player in players:
        # Marktwert zum Gesamtwert addieren
        val = parse_market_value(player.get("marketValue", "0"))
        total_squad_value += val
        
        # Ausfälle prüfen
        player_id = player.get("id")
        all_absences = collector.get_player_injuries(player_id)
        is_out_now = False
        
        if all_absences:
            for absence in all_absences:
                until_str = absence.get("untilDate") or absence.get("until_date")
                from_str = absence.get("fromDate") or absence.get("from_date")
                
                until_date = parse_tm_date(until_str)
                from_date = parse_tm_date(from_str)
                
                if until_date and until_date >= today:
                    is_out_now = True
                elif not until_date and from_date and (today - from_date).days < 250:
                    is_out_now = True
                    
                if is_out_now:
                    missing_squad_value += val
                    missing_players.append({
                        "name": player.get("name"),
                        "value": val,
                        "reason": absence.get("injury", "Unbekannt")
                    })
                    break # Nur einmal pro Spieler zählen!

    # Das magische ML-Feature berechnen
    impact_percentage = (missing_squad_value / total_squad_value * 100) if total_squad_value > 0 else 0.0

    # Schöne Konsolen-Ausgabe
    print("\n" + "="*85)
    print(f"IMPACT-ANALYSE: {team.name.upper()}")
    print("="*85)
    # Tausendertrennzeichen für bessere Lesbarkeit
    print(f"Gesamter Kaderwert:   € {total_squad_value:,.0f}".replace(",", "."))
    print(f"Wert der Ausfälle:    € {missing_squad_value:,.0f}".replace(",", "."))
    print(f"MISSING IMPACT:       {impact_percentage:.2f} %")
    print("-" * 85)
    
    if missing_players:
        print("Fehlende Schlüsselspieler (nach Marktwert sortiert):")
        # Sortiere Liste, damit der teuerste Ausfall oben steht
        missing_players.sort(key=lambda x: x["value"], reverse=True)
        for p in missing_players:
            val_str = f"€ {p['value']:,.0f}".replace(",", ".")
            print(f"- {p['name']:<22} | {val_str:>15} | {p['reason']}")
    else:
        print("Keine Ausfälle. Das Team ist bei 100% Stärke!")
    print("="*85)

    missing_names = [player["name"] for player in missing_players] # (Je nachdem wie deine Variable heißt)
    return impact_percentage, missing_names

if __name__ == "__main__":
    calculate_missing_impact("FC Bayern München")