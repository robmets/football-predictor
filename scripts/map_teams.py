import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import get_session, Team
from src.collectors.transfermarkt_collector import TransfermarktCollector
from src.utils.logger import get_logger

log = get_logger("map_teams")

def get_search_terms(name: str) -> list:
    """Generiert eine Liste von Suchbegriffen: Vom exakten Namen zum groben Namen."""
    terms = [name]
    
    # 1. Zahlen und Punkte sicher entfernen (z.B. "1. FC" -> " FC", "1846" -> "")
    clean1 = re.sub(r'\b\d{1,4}\b', '', name).replace('.', '')
    
    # 2. Nur alleinstehende Kürzel entfernen (\b bedeutet Wortgrenze -> schützt 'FSV')
    generics = ['FC', 'SV', 'TSG', 'VfL', 'VfB', 'SC', 'FSV', 'AFC', 'CF', 'Real', 'United', 'City']
    pattern = r'\b(?:' + '|'.join(generics) + r')\b'
    clean2 = re.sub(pattern, '', clean1, flags=re.IGNORECASE)
    
    clean_final = ' '.join(clean2.split()).strip()
    
    if clean_final and clean_final != name:
        terms.append(clean_final) # z.B. fügt "Mainz" oder "Union Berlin" als Alternative hinzu
        
    return terms

def run_smart_mapping():
    session = get_session()
    collector = TransfermarktCollector()
    
    teams = session.query(Team).all()
    log.info(f"Starte SUPER-DYNAMISCHES Mapping für {len(teams)} Teams...")
    
    print("\n" + "="*95)
    print(f"{'Datenbank Name':<25} | {'TM Suchbegriff':<20} | {'Transfermarkt Name':<25} | {'TM-ID'}")
    print("="*95)

    for team in teams:
        team.transfermarkt_id = None  # Reset für sauberen Durchlauf
        
        search_terms = get_search_terms(team.name)
        tm_id = None
        used_term = ""
        
        for term in search_terms:
            # Wir übergeben den gekürzten Suchbegriff UND den Originalnamen zum Vergleichen
            tm_id = collector.search_club(search_term=term, original_name=team.name)
            if tm_id:
                used_term = term
                break
            
        if tm_id:
            team.transfermarkt_id = str(tm_id)
            session.commit()
            
            try:
                profile = collector._get(f"clubs/{tm_id}/profile")
                tm_name = profile.get("name", "Name nicht gefunden")
                print(f"{team.name:<25} | {used_term:<20} | {tm_name:<25} | {tm_id}")
            except Exception:
                print(f"{team.name:<25} | {used_term:<20} | FEHLER BEI ABFRAGE        | {tm_id}")
        else:
            print(f"{team.name:<25} | {'N/A':<20} | KEEEEIN MAPPING GEFUNDEN  | -")
            
    session.close()
    print("="*95)
    log.info("Mapping abgeschlossen!")

if __name__ == "__main__":
    run_smart_mapping()