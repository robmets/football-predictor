import requests
from datetime import datetime

def debug_player(search_name):
    print(f"\n--- DEBUGGING: {search_name.upper()} ---")
    
    # 1. Spieler suchen
    search_url = f"http://localhost:8000/players/search/{search_name}"
    try:
        search_res = requests.get(search_url).json()
        results = search_res.get("results", [])
        if not results:
            print(f"Spieler {search_name} nicht gefunden!")
            return
            
        player_id = results[0]["id"]
        print(f"TM-ID gefunden: {player_id}")
        
        # 2. Medizinische Verletzungen prüfen
        inj_url = f"http://localhost:8000/players/{player_id}/injuries"
        inj_data = requests.get(inj_url).json().get("injuries", [])
        print(f"Medizinische Ausfälle gefunden: {len(inj_data)}")
        if inj_data:
            latest = inj_data[0]
            print(f"  -> Letzte Verletzung: {latest.get('injury')} | Von: {latest.get('fromDate')} | Bis: {latest.get('untilDate')}")

        # 3. Sperren (Absences) prüfen - HIER STECKT WAHRSCHEINLICH DER FEHLER BEI JACKSON!
        abs_url = f"http://localhost:8000/players/{player_id}/absences"
        try:
            abs_resp = requests.get(abs_url)
            if abs_resp.status_code == 200:
                abs_data = abs_resp.json().get("injuries", [])
                print(f"Sperren/Absences gefunden: {len(abs_data)}")
                if abs_data:
                    latest_abs = abs_data[0]
                    print(f"  -> Letzte Sperre: {latest_abs.get('injury')} | Von: {latest_abs.get('fromDate')} | Bis: {latest_abs.get('untilDate')}")
            else:
                print(f"FEHLER BEIM SPERREN-ENDPUNKT: Status {abs_resp.status_code}")
                print(f"Grund: Hast du die API nach dem Erstellen von absences.py wirklich neu gestartet?")
        except Exception as e:
            print(f"Sperren konnten nicht abgerufen werden: {e}")
            
    except Exception as e:
        print(f"Fehler beim Verbinden zur API: {e}")

if __name__ == "__main__":
    debug_player("Leon Klanac")
    debug_player("Nicolas Jackson")