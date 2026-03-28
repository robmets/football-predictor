from src.collectors.transfermarkt_collector import TransfermarktCollector

collector = TransfermarktCollector()

# 1. Spieler suchen (Wir nehmen Harry Kane als Beispiel)
kane_id = collector.search_player("Harry Kane")
print(f"Transfermarkt ID für Harry Kane: {kane_id}")

if kane_id:
    # 2. Profil und Marktwert abfragen
    profile = collector.get_player_profile(kane_id)
    print(f"Marktwert: {profile.get('marketValue', 'Unbekannt')}")
    
    # 3. Verletzungen abfragen
    injuries = collector.get_player_injuries(kane_id)
    if injuries:
        print(f"Letzte Verletzung/Sperre: {injuries[0].get('injury', 'Unbekannt')} (Verpasst: {injuries[0].get('matchesMissed', 0)} Spiele)")