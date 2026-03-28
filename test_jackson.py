from src.collectors.transfermarkt_collector import TransfermarktCollector

collector = TransfermarktCollector()

# Bayern ID ist 27
players = collector.get_club_players("27")

print(f"Anzahl gefundener Bayern-Spieler: {len(players)}")
print("Liste aller gefundenen Spieler:")

jackson_found = False
for p in players:
    print(f"- {p.get('name')}")
    if "Jackson" in p.get("name", ""):
        jackson_found = True

print("-" * 30)
if jackson_found:
    print("ERGEBNIS: Nicolas Jackson WURDE im Kader gefunden!")
else:
    print("ERGEBNIS: Nicolas Jackson WURDE NICHT im Kader gefunden! (Der Scraper ist schuld)")