from curl_cffi.requests import AsyncSession

async def get_team_players(team_id: int) -> dict:
    """
    Holt den kompletten Kader (alle Spieler) einer Mannschaft über die Sofascore Team-ID.
    """
    url = f"https://www.sofascore.com/api/v1/team/{team_id}/players"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Referer": "https://www.sofascore.com/",
        "Origin": "https://www.sofascore.com"
    }
    
    async with AsyncSession(impersonate="chrome") as client:
        response = await client.get(url, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Fehler beim Abrufen des Kaders für Team {team_id}: {response.status_code}")
            
        data = response.json()
        players_data = data.get("players", [])
        
        # Wir bereinigen die Daten, damit dein Predictor nur das Wichtigste bekommt
        roster = []
        for item in players_data:
            player = item.get("player", {})
            roster.append({
                "id": player.get("id"),
                "name": player.get("name"),
                "position": player.get("position"),
                "shirt_number": player.get("shirtNumber")
            })
            
        return {"team_id": team_id, "roster": roster}

async def get_team_events(team_id: int) -> dict:
    """
    Holt den Spielplan (vergangene und zukünftige Spiele) für ein Team.
    """
    # Sofascore hat verschiedene Event-Endpunkte. Dieser hier liefert die nächsten und letzten Spiele eines Teams.
    url = f"https://www.sofascore.com/api/v1/team/{team_id}/events/next/0"
    
    # Um auch vergangene Spiele zu holen, rufen wir zusätzlich /last/0 ab
    url_last = f"https://www.sofascore.com/api/v1/team/{team_id}/events/last/0"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }
    
    async with AsyncSession(impersonate="chrome") as client:
        # Wir holen zukünftige und vergangene Spiele gleichzeitig
        next_res = await client.get(url, headers=headers)
        last_res = await client.get(url_last, headers=headers)
        
        events_list = []
        
        # Vergangene Spiele verarbeiten
        if last_res.status_code == 200:
            events_list.extend(last_res.json().get("events", []))
            
        # Zukünftige Spiele verarbeiten
        if next_res.status_code == 200:
            events_list.extend(next_res.json().get("events", []))
            
        # Wir filtern die wichtigen Daten für deinen Predictor heraus
        formatted_events = []
        for event in events_list:
            formatted_events.append({
                "match_id": event.get("id"),
                "tournament": event.get("tournament", {}).get("name"),
                "status": event.get("status", {}).get("description"), # z.B. "Ended", "Not started"
                "startTimestamp": event.get("startTimestamp"), # Wann das Spiel startet (Unix Timestamp)
                "home_team": {
                    "id": event.get("homeTeam", {}).get("id"),
                    "name": event.get("homeTeam", {}).get("name")
                },
                "away_team": {
                    "id": event.get("awayTeam", {}).get("id"),
                    "name": event.get("awayTeam", {}).get("name")
                },
                "score": event.get("homeScore", {}).get("current", 0) if event.get("status", {}).get("type") == "finished" else None
            })
            
        # Sortieren nach Datum (älteste zuerst)
        formatted_events = sorted(formatted_events, key=lambda x: x["startTimestamp"])
        
        return {"team_id": team_id, "events": formatted_events}