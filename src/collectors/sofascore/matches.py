from curl_cffi.requests import AsyncSession

async def get_match_lineups_and_ratings(match_id: int) -> dict:
    """
    Ruft die Aufstellungen und Live-Ratings für ein bestimmtes Sofascore-Spiel ab.
    """
    url = f"https://www.sofascore.com/api/v1/event/{match_id}/lineups"
    
    # Noch ein paar zusätzliche Header, die Browser immer mitschicken
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://www.sofascore.com/",
        "Origin": "https://www.sofascore.com",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Cache-Control": "no-cache",
    }
    
    # HIER PASSIERT DIE MAGIE: impersonate="chrome" trickst den Bot-Schutz aus!
    async with AsyncSession(impersonate="chrome") as client:
        response = await client.get(url, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Sofascore API Fehler: Status {response.status_code} - Der Bot-Schutz blockiert uns vielleicht noch.")
            
        data = response.json()
        return parse_player_ratings(data)

def parse_player_ratings(data: dict) -> dict:
    result = {
        "home_team": { "team_live_rating": None, "team_season_rating": None, "players": [] },
        "away_team": { "team_live_rating": None, "team_season_rating": None, "players": [] }
    }
    
    for team_key in ["home", "away"]:
        team_data = data.get(team_key, {})
        dict_key = f"{team_key}_team" # HIER WAR DER FEHLER: Macht aus "home" -> "home_team"
        
        starting_players = team_data.get("players", [])
        bench_players = team_data.get("bench", [])
        all_players = starting_players + bench_players
        
        live_ratings_list = []
        season_ratings_list = []
        
        for p in all_players:
            player_info = p.get("player", {})
            stats = p.get("statistics", {})
            
            live_rating = stats.get("rating", None) 
            season_rating = p.get("avgRating") or player_info.get("avgRating") or player_info.get("averageRating")
            
            if p in starting_players:
                if live_rating: live_ratings_list.append(live_rating)
                if season_rating: season_ratings_list.append(season_rating)
            
            result[dict_key]["players"].append({
                "id": player_info.get("id"),
                "name": player_info.get("name"),
                "position": player_info.get("position", p.get("position")),
                "is_starter": p in starting_players,
                "live_rating": live_rating,
                "season_rating": season_rating
            })
            
        if live_ratings_list:
            result[dict_key]["team_live_rating"] = round(sum(live_ratings_list) / len(live_ratings_list), 2)
        if season_ratings_list:
            result[dict_key]["team_season_rating"] = round(sum(season_ratings_list) / len(season_ratings_list), 2)
            
    return result