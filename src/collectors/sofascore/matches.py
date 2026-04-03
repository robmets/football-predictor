import logging
from curl_cffi.requests import AsyncSession

log = logging.getLogger(__name__)

async def get_match_lineups_and_ratings(match_id: int) -> dict:
    """
    Ruft Aufstellungen für ein Sofascore-Spiel ab.
    Gibt rohe API-Daten zurück (inklusive 'confirmed' Flag).
    """
    url = f"https://www.sofascore.com/api/v1/event/{match_id}/lineups"
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

    async with AsyncSession(impersonate="chrome") as client:
        response = await client.get(url, headers=headers)

        if response.status_code != 200:
            raise Exception(
                f"Sofascore Lineups API Fehler: Status {response.status_code}"
            )

        data = response.json()

        # Debug: zeige rohe Struktur
        confirmed = data.get("confirmed", None)
        home_players = data.get("home", {}).get("players", [])
        away_players = data.get("away", {}).get("players", [])
        log.info(
            f"Sofascore /lineups: confirmed={confirmed}, "
            f"home_players={len(home_players)}, away_players={len(away_players)}"
        )

        return parse_player_ratings(data)


async def get_predicted_lineups(match_id: int) -> dict:
    """
    Versucht voraussichtliche Aufstellungen über pregame-Form zu holen.
    Sofascore-Endpunkte die voraussichtliche Aufstellungen liefern können:
      - /event/{id}/lineups  (confirmed=false aber Spieler vorhanden = Modus B)
      - /event/{id}/pregame-form  (enthält Form-Daten, manchmal Lineup-Hinweise)
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Referer": "https://www.sofascore.com/",
        "Origin": "https://www.sofascore.com",
    }

    async with AsyncSession(impersonate="chrome") as client:
        # Endpunkt 1: Standard-Lineup (funktioniert auch für predicted)
        lineup_url = f"https://www.sofascore.com/api/v1/event/{match_id}/lineups"
        res = await client.get(lineup_url, headers=headers)

        if res.status_code == 200:
            data = res.json()
            confirmed = data.get("confirmed", False)
            home_players = data.get("home", {}).get("players", [])
            away_players = data.get("away", {}).get("players", [])

            log.info(
                f"Predicted Lineups: confirmed={confirmed}, "
                f"home={len(home_players)}, away={len(away_players)}"
            )

            if home_players or away_players:
                parsed = parse_player_ratings(data)
                parsed["confirmed"] = confirmed
                return parsed

        return {"confirmed": False, "home_team": {"players": []}, "away_team": {"players": []}}


def parse_player_ratings(data: dict) -> dict:
    """
    Parst Sofascore-Lineup-Daten.
    Unterstützt bestätigte und voraussichtliche Aufstellungen.
    """
    result = {
        "confirmed": data.get("confirmed", False),
        "home_team": {"team_live_rating": None, "team_season_rating": None, "players": []},
        "away_team": {"team_live_rating": None, "team_season_rating": None, "players": []},
    }

    for team_key in ["home", "away"]:
        team_data = data.get(team_key, {})
        dict_key = f"{team_key}_team"

        starting_players = team_data.get("players", [])
        bench_players    = team_data.get("bench", [])
        all_players      = starting_players + bench_players

        live_ratings_list   = []
        season_ratings_list = []

        for p in all_players:
            player_info  = p.get("player", {})
            stats        = p.get("statistics", {})

            live_rating   = stats.get("rating", None)
            season_rating = (
                p.get("avgRating")
                or player_info.get("avgRating")
                or player_info.get("averageRating")
            )

            is_starter = p in starting_players
            if is_starter:
                if live_rating:   live_ratings_list.append(live_rating)
                if season_rating: season_ratings_list.append(season_rating)

            result[dict_key]["players"].append({
                "id":            player_info.get("id"),
                "name":          player_info.get("name"),
                "position":      player_info.get("position", p.get("position")),
                "is_starter":    is_starter,
                "live_rating":   live_rating,
                "season_rating": season_rating,
            })

        if live_ratings_list:
            result[dict_key]["team_live_rating"] = round(
                sum(live_ratings_list) / len(live_ratings_list), 2
            )
        if season_ratings_list:
            result[dict_key]["team_season_rating"] = round(
                sum(season_ratings_list) / len(season_ratings_list), 2
            )

    return result