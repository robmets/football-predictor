"""
API-Football Collector — api-football.com (v3)
Holt aktuelle Form (letzte 5 Spiele) und Spieler-Ratings.

Nur für Live-Daten kurz vor dem Spiel.
Historische Daten kommen weiterhin von football-data.org.

Budget: 100 Requests/Tag → max 4 Calls pro Vorhersage.
Caching: 2h TTL um wiederholte Calls zu vermeiden.

Usage:
    collector = ApiFootballCollector()
    form = collector.get_team_form("Hamburger SV", league_id=78, n=5)
    ratings = collector.get_player_ratings("Hamburger SV", league_id=78)
"""

import requests
import time
from datetime import datetime, timedelta
from functools import lru_cache
import difflib
from src.utils.logger import get_logger
from config.config import config

log = get_logger(__name__)

import os
BASE_URL = "https://v3.football.api-sports.io"

# League IDs von api-football.com (v3) — verifiziert via Dashboard
LEAGUE_IDS = {
    "BL1": 78,    # Germany Bundesliga
    "PL":  39,    # England Premier League
    "PD":  140,   # Spain La Liga
    "SA":  135,   # Italy Serie A
    "FL1": 61,    # France Ligue 1
    "CL":  2,     # UEFA Champions League
    "WC":  1,     # FIFA World Cup
    "EC":  4,     # UEFA European Championship
}

# Cache: team_name → api-football team_id (vermeidet Search-Calls)
_team_id_cache: dict[str, int] = {}
# Cache: (team_id, season) → form data mit Timestamp
_form_cache: dict[tuple, tuple] = {}
_CACHE_TTL_HOURS = 2


class ApiFootballCollector:
    """
    Holt Live-Daten von api-football.com.
    Minimiert Requests durch Caching und lazy loading.
    """

    def __init__(self):
        # Lese Key direkt aus Umgebungsvariablen (funktioniert auch nach export)
        self.api_key = (
            os.environ.get("API_FOOTBALL_KEY") or
            config.API_FOOTBALL_KEY or
            ""
        )
        self.headers = {"x-apisports-key": self.api_key}
        self._request_count = 0

        if not self.api_key:
            log.warning("API_FOOTBALL_KEY nicht gesetzt — Live-Form deaktiviert")
        else:
            log.info(f"api-football.com bereit (Key: ...{self.api_key[-6:]})")

    # ── Public ───────────────────────────────────────────────────────────────

    def get_team_form(
        self,
        team_name: str,
        league_id: int = 78,
        n: int = 5,
        season: int = None,
    ) -> dict:
        """
        Holt die letzten N Spiele eines Teams direkt von api-football.com.

        Returns:
            {
                team_name, team_id,
                last_5: [{date, home_team, away_team, home_goals, away_goals, result}],
                form_ppg: float,          # gewichtete Punkte pro Spiel (0–3)
                goals_scored_avg: float,
                goals_conceded_avg: float,
                source: "api-football" | "unavailable"
            }
        """
        if not self.api_key:
            return self._unavailable(team_name)

        if season is None:
            from datetime import date
            season = date.today().year if date.today().month >= 7 else date.today().year - 1

        team_id = self._get_team_id(team_name, league_id, season)
        if not team_id:
            return self._unavailable(team_name)

        # Cache check
        cache_key = (team_id, season, n)
        if cache_key in _form_cache:
            data, ts = _form_cache[cache_key]
            if datetime.now() - ts < timedelta(hours=_CACHE_TTL_HOURS):
                log.info(f"Form-Cache hit für {team_name} (noch {_CACHE_TTL_HOURS}h gültig)")
                return data

        log.info(f"Hole Live-Form für {team_name} (ID: {team_id}, letzte {n} Spiele)...")
        data = self._get(f"fixtures", params={
            "team": team_id,
            "last": n,
            "timezone": "Europe/Berlin",
        })

        if not data or not data.get("response"):
            log.warning(f"Keine Spieldaten für {team_name}")
            return self._unavailable(team_name)

        matches = data["response"]
        last_5 = []
        points = []
        goals_scored = []
        goals_conceded = []

        for m in matches:
            home_id    = m["teams"]["home"]["id"]
            home_goals = m["goals"]["home"] or 0
            away_goals = m["goals"]["away"] or 0
            is_home    = home_id == team_id

            scored    = home_goals if is_home else away_goals
            conceded  = away_goals if is_home else home_goals

            if home_goals > away_goals:
                result = "H"
            elif home_goals < away_goals:
                result = "A"
            else:
                result = "D"

            pts = 3 if (is_home and result == "H") or (not is_home and result == "A") else \
                  1 if result == "D" else 0

            last_5.append({
                "date":       m["fixture"]["date"][:10],
                "home_team":  m["teams"]["home"]["name"],
                "away_team":  m["teams"]["away"]["name"],
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result":     result,
                "is_home":    is_home,
                "points":     pts,
            })
            points.append(pts)
            goals_scored.append(scored)
            goals_conceded.append(conceded)

        # Gewichtete PPG (jüngstes Spiel zählt mehr)
        import numpy as np
        weights = np.array([1.0, 0.85, 0.70, 0.55, 0.40][:len(points)])
        weights = weights / weights.sum()
        form_ppg = float(np.dot(points, weights)) if points else 1.5

        result_data = {
            "team_name":         team_name,
            "team_id":           team_id,
            "last_5":            last_5,
            "form_ppg":          round(form_ppg, 3),
            "goals_scored_avg":  round(sum(goals_scored) / len(goals_scored), 2) if goals_scored else 1.3,
            "goals_conceded_avg": round(sum(goals_conceded) / len(goals_conceded), 2) if goals_conceded else 1.3,
            "source":            "api-football",
        }

        _form_cache[cache_key] = (result_data, datetime.now())
        log.success(
            f"Form {team_name}: PPG={form_ppg:.2f}, "
            f"Tore: {result_data['goals_scored_avg']:.1f}/{result_data['goals_conceded_avg']:.1f}"
        )
        return result_data

    def get_player_ratings(
        self,
        team_name: str,
        league_id: int = 78,
        season: int = None,
        top_n: int = 8,
    ) -> list[dict]:
        """
        Holt Spieler-Ratings der letzten Saison für ein Team.
        Gibt die Top-N Spieler nach Rating zurück.

        Returns:
            list of {name, position, rating, goals, assists, minutes_played}
        """
        if not self.api_key:
            return []

        if season is None:
            from datetime import date
            season = date.today().year if date.today().month >= 7 else date.today().year - 1

        team_id = self._get_team_id(team_name, league_id, season)
        if not team_id:
            return []

        log.info(f"Hole Spieler-Ratings für {team_name}...")
        data = self._get("players", params={
            "team":   team_id,
            "season": season,
            "league": league_id,
        })

        if not data or not data.get("response"):
            return []

        players = []
        for p in data["response"]:
            info  = p.get("player", {})
            stats = p.get("statistics", [{}])[0]

            rating = stats.get("games", {}).get("rating")
            if not rating:
                continue

            players.append({
                "name":           info.get("name", "Unknown"),
                "position":       stats.get("games", {}).get("position", "?"),
                "rating":         float(rating),
                "goals":          stats.get("goals", {}).get("total") or 0,
                "assists":        stats.get("goals", {}).get("assists") or 0,
                "minutes_played": stats.get("games", {}).get("minutes") or 0,
                "appearances":    stats.get("games", {}).get("appearences") or 0,
            })

        players.sort(key=lambda x: x["rating"], reverse=True)
        top = players[:top_n]

        if top:
            avg_rating = sum(p["rating"] for p in top) / len(top)
            log.success(f"Spieler-Ratings {team_name}: Ø{avg_rating:.2f} (Top-{len(top)} Spieler)")

        return top

    def get_requests_remaining(self) -> int:
        """Prüft wie viele API-Calls heute noch übrig sind."""
        if not self.api_key:
            return 0
        data = self._get("status")
        if data:
            remaining = data.get("response", {}).get("requests", {}).get("remaining", 0)
            limit     = data.get("response", {}).get("requests", {}).get("limit_day", 100)
            log.info(f"api-football.com: {remaining}/{limit} Requests heute noch verfügbar")
            return remaining
        return 0

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_team_id(self, team_name: str, league_id: int, season: int) -> int | None:
        """Sucht die api-football Team-ID via fuzzy matching. Cached."""
        cache_key = f"{team_name}_{league_id}_{season}"
        if cache_key in _team_id_cache:
            return _team_id_cache[cache_key]

        data = self._get("teams", params={"league": league_id, "season": season})
        if not data or not data.get("response"):
            return None

        teams = data["response"]
        best_score = 0.0
        best_id    = None

        for t in teams:
            tm_name = t["team"]["name"]
            score   = difflib.SequenceMatcher(
                None, team_name.lower(), tm_name.lower()
            ).ratio()
            if score > best_score:
                best_score = score
                best_id    = t["team"]["id"]

        if best_score >= 0.5 and best_id:
            log.info(f"Team-Match: '{team_name}' → ID {best_id} (score={best_score:.2f})")
            _team_id_cache[cache_key] = best_id
            return best_id

        log.warning(f"Kein Team-Match für '{team_name}' in Liga {league_id}")
        return None

    def _get(self, endpoint: str, params: dict = None) -> dict | None:
        """HTTP GET mit Rate-Limit Schutz."""
        if not self.api_key:
            return None

        url = f"{BASE_URL}/{endpoint}"
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=10)
            self._request_count += 1

            # Log remaining requests
            remaining = resp.headers.get("x-ratelimit-requests-remaining", "?")
            log.debug(f"api-football [{endpoint}] → {resp.status_code} (remaining: {remaining})")

            if remaining != "?" and str(remaining).isdigit() and int(remaining) < 10:
                log.warning(f"api-football.com: Nur noch {remaining} Requests heute!")

            if resp.status_code == 401:
                log.error("api-football.com: Ungültiger API-Key (401)")
                return None
            if resp.status_code == 429:
                log.error("api-football.com: Rate Limit erreicht (429)")
                return None

            resp.raise_for_status()
            data = resp.json()

            # Check for API-level errors
            errors = data.get("errors", {})
            if errors:
                log.error(f"api-football.com API-Fehler: {errors}")
                return None

            results = data.get("results", 0)
            log.info(f"api-football [{endpoint}] → {results} Ergebnisse")

            time.sleep(0.3)
            return data

        except requests.RequestException as e:
            log.error(f"api-football.com Verbindungsfehler ({endpoint}): {e}")
            return None

    @staticmethod
    def _unavailable(team_name: str) -> dict:
        return {
            "team_name": team_name, "team_id": None, "last_5": [],
            "form_ppg": 1.5, "goals_scored_avg": 1.3, "goals_conceded_avg": 1.3,
            "source": "unavailable",
        }