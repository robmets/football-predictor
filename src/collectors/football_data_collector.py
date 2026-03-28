"""
Data Collector — football-data.org API
Fetches matches, teams and standings for supported leagues.

Usage:
    collector = FootballDataCollector()
    collector.fetch_matches(league="BL1", seasons=5)
"""

import time
from datetime import datetime, date
import requests
import pandas as pd
from src.utils.logger import get_logger
from src.utils.database import get_session, Match, Team
from config.config import config

log = get_logger(__name__)


class FootballDataCollector:
    """Fetches data from football-data.org API (v4)."""

    BASE_URL = config.FOOTBALL_DATA_BASE_URL
    HEADERS  = {"X-Auth-Token": config.FOOTBALL_DATA_API_KEY}

    # football-data season codes vs. start year
    CURRENT_SEASON = 2024

    def __init__(self):
        config.validate()

    # ── Private Helpers ──────────────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict = None) -> dict:
        """Make a rate-limited GET request."""
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.get(url, headers=self.HEADERS, params=params, timeout=15)

        if response.status_code == 429:
            log.warning("Rate limit hit — waiting 60s...")
            time.sleep(60)
            return self._get(endpoint, params)

        response.raise_for_status()
        time.sleep(0.5)  # be polite to the API
        return response.json()

    # ── Public Methods ───────────────────────────────────────────────────────

    def fetch_matches(self, league: str = "BL1", seasons: int = 5) -> pd.DataFrame:
        """
        Fetch historical matches for a league across N seasons.
        Saves to database and returns a DataFrame.
        """
        if league not in config.SUPPORTED_LEAGUES:
            raise ValueError(f"Unsupported league: {league}. Choose from {list(config.SUPPORTED_LEAGUES)}")

        all_matches = []
        session = get_session()

        for offset in range(seasons):
            season = self.CURRENT_SEASON - offset
            log.info(f"Fetching {config.SUPPORTED_LEAGUES[league]} — season {season}...")

            try:
                data = self._get(f"competitions/{league}/matches", params={"season": season})
            except requests.HTTPError as e:
                log.error(f"Failed fetching season {season}: {e}")
                continue

            for m in data.get("matches", []):
                if m["status"] != "FINISHED":
                    continue

                match = Match(
                    api_id       = m["id"],
                    league       = league,
                    season       = str(season),
                    matchday     = m.get("matchday"),
                    date         = date.fromisoformat(m["utcDate"][:10]),
                    home_team_id = m["homeTeam"]["id"],
                    away_team_id = m["awayTeam"]["id"],
                    home_goals   = m["score"]["fullTime"]["home"],
                    away_goals   = m["score"]["fullTime"]["away"],
                    status       = m["status"],
                )

                # Upsert: skip if already in DB
                existing = session.query(Match).filter_by(api_id=match.api_id).first()
                if not existing:
                    session.add(match)
                    all_matches.append(m)

        session.commit()
        session.close()

        log.success(f"Saved {len(all_matches)} new matches for {league}")
        return self._to_dataframe(all_matches)

    def fetch_teams(self, league: str = "BL1") -> list[dict]:
        """Fetch all teams in a competition and save to DB."""
        log.info(f"Fetching teams for {league}...")
        data = self._get(f"competitions/{league}/teams")
        session = get_session()

        teams = []
        for t in data.get("teams", []):
            team = Team(
                api_id     = t["id"],
                name       = t["name"],
                short_name = t.get("shortName"),
                league     = league,
                country    = t.get("area", {}).get("name"),
            )
            existing = session.query(Team).filter_by(api_id=team.api_id).first()
            if not existing:
                session.add(team)
            teams.append(t)

        session.commit()
        session.close()
        log.success(f"Saved {len(teams)} teams for {league}")
        return teams

    def fetch_upcoming(self, league: str = "BL1") -> pd.DataFrame:
        """Fetch next 10 scheduled matches."""
        log.info(f"Fetching upcoming matches for {league}...")
        data = self._get(
            f"competitions/{league}/matches",
            params={"status": "SCHEDULED", "limit": 10}
        )
        return pd.DataFrame([
            {
                "match_id":   m["id"],
                "date":       m["utcDate"][:10],
                "matchday":   m.get("matchday"),
                "home_team":  m["homeTeam"]["name"],
                "away_team":  m["awayTeam"]["name"],
            }
            for m in data.get("matches", [])
        ])

    # ── Private: Raw → DataFrame ─────────────────────────────────────────────

    @staticmethod
    def _to_dataframe(matches: list[dict]) -> pd.DataFrame:
        rows = []
        for m in matches:
            rows.append({
                "match_id":    m["id"],
                "date":        m["utcDate"][:10],
                "league":      m.get("competition", {}).get("code"),
                "season":      m.get("season", {}).get("startDate", "")[:4],
                "matchday":    m.get("matchday"),
                "home_team":   m["homeTeam"]["name"],
                "away_team":   m["awayTeam"]["name"],
                "home_goals":  m["score"]["fullTime"]["home"],
                "away_goals":  m["score"]["fullTime"]["away"],
                "result":      _get_result(
                                   m["score"]["fullTime"]["home"],
                                   m["score"]["fullTime"]["away"]
                               ),
            })
        return pd.DataFrame(rows)


def _get_result(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    elif home_goals < away_goals:
        return "A"
    return "D"
