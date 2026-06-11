"""
Data Collector — football-data.org API
Fetches matches, teams and standings for supported leagues.

Usage:
    collector = FootballDataCollector()
    collector.fetch_matches(league="BL1", seasons=5)
"""

import json
import os
from pathlib import Path
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

    # football-data season codes vs. start year — dynamisch berechnet
    # Saison 2025/26 startet im Juli 2025 → ab Juli gilt das aktuelle Jahr
    from datetime import date as _date
    CURRENT_SEASON = _date.today().year if _date.today().month >= 7 else _date.today().year - 1

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

        # Turniere (WC/EC) laufen im Kalenderjahr — Saison = Turnierjahr,
        # nicht das Juli-Juli-Schema des Klubfußballs
        base_season = (
            date.today().year if league in config.NEUTRAL_VENUE_LEAGUES
            else self.CURRENT_SEASON
        )

        for offset in range(seasons):
            season = base_season - offset
            log.info(f"Fetching {config.SUPPORTED_LEAGUES[league]} — season {season}...")

            try:
                data = self._get(f"competitions/{league}/matches", params={"season": season})
            except requests.HTTPError as e:
                log.error(f"Failed fetching season {season}: {e}")
                continue

            for m in data.get("matches", []):
                if m["status"] != "FINISHED":
                    continue

                # API-Lag: Spiel als FINISHED markiert, Ergebnis aber noch nicht
                # eingetragen → überspringen, kommt beim nächsten Fetch nach
                if m["score"]["fullTime"]["home"] is None or m["score"]["fullTime"]["away"] is None:
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
                    stage        = m.get("stage"),
                    group_name   = m.get("group"),
                )

                # Upsert: neu anlegen, oder Ergebnis nachtragen falls es beim
                # ersten Fetch noch fehlte (API-Lag direkt nach Abpfiff)
                existing = session.query(Match).filter_by(api_id=match.api_id).first()
                if not existing:
                    session.add(match)
                    all_matches.append(m)
                elif existing.home_goals is None or existing.away_goals is None:
                    existing.home_goals = match.home_goals
                    existing.away_goals = match.away_goals
                    existing.status     = match.status
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

    def get_wc_group_standings(self, season: str = None) -> dict:
        """
        Berechnet WC-Gruppenstandings aus der lokalen DB.
        Gibt dict zurück: {team_name: {position, points, gd, group}} — Position innerhalb der Gruppe.
        """
        from src.utils.database import get_session as _gs, Match as _M, Team as _T
        from sqlalchemy import func as _func
        sess = _gs()
        try:
            # Ohne explizite Saison: nur das AKTUELLSTE Turnier werten —
            # Gruppennamen (GROUP_A …) kollidieren sonst über Jahrzehnte hinweg
            if not season:
                season = sess.query(_func.max(_M.season)).filter(
                    _M.league == "WC", _M.stage == "GROUP_STAGE"
                ).scalar()

            q = sess.query(_M).filter(_M.league == "WC", _M.status == "FINISHED",
                                      _M.stage == "GROUP_STAGE",
                                      _M.home_goals.isnot(None),
                                      _M.away_goals.isnot(None))
            if season:
                q = q.filter(_M.season == str(season))
            matches = q.all()

            tmap = {t.api_id: t.name for t in sess.query(_T).filter(_T.league == "WC").all()}
        finally:
            sess.close()

        if not matches:
            return {}

        # Build standings per group
        group_table: dict[str, dict[str, dict]] = {}
        for m in matches:
            h = tmap.get(m.home_team_id, f"ID:{m.home_team_id}")
            a = tmap.get(m.away_team_id, f"ID:{m.away_team_id}")
            grp = m.group_name or "GROUP_?"

            for team in (h, a):
                if grp not in group_table:
                    group_table[grp] = {}
                if team not in group_table[grp]:
                    group_table[grp][team] = {"points": 0, "gd": 0, "gf": 0, "group": grp}

            hg, ag = m.home_goals or 0, m.away_goals or 0
            # Home team
            group_table[grp][h]["gf"] += hg
            group_table[grp][h]["gd"] += hg - ag
            if hg > ag:
                group_table[grp][h]["points"] += 3
            elif hg == ag:
                group_table[grp][h]["points"] += 1
                group_table[grp][a]["points"] += 1
            else:
                group_table[grp][a]["points"] += 3
            # Away team
            group_table[grp][a]["gf"] += ag
            group_table[grp][a]["gd"] += ag - hg

        # Flatten with position within group
        result = {}
        for grp, teams in group_table.items():
            sorted_teams = sorted(
                teams.items(),
                key=lambda x: (x[1]["points"], x[1]["gd"], x[1]["gf"]),
                reverse=True,
            )
            for pos, (team, stats) in enumerate(sorted_teams, 1):
                result[team] = {**stats, "position": pos}

        return result

    def get_live_standings(self, league_code: str) -> dict:
        """
        Holt die aktuelle Tabelle einer Liga.
        Nutzt einen lokalen Cache (1x am Tag abrufen schont die API).
        WC: berechnet Gruppenstandings aus lokaler DB.
        """
        import json
        from pathlib import Path
        from datetime import datetime

        if league_code not in config.SUPPORTED_LEAGUES:
            return {}

        # Cache-Datei für heute
        today_str = datetime.now().strftime("%Y-%m-%d")
        cache_dir = Path("data/cache/standings")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"standings_{league_code}_{today_str}.json"

        # 1. Haben wir die Tabelle heute schon geladen?
        if cache_file.exists():
            log.info(f"Lade Tabelle für {league_code} aus lokalem Tages-Cache...")
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        # 2. Wenn nein: API anrufen! (Wir nutzen dein schlaues self._get)
        log.info(f"API Call: Lade LIVE Tabelle für {league_code} herunter...")

        try:
            data = self._get(f"competitions/{league_code}/standings")

            # "TOTAL"-Tabellen extrahieren. Bei Turnieren (WC) liefert die API
            # eine Tabelle PRO GRUPPE — wir merken uns die Gruppe und die
            # Position innerhalb der Gruppe.
            standings_data = {}
            for standing in data.get("standings", []):
                if standing.get("type") != "TOTAL":
                    continue
                group = standing.get("group")  # z.B. "GROUP_A" oder None bei Ligen
                if group:
                    group = group.upper().replace("GROUP ", "GROUP_")
                for row in standing.get("table", []):
                    team_name = row.get("team", {}).get("name")
                    entry = {
                        "position": row.get("position"),
                        "points": row.get("points"),
                        "playedGames": row.get("playedGames"),
                        "goalDifference": row.get("goalDifference"),
                    }
                    if group:
                        entry["group"] = group
                        entry["gd"] = row.get("goalDifference", 0)
                    standings_data[team_name] = entry

            # 3. Im Cache speichern
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(standings_data, f, ensure_ascii=False, indent=4)

            return standings_data

        except Exception as e:
            log.error(f"Fehler beim Laden der Tabelle: {e}")
            # WC-Fallback: Gruppenstandings aus der lokalen DB berechnen
            if league_code == "WC":
                return self.get_wc_group_standings()
            return {}

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
                "stage":       m.get("stage"),
                "group_name":  m.get("group"),
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