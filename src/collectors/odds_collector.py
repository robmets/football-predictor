"""
Odds Collector — The Odds API
Fetches live bookmaker odds and converts them to implied probabilities.

Usage:
    collector = OddsCollector()
    odds = collector.fetch_odds(league="BL1")
"""

import requests
import pandas as pd
from src.utils.logger import get_logger
from config.config import config

log = get_logger(__name__)

# football-data league code → Odds API sport key
LEAGUE_TO_SPORT = {
    "BL1": "soccer_germany_bundesliga",
    "PL":  "soccer_epl",
    "PD":  "soccer_spain_la_liga",
    "SA":  "soccer_italy_serie_a",
    "FL1": "soccer_france_ligue_one",
    "CL":  "soccer_uefa_champs_league",
}

# Bookmakers to include (in priority order)
PREFERRED_BOOKMAKERS = [
    "bet365", "bwin", "unibet", "williamhill",
    "pinnacle", "betway", "draftkings",
]


class OddsCollector:
    """
    Fetches match odds from The Odds API and converts them
    to implied win probabilities (removing the bookmaker margin).
    """

    def __init__(self):
        self.api_key = config.ODDS_API_KEY
        self.base_url = config.ODDS_API_BASE_URL

        if not self.api_key:
            log.warning("ODDS_API_KEY not set — odds features disabled")

    # ── Public ───────────────────────────────────────────────────────────────

    def fetch_odds(self, league: str = "BL1") -> pd.DataFrame:
        """
        Fetch upcoming match odds for a league.

        Returns a DataFrame with columns:
            match_id, home_team, away_team, commence_time,
            bookmaker, odds_home, odds_draw, odds_away,
            implied_home, implied_draw, implied_away,
            margin (bookmaker's edge in %)
        """
        if not self.api_key:
            log.error("No ODDS_API_KEY configured.")
            return pd.DataFrame()

        sport = LEAGUE_TO_SPORT.get(league)
        if not sport:
            log.error(f"No sport key for league {league}")
            return pd.DataFrame()

        log.info(f"Fetching odds for {league} ({sport})...")

        try:
            resp = requests.get(
                f"{self.base_url}/sports/{sport}/odds",
                params={
                    "apiKey":  self.api_key,
                    "regions": "eu",
                    "markets": "h2h",
                    "oddsFormat": "decimal",
                },
                timeout=15,
            )
            resp.raise_for_status()
        except requests.HTTPError as e:
            log.error(f"Odds API error: {e}")
            return pd.DataFrame()

        data = resp.json()
        remaining = resp.headers.get("x-requests-remaining", "?")
        log.info(f"Odds API requests remaining: {remaining}")

        rows = []
        for match in data:
            home = match["home_team"]
            away = match["away_team"]
            commence = match["commence_time"][:10]

            for bm in match.get("bookmakers", []):
                if bm["key"] not in PREFERRED_BOOKMAKERS:
                    continue
                markets = {m["key"]: m for m in bm.get("markets", [])}
                h2h = markets.get("h2h")
                if not h2h:
                    continue

                outcome_map = {o["name"]: o["price"] for o in h2h["outcomes"]}
                odds_h = outcome_map.get(home)
                odds_d = outcome_map.get("Draw")
                odds_a = outcome_map.get(away)

                if not all([odds_h, odds_d, odds_a]):
                    continue

                implied_h = 1 / odds_h
                implied_d = 1 / odds_d
                implied_a = 1 / odds_a
                total     = implied_h + implied_d + implied_a
                margin    = round((total - 1) * 100, 2)

                # Remove margin → fair probabilities
                rows.append({
                    "home_team":    home,
                    "away_team":    away,
                    "commence":     commence,
                    "bookmaker":    bm["title"],
                    "odds_home":    odds_h,
                    "odds_draw":    odds_d,
                    "odds_away":    odds_a,
                    "implied_home": round(implied_h / total, 4),
                    "implied_draw": round(implied_d / total, 4),
                    "implied_away": round(implied_a / total, 4),
                    "margin_pct":   margin,
                })

        df = pd.DataFrame(rows)
        log.success(f"Fetched odds for {len(df)} bookmaker/match combos")
        return df

    def get_consensus_odds(self, league: str = "BL1") -> pd.DataFrame:
        """
        Average implied probabilities across all available bookmakers
        per match → consensus market view.
        """
        df = self.fetch_odds(league)
        if df.empty:
            return df

        consensus = (
            df.groupby(["home_team", "away_team", "commence"])
            .agg(
                market_home=("implied_home", "mean"),
                market_draw=("implied_draw", "mean"),
                market_away=("implied_away", "mean"),
                avg_margin=("margin_pct", "mean"),
                bookmakers=("bookmaker", "count"),
            )
            .reset_index()
        )
        return consensus