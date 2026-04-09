"""
Odds Collector — The Odds API
Fetches live bookmaker odds and converts them to implied probabilities.

Usage:
    collector = OddsCollector()
    all_markets = collector.fetch_all_markets_cached(league="BL1")
"""

import requests
import pandas as pd
import difflib
from datetime import datetime, timedelta
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
    "EL":  "soccer_uefa_europa_league",
    "UECL": "soccer_uefa_europa_conference_league",
}

# Bookmakers to include (in priority order)
PREFERRED_BOOKMAKERS = [
    "bet365", "bwin", "unibet", "williamhill",
    "pinnacle", "betway", "draftkings",
]

# In-Memory Cache (Schont das API-Budget)
_ODDS_CACHE = {}  
CACHE_TTL_HOURS = 6


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

    # ── Caching ──────────────────────────────────────────────────────────────

    def _get_cache_key(self, league: str) -> str:
        # Simpler Ansatz: Caching pro Liga und aktuellem Datum
        today = datetime.now().strftime("%Y-%m-%d")
        return f"{league}_{today}"

    def fetch_all_markets_cached(self, league: str = "BL1") -> dict:
        """Holt Quoten aus dem Cache oder macht einen neuen API Call."""
        cache_key = self._get_cache_key(league)

        if cache_key in _ODDS_CACHE:
            cached = _ODDS_CACHE[cache_key]
            # Prüfen ob der Cache jünger als CACHE_TTL_HOURS ist
            if datetime.now() - cached['timestamp'] < timedelta(hours=CACHE_TTL_HOURS):
                log.info(f"Nutze Odds-Cache für {league}")
                return cached['data']

        # Cache miss → API Call
        data = self.fetch_all_markets(league)
        _ODDS_CACHE[cache_key] = {'timestamp': datetime.now(), 'data': data}
        return data

    # ── API Calls ────────────────────────────────────────────────────────────

    def fetch_all_markets(self, league: str = "BL1") -> dict:
        """
        Holt ALLE verfügbaren Märkte in einem Call:
        - h2h (1×2)
        - totals (Over/Under)
        - btts (Both Teams to Score)
        
        Returns dict of DataFrames: {'h2h': df, 'totals': df, 'btts': df}
        """
        empty_res = {'h2h': pd.DataFrame(), 'totals': pd.DataFrame(), 'btts': pd.DataFrame()}

        if not self.api_key:
            log.error("No ODDS_API_KEY configured.")
            return empty_res

        sport = LEAGUE_TO_SPORT.get(league)
        if not sport:
            log.error(f"No sport key for league {league}")
            return empty_res

        log.info(f"Fetching extended odds for {league} ({sport})...")

        try:
            resp = requests.get(
                f"{self.base_url}/sports/{sport}/odds",
                params={
                    "apiKey":  self.api_key,
                    "regions": "eu",
                    "markets": "h2h,totals",
                    "oddsFormat": "decimal",
                },
                timeout=15,
            )
            resp.raise_for_status()
        except requests.HTTPError as e:
            log.error(f"Odds API error: {e}")
            return empty_res

        data = resp.json()
        remaining = resp.headers.get("x-requests-remaining", "?")
        log.info(f"Odds API requests remaining: {remaining}")

        rows_h2h = []
        rows_totals = []
        rows_btts = []

        for match in data:
            home = match["home_team"]
            away = match["away_team"]
            commence = match["commence_time"][:10]

            for bm in match.get("bookmakers", []):
                if bm["key"] not in PREFERRED_BOOKMAKERS:
                    continue
                markets = {m["key"]: m for m in bm.get("markets", [])}

                # --- 1x2 (H2H) ---
                if "h2h" in markets:
                    outcome_map = {o["name"]: o["price"] for o in markets["h2h"]["outcomes"]}
                    odds_h = outcome_map.get(home)
                    odds_d = outcome_map.get("Draw")
                    odds_a = outcome_map.get(away)

                    if all([odds_h, odds_d, odds_a]):
                        implied_h = 1 / odds_h
                        implied_d = 1 / odds_d
                        implied_a = 1 / odds_a
                        total     = implied_h + implied_d + implied_a
                        margin    = round((total - 1) * 100, 2)

                        rows_h2h.append({
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

                # --- Over/Under (Totals) ---
                if "totals" in markets:
                    point_groups = {}
                    # Gruppiere Over/Under nach dem "point" (z.B. 1.5, 2.5)
                    for o in markets["totals"]["outcomes"]:
                        pt = o.get("point", 2.5)
                        if pt not in point_groups:
                            point_groups[pt] = {}
                        point_groups[pt][o["name"]] = o["price"]

                    for pt, odds_dict in point_groups.items():
                        odds_over = odds_dict.get("Over")
                        odds_under = odds_dict.get("Under")
                        
                        if odds_over and odds_under:
                            imp_o = 1 / odds_over
                            imp_u = 1 / odds_under
                            total = imp_o + imp_u
                            margin = round((total - 1) * 100, 2)

                            rows_totals.append({
                                "home_team": home,
                                "away_team": away,
                                "commence": commence,
                                "bookmaker": bm["title"],
                                "point": pt,
                                "odds_over": odds_over,
                                "odds_under": odds_under,
                                "implied_over": round(imp_o / total, 4),
                                "implied_under": round(imp_u / total, 4),
                                "margin_pct": margin,
                            })

                # --- BTTS (Both Teams To Score) ---
                if "btts" in markets:
                    outcome_map = {o["name"]: o["price"] for o in markets["btts"]["outcomes"]}
                    odds_yes = outcome_map.get("Yes")
                    odds_no = outcome_map.get("No")
                    
                    if odds_yes and odds_no:
                        imp_y = 1 / odds_yes
                        imp_n = 1 / odds_no
                        total = imp_y + imp_n
                        margin = round((total - 1) * 100, 2)

                        rows_btts.append({
                            "home_team": home,
                            "away_team": away,
                            "commence": commence,
                            "bookmaker": bm["title"],
                            "odds_yes": odds_yes,
                            "odds_no": odds_no,
                            "implied_yes": round(imp_y / total, 4),
                            "implied_no": round(imp_n / total, 4),
                            "margin_pct": margin,
                        })

        res = {
            'h2h': pd.DataFrame(rows_h2h),
            'totals': pd.DataFrame(rows_totals),
            'btts': pd.DataFrame(rows_btts)
        }
        log.success(f"Fetched markets: {len(res['h2h'])} H2H, {len(res['totals'])} Totals, {len(res['btts'])} BTTS")
        return res

    # ── Legacy Wrapper ───────────────────────────────────────────────────────
    # Behält die Abwärtskompatibilität für das bestehende Dashboard (Value Bets Tab)
    
    def fetch_odds(self, league: str = "BL1") -> pd.DataFrame:
        """Wrapper für alten Code: Liefert nur das H2H DataFrame."""
        return self.fetch_all_markets(league).get('h2h', pd.DataFrame())

    def get_consensus_odds(self, league: str = "BL1") -> pd.DataFrame:
        """Wrapper für alten Code: Berechnet den Consensus nur für den H2H Markt."""
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

    def find_match(
        self,
        consensus_df: "pd.DataFrame",
        home_team: str,
        away_team: str,
        threshold: float = 0.55,
    ) -> "pd.Series | None":
        """Unveränderte Fuzzy-Matching Logik."""
        if consensus_df.empty:
            return None

        best_score = 0.0
        best_row   = None

        for _, row in consensus_df.iterrows():
            score_h = difflib.SequenceMatcher(
                None, home_team.lower(), row["home_team"].lower()
            ).ratio()
            score_a = difflib.SequenceMatcher(
                None, away_team.lower(), row["away_team"].lower()
            ).ratio()
            combined = (score_h + score_a) / 2

            if combined > best_score:
                best_score = combined
                best_row   = row

        if best_score >= threshold:
            log.info(
                f"Match found: '{home_team}' vs '{away_team}' → "
                f"'{best_row['home_team']}' vs '{best_row['away_team']}' "
                f"(score={best_score:.2f})"
            )
            return best_row

        log.warning(
            f"No odds match for '{home_team}' vs '{away_team}' "
            f"(best score={best_score:.2f} < {threshold})"
        )
        return None