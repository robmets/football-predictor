"""
Feature Engineering — Extracts all model features from raw match data.

Features computed per match:
  - Home / Away form (last 5 games, weighted)
  - Goals scored / conceded averages
  - Head-to-head record
  - Home advantage coefficient
  - Days rest since last match
  - League position & points

Usage:
    from src.features.feature_builder import FeatureBuilder
    fb = FeatureBuilder(df_matches)
    features = fb.build_features()
"""

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

log = get_logger(__name__)

# Weights for form calculation: most recent game counts most
FORM_WEIGHTS = np.array([1.0, 0.85, 0.70, 0.55, 0.40])


class FeatureBuilder:
    """
    Transforms a raw matches DataFrame into a feature matrix
    ready for model training.

    Expected input columns:
        match_id, date, league, season, matchday,
        home_team, away_team, home_goals, away_goals, result
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df = self.df.sort_values("date").reset_index(drop=True)
        log.info(f"FeatureBuilder loaded {len(self.df)} matches")

    # ── Public ───────────────────────────────────────────────────────────────

    def build_features(self) -> pd.DataFrame:
        """
        Main entry point. Returns one row per match with all features attached.
        Call this to get the full feature matrix for training.
        """
        log.info("Building features...")
        rows = []

        for idx, match in self.df.iterrows():
            past = self.df[self.df["date"] < match["date"]]
            features = self._extract(match, past)
            rows.append(features)

        result_df = pd.DataFrame(rows)
        log.success(f"Built {len(result_df)} feature rows, {result_df.shape[1]} columns")
        return result_df

    # ── Core Extractors ──────────────────────────────────────────────────────

    def _extract(self, match: pd.Series, past: pd.DataFrame) -> dict:
        home = match["home_team"]
        away = match["away_team"]

        home_form   = self._team_form(home, past, n=5)
        away_form   = self._team_form(away, past, n=5)
        home_specific = self._team_form_specific(home, past, venue="home", n=5)
        away_specific = self._team_form_specific(away, past, venue="away", n=5)
        home_stats  = self._goal_stats(home, past, n=10)
        away_stats  = self._goal_stats(away, past, n=10)
        h2h         = self._head_to_head(home, away, past, n=6)
        home_adv    = self._home_advantage(home, past, n=20)
        rest        = self._days_rest(home, away, past, match["date"])
        standings   = self._standings(past, match.get("season"))

        home_pos = standings.get(home, {}).get("position", 10)
        away_pos = standings.get(away, {}).get("position", 10)
        home_pts = standings.get(home, {}).get("points", 0)
        away_pts = standings.get(away, {}).get("points", 0)

        return {
            # --- Identifiers ---
            "match_id":             match.get("match_id"),
            "date":                 match["date"],
            "season":               match.get("season"),
            "matchday":             match.get("matchday"),
            "home_team":            home,
            "away_team":            away,

            # --- Target variable ---
            "result":               match.get("result"),          # H / D / A
            "home_goals":           match.get("home_goals"),
            "away_goals":           match.get("away_goals"),

            # --- Form (weighted points per game, 0–3 scale) ---
            "home_form_ppg":        home_form["ppg"],
            "away_form_ppg":        away_form["ppg"],
            "form_diff":            home_form["ppg"] - away_form["ppg"],
            "home_form_ppg_specific": home_specific["ppg"],
            "away_form_ppg_specific": away_specific["ppg"],

            # --- Goals scored ---
            "home_goals_scored_avg":    home_stats["scored_avg"],
            "home_goals_conceded_avg":  home_stats["conceded_avg"],
            "away_goals_scored_avg":    away_stats["scored_avg"],
            "away_goals_conceded_avg":  away_stats["conceded_avg"],

            # --- Attack vs Defence matchup ---
            "home_attack_vs_away_def":  home_stats["scored_avg"] - away_stats["conceded_avg"],
            "away_attack_vs_home_def":  away_stats["scored_avg"] - home_stats["conceded_avg"],

            # --- Head-to-head ---
            "h2h_home_wins":        h2h["home_wins"],
            "h2h_draws":            h2h["draws"],
            "h2h_away_wins":        h2h["away_wins"],
            "h2h_home_win_rate":    h2h["home_win_rate"],
            "h2h_games_played":     h2h["games_played"],

            # --- Home advantage ---
            "home_win_rate_at_home": home_adv["home_win_rate"],
            "home_goals_home_avg":   home_adv["goals_home_avg"],

            # --- Rest days ---
            "home_days_rest":       rest["home_days_rest"],
            "away_days_rest":       rest["away_days_rest"],
            "rest_advantage":       rest["home_days_rest"] - rest["away_days_rest"],

            # --- League standings ---
            "home_league_position": home_pos,
            "away_league_position": away_pos,
            "position_diff":        away_pos - home_pos,   # positive = home team higher
            "home_points":          home_pts,
            "away_points":          away_pts,
            "points_diff":          home_pts - away_pts,
        }

    # ── Feature Helpers ──────────────────────────────────────────────────────

    def _team_form(self, team: str, past: pd.DataFrame, n: int = 5) -> dict:
        """
        Weighted points-per-game from last N matches (home or away).
        Returns 0s if not enough data.
        """
        home_games = past[past["home_team"] == team].copy()
        away_games = past[past["away_team"] == team].copy()

        home_games["pts"] = home_games["result"].map({"H": 3, "D": 1, "A": 0})
        away_games["pts"] = away_games["result"].map({"A": 3, "D": 1, "H": 0})

        all_games = pd.concat([
            home_games[["date", "pts"]],
            away_games[["date", "pts"]],
        ]).sort_values("date", ascending=False).head(n)

        if len(all_games) == 0:
            return {"ppg": 1.5, "games": 0}  # fallback = average

        pts   = all_games["pts"].values
        wts   = FORM_WEIGHTS[:len(pts)]
        wts   = wts / wts.sum()
        w_ppg = float(np.dot(pts, wts))

        return {"ppg": w_ppg, "games": len(all_games)}

    def _team_form_specific(self, team: str, past: pd.DataFrame, venue: str, n: int = 5) -> dict:
        """Spezifische Form: Nur Heim- oder nur Auswärtsspiele."""
        if venue == "home":
            games = past[past["home_team"] == team].copy()
            games["pts"] = games["result"].map({"H": 3, "D": 1, "A": 0})
        else:
            games = past[past["away_team"] == team].copy()
            games["pts"] = games["result"].map({"A": 3, "D": 1, "H": 0})

        all_games = games.sort_values("date", ascending=False).head(n)

        if len(all_games) == 0:
            return {"ppg": 1.5, "games": 0}  # Liga-Schnitt Fallback

        pts   = all_games["pts"].values
        wts   = FORM_WEIGHTS[:len(pts)]
        wts   = wts / wts.sum()
        w_ppg = float(np.dot(pts, wts))

        return {"ppg": w_ppg, "games": len(all_games)}

    def _goal_stats(self, team: str, past: pd.DataFrame, n: int = 10) -> dict:
        """Average goals scored and conceded over last N matches."""
        home_games = past[past["home_team"] == team][["date", "home_goals", "away_goals"]].copy()
        away_games = past[past["away_team"] == team][["date", "home_goals", "away_goals"]].copy()

        home_games.rename(columns={"home_goals": "scored", "away_goals": "conceded"}, inplace=True)
        away_games.rename(columns={"away_goals": "scored", "home_goals": "conceded"}, inplace=True)

        games = pd.concat([home_games, away_games]).sort_values("date", ascending=False).head(n)

        if len(games) == 0:
            return {"scored_avg": 1.3, "conceded_avg": 1.3}  # league avg fallback

        return {
            "scored_avg":   float(games["scored"].mean()),
            "conceded_avg": float(games["conceded"].mean()),
        }

    def _head_to_head(self, home: str, away: str, past: pd.DataFrame, n: int = 6) -> dict:
        """Direct encounter record between the two teams (last N meetings)."""
        h2h = past[
            ((past["home_team"] == home) & (past["away_team"] == away)) |
            ((past["home_team"] == away) & (past["away_team"] == home))
        ].sort_values("date", ascending=False).head(n)

        if len(h2h) == 0:
            return {"home_wins": 0, "draws": 0, "away_wins": 0,
                    "home_win_rate": 0.33, "games_played": 0}

        home_wins = int(((h2h["home_team"] == home) & (h2h["result"] == "H")).sum() +
                        ((h2h["away_team"] == home) & (h2h["result"] == "A")).sum())
        away_wins = int(((h2h["home_team"] == away) & (h2h["result"] == "H")).sum() +
                        ((h2h["away_team"] == away) & (h2h["result"] == "A")).sum())
        draws     = int(len(h2h) - home_wins - away_wins)

        return {
            "home_wins":      home_wins,
            "draws":          draws,
            "away_wins":      away_wins,
            "home_win_rate":  round(home_wins / len(h2h), 3),
            "games_played":   len(h2h),
        }

    def _home_advantage(self, team: str, past: pd.DataFrame, n: int = 20) -> dict:
        """Win rate and avg goals when playing at home."""
        home_games = past[past["home_team"] == team].sort_values("date", ascending=False).head(n)

        if len(home_games) == 0:
            return {"home_win_rate": 0.45, "goals_home_avg": 1.5}

        win_rate  = float((home_games["result"] == "H").mean())
        goals_avg = float(home_games["home_goals"].mean())

        return {"home_win_rate": win_rate, "goals_home_avg": goals_avg}

    def _days_rest(self, home: str, away: str, past: pd.DataFrame, match_date) -> dict:
        """Days since last match for each team — more rest = fresher legs."""
        def last_game_date(team):
            games = past[(past["home_team"] == team) | (past["away_team"] == team)]
            if len(games) == 0:
                return None
            return games["date"].max()

        def rest_days(team):
            last = last_game_date(team)
            if last is None:
                return 7  # assume normal rest if no data
            return int((match_date - last).days)

        return {
            "home_days_rest": rest_days(home),
            "away_days_rest": rest_days(away),
        }

    def _standings(self, past: pd.DataFrame, season: str) -> dict:
        """
        Compute live standings from past matches within the same season.
        Returns dict: team → {points, gd, position}
        """
        season_matches = past[past["season"] == season] if season else past

        if len(season_matches) == 0:
            return {}

        teams = set(season_matches["home_team"]) | set(season_matches["away_team"])
        table = {}

        for team in teams:
            home = season_matches[season_matches["home_team"] == team]
            away = season_matches[season_matches["away_team"] == team]

            pts  = (home["result"] == "H").sum() * 3 + (home["result"] == "D").sum()
            pts += (away["result"] == "A").sum() * 3 + (away["result"] == "D").sum()

            gf = home["home_goals"].sum() + away["away_goals"].sum()
            ga = home["away_goals"].sum() + away["home_goals"].sum()

            table[team] = {"points": int(pts), "gd": int(gf - ga), "gf": int(gf)}

        # Sort by points → goal difference → goals scored
        sorted_teams = sorted(
            table.items(),
            key=lambda x: (x[1]["points"], x[1]["gd"], x[1]["gf"]),
            reverse=True
        )
        for pos, (team, _) in enumerate(sorted_teams, 1):
            table[team]["position"] = pos

        return table