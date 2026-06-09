"""
Live Form Calculator
Berechnet Lambda-Anpassungsfaktoren aus lokaler DB + Sofascore.

Verbesserungen:
  1. Cross-Competition Form: Für CL-Spiele werden auch Liga-Daten berücksichtigt
  2. Specific Form Blend: Heim/Auswärts-spezifische Form fließt zu 60% ein
  3. H2H-Faktor: Historische Duelle adjustieren Lambda (±8% max)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import get_logger

log = get_logger(__name__)

LEAGUE_AVG_PPG   = 1.5
SOFASCORE_BASELINE    = 50.0
SOFASCORE_SENSITIVITY = 0.01


class LiveFormCalculator:

    def __init__(self):
        pass

    # ── Cross-Competition Datenbasis ─────────────────────────────────────────

    def _load_cross_competition_features(self, team_name: str, primary_df: pd.DataFrame, primary_league: str) -> pd.DataFrame:
        """
        Lädt Spiele des Teams aus ALLEN verfügbaren Ligen.
        Wichtig für: CL-Teams die auch Liga-Form berücksichtigen sollen,
        oder H2H von Teams die sich in mehreren Wettbewerben begegnen.
        """
        all_dfs = []
        if primary_df is not None and not primary_df.empty:
            all_dfs.append(primary_df)

        for csv in sorted(Path("data/processed").glob("features_*.csv")):
            league_code = csv.stem.replace("features_", "")
            if league_code == primary_league:
                continue  # schon drin
            try:
                df_other = pd.read_csv(csv, parse_dates=["date"])
                if (team_name in df_other["home_team"].values or
                        team_name in df_other["away_team"].values):
                    all_dfs.append(df_other)
            except Exception:
                pass

        if not all_dfs:
            return primary_df if primary_df is not None else pd.DataFrame()

        combined = pd.concat(all_dfs)
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        combined = (
            combined
            .drop_duplicates(subset=["match_id"])
            .sort_values("date")
            .reset_index(drop=True)
        )
        return combined

    # ── H2H-Faktor ───────────────────────────────────────────────────────────

    def get_h2h_factor(self, home_team: str, away_team: str, league: str, features_df=None) -> dict:
        """
        Berechnet H2H-Faktor aus historischen Duellen — über alle Wettbewerbe.
        
        Für Atletico vs Barcelona in der CL werden auch alle La Liga Duelle
        der letzten Jahre mitgezählt.

        Returns dict:
          factor (float): Lambda-Anpassung für Heimteam (1.0 = neutral)
          games (int): Anzahl gefundener Duelle
          home_win_rate (float): Siegquote Heimteam in H2H
        """
        # Daten aus allen Wettbewerben laden (cross-competition!)
        combined = self._load_cross_competition_features(home_team, features_df, league)

        if combined.empty:
            return {"factor": 1.0, "games": 0, "home_win_rate": 0.33}

        h2h = combined[
            ((combined["home_team"] == home_team) & (combined["away_team"] == away_team)) |
            ((combined["home_team"] == away_team) & (combined["away_team"] == home_team))
        ].sort_values("date", ascending=False).head(8)

        if len(h2h) < 3:
            return {"factor": 1.0, "games": len(h2h), "home_win_rate": 0.33}

        home_wins = 0
        for _, row in h2h.iterrows():
            if row["home_team"] == home_team and row["result"] == "H":
                home_wins += 1
            elif row["home_team"] == away_team and row["result"] == "A":
                home_wins += 1

        home_win_rate = home_wins / len(h2h)
        # ±8% Lambda-Anpassung maximal
        factor = 1.0 + (home_win_rate - 0.33) * 0.12
        factor = round(max(0.92, min(1.08, factor)), 3)

        log.info(
            f"H2H {home_team} vs {away_team}: {home_wins}/{len(h2h)} Heimsiege "
            f"({home_win_rate:.0%}) → Faktor ×{factor:.3f} (aus {len(h2h)} Duellen in allen Wettbewerben)"
        )
        return {"factor": factor, "games": len(h2h), "home_win_rate": round(home_win_rate, 3)}

    # ── Lambda Adjustment ────────────────────────────────────────────────────

    def get_lambda_adjustment(
        self,
        team_name: str,
        league: str = "BL1",
        injury_impact: float = 0.0,
        features_df=None,
        sofascore_team_rating: float = None,
        is_home: bool = True,
    ) -> dict:
        """
        Berechnet Gesamt-Lambda-Faktor.
        Nutzt cross-competition Form (Liga + CL) und blendet spezifische H/A-Form ein.
        """
        # Primäre Features laden
        if features_df is None:
            csv_path = Path(f"data/processed/features_{league}.csv")
            if csv_path.exists():
                features_df = pd.read_csv(csv_path, parse_dates=["date"])
            else:
                return self._fallback(team_name)

        # Cross-Competition: Spiele aus allen Wettbewerben holen
        combined = self._load_cross_competition_features(team_name, features_df, league)

        home_games = combined[combined["home_team"] == team_name].copy()
        away_games = combined[combined["away_team"] == team_name].copy()

        home_games["pts"]      = home_games["result"].map({"H": 3, "D": 1, "A": 0})
        home_games["scored"]   = home_games["home_goals"]
        home_games["conceded"] = home_games["away_goals"]
        home_games["is_home"]  = True

        away_games["pts"]      = away_games["result"].map({"A": 3, "D": 1, "H": 0})
        away_games["scored"]   = away_games["away_goals"]
        away_games["conceded"] = away_games["home_goals"]
        away_games["is_home"]  = False

        all_games = pd.concat([
            home_games[["date", "home_team", "away_team", "home_goals", "away_goals",
                         "result", "pts", "scored", "conceded", "is_home"]],
            away_games[["date", "home_team", "away_team", "home_goals", "away_goals",
                         "result", "pts", "scored", "conceded", "is_home"]],
        ]).sort_values("date", ascending=False).head(5)

        if len(all_games) == 0:
            return self._fallback(team_name)

        # Gewichtete PPG (allgemein, letzte 5 Spiele)
        weights = np.array([1.0, 0.85, 0.70, 0.55, 0.40][:len(all_games)])
        weights = weights / weights.sum()
        form_ppg           = float(np.dot(all_games["pts"].values, weights))
        goals_scored_avg   = float(all_games["scored"].mean())
        goals_conceded_avg = float(all_games["conceded"].mean())

        # Spezifische Form: nur Heim oder nur Auswärts
        specific_games = home_games if is_home else away_games
        specific_games = specific_games.sort_values("date", ascending=False).head(5)
        if len(specific_games) >= 3:
            w_spec       = np.array([1.0, 0.85, 0.70, 0.55, 0.40][:len(specific_games)])
            w_spec       = w_spec / w_spec.sum()
            specific_ppg = float(np.dot(specific_games["pts"].values, w_spec))
        else:
            specific_ppg = form_ppg  # zu wenig spezifische Spiele → allgemeine Form

        # Specific Form Blend: 60% spezifisch + 40% allgemein (wenn genug Daten)
        if len(specific_games) >= 3:
            blended_ppg = 0.6 * specific_ppg + 0.4 * form_ppg
            log.info(
                f"Form-Blend {team_name} ({'Heim' if is_home else 'Auswärts'}): "
                f"{specific_ppg:.2f} spezifisch × 60% + {form_ppg:.2f} allgemein × 40% = {blended_ppg:.2f}"
            )
        else:
            blended_ppg = form_ppg

        # Form-Faktor aus geblendeter PPG
        form_factor = 1.0 + (blended_ppg - LEAGUE_AVG_PPG) / LEAGUE_AVG_PPG * 0.10
        form_factor = round(max(0.80, min(1.20, form_factor)), 3)

        # Sofascore Rating-Faktor
        if sofascore_team_rating is not None:
            rating_factor = 1.0 + (sofascore_team_rating - SOFASCORE_BASELINE) * SOFASCORE_SENSITIVITY
            rating_factor = round(max(0.85, min(1.15, rating_factor)), 3)
            log.info(
                f"Sofascore Rating-Faktor {team_name}: "
                f"{sofascore_team_rating:.1f} vs Basis {SOFASCORE_BASELINE} → ×{rating_factor:.3f}"
            )
        else:
            rating_factor = 1.0

        attack_factor = round(max(0.75, min(1.25, form_factor * rating_factor)), 3)

        # Last 5 für Anzeige
        last_5 = []
        for _, row in all_games.iterrows():
            last_5.append({
                "date":       str(row["date"])[:10],
                "home_team":  row["home_team"],
                "away_team":  row["away_team"],
                "home_goals": int(row["home_goals"]) if pd.notna(row["home_goals"]) else 0,
                "away_goals": int(row["away_goals"]) if pd.notna(row["away_goals"]) else 0,
                "result":     row["result"],
                "is_home":    bool(row["is_home"]),
                "points":     int(row["pts"]),
            })

        result = {
            "attack_factor":      attack_factor,
            "form_ppg":           round(form_ppg, 3),
            "specific_ppg":       round(specific_ppg, 3),
            "blended_ppg":        round(blended_ppg, 3),
            "goals_scored_avg":   round(goals_scored_avg, 2),
            "goals_conceded_avg": round(goals_conceded_avg, 2),
            "avg_player_rating":  sofascore_team_rating,
            "top_players":        [],
            "source":             "local-db",
            "last_5":             last_5,
            "breakdown": {
                "form_factor":   form_factor,
                "rating_factor": rating_factor,
                "injury_factor": round(1.0 - min(injury_impact / 100 * 0.5, 0.30), 3),
                "combined":      attack_factor,
            }
        }

        log.info(
            f"Lambda-Faktor {team_name}: {attack_factor:.3f} "
            f"(Form PPG: {form_ppg:.2f}, letzte {len(all_games)} Spiele aus DB, cross-competition)"
        )
        return result

    def _fallback(self, team_name: str) -> dict:
        return {
            "attack_factor": 1.0, "form_ppg": 1.5, "specific_ppg": 1.5, "blended_ppg": 1.5,
            "goals_scored_avg": 1.3, "goals_conceded_avg": 1.3,
            "avg_player_rating": None, "top_players": [],
            "source": "fallback", "last_5": [],
            "breakdown": {"form_factor": 1.0, "rating_factor": 1.0,
                          "injury_factor": 1.0, "combined": 1.0},
        }

    def print_summary(self, team_name: str, adjustment: dict) -> None:
        src = adjustment["source"]
        print(f"\n{'─'*60}")
        print(f"  LIVE-FORM: {team_name.upper()} [{src}]")
        print(f"{'─'*60}")
        print(f"  Aktuelle PPG:      {adjustment['form_ppg']:.2f} / 3.00")
        if adjustment.get("blended_ppg") and adjustment["blended_ppg"] != adjustment["form_ppg"]:
            print(f"  Blended PPG:       {adjustment['blended_ppg']:.2f} (spez. + allg.)")
        print(f"  Tore geschossen:   Ø {adjustment['goals_scored_avg']:.2f}/Spiel")
        print(f"  Tore kassiert:     Ø {adjustment['goals_conceded_avg']:.2f}/Spiel")
        if adjustment.get("avg_player_rating"):
            print(f"  Ø Sofascore-Rating: {adjustment['avg_player_rating']:.1f} / 100")

        b = adjustment["breakdown"]
        form_pct   = (b['form_factor']   - 1.0) * 100
        rating_pct = (b['rating_factor'] - 1.0) * 100
        total_pct  = (b['combined']      - 1.0) * 100
        print(f"\n  Lambda-Anpassung:")
        print(f"    Form-Faktor:   {form_pct:+.1f}%")
        print(f"    Rating-Faktor: {rating_pct:+.1f}%")
        print(f"    → Gesamt:      {total_pct:+.1f}%")
        print(f"{'─'*60}")

        if adjustment["last_5"]:
            print(f"  Letzte 5 Spiele (alle Wettbewerbe):")
            for m in adjustment["last_5"]:
                loc    = "H" if m.get("is_home") else "A"
                pts    = m.get("points", 0)
                symbol = "W" if pts == 3 else ("D" if pts == 1 else "L")
                print(f"    {m['date']}  [{loc}] {m['home_team']} {m['home_goals']}:{m['away_goals']} {m['away_team']}  {symbol}")