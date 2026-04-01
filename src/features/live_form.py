"""
Live Form Calculator
Kombiniert api-football.com Live-Form mit historischen Daten.

Strategie:
  - Letzte 5 Spiele: api-football.com (live, tagesaktuell)
  - Angriffs/Abwehrstärke: Poisson-Modell aus historischen Daten (854 Spiele)
  - Spieler-Ratings: api-football.com → Anpassung des Lambda

Der Lambda-Anpassungsfaktor kombiniert:
  1. Aktuelle Form-PPG vs. historischer Durchschnitt
  2. Spieler-Rating-Faktor (Ø-Rating der Top-Spieler)
  3. Verletzungs-Impact (von Transfermarkt)
  4. Wetter-Impact

Usage:
    calc = LiveFormCalculator()
    adjustment = calc.get_lambda_adjustment("Hamburger SV", league="BL1")
    # adjustment = {"attack_factor": 0.92, "form_ppg": 1.8, ...}
"""

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

log = get_logger(__name__)

# Referenz-PPG aus historischen Daten (Bundesliga-Durchschnitt)
LEAGUE_AVG_PPG = 1.5   # entspricht ~50% Siegquote

# Spieler-Rating-Referenz: 7.0 ist Bundesliga-Durchschnitt
RATING_BASELINE = 7.0
RATING_SENSITIVITY = 0.15   # ±0.15 Lambda pro Rating-Punkt über/unter Baseline


class LiveFormCalculator:
    """
    Berechnet Lambda-Anpassungsfaktoren basierend auf Live-Daten.
    Wird VOR der Monte Carlo Simulation aufgerufen.
    """

    def __init__(self):
        pass  # Nutzt lokale DB — kein externer API-Call nötig

    def get_lambda_adjustment(
        self,
        team_name: str,
        league: str = "BL1",
        injury_impact: float = 0.0,
        features_df=None,
    ) -> dict:
        """
        Berechnet den Gesamt-Lambda-Faktor aus unserer eigenen DB.
        Kein externer API-Call — funktioniert immer.

        Benutzt die letzten 5 Spiele aus features_df (oder lädt sie aus CSV).
        """
        import pandas as pd
        import numpy as np
        from pathlib import Path

        # Features laden falls nicht übergeben
        if features_df is None:
            csv_path = Path(f"data/processed/features_{league}.csv")
            if csv_path.exists():
                features_df = pd.read_csv(csv_path, parse_dates=["date"])
            else:
                return self._fallback(team_name)

        # Letzte 5 Spiele des Teams aus DB holen
        home_games = features_df[features_df["home_team"] == team_name].copy()
        away_games = features_df[features_df["away_team"] == team_name].copy()

        home_games["pts"] = home_games["result"].map({"H": 3, "D": 1, "A": 0})
        home_games["scored"]   = home_games["home_goals"]
        home_games["conceded"] = home_games["away_goals"]
        home_games["is_home"]  = True

        away_games["pts"] = away_games["result"].map({"A": 3, "D": 1, "H": 0})
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

        # Gewichtete PPG
        weights = np.array([1.0, 0.85, 0.70, 0.55, 0.40][:len(all_games)])
        weights = weights / weights.sum()
        form_ppg = float(np.dot(all_games["pts"].values, weights))
        goals_scored_avg   = float(all_games["scored"].mean())
        goals_conceded_avg = float(all_games["conceded"].mean())

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

        # Form-Faktor berechnen
        form_factor = 1.0 + (form_ppg - LEAGUE_AVG_PPG) / LEAGUE_AVG_PPG * 0.10
        form_factor = round(max(0.80, min(1.20, form_factor)), 3)
        attack_factor = form_factor  # Nur Form, kein Rating-Faktor

        result = {
            "attack_factor":      attack_factor,
            "form_ppg":           round(form_ppg, 3),
            "goals_scored_avg":   round(goals_scored_avg, 2),
            "goals_conceded_avg": round(goals_conceded_avg, 2),
            "avg_player_rating":  None,
            "top_players":        [],
            "source":             "local-db",
            "last_5":             last_5,
            "breakdown": {
                "form_factor":  form_factor,
                "rating_factor": 1.0,
                "injury_factor": round(1.0 - min(injury_impact / 100 * 0.5, 0.30), 3),
                "combined":     attack_factor,
            }
        }

        log.info(
            f"Lambda-Faktor {team_name}: {attack_factor:.3f} "
            f"(Form PPG: {form_ppg:.2f}, letzte {len(all_games)} Spiele aus DB)"
        )
        return result

    def _fallback(self, team_name: str) -> dict:
        return {
            "attack_factor": 1.0, "form_ppg": 1.5,
            "goals_scored_avg": 1.3, "goals_conceded_avg": 1.3,
            "avg_player_rating": None, "top_players": [],
            "source": "fallback", "last_5": [],
            "breakdown": {"form_factor": 1.0, "rating_factor": 1.0,
                         "injury_factor": 1.0, "combined": 1.0},
        }

    def print_summary(self, team_name: str, adjustment: dict) -> None:
        """Gibt eine lesbare Zusammenfassung im Terminal aus."""
        src = adjustment["source"]
        print(f"\n{'─'*60}")
        print(f"  LIVE-FORM: {team_name.upper()} [{src}]")
        print(f"{'─'*60}")
        print(f"  Aktuelle PPG:      {adjustment['form_ppg']:.2f} / 3.00")
        print(f"  Tore geschossen:   Ø {adjustment['goals_scored_avg']:.2f}/Spiel")
        print(f"  Tore kassiert:     Ø {adjustment['goals_conceded_avg']:.2f}/Spiel")

        if adjustment.get("avg_player_rating"):
            print(f"  Ø Spieler-Rating:  {adjustment['avg_player_rating']:.2f} / 10")

        if adjustment["top_players"]:
            print(f"  Top-Spieler:")
            for p in adjustment["top_players"][:3]:
                print(f"    {p['name']:<22} Rating: {p['rating']:.1f} | "
                      f"{p['goals']}G {p['assists']}A")

        b = adjustment["breakdown"]
        # Faktor 1.05 = +5% Änderung, 1.0 = keine Änderung (0%)
        form_pct   = (b['form_factor']   - 1.0) * 100
        rating_pct = (b['rating_factor'] - 1.0) * 100
        total_pct  = (b['combined']      - 1.0) * 100
        print(f"\n  Lambda-Anpassung:")
        print(f"    Form-Faktor:   {form_pct:+.1f}%")
        print(f"    Rating-Faktor: {rating_pct:+.1f}%")
        print(f"    → Gesamt:      {total_pct:+.1f}%")
        print(f"{'─'*60}")

        if adjustment["last_5"]:
            print(f"  Letzte 5 Spiele:")
            for m in adjustment["last_5"]:
                loc    = "H" if m.get("is_home") else "A"
                pts    = m.get("points", 0)
                symbol = "W" if pts == 3 else ("D" if pts == 1 else "L")
                print(f"    {m['date']}  [{loc}] {m['home_team']} {m['home_goals']}:{m['away_goals']} {m['away_team']}  {symbol}")