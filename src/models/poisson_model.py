"""
Poisson Goal Model — predicts match outcomes using Dixon-Coles style
attack/defence strength estimation.

How it works:
  1. Estimate each team's attack & defence strength from historical data
  2. Compute expected goals (lambda) for home & away team
  3. Build a score matrix using the Poisson distribution
  4. Sum probabilities → P(Home Win), P(Draw), P(Away Win)

Usage:
    model = PoissonModel()
    model.fit(df_features)
    pred  = model.predict("Bayern München", "Borussia Dortmund")
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from src.utils.logger import get_logger

log = get_logger(__name__)

MAX_GOALS = 10   # score matrix goes from 0..MAX_GOALS x 0..MAX_GOALS


class PoissonModel:
    """
    Dixon-Coles inspired Poisson model.
    Fits attack/defence ratings per team + home advantage via MLE.
    """

    def __init__(self):
        self.attack:  dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.home_advantage: float = 1.35   # typical Bundesliga home boost
        self.avg_goals_home: float = 1.5
        self.avg_goals_away: float = 1.2
        self.teams: list[str] = []
        self._fitted = False

    # ── Fitting ─────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, min_games: int = 5) -> "PoissonModel":
        """
        Estimate team strengths from historical match data.

        Args:
            df: DataFrame with columns home_team, away_team,
                home_goals, away_goals
            min_games: teams with fewer games are assigned league average
        """
        df = df.dropna(subset=["home_goals", "away_goals"]).copy()
        df["home_goals"] = df["home_goals"].astype(int)
        df["away_goals"] = df["away_goals"].astype(int)

        self.teams = sorted(
            set(df["home_team"].unique()) | set(df["away_team"].unique())
        )
        n = len(self.teams)
        idx = {t: i for i, t in enumerate(self.teams)}

        # --- Global averages (used as baseline) ---
        self.avg_goals_home = df["home_goals"].mean()
        self.avg_goals_away = df["away_goals"].mean()

        log.info(f"Fitting Poisson model on {len(df)} matches, {n} teams")
        log.info(f"Avg goals — Home: {self.avg_goals_home:.2f}  Away: {self.avg_goals_away:.2f}")

        # --- MLE optimisation ---
        # Parameters: [attack_0..n-1, defence_0..n-1, home_advantage]
        x0 = np.concatenate([
            np.ones(n),           # attack strengths
            np.ones(n),           # defence strengths
            [self.home_advantage] # home advantage
        ])

        bounds = (
            [(0.1, 4.0)] * n +   # attack > 0
            [(0.1, 4.0)] * n +   # defence > 0
            [(1.0, 2.0),]        # home advantage
        )

        # --- Dixon-Coles Zeitgewichtung ---
        # Neuere Spiele zählen mehr: exp(-xi * Tage)
        # xi=0.003 → 1 Jahr altes Spiel hat Gewicht ~0.33

        from datetime import date as _today_date
        ref_date = pd.Timestamp(_today_date.today())
        if "date" in df.columns:
            df["_days_ago"] = (ref_date - pd.to_datetime(df["date"])).dt.days.clip(lower=0)
            weights = np.exp(-0.003 * df["_days_ago"].values)
        else:
            weights = np.ones(len(df))

        result = minimize(
            fun=self._neg_log_likelihood,
            x0=x0,
            args=(df, idx, n, weights),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-9},
        )

        if not result.success:
            log.warning(f"Optimisation warning: {result.message}")

        params = result.x
        self.attack        = {t: params[idx[t]]     for t in self.teams}
        self.defence       = {t: params[n + idx[t]] for t in self.teams}
        self.home_advantage = float(params[2 * n])

        self._fitted = True
        log.success(
            f"Model fitted — home advantage: {self.home_advantage:.3f}  "
            f"Best attack: {max(self.attack, key=self.attack.get)}  "
            f"Best defence: {min(self.defence, key=self.defence.get)}"
        )
        return self

    @staticmethod
    def _neg_log_likelihood(params, df, idx, n, weights=None):
        """Negative log-likelihood for home/away goals under Poisson.
        weights: array of per-match time decay weights (Dixon-Coles).
        """
        attack   = params[:n]
        defence  = params[n:2*n]
        home_adv = params[2*n]

        log_lik = 0.0
        for i, (_, row) in enumerate(df.iterrows()):
            hi = idx.get(row["home_team"])
            ai = idx.get(row["away_team"])
            if hi is None or ai is None:
                continue

            lambda_h = max(attack[hi] * defence[ai] * home_adv, 1e-6)
            lambda_a = max(attack[ai] * defence[hi], 1e-6)

            w = weights[i] if weights is not None else 1.0
            log_lik += w * poisson.logpmf(int(row["home_goals"]), lambda_h)
            log_lik += w * poisson.logpmf(int(row["away_goals"]), lambda_a)

        return -log_lik

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict(self, home_team: str, away_team: str) -> dict:
        """
        Predict match outcome probabilities.

        Returns:
            {
                prob_home_win, prob_draw, prob_away_win,
                expected_home_goals, expected_away_goals,
                score_matrix: dict of (h_goals, a_goals) → probability
            }
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted yet. Call .fit() first.")

        lambda_h, lambda_a = self._expected_goals(home_team, away_team)

        # Build score probability matrix
        score_matrix = {}
        prob_home = 0.0
        prob_draw = 0.0
        prob_away = 0.0

        for h in range(MAX_GOALS + 1):
            for a in range(MAX_GOALS + 1):
                p = poisson.pmf(h, lambda_h) * poisson.pmf(a, lambda_a)
                score_matrix[(h, a)] = round(p, 6)
                if h > a:
                    prob_home += p
                elif h == a:
                    prob_draw += p
                else:
                    prob_away += p

        # Normalise (captures truncation at MAX_GOALS)
        total = prob_home + prob_draw + prob_away
        prob_home /= total
        prob_draw /= total
        prob_away /= total

        # Most likely exact scores (top 5)
        top_scores = sorted(score_matrix.items(), key=lambda x: -x[1])[:5]

        return {
            "home_team":           home_team,
            "away_team":           away_team,
            "expected_home_goals": round(lambda_h, 3),
            "expected_away_goals": round(lambda_a, 3),
            "prob_home_win":       round(prob_home, 4),
            "prob_draw":           round(prob_draw, 4),
            "prob_away_win":       round(prob_away, 4),
            "top_scores":          [
                {"score": f"{h}:{a}", "probability": round(p * 100, 2)}
                for (h, a), p in top_scores
            ],
            "score_matrix":        score_matrix,
        }

    def _expected_goals(self, home_team: str, away_team: str):
        """
        Compute lambda (expected goals) for both teams.
        Falls back to league average for unknown teams.
        """
        home_att = self.attack.get(home_team,  np.mean(list(self.attack.values()))  if self.attack  else 1.0)
        home_def = self.defence.get(home_team, np.mean(list(self.defence.values())) if self.defence else 1.0)
        away_att = self.attack.get(away_team,  np.mean(list(self.attack.values()))  if self.attack  else 1.0)
        away_def = self.defence.get(away_team, np.mean(list(self.defence.values())) if self.defence else 1.0)

        lambda_h = home_att * away_def * self.home_advantage
        lambda_a = away_att * home_def

        return lambda_h, lambda_a

    # ── Team Ratings ─────────────────────────────────────────────────────────

    def team_ratings(self) -> pd.DataFrame:
        """Return a DataFrame with all team ratings, sorted by overall strength."""
        if not self._fitted:
            raise RuntimeError("Model not fitted yet.")

        rows = []
        for team in self.teams:
            rows.append({
                "team":    team,
                "attack":  round(self.attack[team], 3),
                "defence": round(self.defence[team], 3),
                "overall": round(self.attack[team] / self.defence[team], 3),
            })

        return (
            pd.DataFrame(rows)
            .sort_values("overall", ascending=False)
            .reset_index(drop=True)
        )