"""
Monte Carlo Simulation Engine.
Runs N simulated matches using Poisson-distributed goal draws,
then aggregates results into probability distributions.

Usage:
    from src.simulation.monte_carlo import MonteCarloSimulator
    sim = MonteCarloSimulator(poisson_model)
    result = sim.simulate("Bayern München", "Borussia Dortmund", n=10000)
"""

import numpy as np
import pandas as pd
from collections import Counter
from src.utils.logger import get_logger

log = get_logger(__name__)


class MonteCarloSimulator:
    """
    Runs thousands of match simulations using the Poisson model's
    expected goals as the distribution parameter.
    """

    def __init__(self, model):
        """
        Args:
            model: Fitted PoissonModel instance
        """
        self.model = model

    def simulate(
        self,
        home_team: str,
        away_team: str,
        n: int = 10_000,
        seed: int = None,
    ) -> dict:
        """
        Simulate a match N times and return aggregated probabilities.

        Args:
            home_team: Name of the home team
            away_team: Name of the away team
            n:         Number of simulations (default 10,000)
            seed:      Random seed for reproducibility

        Returns:
            Full prediction dict with simulation stats
        """
        if seed is not None:
            np.random.seed(seed)

        lambda_h, lambda_a = self.model._expected_goals(home_team, away_team)
        log.info(
            f"Simulating {n:,} matches: {home_team} (λ={lambda_h:.2f}) "
            f"vs {away_team} (λ={lambda_a:.2f})"
        )

        # Draw goals from Poisson distribution for all simulations at once
        home_goals_sim = np.random.poisson(lambda_h, n)
        away_goals_sim = np.random.poisson(lambda_a, n)

        # Count outcomes
        home_wins = int(np.sum(home_goals_sim > away_goals_sim))
        draws     = int(np.sum(home_goals_sim == away_goals_sim))
        away_wins = int(np.sum(home_goals_sim < away_goals_sim))

        prob_home = home_wins / n
        prob_draw = draws     / n
        prob_away = away_wins / n

        # Score frequency distribution
        score_counts = Counter(zip(home_goals_sim, away_goals_sim))
        top_scores = [
            {
                "score":       f"{h}:{a}",
                "count":       count,
                "probability": round(count / n * 100, 2),
            }
            for (h, a), count in score_counts.most_common(8)
        ]

        # Goal distribution stats
        avg_home_goals = float(np.mean(home_goals_sim))
        avg_away_goals = float(np.mean(away_goals_sim))

        # Margin of victory distribution
        goal_diffs = home_goals_sim - away_goals_sim
        avg_margin  = float(np.mean(goal_diffs))
        std_margin  = float(np.std(goal_diffs))

        # Both teams score probability
        btts = float(np.mean((home_goals_sim > 0) & (away_goals_sim > 0)))

        # Over/Under 2.5 goals
        total_goals = home_goals_sim + away_goals_sim
        over_2_5    = float(np.mean(total_goals > 2.5))
        over_3_5    = float(np.mean(total_goals > 3.5))

        result = {
            "home_team":           home_team,
            "away_team":           away_team,
            "simulations":         n,
            "expected_home_goals": round(avg_home_goals, 3),
            "expected_away_goals": round(avg_away_goals, 3),

            # Core probabilities
            "prob_home_win":       round(prob_home, 4),
            "prob_draw":           round(prob_draw, 4),
            "prob_away_win":       round(prob_away, 4),

            # Additional markets
            "prob_btts":           round(btts, 4),
            "prob_over_2_5":       round(over_2_5, 4),
            "prob_over_3_5":       round(over_3_5, 4),
            "avg_goal_margin":     round(avg_margin, 3),
            "std_goal_margin":     round(std_margin, 3),

            # Most likely exact scores
            "top_scores":          top_scores,

            # Confidence: how dominant is the favourite?
            "favourite":           home_team if prob_home > prob_away else away_team,
            "favourite_prob":      round(max(prob_home, prob_away), 4),
            "confidence":          _confidence_label(max(prob_home, prob_away)),
        }

        log.success(
            f"Result: {home_team} {prob_home:.1%} | "
            f"Draw {prob_draw:.1%} | "
            f"{away_team} {prob_away:.1%}  "
            f"[{result['confidence']}]"
        )
        return result


def _confidence_label(prob: float) -> str:
    """Human-readable confidence based on favourite's win probability."""
    if prob >= 0.70:
        return "HIGH"
    elif prob >= 0.55:
        return "MEDIUM"
    elif prob >= 0.45:
        return "LOW"
    return "TOSS-UP"