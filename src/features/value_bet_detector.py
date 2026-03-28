"""
Value Bet Detector
Compares model probabilities vs. bookmaker implied probabilities.
A "value bet" exists when our model sees higher probability than the market.

Usage:
    detector = ValueBetDetector()
    value_bets = detector.analyze(model_pred, market_odds)
"""

import pandas as pd
from src.utils.logger import get_logger

log = get_logger(__name__)

# Minimum edge to flag as a value bet (in percentage points)
MIN_EDGE_PCT = 5.0


class ValueBetDetector:
    """
    Compares model output vs. bookmaker consensus odds.
    Computes the edge (our prob − market implied prob) for each outcome.
    """

    def analyze(
        self,
        home_team:     str,
        away_team:     str,
        model_home:    float,
        model_draw:    float,
        model_away:    float,
        market_home:   float,
        market_draw:   float,
        market_away:   float,
        avg_margin:    float = 0.0,
    ) -> dict:
        """
        Run value bet analysis for a single match.

        Args:
            model_*:  Our model's win probabilities (0–1)
            market_*: Bookmaker's implied probabilities (margin-adjusted, 0–1)
            avg_margin: Bookmaker's average margin (%)

        Returns:
            Analysis dict with edge per outcome and overall verdict
        """
        edge_home = round((model_home - market_home) * 100, 2)
        edge_draw = round((model_draw - market_draw) * 100, 2)
        edge_away = round((model_away - market_away) * 100, 2)

        bets = []
        if edge_home >= MIN_EDGE_PCT:
            bets.append({
                "outcome":      "Heimsieg",
                "model_prob":   round(model_home * 100, 1),
                "market_prob":  round(market_home * 100, 1),
                "edge_pct":     edge_home,
                "rating":       _rating(edge_home),
            })
        if edge_draw >= MIN_EDGE_PCT:
            bets.append({
                "outcome":      "Unentschieden",
                "model_prob":   round(model_draw * 100, 1),
                "market_prob":  round(market_draw * 100, 1),
                "edge_pct":     edge_draw,
                "rating":       _rating(edge_draw),
            })
        if edge_away >= MIN_EDGE_PCT:
            bets.append({
                "outcome":      "Auswärtssieg",
                "model_prob":   round(model_away * 100, 1),
                "market_prob":  round(market_away * 100, 1),
                "edge_pct":     edge_away,
                "rating":       _rating(edge_away),
            })

        # Sort by edge descending
        bets.sort(key=lambda x: -x["edge_pct"])

        # Overall model confidence vs market
        model_favourite  = max(["home", "draw", "away"],
                               key=lambda x: {"home": model_home, "draw": model_draw, "away": model_away}[x])
        market_favourite = max(["home", "draw", "away"],
                               key=lambda x: {"home": market_home, "draw": market_draw, "away": market_away}[x])

        return {
            "home_team":       home_team,
            "away_team":       away_team,
            "model_home":      round(model_home * 100, 1),
            "model_draw":      round(model_draw * 100, 1),
            "model_away":      round(model_away * 100, 1),
            "market_home":     round(market_home * 100, 1),
            "market_draw":     round(market_draw * 100, 1),
            "market_away":     round(market_away * 100, 1),
            "edge_home":       edge_home,
            "edge_draw":       edge_draw,
            "edge_away":       edge_away,
            "value_bets":      bets,
            "has_value":       len(bets) > 0,
            "avg_margin_pct":  round(avg_margin, 2),
            "model_favourite": model_favourite,
            "market_favourite": market_favourite,
            "disagreement":    model_favourite != market_favourite,
        }

    def batch_analyze(
        self,
        predictions: list[dict],
        consensus_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Analyze value bets for multiple matches.

        Args:
            predictions: List of model prediction dicts (from simulate())
            consensus_df: DataFrame from OddsCollector.get_consensus_odds()

        Returns:
            DataFrame of all value bets found
        """
        results = []
        for pred in predictions:
            match = consensus_df[
                (consensus_df["home_team"].str.contains(pred["home_team"][:8], case=False)) &
                (consensus_df["away_team"].str.contains(pred["away_team"][:8], case=False))
            ]
            if match.empty:
                log.warning(f"No market odds found for {pred['home_team']} vs {pred['away_team']}")
                continue

            row = match.iloc[0]
            analysis = self.analyze(
                home_team=pred["home_team"],
                away_team=pred["away_team"],
                model_home=pred["prob_home_win"],
                model_draw=pred["prob_draw"],
                model_away=pred["prob_away_win"],
                market_home=row["market_home"],
                market_draw=row["market_draw"],
                market_away=row["market_away"],
                avg_margin=row.get("avg_margin", 0),
            )
            if analysis["has_value"]:
                results.extend([
                    {**vb, "home_team": pred["home_team"], "away_team": pred["away_team"]}
                    for vb in analysis["value_bets"]
                ])

        return pd.DataFrame(results) if results else pd.DataFrame()


def _rating(edge: float) -> str:
    """Star rating based on edge size."""
    if edge >= 15:
        return "⭐⭐⭐ STARK"
    elif edge >= 10:
        return "⭐⭐ GUT"
    return "⭐ SCHWACH"