"""
Script: Predict a match using the full pipeline.
Loads data → fits Poisson model → runs Monte Carlo → prints results.

Usage:
    python scripts/predict.py --home "Bayern München" --away "Borussia Dortmund"
    python scripts/predict.py --home "Bayer 04 Leverkusen" --away "RB Leipzig" --sims 20000
"""

import sys
from pathlib import Path
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.poisson_model import PoissonModel
from src.simulation.monte_carlo import MonteCarloSimulator
from src.utils.logger import get_logger
from config.config import config

log = get_logger("predict")
app = typer.Typer()

FEATURES_PATH = "data/processed/features_BL1.csv"


def _bar(prob: float, width: int = 30) -> str:
    filled = int(round(prob * width))
    return "█" * filled + "░" * (width - filled)


def _print_prediction(result: dict):
    hw = result["prob_home_win"]
    dr = result["prob_draw"]
    aw = result["prob_away_win"]
    home = result["home_team"]
    away = result["away_team"]

    print("\n" + "═" * 58)
    print(f"  ⚽  {home}  vs  {away}")
    print("═" * 58)
    print(f"\n  Expected goals:  {home[:20]:<20} {result['expected_home_goals']:.2f}")
    print(f"                   {away[:20]:<20} {result['expected_away_goals']:.2f}")
    print(f"\n  {'Outcome':<22} {'Probability':>10}   {'':30}")
    print(f"  {'─'*64}")
    print(f"  {'Home win  ' + home[:16]:<22} {hw:>9.1%}   {_bar(hw)}")
    print(f"  {'Draw':<22} {dr:>9.1%}   {_bar(dr)}")
    print(f"  {'Away win  ' + away[:16]:<22} {aw:>9.1%}   {_bar(aw)}")
    print(f"\n  Additional markets:")
    print(f"    Both teams score:   {result['prob_btts']:.1%}")
    print(f"    Over 2.5 goals:     {result['prob_over_2_5']:.1%}")
    print(f"    Over 3.5 goals:     {result['prob_over_3_5']:.1%}")
    print(f"\n  Most likely exact scores:")
    for s in result["top_scores"][:5]:
        print(f"    {s['score']:>5}   {s['probability']:>5.2f}%  {_bar(s['probability'] / 100, 20)}")
    print(f"\n  Favourite:   {result['favourite']}")
    print(f"  Confidence:  {result['confidence']}  ({result['favourite_prob']:.1%})")
    print(f"  Simulations: {result['simulations']:,}")
    print("═" * 58 + "\n")


@app.command()
def predict(
    home: str = typer.Option(..., "--home", "-h", help="Home team name"),
    away: str = typer.Option(..., "--away", "-a", help="Away team name"),
    league: str = typer.Option("BL1", "--league", "-l"),
    sims: int = typer.Option(config.SIMULATION_RUNS, "--sims", "-n"),
    features_path: str = typer.Option(FEATURES_PATH, "--features"),
):
    # Load feature data
    path = features_path.replace("BL1", league)
    if not Path(path).exists():
        log.error(f"Features not found at {path}. Run build_features.py first.")
        raise typer.Exit(1)

    df = pd.read_csv(path)
    log.info(f"Loaded {len(df)} matches for model fitting")

    # Fit Poisson model
    model = PoissonModel()
    model.fit(df)

    # Print team ratings
    ratings = model.team_ratings()
    print(f"\n  Team Ratings (top 5):")
    print(ratings.head(5).to_string(index=False))

    # Run Monte Carlo simulation
    sim = MonteCarloSimulator(model)
    result = sim.simulate(home, away, n=sims)

    _print_prediction(result)


if __name__ == "__main__":
    app()