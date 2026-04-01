"""
Script: Predict a match using the full pipeline.

Usage:
    python scripts/predict.py --home "Bayern München" --away "Borussia Dortmund"
    python scripts/predict.py --home "Liverpool FC" --away "Arsenal FC" --league PL
    python scripts/predict.py --home "Real Madrid CF" --away "FC Barcelona" --league PD
"""

import sys
from pathlib import Path

# MUSS als erstes stehen — vor allen src-Imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import typer

from src.models.poisson_model import PoissonModel
from src.simulation.monte_carlo import MonteCarloSimulator
from src.utils.logger import get_logger
from src.features.injury_impact import calculate_missing_impact
from src.collectors.weather_collector import WeatherCollector
from src.features.live_form import LiveFormCalculator
from config.config import config
from src.models.xgboost_model import XGBoostFeedbackModel

log = get_logger("predict")
app = typer.Typer()


def _bar(prob: float, width: int = 30) -> str:
    filled = int(round(prob * width))
    return "█" * filled + "░" * (width - filled)


def _print_prediction(result: dict):
    hw   = result["prob_home_win"]
    dr   = result["prob_draw"]
    aw   = result["prob_away_win"]
    home = result["home_team"]
    away = result["away_team"]

    print("\n" + "═" * 62)
    print(f"  ⚽  {home}  vs  {away}")
    print("═" * 62)
    print(f"\n  Expected goals:  {home[:20]:<20} {result['expected_home_goals']:.2f}")
    print(f"                   {away[:20]:<20} {result['expected_away_goals']:.2f}")

    hf = result.get("home_form", {})
    af = result.get("away_form", {})
    if hf.get("source") == "api-football":
        print(f"\n  Live-Form:       {home[:20]:<20} PPG={hf['form_ppg']:.2f}  "
              f"Faktor={hf['breakdown']['combined']:+.1%}")
        print(f"                   {away[:20]:<20} PPG={af['form_ppg']:.2f}  "
              f"Faktor={af['breakdown']['combined']:+.1%}")

    print(f"\n  {'Outcome':<22} {'Probability':>10}   {'':30}")
    print(f"  {'─' * 64}")
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
    print("═" * 62 + "\n")


@app.command()
def predict(
    home:          str = typer.Option(...,    "--home",   "-h"),
    away:          str = typer.Option(...,    "--away",   "-a"),
    league:        str = typer.Option("BL1", "--league", "-l"),
    sims:          int = typer.Option(config.SIMULATION_RUNS, "--sims", "-n"),
):
    path = Path(f"data/processed/features_{league}.csv")
    if not path.exists():
        log.error(f"Features nicht gefunden: {path}")
        log.error(f"Bitte ausführen: python scripts/update.py --league {league}")
        raise typer.Exit(1)

    df = pd.read_csv(path)
    log.info(f"Loaded {len(df)} matches für {config.SUPPORTED_LEAGUES.get(league, league)}")

    # Verletzungen
    log.info(f"Hole Verletzungsdaten für {home}...")
    home_impact = calculate_missing_impact(home)
    log.info(f"Hole Verletzungsdaten für {away}...")
    away_impact = calculate_missing_impact(away)

    if home_impact and home_impact > 15.0:
        log.warning(f"ACHTUNG: {home} stark geschwächt! ({home_impact:.1f}%)")
    if away_impact and away_impact > 15.0:
        log.warning(f"ACHTUNG: {away} stark geschwächt! ({away_impact:.1f}%)")

    # Live-Form
    form_calc = LiveFormCalculator()
    home_form = form_calc.get_lambda_adjustment(home, league=league, injury_impact=home_impact or 0.0, features_df=df)
    form_calc.print_summary(home, home_form)
    away_form = form_calc.get_lambda_adjustment(away, league=league, injury_impact=away_impact or 0.0, features_df=df)
    form_calc.print_summary(away, away_form)

    # Wetter
    weather = WeatherCollector().get_match_weather(home)
    if weather["goal_impact_factor"] != 0:
        log.info(f"Wetter: {weather['description']} ({weather['goal_impact_factor']:+.0%})")

    # Modell & Simulation
    model = PoissonModel()
    model.fit(df)
    print(f"\n  Team Ratings Top 5 ({config.SUPPORTED_LEAGUES.get(league, league)}):")
    print(model.team_ratings().head(5).to_string(index=False))

    sim    = MonteCarloSimulator(model)
    result = sim.simulate(
        home, away, n=sims,
        home_injury_impact=home_impact or 0.0,
        away_injury_impact=away_impact or 0.0,
        weather_impact=weather["goal_impact_factor"],
        home_form_factor=home_form["attack_factor"],
        away_form_factor=away_form["attack_factor"],
    )
    result["home_form"] = home_form
    result["away_form"] = away_form
    result["home_injury_impact"] = home_impact or 0.0
    result["away_injury_impact"] = away_impact or 0.0
    result["home_form_ppg"] = home_form["form_ppg"]
    result["away_form_ppg"] = away_form["form_ppg"]

    # XGBoost Ensemble (falls Modell trainiert)
    xgb = XGBoostFeedbackModel()
    if xgb.is_trained:
        result = xgb.get_ensemble(result)
        log.info(f"Ensemble: Poisson {1-result['ensemble_weight_xgb']:.0%} + XGBoost {result['ensemble_weight_xgb']:.0%}")

    # Vorhersage in DB speichern
    pred_id = XGBoostFeedbackModel.save_prediction(result, league=league)
    log.info(f"Vorhersage-ID: {pred_id} — nach dem Spiel eintragen:")
    log.info(f"  python scripts/enter_result.py --id {pred_id} --home-goals X --away-goals Y")

    _print_prediction(result)


if __name__ == "__main__":
    app()