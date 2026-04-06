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
from src.collectors.sofascore_collector import SofascoreCollector

log = get_logger("predict")
app = typer.Typer()
sofascore = SofascoreCollector()


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

    # ---------------------------------------------------------
    # 1. TRANSFERMARKT: Verletzungen
    # ---------------------------------------------------------
    log.info(f"Hole Verletzungsdaten für {home}...")
    home_impact, home_missing_names = calculate_missing_impact(home)
    log.info(f"Hole Verletzungsdaten für {away}...")
    away_impact, away_missing_names = calculate_missing_impact(away)

    if home_impact and home_impact > 15.0:
        log.warning(f"ACHTUNG: {home} stark geschwächt! ({home_impact:.1f}%)")
    if away_impact and away_impact > 15.0:
        log.warning(f"ACHTUNG: {away} stark geschwächt! ({away_impact:.1f}%)")

    # ---------------------------------------------------------
    # 2. SOFASCORE: Ratings & Mismatch
    # ---------------------------------------------------------
    log.info("Mappe Teams in Sofascore...")
    home_sofascore_id = sofascore.get_team_id(home)
    away_sofascore_id = sofascore.get_team_id(away)
    
    # Initiale Modifikatoren für Sofascore
    home_sofascore_penalty = 0.0
    away_sofascore_penalty = 0.0
    
    if home_sofascore_id and away_sofascore_id:
        match_id = sofascore.get_match_id(home_sofascore_id, away)
        
        if match_id:
            match_ratings = sofascore.get_match_ratings(match_id)
            
            if match_ratings:
                # DEBUGGING: Lass uns schauen, was wir von Sofascore bekommen!
                log.info(f"DEBUG: Anzahl Spieler im Home-Team (Sofascore Lineup API): {len(match_ratings['home_team']['players'])}")
                if match_ratings['home_team']['players']:
                     log.info(f"DEBUG: Erstes Spieler-Objekt: {match_ratings['home_team']['players'][0]}")

                # Wir versuchen, die Starter zu finden (voraussichtlich oder bestätigt)
                # Manchmal schickt Sofascore für "mögliche" Aufstellungen einfach die ersten 11 Spieler in der Liste,
                # auch ohne is_starter flag.
                
                home_players = match_ratings["home_team"]["players"]
                away_players = match_ratings["away_team"]["players"]
                
                home_starters = [p for p in home_players if p.get("is_starter")]
                away_starters = [p for p in away_players if p.get("is_starter")]
                
                # Wenn wir keine 11 Spieler mit 'is_starter' haben, prüfen wir, ob wir überhaupt 
                # genau 11 Spieler (z.B. als Predicted Lineup) bekommen haben
                if len(home_starters) != 11 and len(home_players) >= 11:
                     log.info("Nutze die ersten 11 Spieler als voraussichtliche Aufstellung (Predicted Lineup)")
                     # Wir nehmen einfach die ersten 11 (die Startelf)
                     home_starters = home_players[:11]
                     away_starters = away_players[:11]

                if len(home_starters) == 11:
                    log.info("Aufstellung (voraussichtlich oder live) gefunden! Starte Modus A/B...")
                else:
                    log.info("Keine Match-Aufstellung verfügbar. Starte Modus C (Kader-Schnitt)...")
                    
                home_sofascore_penalty = sofascore.get_injured_player_impact(home_sofascore_id, home_missing_names, match_starters=home_starters)
                away_sofascore_penalty = sofascore.get_injured_player_impact(away_sofascore_id, away_missing_names, match_starters=away_starters)
                    
    # Sofascore Penalty zum Transfermarkt Impact addieren
    home_total_impact = (home_impact or 0.0) + home_sofascore_penalty
    away_total_impact = (away_impact or 0.0) + away_sofascore_penalty


    # ---------------------------------------------------------
    # 3. LIVE-FORM, H2H & WETTER
    # ---------------------------------------------------------
    form_calc = LiveFormCalculator()
    home_form = form_calc.get_lambda_adjustment(
        home, league=league, injury_impact=home_total_impact,
        sofascore_team_rating=home_sofascore_rating, is_home=True, features_df=df,
    )
    form_calc.print_summary(home, home_form)

    away_form = form_calc.get_lambda_adjustment(
        away, league=league, injury_impact=away_total_impact,
        sofascore_team_rating=away_sofascore_rating, is_home=False, features_df=df,
    )
    form_calc.print_summary(away, away_form)

    # H2H über alle Wettbewerbe (z.B. La Liga + CL für Atletico vs Barcelona)
    h2h = form_calc.get_h2h_factor(home, away, league, features_df=df)

    weather = WeatherCollector().get_match_weather(home)
    if weather["goal_impact_factor"] != 0:
        log.info(f"Wetter: {weather['description']} ({weather['goal_impact_factor']:+.0%})")


    # ---------------------------------------------------------
    # 4. MODELL & SIMULATION
    # ---------------------------------------------------------
    model = PoissonModel()
    model.fit(df)
    print(f"\n  Team Ratings Top 5 ({config.SUPPORTED_LEAGUES.get(league, league)}):")
    print(model.team_ratings().head(5).to_string(index=False))

    sim    = MonteCarloSimulator(model)
    result = sim.simulate(
        home, away, n=sims,
        home_injury_impact=home_total_impact,
        away_injury_impact=away_total_impact,
        weather_impact=weather["goal_impact_factor"],
        home_form_factor=home_form["attack_factor"],
        away_form_factor=away_form["attack_factor"],
        h2h_factor=h2h["factor"],
        h2h_games=h2h["games"],
    )
    result["home_form"] = home_form
    result["away_form"] = away_form
    
    # HIER EBENFALLS TOTAL IMPACT SPEICHERN
    result["home_injury_impact"] = home_total_impact
    result["away_injury_impact"] = away_total_impact
    
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