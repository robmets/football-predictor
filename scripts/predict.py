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
from src.features.context_engine import ContextEngine
from src.collectors.football_data_collector import FootballDataCollector
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
    league:        str = typer.Option("BL1",  "--league", "-l"),
    sims:          int = typer.Option(config.SIMULATION_RUNS, "--sims", "-n"),
    stage:         str = typer.Option(None,   "--stage",  help="WC stage: GROUP_STAGE|LAST_16|QUARTER_FINALS|SEMI_FINALS|FINAL"),
    group:         str = typer.Option(None,   "--group",  help="WC group: GROUP_A … GROUP_L"),
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
    home_impact, home_missing_names = calculate_missing_impact(home, league=league) or (0.0, [])
    log.info(f"Hole Verletzungsdaten für {away}...")
    away_impact, away_missing_names = calculate_missing_impact(away, league=league) or (0.0, [])

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
    home_sofascore_rating  = None
    away_sofascore_rating  = None
    home_starters = []
    away_starters = []
    modus = "C"

    if home_sofascore_id and away_sofascore_id:
        match_id = sofascore.get_match_id(home_sofascore_id, away)

        if match_id:
            # Modus A: bestätigte Live-Aufstellung
            confirmed = sofascore.get_match_ratings(match_id)
            if confirmed:
                h_st = [p for p in confirmed["home_team"]["players"] if p.get("is_starter")]
                a_st = [p for p in confirmed["away_team"]["players"] if p.get("is_starter")]
                if len(h_st) == 11 and confirmed.get("confirmed", False):
                    home_starters, away_starters, modus = h_st, a_st, "A"

            # Modus B: voraussichtliche Aufstellung
            if modus != "A":
                predicted = sofascore.get_predicted_match_ratings(match_id)
                if predicted:
                    h_pl = predicted["home_team"]["players"]
                    a_pl = predicted["away_team"]["players"]
                    h_st = [p for p in h_pl if p.get("is_starter")]
                    a_st = [p for p in a_pl if p.get("is_starter")]
                    if len(h_st) < 11 and len(h_pl) >= 11:
                        h_st, a_st = h_pl[:11], a_pl[:11]
                    if len(h_st) >= 11:
                        home_starters, away_starters = h_st[:11], a_st[:11]
                        modus = "B"

        if home_sofascore_id:
            home_ssc = sofascore.get_injured_player_impact(
                home_sofascore_id, home_missing_names,
                match_starters=home_starters if modus in ("A", "B") else None,
            )
            home_sofascore_penalty = home_ssc["penalty"]
            home_sofascore_rating  = home_ssc["team_avg_rating"]

        if away_sofascore_id:
            away_ssc = sofascore.get_injured_player_impact(
                away_sofascore_id, away_missing_names,
                match_starters=away_starters if modus in ("A", "B") else None,
            )
            away_sofascore_penalty = away_ssc["penalty"]
            away_sofascore_rating  = away_ssc["team_avg_rating"]

        log.info(f"Sofascore Modus {modus}: Home Rating={home_sofascore_rating} Away Rating={away_sofascore_rating}")

    # Sofascore Penalty zum Transfermarkt Impact addieren
    home_total_impact = (home_impact or 0.0) + home_sofascore_penalty
    away_total_impact = (away_impact or 0.0) + away_sofascore_penalty


    # ---------------------------------------------------------
    # 3. LIVE-FORM, H2H, CONTEXT & WETTER
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

    # H2H über alle Wettbewerbe
    h2h = form_calc.get_h2h_factor(home, away, league, features_df=df)

    # Context Engine: Tabelle + Derby + CL Knockout
    try:
        standings = FootballDataCollector().get_live_standings(league)
    except Exception:
        standings = {}

    ctx_engine = ContextEngine()
    if league == "WC":
        # WC: matchday from group position, stage from CLI arg
        wc_stage = stage or "GROUP_STAGE"
        wc_group = group
        matchday = 2  # default mid-group
        context = ctx_engine.calculate_context(
            home, away, league, matchday, standings,
            stage=wc_stage, group_name=wc_group,
        )
    else:
        matchday = 20
        if standings and home in standings:
            matchday = standings[home].get("playedGames", 19) + 1
        context = ctx_engine.calculate_context(home, away, league, matchday, standings)

    # Knockout-Boost oder Tabellen-Boost auf Form-Faktor anwenden
    home_form["attack_factor"] = round(home_form["attack_factor"] * context["home_motivation"], 3)
    away_form["attack_factor"] = round(away_form["attack_factor"] * context["away_motivation"], 3)

    if context.get("is_knockout"):
        log.info(f"🏆 {context['knockout_stage']} — Motivation ×{context['home_motivation']:.3f} für beide Teams")
    if context.get("is_derby"):
        log.info(f"🔥 DERBY — Boost aktiv")

    weather = WeatherCollector().get_match_weather(home)
    if weather["goal_impact_factor"] != 0:
        log.info(f"Wetter: {weather['description']} ({weather['goal_impact_factor']:+.0%})")


    # ---------------------------------------------------------
    # 4. MODELL & SIMULATION
    # ---------------------------------------------------------
    model = PoissonModel()
    model.fit(df, league=league)
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
    result["home_form"]          = home_form
    result["away_form"]          = away_form
    result["home_injury_impact"] = home_total_impact
    result["away_injury_impact"] = away_total_impact
    result["home_form_ppg"]      = home_form["form_ppg"]
    result["away_form_ppg"]      = away_form["form_ppg"]
    result["is_derby"]           = context["is_derby"]
    result["knockout_stage"]     = context.get("knockout_stage")
    result["is_knockout"]        = context.get("is_knockout", False)

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