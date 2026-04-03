"""
Script: Predict a match using the full pipeline.

Usage:
    python scripts/predict.py --home "Bayern München" --away "Borussia Dortmund"
    python scripts/predict.py --home "Liverpool FC" --away "Arsenal FC" --league PL
    python scripts/predict.py --home "Real Madrid CF" --away "FC Barcelona" --league PD
"""

import sys
from pathlib import Path
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
from src.collectors.football_data_collector import FootballDataCollector
from src.features.context_engine import ContextEngine

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
    if result.get("is_derby"):
        print(f"  🔥 DERBY")
    if result.get("home_motivation", 1.0) > 1.01:
        print(f"  Motivation {result['home_team'][:15]}: ×{result['home_motivation']}")
    if result.get("away_motivation", 1.0) > 1.01:
        print(f"  Motivation {result['away_team'][:15]}: ×{result['away_motivation']}")
    print("═" * 62 + "\n")


@app.command()
def predict(
    home:   str = typer.Option(...,    "--home",   "-h"),
    away:   str = typer.Option(...,    "--away",   "-a"),
    league: str = typer.Option("BL1", "--league", "-l"),
    sims:   int = typer.Option(config.SIMULATION_RUNS, "--sims", "-n"),
):
    path = Path(f"data/processed/features_{league}.csv")
    if not path.exists():
        log.error(f"Features nicht gefunden: {path}")
        raise typer.Exit(1)

    df = pd.read_csv(path)
    log.info(f"Loaded {len(df)} matches für {config.SUPPORTED_LEAGUES.get(league, league)}")

    # ── 1. TRANSFERMARKT: Verletzungen ────────────────────────────────────────
    log.info(f"Hole Verletzungsdaten für {home}...")
    _home_result = calculate_missing_impact(home)
    home_impact, home_missing_names = _home_result if _home_result else (0.0, [])

    log.info(f"Hole Verletzungsdaten für {away}...")
    _away_result = calculate_missing_impact(away)
    away_impact, away_missing_names = _away_result if _away_result else (0.0, [])

    if home_impact > 15.0:
        log.warning(f"ACHTUNG: {home} stark geschwächt! ({home_impact:.1f}%)")
    if away_impact > 15.0:
        log.warning(f"ACHTUNG: {away} stark geschwächt! ({away_impact:.1f}%)")

    # ── 2. SOFASCORE: Ratings & Verletzungspenalty ────────────────────────────
    log.info("Mappe Teams in Sofascore...")
    home_sofascore_id = sofascore.get_team_id(home)
    away_sofascore_id = sofascore.get_team_id(away)

    home_sofascore_penalty = 0.0
    away_sofascore_penalty = 0.0
    home_sofascore_rating  = None
    away_sofascore_rating  = None

    if home_sofascore_id and away_sofascore_id:
        match_id     = sofascore.get_match_id(home_sofascore_id, away)
        home_starters = []
        away_starters = []
        modus = "C"

        if match_id:
            # Modus A: bestätigte Live-Aufstellung
            confirmed = sofascore.get_match_ratings(match_id)
            if confirmed:
                h_st = [p for p in confirmed["home_team"]["players"] if p.get("is_starter")]
                a_st = [p for p in confirmed["away_team"]["players"] if p.get("is_starter")]
                if len(h_st) == 11 and confirmed.get("confirmed", False):
                    home_starters, away_starters, modus = h_st, a_st, "A"
                    log.info("✅ Modus A: Bestätigte Live-Aufstellung (11 Spieler)")

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
                        log.info("✅ Modus B: Voraussichtliche Aufstellung (11 Spieler)")
        else:
            log.info("Kein Sofascore Match-ID gefunden — Modus C (Kader-Schnitt)")

        if modus == "C":
            log.info("✅ Modus C: Kein Spiel gefunden oder keine Aufstellung — nutze Kader-Schnitt")

        home_ssc = sofascore.get_injured_player_impact(
            home_sofascore_id, home_missing_names,
            match_starters=home_starters if modus in ("A", "B") else None,
        )
        away_ssc = sofascore.get_injured_player_impact(
            away_sofascore_id, away_missing_names,
            match_starters=away_starters if modus in ("A", "B") else None,
        )
        home_sofascore_penalty = home_ssc["penalty"]
        away_sofascore_penalty = away_ssc["penalty"]
        home_sofascore_rating  = home_ssc["team_avg_rating"]
        away_sofascore_rating  = away_ssc["team_avg_rating"]
        if home_sofascore_penalty > 0:
            log.info(f"Sofascore Gesamt-Penalty {home}: {home_sofascore_penalty:.2f}%")
        if away_sofascore_penalty > 0:
            log.info(f"Sofascore Gesamt-Penalty {away}: {away_sofascore_penalty:.2f}%")

    home_total_impact = home_impact + home_sofascore_penalty
    away_total_impact = away_impact + away_sofascore_penalty

    # ── 2.5 CONTEXT ENGINE (Tabellen, Motivation, Derbys) ────────────────────
    try:
        fd_collector = FootballDataCollector()
        standings = fd_collector.get_live_standings(league)
    except Exception as e:
        log.warning(f"Tabelle konnte nicht geladen werden: {e} — Context Engine neutral")
        standings = {}

    if not standings:
        log.warning(f"Tabelle für {league} leer — Context Engine ohne Standings (alle Boosts = 1.0)")

    matchday = 20
    if standings and home in standings:
        matchday = standings[home].get("playedGames", 19) + 1

    ctx_engine = ContextEngine()
    context = ctx_engine.calculate_context(home, away, league, matchday, standings)

    log.info(
        f"🧠 Context Engine: Derby={bool(context['is_derby'])} | "
        f"Urgency={context['urgency']:.2f} | "
        f"Home-Boost=×{context['home_motivation']} | "
        f"Away-Boost=×{context['away_motivation']}"
    )

    # ── 3. LIVE-FORM & WETTER ────────────────────────────────────────────────
    form_calc = LiveFormCalculator()
    home_form = form_calc.get_lambda_adjustment(
        home, league=league, injury_impact=home_total_impact,
        sofascore_team_rating=home_sofascore_rating, is_home=True,
        features_df=df,
    )
    away_form = form_calc.get_lambda_adjustment(
        away, league=league, injury_impact=away_total_impact,
        sofascore_team_rating=away_sofascore_rating, is_home=False,
        features_df=df,
    )
    form_calc.print_summary(home, home_form)
    form_calc.print_summary(away, away_form)

    log.info(f"📈 Spezifische Form {home} (Heim): {home_form.get('specific_ppg', 0):.2f} PPG")
    log.info(f"📈 Spezifische Form {away} (Auswärts): {away_form.get('specific_ppg', 0):.2f} PPG")

    # Context-Boost auf attack_factor anwenden
    home_form["attack_factor"] = round(home_form["attack_factor"] * context["home_motivation"], 3)
    away_form["attack_factor"] = round(away_form["attack_factor"] * context["away_motivation"], 3)

    weather = WeatherCollector().get_match_weather(home)
    if weather["goal_impact_factor"] != 0:
        log.info(f"Wetter: {weather['description']} ({weather['goal_impact_factor']:+.0%})")

    # ── 4. MODELL & SIMULATION ────────────────────────────────────────────────
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
    )
    result["home_form"]          = home_form
    result["away_form"]          = away_form
    result["home_injury_impact"] = home_total_impact
    result["away_injury_impact"] = away_total_impact
    result["home_form_ppg"]      = home_form["form_ppg"]
    result["away_form_ppg"]      = away_form["form_ppg"]
    result["home_specific_ppg"]  = home_form.get("specific_ppg", 1.5)
    result["away_specific_ppg"]  = away_form.get("specific_ppg", 1.5)
    result["is_derby"]           = context["is_derby"]
    result["match_urgency"]      = context["urgency"]
    result["home_motivation"]    = context["home_motivation"]
    result["away_motivation"]    = context["away_motivation"]

    # XGBoost Ensemble
    xgb = XGBoostFeedbackModel()
    if xgb.is_trained:
        result = xgb.get_ensemble(result)
        log.info(f"Ensemble: Poisson {1-result['ensemble_weight_xgb']:.0%} + XGBoost {result['ensemble_weight_xgb']:.0%}")

    pred_id = XGBoostFeedbackModel.save_prediction(result, league=league)
    log.info(f"Vorhersage-ID: {pred_id} — nach dem Spiel eintragen:")
    log.info(f"  python scripts/enter_result.py --id {pred_id} --home-goals X --away-goals Y")

    _print_prediction(result)


if __name__ == "__main__":
    app()