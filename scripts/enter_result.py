"""
Script: Ergebnis nach dem Spiel eintragen.

Usage:
    python scripts/enter_result.py --id 5 --home-goals 2 --away-goals 1
    python scripts/enter_result.py --list          # Zeigt offene Vorhersagen
    python scripts/enter_result.py --history       # Zeigt alle Ergebnisse
    python scripts/enter_result.py --train         # XGBoost neu trainieren
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from src.models.xgboost_model import XGBoostFeedbackModel
from src.utils.database import get_session, Prediction
from src.utils.logger import get_logger

log = get_logger("enter_result")
app = typer.Typer()


def _print_open(preds):
    print("\n" + "═" * 72)
    print(f"  {'ID':>3}  {'Datum':<12} {'Heim':<22} {'Auswärts':<22} {'Prognose'}")
    print("═" * 72)
    for p in preds:
        datum = str(p.created_at)[:10]
        winner = {"H": p.home_team[:10], "D": "Unentsch.", "A": p.away_team[:10]}.get(p.predicted_winner, "?")
        conf = p.confidence or "?"
        print(f"  {p.id:>3}  {datum:<12} {p.home_team[:20]:<22} {p.away_team[:20]:<22} {winner} [{conf}]")
    print("═" * 72 + "\n")


def _print_history(preds):
    correct = sum(1 for p in preds if p.prediction_correct is True)
    total   = len(preds)
    print("\n" + "═" * 80)
    print(f"  {'ID':>3}  {'Datum':<10} {'Spiel':<35} {'Prognose':<10} {'Ergebnis':<10} {'OK'}")
    print("═" * 80)
    for p in preds:
        spiel  = f"{p.home_team[:15]} vs {p.away_team[:15]}"
        result = f"{p.actual_home_goals}:{p.actual_away_goals}" if p.actual_home_goals is not None else "—"
        ok     = "✅" if p.prediction_correct else "❌"
        datum  = str(p.created_at)[:10]
        print(f"  {p.id:>3}  {datum:<10} {spiel:<35} {p.predicted_winner:<10} {result:<10} {ok}")
    print("═" * 80)
    if total > 0:
        print(f"\n  Trefferquote: {correct}/{total} = {correct/total:.1%}")
    print()


@app.command()
def main(
    id:          int  = typer.Option(None,  "--id",          "-i",  help="Vorhersage-ID"),
    home_goals:  int  = typer.Option(None,  "--home-goals",  "-hg", help="Tore Heimteam"),
    away_goals:  int  = typer.Option(None,  "--away-goals",  "-ag", help="Tore Auswärtsteam"),
    list_open:   bool = typer.Option(False, "--list",        "-l",  help="Offene Vorhersagen anzeigen"),
    history:     bool = typer.Option(False, "--history",     "-H",  help="Alle Ergebnisse anzeigen"),
    train:       bool = typer.Option(False, "--train",       "-t",  help="XGBoost trainieren"),
):
    session = get_session()

    if list_open:
        preds = session.query(Prediction).filter(
            Prediction.actual_result == None
        ).order_by(Prediction.created_at.desc()).all()
        session.close()
        if not preds:
            print("\n  Keine offenen Vorhersagen — erst predict.py ausführen!\n")
        else:
            print(f"\n  {len(preds)} offene Vorhersagen:")
            _print_open(preds)
        return

    if history:
        preds = session.query(Prediction).filter(
            Prediction.actual_result != None
        ).order_by(Prediction.created_at.desc()).all()
        session.close()
        if not preds:
            print("\n  Noch keine Ergebnisse eingetragen.\n")
        else:
            _print_history(preds)
        return

    if train:
        session.close()
        xgb = XGBoostFeedbackModel()
        result = xgb.train()
        if result["success"]:
            print(f"\n  ✅ XGBoost trainiert!")
            print(f"     Samples:  {result['n_samples']}")
            print(f"     Accuracy: {result['accuracy']:.1%}")
            print(f"     Top Features:")
            for feat, imp in result["top_features"]:
                print(f"       {feat:<25} {imp:.3f}")
        else:
            print(f"\n  ❌ Training fehlgeschlagen: {result['error']}")
            n = result.get("n_samples", 0)
            from src.models.xgboost_model import MIN_SAMPLES
            print(f"     {n} / {MIN_SAMPLES} Ergebnisse vorhanden")
        return

    # Ergebnis eintragen
    if id is None or home_goals is None or away_goals is None:
        print("\n  Bitte ID und Tore angeben:")
        print("    python scripts/enter_result.py --id 5 --home-goals 2 --away-goals 1")
        print("\n  Oder offene Vorhersagen anzeigen:")
        print("    python scripts/enter_result.py --list\n")
        session.close()
        return

    session.close()
    success = XGBoostFeedbackModel.enter_result(id, home_goals, away_goals)
    if success:
        # Stats anzeigen
        session2 = get_session()
        all_results = session2.query(Prediction).filter(
            Prediction.actual_result != None
        ).all()
        session2.close()
        correct = sum(1 for p in all_results if p.prediction_correct is True)
        total   = len(all_results)
        from src.models.xgboost_model import MIN_SAMPLES
        print(f"\n  Gesamte Trefferquote: {correct}/{total} = {correct/total:.1%}" if total > 0 else "")
        if total >= MIN_SAMPLES:
            print(f"  💡 Genug Daten für Training! Führe aus:")
            print(f"     python scripts/enter_result.py --train")
        else:
            remaining = MIN_SAMPLES - total
            print(f"  ℹ️  Noch {remaining} Ergebnisse bis zum ersten XGBoost-Training")


if __name__ == "__main__":
    app()