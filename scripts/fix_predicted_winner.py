"""
Migration: Korrigiert predicted_winner für alle bestehenden Vorhersagen.

Das alte System hat predicted_winner aus Expected Goals berechnet (nie "D").
Das neue System nimmt die höchste Wahrscheinlichkeit (kann "D" sein).

Führt auch prediction_correct neu aus für alle Spiele mit bekanntem Ergebnis.

Usage:
    python scripts/fix_predicted_winner.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import get_session, Prediction
from src.utils.logger import get_logger

log = get_logger("fix_predicted_winner")


def fix():
    session = get_session()
    all_preds = session.query(Prediction).all()

    fixed = 0
    corrected = 0
    draw_found = 0

    for p in all_preds:
        # Sicherheitscheck — Wahrscheinlichkeiten müssen vorhanden sein
        prob_h = p.prob_home_win or 0
        prob_d = p.prob_draw or 0
        prob_a = p.prob_away_win or 0

        if prob_h == 0 and prob_d == 0 and prob_a == 0:
            continue  # Keine Wahrscheinlichkeiten gespeichert — überspringen

        # Neue Logik: höchste Wahrscheinlichkeit gewinnt
        if prob_d > prob_h and prob_d > prob_a:
            new_winner = "D"
            draw_found += 1
        elif prob_h >= prob_a:
            new_winner = "H"
        else:
            new_winner = "A"

        old_winner = p.predicted_winner

        if old_winner != new_winner:
            p.predicted_winner = new_winner
            fixed += 1

            # prediction_correct neu berechnen wenn Ergebnis bekannt
            if p.actual_result is not None:
                old_correct = p.prediction_correct
                p.prediction_correct = (new_winner == p.actual_result)
                if old_correct != p.prediction_correct:
                    corrected += 1

        elif p.actual_result is not None:
            # Auch wenn winner gleich bleibt — prediction_correct sicherstellen
            p.prediction_correct = (new_winner == p.actual_result)

    session.commit()
    session.close()

    # Statistiken
    session2 = get_session()
    with_result = session2.query(Prediction).filter(Prediction.actual_result != None).all()
    correct = sum(1 for p in with_result if p.prediction_correct)
    total = len(with_result)
    
    from collections import Counter
    winner_dist = Counter(p.predicted_winner for p in session2.query(Prediction).all())
    result_dist = Counter(p.actual_result for p in with_result)
    session2.close()

    print(f"\n{'='*60}")
    print(f"  MIGRATION ABGESCHLOSSEN")
    print(f"{'='*60}")
    print(f"  predicted_winner geändert:  {fixed} Vorhersagen")
    print(f"  davon jetzt 'D' (Draw):     {draw_found}")
    print(f"  prediction_correct geändert: {corrected} Vorhersagen")
    print(f"\n  Neue Verteilung predicted_winner:")
    for k, v in sorted(winner_dist.items()):
        label = {"H": "Heimsieg", "D": "Unentschieden", "A": "Auswärtssieg"}.get(k, k)
        print(f"    {label}: {v}")
    print(f"\n  Echte Ergebnisse (zum Vergleich):")
    for k, v in sorted(result_dist.items()):
        label = {"H": "Heimsieg", "D": "Unentschieden", "A": "Auswärtssieg"}.get(k, k)
        print(f"    {label}: {v}")
    print(f"\n  Neue Trefferquote: {correct}/{total} = {correct/total:.1%}" if total > 0 else "")
    print(f"{'='*60}\n")
    print(f"  Nächster Schritt: XGBoost neu trainieren!")
    print(f"  → Dashboard → Feedback & Training → Training starten")
    print(f"  → oder: python scripts/enter_result.py --train\n")


if __name__ == "__main__":
    fix()