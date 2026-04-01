"""
XGBoost Feedback Model
Lernt aus den Fehlern des Poisson-Modells.

Workflow:
  1. Poisson macht Vorhersage → wird in DB gespeichert
  2. Echtes Ergebnis wird eingetragen
  3. XGBoost trainiert auf Feature-Matrix + Prediction-Error
  4. Beim nächsten Predict: Ensemble aus Poisson + XGBoost

Features für XGBoost:
  - Alle Poisson-Features (Form, Verletzung, Wetter, etc.)
  - Poisson-Wahrscheinlichkeiten als Meta-Features
  - Target: tatsächliches Ergebnis (H/D/A)

Usage:
    model = XGBoostFeedbackModel()
    model.train(predictions_with_results)
    proba = model.predict_proba(features)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib
from src.utils.database import get_session, Prediction
from src.utils.logger import get_logger

log = get_logger(__name__)

MODEL_PATH = Path("data/processed/xgboost_model.pkl")
MIN_SAMPLES = 10  # Mindestanzahl Ergebnisse für Training


class XGBoostFeedbackModel:
    """
    XGBoost-Modell das auf echten Vorhersage-Fehlern trainiert.
    Ergänzt das Poisson-Modell mit gelernten Korrekturfaktoren.
    """

    def __init__(self):
        self.model = None
        self.is_trained = False
        self.n_training_samples = 0
        self.accuracy = None
        self.trained_at = None

        # Versuche gespeichertes Modell zu laden
        self._load()

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self) -> dict:
        """
        Trainiert XGBoost auf allen Vorhersagen mit bekanntem Ergebnis.
        Gibt Training-Report zurück.
        """
        try:
            from xgboost import XGBClassifier
        except ImportError:
            log.error("xgboost nicht installiert. Führe aus: pip install xgboost")
            return {"success": False, "error": "xgboost not installed"}

        # Lade Vorhersagen mit Ergebnissen aus DB
        df = self._load_training_data()

        if len(df) < MIN_SAMPLES:
            msg = f"Zu wenig Daten: {len(df)} / {MIN_SAMPLES} Ergebnisse nötig"
            log.warning(msg)
            return {"success": False, "error": msg, "n_samples": len(df)}

        log.info(f"Trainiere XGBoost auf {len(df)} Vorhersagen...")

        X, y = self._build_features(df)

        # Train/Test Split (letzte 20% als Test)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss",
            use_label_encoder=False,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred = self.model.predict(X_test)
        self.accuracy = float(np.mean(y_pred == y_test))
        self.n_training_samples = len(df)
        self.is_trained = True
        self.trained_at = datetime.now()

        # Feature Importance
        importance = dict(zip(
            self._feature_names(),
            self.model.feature_importances_
        ))
        top_features = sorted(importance.items(), key=lambda x: -x[1])[:5]

        self._save()

        result = {
            "success":          True,
            "n_samples":        len(df),
            "train_size":       split,
            "test_size":        len(X_test),
            "accuracy":         round(self.accuracy, 3),
            "top_features":     top_features,
            "trained_at":       self.trained_at.strftime("%d.%m.%Y %H:%M"),
        }

        log.success(
            f"XGBoost trainiert: Accuracy={self.accuracy:.1%} "
            f"auf {len(X_test)} Test-Spielen"
        )
        return result

    def predict_proba(
        self,
        poisson_home: float,
        poisson_draw: float,
        poisson_away: float,
        home_form_ppg: float = 1.5,
        away_form_ppg: float = 1.5,
        home_injury: float = 0.0,
        away_injury: float = 0.0,
        weather_impact: float = 0.0,
        home_form_factor: float = 1.0,
        away_form_factor: float = 1.0,
    ) -> dict | None:
        """
        Gibt XGBoost-Wahrscheinlichkeiten zurück wenn Modell trainiert.
        Returns None wenn nicht genug Daten.
        """
        if not self.is_trained or self.model is None:
            return None

        X = np.array([[
            poisson_home, poisson_draw, poisson_away,
            home_form_ppg, away_form_ppg,
            home_injury, away_injury,
            weather_impact,
            home_form_factor, away_form_factor,
            home_form_ppg - away_form_ppg,
            home_injury - away_injury,
        ]])

        proba = self.model.predict_proba(X)[0]
        # XGBoost labels: 0=A, 1=D, 2=H
        return {
            "prob_home_win": round(float(proba[2]), 4),
            "prob_draw":     round(float(proba[1]), 4),
            "prob_away_win": round(float(proba[0]), 4),
        }

    def get_ensemble(
        self,
        poisson_result: dict,
        weight_xgb: float = 0.35,
    ) -> dict:
        """
        Kombiniert Poisson + XGBoost zu Ensemble-Vorhersage.
        weight_xgb = Anteil XGBoost (0.35 = 35% XGB, 65% Poisson)
        """
        xgb = self.predict_proba(
            poisson_home=poisson_result["prob_home_win"],
            poisson_draw=poisson_result["prob_draw"],
            poisson_away=poisson_result["prob_away_win"],
            home_form_ppg=poisson_result.get("home_form_ppg", 1.5),
            away_form_ppg=poisson_result.get("away_form_ppg", 1.5),
            home_injury=poisson_result.get("home_injury_impact", 0),
            away_injury=poisson_result.get("away_injury_impact", 0),
            weather_impact=poisson_result.get("weather_impact", 0),
            home_form_factor=poisson_result.get("home_form_factor", 1.0),
            away_form_factor=poisson_result.get("away_form_factor", 1.0),
        )

        if xgb is None:
            return poisson_result  # Nur Poisson wenn kein XGBoost

        w_p = 1 - weight_xgb
        ensemble = {
            **poisson_result,
            "prob_home_win": round(w_p * poisson_result["prob_home_win"] + weight_xgb * xgb["prob_home_win"], 4),
            "prob_draw":     round(w_p * poisson_result["prob_draw"]     + weight_xgb * xgb["prob_draw"],     4),
            "prob_away_win": round(w_p * poisson_result["prob_away_win"] + weight_xgb * xgb["prob_away_win"], 4),
            "xgb_proba":     xgb,
            "ensemble_weight_xgb": weight_xgb,
            "model": "ensemble",
        }
        return ensemble

    # ── Vorhersagen speichern ─────────────────────────────────────────────────

    @staticmethod
    def save_prediction(result: dict, league: str = "BL1") -> int:
        """Speichert eine Vorhersage in der DB. Gibt die ID zurück."""
        from datetime import datetime as dt

        def _res(h, a):
            if h > a: return "H"
            if h < a: return "A"
            return "D"

        pred_result = _res(
            result.get("expected_home_goals", 1),
            result.get("expected_away_goals", 1)
        )

        session = get_session()
        p = Prediction(
            created_at          = dt.now(),
            league              = league,
            home_team           = result["home_team"],
            away_team           = result["away_team"],
            prob_home_win       = result.get("prob_home_win"),
            prob_draw           = result.get("prob_draw"),
            prob_away_win       = result.get("prob_away_win"),
            expected_home_goals = result.get("expected_home_goals"),
            expected_away_goals = result.get("expected_away_goals"),
            predicted_winner    = pred_result,
            confidence          = result.get("confidence"),
            simulation_runs     = result.get("simulations"),
            home_injury_impact  = result.get("home_injury_impact", 0),
            away_injury_impact  = result.get("away_injury_impact", 0),
            home_form_ppg       = result.get("home_form", {}).get("form_ppg", 1.5),
            away_form_ppg       = result.get("away_form", {}).get("form_ppg", 1.5),
            weather_impact      = result.get("weather_impact", 0),
            home_form_factor    = result.get("home_form_factor", 1.0),
            away_form_factor    = result.get("away_form_factor", 1.0),
        )
        session.add(p)
        session.commit()
        pred_id = p.id
        session.close()
        log.success(f"Vorhersage gespeichert: ID={pred_id} ({result['home_team']} vs {result['away_team']})")
        return pred_id

    @staticmethod
    def enter_result(prediction_id: int, home_goals: int, away_goals: int) -> bool:
        """Trägt das echte Ergebnis nach dem Spiel ein."""
        from datetime import datetime as dt

        def _res(h, a):
            if h > a: return "H"
            if h < a: return "A"
            return "D"

        session = get_session()
        pred = session.query(Prediction).filter_by(id=prediction_id).first()

        if not pred:
            log.error(f"Vorhersage ID={prediction_id} nicht gefunden")
            session.close()
            return False

        actual = _res(home_goals, away_goals)
        pred.actual_home_goals  = home_goals
        pred.actual_away_goals  = away_goals
        pred.actual_result      = actual
        pred.result_entered_at  = dt.now()
        pred.prediction_correct = (pred.predicted_winner == actual)

        session.commit()
        correct = pred.prediction_correct
        session.close()

        log.success(
            f"Ergebnis eingetragen: {pred.home_team} {home_goals}:{away_goals} {pred.away_team} "
            f"→ {'✅ RICHTIG' if correct else '❌ FALSCH'}"
        )
        return True

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_training_data(self) -> pd.DataFrame:
        """Lädt alle Vorhersagen mit eingetragenem Ergebnis."""
        session = get_session()
        preds = session.query(Prediction).filter(
            Prediction.actual_result != None
        ).all()
        session.close()

        rows = []
        for p in preds:
            rows.append({
                "prob_home_win":    p.prob_home_win or 0.33,
                "prob_draw":        p.prob_draw or 0.33,
                "prob_away_win":    p.prob_away_win or 0.33,
                "home_form_ppg":    p.home_form_ppg or 1.5,
                "away_form_ppg":    p.away_form_ppg or 1.5,
                "home_injury":      p.home_injury_impact or 0,
                "away_injury":      p.away_injury_impact or 0,
                "weather_impact":   p.weather_impact or 0,
                "home_form_factor": p.home_form_factor or 1.0,
                "away_form_factor": p.away_form_factor or 1.0,
                "form_diff":        (p.home_form_ppg or 1.5) - (p.away_form_ppg or 1.5),
                "injury_diff":      (p.home_injury_impact or 0) - (p.away_injury_impact or 0),
                "result":           p.actual_result,
            })

        return pd.DataFrame(rows)

    def _build_features(self, df: pd.DataFrame):
        feature_cols = self._feature_names()
        X = df[feature_cols].values
        y = df["result"].map({"H": 2, "D": 1, "A": 0}).values
        return X, y

    @staticmethod
    def _feature_names():
        return [
            "prob_home_win", "prob_draw", "prob_away_win",
            "home_form_ppg", "away_form_ppg",
            "home_injury", "away_injury",
            "weather_impact", "home_form_factor", "away_form_factor",
            "form_diff", "injury_diff",
        ]

    def _save(self):
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model":    self.model,
            "accuracy": self.accuracy,
            "n_samples": self.n_training_samples,
            "trained_at": self.trained_at,
        }, MODEL_PATH)
        log.info(f"Modell gespeichert → {MODEL_PATH}")

    def _load(self):
        if MODEL_PATH.exists():
            try:
                data = joblib.load(MODEL_PATH)
                self.model              = data["model"]
                self.accuracy           = data["accuracy"]
                self.n_training_samples = data["n_samples"]
                self.trained_at         = data["trained_at"]
                self.is_trained         = True
                log.info(
                    f"XGBoost geladen: Accuracy={self.accuracy:.1%}, "
                    f"{self.n_training_samples} Trainingssamples"
                )
            except Exception as e:
                log.warning(f"Konnte XGBoost-Modell nicht laden: {e}")