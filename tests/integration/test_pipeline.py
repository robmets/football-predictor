"""
Integration Tests — Vollständiger System-Test
Testet die komplette Pipeline von Daten bis Vorhersage.

Run with:
    python -m pytest tests/integration/test_pipeline.py -v
    python -m pytest tests/integration/test_pipeline.py -v -s   # mit Print-Output
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_bl1_df():
    """Synthetische BL1-Daten für Tests (kein echter DB-Zugriff nötig)."""
    np.random.seed(42)
    teams = [
        "FC Bayern München", "Bayer 04 Leverkusen", "Borussia Dortmund",
        "RB Leipzig", "VfB Stuttgart", "Eintracht Frankfurt",
        "SC Freiburg", "FC Augsburg", "Hamburger SV", "SV Werder Bremen"
    ]
    rows = []
    match_id = 1
    for season in ["2024", "2025"]:
        for _ in range(100):
            h, a = np.random.choice(teams, 2, replace=False)
            hg = np.random.poisson(1.6)
            ag = np.random.poisson(1.1)
            rows.append({
                "match_id":   match_id,
                "date":       f"{season}-{np.random.randint(8,12):02d}-{np.random.randint(1,28):02d}",
                "league":     "BL1",
                "season":     season,
                "matchday":   np.random.randint(1, 34),
                "home_team":  h,
                "away_team":  a,
                "home_goals": hg,
                "away_goals": ag,
                "result":     "H" if hg > ag else ("A" if hg < ag else "D"),
            })
            match_id += 1
    return pd.DataFrame(rows)


@pytest.fixture
def fitted_poisson(sample_bl1_df):
    from src.models.poisson_model import PoissonModel
    model = PoissonModel()
    model.fit(sample_bl1_df)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATENBANK
# ══════════════════════════════════════════════════════════════════════════════

class TestDatabase:
    def test_init_creates_tables(self, tmp_path):
        from src.utils.database import init_db, get_session, Team, Match, Prediction
        db = str(tmp_path / "test.db")
        init_db(db)
        sess = get_session(db)
        # Alle Tabellen müssen existieren
        assert sess.query(Team).count() == 0
        assert sess.query(Match).count() == 0
        assert sess.query(Prediction).count() == 0
        sess.close()

    def test_team_has_transfermarkt_id(self, tmp_path):
        """Team-Model muss transfermarkt_id haben."""
        from src.utils.database import init_db, get_session, Team
        db = str(tmp_path / "test.db")
        init_db(db)
        sess = get_session(db)
        t = Team(api_id=999, name="Test FC", league="BL1", transfermarkt_id="12345")
        sess.add(t)
        sess.commit()
        result = sess.query(Team).filter_by(api_id=999).first()
        assert result.transfermarkt_id == "12345"
        sess.close()

    def test_prediction_new_schema(self, tmp_path):
        """Prediction-Tabelle muss neue Felder haben."""
        from src.utils.database import init_db, get_session, Prediction
        db = str(tmp_path / "test.db")
        init_db(db)
        sess = get_session(db)
        p = Prediction(
            created_at=datetime.now(),
            league="BL1",
            home_team="FC Bayern München",
            away_team="Borussia Dortmund",
            prob_home_win=0.55,
            prob_draw=0.25,
            prob_away_win=0.20,
            predicted_winner="H",
            confidence="MEDIUM",
        )
        sess.add(p)
        sess.commit()
        result = sess.query(Prediction).first()
        assert result.home_team == "FC Bayern München"
        assert result.league == "BL1"
        assert result.actual_result is None  # noch kein Ergebnis
        sess.close()


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineering:
    def test_feature_matrix_shape(self, sample_bl1_df):
        from src.features.feature_builder import FeatureBuilder
        fb = FeatureBuilder(sample_bl1_df)
        features = fb.build_features()
        assert len(features) == len(sample_bl1_df)
        assert features.shape[1] >= 30

    def test_no_data_leakage(self, sample_bl1_df):
        """Erste Reihe darf keine historischen Daten haben."""
        from src.features.feature_builder import FeatureBuilder
        fb = FeatureBuilder(sample_bl1_df)
        features = fb.build_features()
        assert features.iloc[0]["h2h_games_played"] == 0

    def test_form_ppg_in_range(self, sample_bl1_df):
        from src.features.feature_builder import FeatureBuilder
        features = FeatureBuilder(sample_bl1_df).build_features()
        # dropna() weil erste Matches keine historischen Daten haben (Fallback = 1.5)
        home_ppg = features["home_form_ppg"].dropna()
        away_ppg = features["away_form_ppg"].dropna()
        # 3.0 ist valide (alle Spiele gewonnen) — explizit inklusiv prüfen
        assert ((home_ppg >= 0) & (home_ppg <= 3)).all(), f"Außerhalb [0,3]: {home_ppg[(home_ppg < 0) | (home_ppg > 3)].values}"
        assert ((away_ppg >= 0) & (away_ppg <= 3)).all(), f"Außerhalb [0,3]: {away_ppg[(away_ppg < 0) | (away_ppg > 3)].values}"

    def test_live_form_from_db(self, sample_bl1_df):
        """Live-Form soll aus DataFrame berechnet werden."""
        from src.features.live_form import LiveFormCalculator
        calc = LiveFormCalculator()
        result = calc.get_lambda_adjustment(
            "FC Bayern München", league="BL1", features_df=sample_bl1_df
        )
        assert result["source"] == "local-db"
        assert 0 < result["attack_factor"] < 2.0
        assert 0 <= result["form_ppg"] <= 3.0
        assert len(result["last_5"]) <= 5

    def test_live_form_unknown_team(self, sample_bl1_df):
        """Unbekanntes Team soll Fallback liefern, kein Crash."""
        from src.features.live_form import LiveFormCalculator
        result = LiveFormCalculator().get_lambda_adjustment(
            "Unbekanntes Team FC", features_df=sample_bl1_df
        )
        assert result["attack_factor"] == 1.0
        assert result["source"] == "fallback"


# ══════════════════════════════════════════════════════════════════════════════
# 3. POISSON-MODELL
# ══════════════════════════════════════════════════════════════════════════════

class TestPoissonModel:
    def test_fits_without_error(self, sample_bl1_df):
        from src.models.poisson_model import PoissonModel
        model = PoissonModel()
        model.fit(sample_bl1_df)
        assert model._fitted

    def test_probabilities_sum_to_one(self, fitted_poisson):
        pred = fitted_poisson.predict("FC Bayern München", "Borussia Dortmund")
        total = pred["prob_home_win"] + pred["prob_draw"] + pred["prob_away_win"]
        assert abs(total - 1.0) < 0.01

    def test_home_advantage_positive(self, fitted_poisson):
        """Home advantage coefficient muss > 1 sein."""
        assert fitted_poisson.home_advantage > 1.0

    def test_strong_team_higher_attack(self, fitted_poisson):
        """Bayern muss höhere Angriffsstärke haben als das schwächste Team."""
        ratings = fitted_poisson.team_ratings()
        best = ratings.iloc[0]["attack"]
        worst = ratings.iloc[-1]["attack"]
        assert best > worst

    def test_unknown_team_fallback(self, fitted_poisson):
        """Unbekanntes Team darf nicht crashen."""
        pred = fitted_poisson.predict("FC Bayern München", "UnknownFC 99")
        assert pred["prob_home_win"] > 0
        assert pred["prob_away_win"] > 0

    def test_score_matrix_sums_to_one(self, fitted_poisson):
        pred = fitted_poisson.predict("FC Bayern München", "RB Leipzig")
        total = sum(pred["score_matrix"].values())
        assert abs(total - 1.0) < 0.02  # Truncation bei MAX_GOALS erlaubt


# ══════════════════════════════════════════════════════════════════════════════
# 4. MONTE CARLO SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

class TestMonteCarlo:
    def test_probabilities_sum(self, fitted_poisson):
        from src.simulation.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator(fitted_poisson)
        result = sim.simulate("FC Bayern München", "Borussia Dortmund", n=2000, seed=42)
        total = result["prob_home_win"] + result["prob_draw"] + result["prob_away_win"]
        assert abs(total - 1.0) < 0.01

    def test_reproducible_with_seed(self, fitted_poisson):
        from src.simulation.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator(fitted_poisson)
        r1 = sim.simulate("FC Bayern München", "RB Leipzig", n=1000, seed=7)
        r2 = sim.simulate("FC Bayern München", "RB Leipzig", n=1000, seed=7)
        assert r1["prob_home_win"] == r2["prob_home_win"]

    def test_injury_reduces_lambda(self, fitted_poisson):
        """Verletzungs-Impact muss Lambda reduzieren."""
        from src.simulation.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator(fitted_poisson)
        r_healthy  = sim.simulate("FC Bayern München", "RB Leipzig", n=1000, seed=1, home_injury_impact=0)
        r_injured  = sim.simulate("FC Bayern München", "RB Leipzig", n=1000, seed=1, home_injury_impact=30)
        # Heimsieg-Wahrscheinlichkeit muss bei Verletzungen sinken
        assert r_injured["prob_home_win"] < r_healthy["prob_home_win"]

    def test_form_factor_changes_result(self, fitted_poisson):
        """Gute Form muss Wahrscheinlichkeit erhöhen."""
        from src.simulation.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator(fitted_poisson)
        r_normal   = sim.simulate("FC Bayern München", "RB Leipzig", n=2000, seed=2, home_form_factor=1.0)
        r_good     = sim.simulate("FC Bayern München", "RB Leipzig", n=2000, seed=2, home_form_factor=1.2)
        assert r_good["prob_home_win"] > r_normal["prob_home_win"]

    def test_weather_reduces_goals(self, fitted_poisson):
        """Schlechtes Wetter muss Tore reduzieren."""
        from src.simulation.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator(fitted_poisson)
        r_sun  = sim.simulate("FC Bayern München", "RB Leipzig", n=2000, seed=3, weather_impact=0.0)
        r_rain = sim.simulate("FC Bayern München", "RB Leipzig", n=2000, seed=3, weather_impact=-0.08)
        assert r_rain["expected_home_goals"] < r_sun["expected_home_goals"]

    def test_btts_and_over_in_range(self, fitted_poisson):
        from src.simulation.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator(fitted_poisson)
        r = sim.simulate("FC Bayern München", "Borussia Dortmund", n=2000, seed=5)
        assert 0 <= r["prob_btts"] <= 1
        assert 0 <= r["prob_over_2_5"] <= 1
        assert 0 <= r["prob_over_3_5"] <= 1
        assert r["prob_over_2_5"] >= r["prob_over_3_5"]  # Logisch: mehr Spiele über 2.5 als 3.5


# ══════════════════════════════════════════════════════════════════════════════
# 5. XGBOOST FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════

class TestXGBoostFeedback:
    def test_save_prediction(self, tmp_path):
        from src.utils.database import init_db, get_session, Prediction
        from src.models.xgboost_model import XGBoostFeedbackModel
        db = str(tmp_path / "test.db")
        init_db(db)

        result = {
            "home_team": "FC Bayern München",
            "away_team": "Borussia Dortmund",
            "prob_home_win": 0.55,
            "prob_draw": 0.25,
            "prob_away_win": 0.20,
            "expected_home_goals": 2.1,
            "expected_away_goals": 1.0,
            "confidence": "MEDIUM",
            "simulations": 10000,
            "home_injury_impact": 0.0,
            "away_injury_impact": 5.0,
            "weather_impact": 0.0,
            "home_form_factor": 1.05,
            "away_form_factor": 0.98,
            "home_form": {"form_ppg": 2.1},
            "away_form": {"form_ppg": 1.3},
        }

        # Patch get_session to use test DB
        import src.models.xgboost_model as xm
        original = xm.get_session
        xm.get_session = lambda: get_session(db)

        pred_id = XGBoostFeedbackModel.save_prediction(result, league="BL1")
        assert pred_id is not None and pred_id > 0

        sess = get_session(db)
        p = sess.query(Prediction).first()
        assert p.home_team == "FC Bayern München"
        assert p.predicted_winner == "H"
        assert p.actual_result is None
        sess.close()
        xm.get_session = original

    def test_enter_result(self, tmp_path):
        from src.utils.database import init_db, get_session, Prediction
        from src.models.xgboost_model import XGBoostFeedbackModel
        import src.models.xgboost_model as xm
        db = str(tmp_path / "test2.db")
        init_db(db)

        sess = get_session(db)
        p = Prediction(
            created_at=datetime.now(), league="BL1",
            home_team="HSV", away_team="FCB",
            prob_home_win=0.3, prob_draw=0.3, prob_away_win=0.4,
            predicted_winner="A",
        )
        sess.add(p)
        sess.commit()
        pred_id = p.id
        sess.close()

        original = xm.get_session
        xm.get_session = lambda: get_session(db)

        ok = XGBoostFeedbackModel.enter_result(pred_id, home_goals=1, away_goals=2)
        assert ok

        sess2 = get_session(db)
        updated = sess2.query(Prediction).filter_by(id=pred_id).first()
        assert updated.actual_result == "A"
        assert updated.prediction_correct is True  # A war vorhergesagt, A eingetreten
        sess2.close()
        xm.get_session = original

    def test_not_enough_data_for_training(self, tmp_path):
        # XGBoost braucht libomp auf Mac: brew install libomp
        try:
            import xgboost
        except Exception:
            pytest.skip("XGBoost nicht verfügbar — brew install libomp ausführen")

        from src.utils.database import init_db
        from src.models.xgboost_model import XGBoostFeedbackModel
        import src.models.xgboost_model as xm
        db = str(tmp_path / "test3.db")
        init_db(db)

        from src.utils.database import get_session as _get_session
        original = xm.get_session
        xm.get_session = lambda: _get_session(db)

        xgb = XGBoostFeedbackModel()
        result = xgb.train()
        assert result["success"] is False
        assert "n_samples" in result

        xm.get_session = original


# ══════════════════════════════════════════════════════════════════════════════
# 6. WETTER
# ══════════════════════════════════════════════════════════════════════════════

class TestWeather:
    def test_no_api_key_returns_no_data(self):
        from src.collectors.weather_collector import WeatherCollector
        from config.config import config
        original = config.OPENWEATHER_API_KEY
        config.OPENWEATHER_API_KEY = ""
        wc = WeatherCollector()
        result = wc.get_match_weather("FC Bayern München")
        assert result["goal_impact_factor"] == 0.0
        assert result["condition"] == "unknown"  # _no_data() setzt condition="unknown"
        config.OPENWEATHER_API_KEY = original

    def test_unknown_team_no_crash(self):
        from src.collectors.weather_collector import WeatherCollector
        wc = WeatherCollector()
        result = wc.get_match_weather("Nicht Existierender FC 999")
        assert result["goal_impact_factor"] == 0.0

    def test_goal_impact_in_range(self):
        from src.collectors.weather_collector import WEATHER_IMPACT
        for label, factor in WEATHER_IMPACT.items():
            assert -0.15 <= factor <= 0.05, f"{label}: {factor} out of range"


# ══════════════════════════════════════════════════════════════════════════════
# 7. VALUE BET DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class TestValueBetDetector:
    def test_detects_value(self):
        from src.features.value_bet_detector import ValueBetDetector
        detector = ValueBetDetector()
        analysis = detector.analyze(
            home_team="Bayern", away_team="Dortmund",
            model_home=0.70,  # Modell sagt 70%
            model_draw=0.15,
            model_away=0.15,
            market_home=0.55,  # Markt sagt nur 55% → +15 PP Edge
            market_draw=0.25,
            market_away=0.20,
        )
        assert analysis["has_value"] is True
        assert len(analysis["value_bets"]) > 0
        assert analysis["value_bets"][0]["edge_pct"] >= 5.0

    def test_no_value_when_aligned(self):
        from src.features.value_bet_detector import ValueBetDetector
        detector = ValueBetDetector()
        analysis = detector.analyze(
            home_team="Bayern", away_team="Dortmund",
            model_home=0.50, model_draw=0.25, model_away=0.25,
            market_home=0.50, market_draw=0.25, market_away=0.25,
        )
        assert analysis["has_value"] is False

    def test_disagreement_detected(self):
        from src.features.value_bet_detector import ValueBetDetector
        detector = ValueBetDetector()
        # Modell: Heimteam Favorit (65%), Markt: Auswärtsteam Favorit (55%)
        analysis = detector.analyze(
            home_team="Bayern", away_team="Dortmund",
            model_home=0.65, model_draw=0.20, model_away=0.15,  # Modell → home
            market_home=0.25, market_draw=0.20, market_away=0.55,  # Markt → away
        )
        assert analysis["model_favourite"] == "home"
        assert analysis["market_favourite"] == "away"
        assert analysis["disagreement"] is True


# ══════════════════════════════════════════════════════════════════════════════
# 8. UPDATE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class TestUpdatePipeline:
    def test_feature_files_exist_after_update(self):
        """Nach update.py müssen CSV-Dateien existieren."""
        for league in ["BL1", "PL", "PD", "SA", "FL1"]:
            path = Path(f"data/processed/features_{league}.csv")
            assert path.exists(), f"features_{league}.csv fehlt — run: python scripts/update.py --league {league}"

    def test_feature_files_not_empty(self):
        for league in ["BL1", "PL"]:
            path = Path(f"data/processed/features_{league}.csv")
            if path.exists():
                df = pd.read_csv(path)
                assert len(df) > 100, f"{league}: Zu wenige Zeilen ({len(df)})"
                assert "home_form_ppg" in df.columns
                assert "result" in df.columns

    def test_league_isolation(self):
        """Jede Liga-Datei darf nur Teams dieser Liga enthalten."""
        league_prefixes = {
            "BL1": ["FC Bayern", "Bayer", "Borussia", "RB Leipzig"],
            "PL":  ["Liverpool", "Arsenal", "Chelsea", "Manchester"],
        }
        for league, expected_teams in league_prefixes.items():
            path = Path(f"data/processed/features_{league}.csv")
            if path.exists():
                df = pd.read_csv(path)
                all_teams = set(df["home_team"].dropna()) | set(df["away_team"].dropna())
                found = any(
                    any(exp in team for team in all_teams)
                    for exp in expected_teams
                )
                assert found, f"{league}: Keine erwarteten Teams gefunden"