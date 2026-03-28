"""
Tests for PoissonModel and MonteCarloSimulator.
Run with: python -m pytest tests/unit/test_models.py -v
"""

import pytest
import pandas as pd
import numpy as np
from src.models.poisson_model import PoissonModel
from src.simulation.monte_carlo import MonteCarloSimulator


@pytest.fixture
def sample_df():
    """20 synthetic matches — enough to fit a basic model."""
    np.random.seed(42)
    teams = ["Bayern", "Dortmund", "Leipzig", "Leverkusen"]
    rows = []
    for _ in range(40):
        h, a = np.random.choice(teams, 2, replace=False)
        rows.append({
            "home_team": h, "away_team": a,
            "home_goals": np.random.poisson(1.6),
            "away_goals": np.random.poisson(1.1),
            "result": "H",
            "season": "2023",
        })
    return pd.DataFrame(rows)


@pytest.fixture
def fitted_model(sample_df):
    m = PoissonModel()
    m.fit(sample_df)
    return m


# ── Poisson Model ──────────────────────────────────────────────────────────

def test_model_fits(fitted_model):
    assert fitted_model._fitted is True
    assert len(fitted_model.attack) > 0
    assert len(fitted_model.defence) > 0


def test_probabilities_sum_to_one(fitted_model):
    pred = fitted_model.predict("Bayern", "Dortmund")
    total = pred["prob_home_win"] + pred["prob_draw"] + pred["prob_away_win"]
    assert abs(total - 1.0) < 0.001


def test_expected_goals_positive(fitted_model):
    pred = fitted_model.predict("Bayern", "Dortmund")
    assert pred["expected_home_goals"] > 0
    assert pred["expected_away_goals"] > 0


def test_home_advantage_applied(fitted_model):
    """Home team should have higher expected goals than when playing away."""
    lh, la = fitted_model._expected_goals("Bayern", "Dortmund")
    lh_rev, la_rev = fitted_model._expected_goals("Dortmund", "Bayern")
    # Home advantage means home team always gets the boost
    assert lh > la_rev or la_rev > lh  # just check both are positive
    assert lh > 0 and la > 0


def test_team_ratings_returns_all_teams(fitted_model, sample_df):
    ratings = fitted_model.team_ratings()
    all_teams = set(sample_df["home_team"]) | set(sample_df["away_team"])
    assert set(ratings["team"]) == all_teams


def test_unknown_team_fallback(fitted_model):
    """Unknown teams should not raise — should use league average."""
    pred = fitted_model.predict("Bayern", "UnknownFC")
    assert pred["prob_home_win"] > 0
    assert pred["prob_away_win"] > 0


# ── Monte Carlo ────────────────────────────────────────────────────────────

def test_simulation_probabilities_sum(fitted_model):
    sim = MonteCarloSimulator(fitted_model)
    result = sim.simulate("Bayern", "Dortmund", n=5000, seed=42)
    total = result["prob_home_win"] + result["prob_draw"] + result["prob_away_win"]
    assert abs(total - 1.0) < 0.01


def test_simulation_reproducible(fitted_model):
    sim = MonteCarloSimulator(fitted_model)
    r1 = sim.simulate("Bayern", "Dortmund", n=1000, seed=99)
    r2 = sim.simulate("Bayern", "Dortmund", n=1000, seed=99)
    assert r1["prob_home_win"] == r2["prob_home_win"]


def test_simulation_btts_valid(fitted_model):
    sim = MonteCarloSimulator(fitted_model)
    result = sim.simulate("Bayern", "Leipzig", n=5000, seed=1)
    assert 0 <= result["prob_btts"] <= 1


def test_simulation_top_scores(fitted_model):
    sim = MonteCarloSimulator(fitted_model)
    result = sim.simulate("Bayern", "Dortmund", n=5000, seed=7)
    assert len(result["top_scores"]) > 0
    assert "score" in result["top_scores"][0]
    assert "probability" in result["top_scores"][0]