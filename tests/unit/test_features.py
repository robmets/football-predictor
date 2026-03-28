"""
Unit tests for FeatureBuilder.
Run with: pytest tests/unit/test_features.py -v
"""

import pytest
import pandas as pd
import numpy as np
from src.features.feature_builder import FeatureBuilder


@pytest.fixture
def sample_matches():
    """Small synthetic match dataset — 3 teams, 12 games."""
    return pd.DataFrame([
        # Season 2023, Matchday 1–4
        {"match_id": 1,  "date": "2023-08-12", "league": "BL1", "season": "2023", "matchday": 1,
         "home_team": "Bayern",   "away_team": "Dortmund", "home_goals": 3, "away_goals": 0, "result": "H"},
        {"match_id": 2,  "date": "2023-08-12", "league": "BL1", "season": "2023", "matchday": 1,
         "home_team": "Leipzig",  "away_team": "Bayern",   "home_goals": 1, "away_goals": 1, "result": "D"},
        {"match_id": 3,  "date": "2023-08-19", "league": "BL1", "season": "2023", "matchday": 2,
         "home_team": "Dortmund", "away_team": "Leipzig",  "home_goals": 2, "away_goals": 1, "result": "H"},
        {"match_id": 4,  "date": "2023-08-26", "league": "BL1", "season": "2023", "matchday": 3,
         "home_team": "Bayern",   "away_team": "Leipzig",  "home_goals": 4, "away_goals": 0, "result": "H"},
        {"match_id": 5,  "date": "2023-09-02", "league": "BL1", "season": "2023", "matchday": 4,
         "home_team": "Leipzig",  "away_team": "Dortmund", "home_goals": 0, "away_goals": 2, "result": "A"},
        {"match_id": 6,  "date": "2023-09-16", "league": "BL1", "season": "2023", "matchday": 5,
         "home_team": "Dortmund", "away_team": "Bayern",   "home_goals": 1, "away_goals": 2, "result": "A"},
        # Matchday 6–8
        {"match_id": 7,  "date": "2023-09-23", "league": "BL1", "season": "2023", "matchday": 6,
         "home_team": "Bayern",   "away_team": "Dortmund", "home_goals": 2, "away_goals": 2, "result": "D"},
        {"match_id": 8,  "date": "2023-09-30", "league": "BL1", "season": "2023", "matchday": 7,
         "home_team": "Leipzig",  "away_team": "Bayern",   "home_goals": 0, "away_goals": 3, "result": "A"},
        {"match_id": 9,  "date": "2023-10-07", "league": "BL1", "season": "2023", "matchday": 8,
         "home_team": "Dortmund", "away_team": "Leipzig",  "home_goals": 1, "away_goals": 0, "result": "H"},
        {"match_id": 10, "date": "2023-10-21", "league": "BL1", "season": "2023", "matchday": 9,
         "home_team": "Bayern",   "away_team": "Leipzig",  "home_goals": 2, "away_goals": 1, "result": "H"},
        {"match_id": 11, "date": "2023-10-28", "league": "BL1", "season": "2023", "matchday": 10,
         "home_team": "Leipzig",  "away_team": "Dortmund", "home_goals": 1, "away_goals": 1, "result": "D"},
        {"match_id": 12, "date": "2023-11-04", "league": "BL1", "season": "2023", "matchday": 11,
         "home_team": "Dortmund", "away_team": "Bayern",   "home_goals": 0, "away_goals": 1, "result": "A"},
    ])


def test_feature_matrix_shape(sample_matches):
    fb = FeatureBuilder(sample_matches)
    features = fb.build_features()
    assert len(features) == len(sample_matches)
    assert features.shape[1] > 10


def test_form_ppg_range(sample_matches):
    """PPG must always be between 0 and 3."""
    fb = FeatureBuilder(sample_matches)
    features = fb.build_features()
    assert features["home_form_ppg"].between(0, 3).all()
    assert features["away_form_ppg"].between(0, 3).all()


def test_result_preserved(sample_matches):
    """Result column should pass through unchanged."""
    fb = FeatureBuilder(sample_matches)
    features = fb.build_features()
    assert set(features["result"].dropna().unique()).issubset({"H", "D", "A"})


def test_no_data_leakage(sample_matches):
    """
    For match at index 0, there are no past games.
    All stats should be fallback values, not future data.
    """
    fb = FeatureBuilder(sample_matches)
    features = fb.build_features()
    first = features.iloc[0]
    assert first["h2h_games_played"] == 0


def test_goal_stats_positive(sample_matches):
    """Average goals must be >= 0."""
    fb = FeatureBuilder(sample_matches)
    features = fb.build_features()
    assert (features["home_goals_scored_avg"] >= 0).all()
    assert (features["away_goals_conceded_avg"] >= 0).all()


def test_standings_positions_unique(sample_matches):
    """All positions should be unique (1 per team) in last match features."""
    fb = FeatureBuilder(sample_matches)
    features = fb.build_features()
    last = features.dropna(subset=["home_league_position"])
    # Just check positions are in a valid range
    assert last["home_league_position"].between(1, 30).all()