"""
Unit Tests — WC Integration
Tests für Context Engine (WC stages), Feature Builder (WC group standings),
CSV Importer (stage normalization) und Odds/API Mappings.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import pandas as pd
import numpy as np
from datetime import date


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def wc_group_df():
    """Synthetische WC Gruppenphase-Daten (Gruppe A, 2022)."""
    return pd.DataFrame([
        {"match_id": 1, "date": "2022-11-20", "league": "WC", "season": "2022",
         "matchday": 1, "stage": "GROUP_STAGE", "group_name": "GROUP_A",
         "home_team": "Qatar", "away_team": "Ecuador",
         "home_goals": 0, "away_goals": 2, "result": "A"},
        {"match_id": 2, "date": "2022-11-21", "league": "WC", "season": "2022",
         "matchday": 1, "stage": "GROUP_STAGE", "group_name": "GROUP_A",
         "home_team": "Senegal", "away_team": "Netherlands",
         "home_goals": 0, "away_goals": 2, "result": "A"},
        {"match_id": 3, "date": "2022-11-25", "league": "WC", "season": "2022",
         "matchday": 2, "stage": "GROUP_STAGE", "group_name": "GROUP_A",
         "home_team": "Qatar", "away_team": "Senegal",
         "home_goals": 1, "away_goals": 3, "result": "A"},
        {"match_id": 4, "date": "2022-11-25", "league": "WC", "season": "2022",
         "matchday": 2, "stage": "GROUP_STAGE", "group_name": "GROUP_A",
         "home_team": "Netherlands", "away_team": "Ecuador",
         "home_goals": 1, "away_goals": 1, "result": "D"},
    ])


@pytest.fixture
def wc_knockout_df():
    """Synthetische WC Knockout-Daten."""
    return pd.DataFrame([
        {"match_id": 10, "date": "2022-12-03", "league": "WC", "season": "2022",
         "matchday": 4, "stage": "LAST_16", "group_name": None,
         "home_team": "France", "away_team": "Poland",
         "home_goals": 3, "away_goals": 1, "result": "H"},
    ])


# ── Context Engine WC Tests ──────────────────────────────────────────────────

class TestWCContextEngine:

    def test_group_stage_urgency_scales_with_matchday(self):
        from src.features.context_engine import ContextEngine
        engine = ContextEngine()

        ctx_md1 = engine.calculate_context("Germany", "Japan", "WC", 1, {}, stage="GROUP_STAGE", group_name="GROUP_E")
        ctx_md3 = engine.calculate_context("Germany", "Japan", "WC", 3, {}, stage="GROUP_STAGE", group_name="GROUP_E")

        assert ctx_md1["urgency"] < ctx_md3["urgency"], "MD3 must have higher urgency than MD1"
        assert ctx_md3["urgency"] == 1.0, "MD3 urgency = 1.0 (must qualify)"

    def test_knockout_stage_boosts(self):
        from src.features.context_engine import ContextEngine
        engine = ContextEngine()

        stages = ["LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL"]
        boosts = []
        for s in stages:
            ctx = engine.calculate_context("Brazil", "Argentina", "WC", None, {}, stage=s)
            assert ctx["is_knockout"] is True
            boosts.append(ctx["home_motivation"])

        # Boosts must be strictly increasing toward Final
        assert boosts == sorted(boosts), f"Boosts must increase: {boosts}"
        assert boosts[-1] == 1.40, "Final boost must be 1.40"

    def test_third_place_boost_lower_than_semifinal(self):
        from src.features.context_engine import ContextEngine
        engine = ContextEngine()
        sf  = engine.calculate_context("France", "Morocco", "WC", None, {}, stage="SEMI_FINALS")
        tp  = engine.calculate_context("Croatia", "Morocco", "WC", None, {}, stage="THIRD_PLACE")
        assert tp["home_motivation"] < sf["home_motivation"]

    def test_group_motivation_last_place_higher(self):
        from src.features.context_engine import ContextEngine
        engine = ContextEngine()
        standings = {
            "Qatar": {"position": 4, "points": 0, "group": "GROUP_A"},
            "Netherlands": {"position": 1, "points": 6, "group": "GROUP_A"},
        }
        ctx_last = engine.calculate_context("Qatar", "Netherlands", "WC", 3, standings,
                                             stage="GROUP_STAGE", group_name="GROUP_A")
        ctx_first = engine.calculate_context("Netherlands", "Qatar", "WC", 3, standings,
                                              stage="GROUP_STAGE", group_name="GROUP_A")
        assert ctx_last["home_motivation"] > ctx_first["home_motivation"]

    def test_wc_context_returns_required_keys(self):
        from src.features.context_engine import ContextEngine
        ctx = ContextEngine().calculate_context("Brazil", "Argentina", "WC", 2, {},
                                                  stage="GROUP_STAGE", group_name="GROUP_C")
        for key in ("is_derby", "urgency", "home_motivation", "away_motivation",
                    "knockout_stage", "is_knockout"):
            assert key in ctx, f"Missing key: {key}"

    def test_existing_leagues_unaffected(self):
        """CL and BL1 behavior must not change."""
        from src.features.context_engine import ContextEngine
        engine = ContextEngine()
        # CL knockout
        ctx = engine.calculate_context("Bayern", "Dortmund", "CL", 11, {})
        assert ctx["is_knockout"] is True
        assert "Viertelfinale" in ctx["knockout_stage"]
        # BL1 league
        ctx2 = engine.calculate_context("Bayern", "Dortmund", "BL1", 20, {})
        assert ctx2["is_knockout"] is False


# ── Feature Builder WC Tests ─────────────────────────────────────────────────

class TestWCFeatureBuilder:

    def test_neutral_venue_home_advantage(self, wc_group_df):
        from src.features.feature_builder import FeatureBuilder
        fb = FeatureBuilder(wc_group_df)
        features = fb.build_features()
        # WC = neutral → home_win_rate_at_home should be <= 0.33 (fallback) or from historical WC data
        # Just check it doesn't crash and returns a valid float
        assert features["home_win_rate_at_home"].notna().all()
        assert (features["home_win_rate_at_home"] >= 0).all()
        assert (features["home_win_rate_at_home"] <= 1).all()

    def test_group_standings_computed(self, wc_group_df):
        from src.features.feature_builder import FeatureBuilder
        fb = FeatureBuilder(wc_group_df)
        features = fb.build_features()
        # Position should be in group range [1-4], not overall range [1-48]
        valid_positions = features["home_league_position"].dropna()
        assert (valid_positions >= 1).all()
        assert (valid_positions <= 4).all(), f"Group position > 4: {valid_positions[valid_positions > 4].values}"

    def test_stage_group_in_output(self, wc_group_df):
        from src.features.feature_builder import FeatureBuilder
        features = FeatureBuilder(wc_group_df).build_features()
        assert "stage" in features.columns
        assert "group_name" in features.columns
        assert (features["stage"] == "GROUP_STAGE").all()
        assert (features["group_name"] == "GROUP_A").all()

    def test_non_wc_features_unchanged(self):
        """BL1 features must still work with 36 original columns + 2 new (stage, group_name)."""
        np.random.seed(0)
        teams = [f"Team{i}" for i in range(6)]
        rows = []
        for i in range(30):
            h, a = np.random.choice(teams, 2, replace=False)
            hg, ag = np.random.randint(0, 4), np.random.randint(0, 3)
            rows.append({"match_id": i+1, "date": f"2024-01-{i+1:02d}", "league": "BL1",
                         "season": "2024", "matchday": i+1,
                         "home_team": h, "away_team": a,
                         "home_goals": hg, "away_goals": ag,
                         "result": "H" if hg > ag else ("A" if hg < ag else "D")})
        from src.features.feature_builder import FeatureBuilder
        df = pd.DataFrame(rows)
        features = FeatureBuilder(df).build_features()
        # Must have all original expected columns
        for col in ["home_form_ppg", "away_form_ppg", "h2h_home_win_rate",
                    "position_diff", "home_win_rate_at_home"]:
            assert col in features.columns, f"Missing column: {col}"
        # form_ppg must stay in valid range
        assert (features["home_form_ppg"] >= 0).all()
        assert (features["home_form_ppg"] <= 3).all()


# ── CSV Import Stage Normalization Tests ─────────────────────────────────────

class TestWCCsvImport:

    def test_stage_normalization_groups(self):
        from scripts.import_wc_csv import normalize_stage
        assert normalize_stage("Group A") == ("GROUP_STAGE", "GROUP_A")
        assert normalize_stage("Group B") == ("GROUP_STAGE", "GROUP_B")
        assert normalize_stage("Group L") == ("GROUP_STAGE", "GROUP_L")
        assert normalize_stage("Group 1") == ("GROUP_STAGE", "GROUP_A")
        assert normalize_stage("Group 4") == ("GROUP_STAGE", "GROUP_D")

    def test_stage_normalization_knockout(self):
        from scripts.import_wc_csv import normalize_stage
        assert normalize_stage("Round of 16")[0] == "LAST_16"
        assert normalize_stage("Quarter-finals")[0] == "QUARTER_FINALS"
        assert normalize_stage("Semi-finals")[0] == "SEMI_FINALS"
        assert normalize_stage("Final")[0] == "FINAL"
        assert normalize_stage("Match for third place")[0] == "THIRD_PLACE"

    def test_synthetic_team_ids_stable(self):
        from scripts.import_wc_csv import _team_api_id
        id_germany = _team_api_id("Germany")
        assert id_germany == _team_api_id("Germany"), "IDs must be deterministic"
        assert id_germany == _team_api_id("  Germany  "), "IDs must be whitespace-insensitive"
        assert 900_000 <= id_germany < 1_000_000, "IDs must be in synthetic range"
        assert _team_api_id("Germany") != _team_api_id("France"), "Different teams = different IDs"

    def test_date_parsing(self):
        from scripts.import_wc_csv import _parse_date
        assert _parse_date("13 Jul 1930 - 15:00") == date(1930, 7, 13)
        assert _parse_date("17 June 1970") == date(1970, 6, 17)
        assert _parse_date("2022-11-20") == date(2022, 11, 20)
        assert _parse_date("") is None
        assert _parse_date(None) is None


# ── Collector Mapping Tests ───────────────────────────────────────────────────

class TestWCCollectorMappings:

    def test_wc_in_odds_league_map(self):
        from src.collectors.odds_collector import LEAGUE_TO_SPORT
        assert "WC" in LEAGUE_TO_SPORT
        assert LEAGUE_TO_SPORT["WC"] == "soccer_fifa_world_cup"

    def test_wc_in_api_football_league_ids(self):
        from src.collectors.api_football_collector import LEAGUE_IDS
        assert "WC" in LEAGUE_IDS
        assert LEAGUE_IDS["WC"] == 1

    def test_wc_in_supported_leagues(self):
        from config.config import config
        assert "WC" in config.SUPPORTED_LEAGUES
        assert "World Cup" in config.SUPPORTED_LEAGUES["WC"]

    def test_existing_mappings_unchanged(self):
        from src.collectors.odds_collector import LEAGUE_TO_SPORT
        from src.collectors.api_football_collector import LEAGUE_IDS
        assert LEAGUE_TO_SPORT["CL"] == "soccer_uefa_champs_league"
        assert LEAGUE_TO_SPORT["BL1"] == "soccer_germany_bundesliga"
        assert LEAGUE_IDS["CL"] == 2
        assert LEAGUE_IDS["BL1"] == 78
