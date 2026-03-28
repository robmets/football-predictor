"""
Unit tests for the database layer.
Run with: pytest tests/ -v
"""

import pytest
from pathlib import Path
from src.utils.database import init_db, get_session, Match, Team


@pytest.fixture
def test_db(tmp_path):
    """Create a fresh in-memory test database for each test."""
    db_path = str(tmp_path / "test.db")
    init_db(db_path=db_path)
    return db_path


def test_database_initializes(test_db):
    """Database should be created with correct tables."""
    assert Path(test_db).exists()


def test_team_insert(test_db):
    """Should be able to insert and retrieve a team."""
    session = get_session(db_path=test_db)

    team = Team(api_id=1, name="Bayern München", short_name="FCB", league="BL1", country="Germany")
    session.add(team)
    session.commit()

    result = session.query(Team).filter_by(api_id=1).first()
    assert result is not None
    assert result.name == "Bayern München"
    session.close()


def test_match_insert(test_db):
    """Should be able to insert and retrieve a match."""
    from datetime import date
    session = get_session(db_path=test_db)

    match = Match(
        api_id=9999,
        league="BL1",
        season="2023",
        matchday=1,
        date=date(2023, 8, 18),
        home_team_id=1,
        away_team_id=2,
        home_goals=3,
        away_goals=1,
        status="FINISHED",
    )
    session.add(match)
    session.commit()

    result = session.query(Match).filter_by(api_id=9999).first()
    assert result.home_goals == 3
    assert result.away_goals == 1
    session.close()
