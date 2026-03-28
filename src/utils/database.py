"""
Database layer — SQLite via SQLAlchemy.
Creates all tables on first run.
"""

from sqlalchemy import (
    create_engine, text,
    Column, Integer, String, Float, Date, DateTime, Boolean,
    ForeignKey, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from pathlib import Path
from src.utils.logger import get_logger

log = get_logger(__name__)


class Base(DeclarativeBase):
    pass


# ── ORM Models ──────────────────────────────────────────────────────────────

class Team(Base):
    __tablename__ = "teams"

    id          = Column(Integer, primary_key=True)
    api_id      = Column(Integer, unique=True, nullable=False)
    name        = Column(String, nullable=False)
    short_name  = Column(String)
    league      = Column(String)
    country     = Column(String)


class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (UniqueConstraint("api_id"),)

    id              = Column(Integer, primary_key=True)
    api_id          = Column(Integer, nullable=False)
    league          = Column(String, nullable=False)
    season          = Column(String, nullable=False)
    matchday        = Column(Integer)
    date            = Column(Date)
    home_team_id    = Column(Integer, ForeignKey("teams.api_id"))
    away_team_id    = Column(Integer, ForeignKey("teams.api_id"))
    home_goals      = Column(Integer)
    away_goals      = Column(Integer)
    status          = Column(String)   # FINISHED, SCHEDULED, LIVE


class Prediction(Base):
    __tablename__ = "predictions"

    id              = Column(Integer, primary_key=True)
    match_id        = Column(Integer, ForeignKey("matches.api_id"))
    created_at      = Column(DateTime)
    model_version   = Column(String)
    prob_home_win   = Column(Float)
    prob_draw       = Column(Float)
    prob_away_win   = Column(Float)
    expected_home_goals = Column(Float)
    expected_away_goals = Column(Float)
    confidence      = Column(Float)
    simulation_runs = Column(Integer)


# ── Engine & Session ─────────────────────────────────────────────────────────

def get_engine(db_path: str = "data/football.db"):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    return engine


def init_db(db_path: str = "data/football.db"):
    """Create all tables if they don't exist."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    log.info(f"Database initialized at {db_path}")
    return engine


def get_session(db_path: str = "data/football.db") -> Session:
    engine = get_engine(db_path)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()
