"""
Auto-Update Pipeline — alle Ligen
Holt neue Spiele und baut Features neu.

Usage:
    python scripts/update.py                    # BL1 updaten
    python scripts/update.py --league PL        # Premier League
    python scripts/update.py --all              # Alle 6 Ligen
    python scripts/update.py --check            # Nur Status anzeigen

Cron (jeden Samstag 23:00):
    0 23 * * 6 cd ~/Desktop/football-predictor && source venv/bin/activate && python scripts/update.py --all
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
import pandas as pd
import typer

from src.collectors.football_data_collector import FootballDataCollector
from src.utils.database import get_session, Match, Team
from src.features.feature_builder import FeatureBuilder
from src.utils.logger import get_logger
from config.config import config

log = get_logger("update")
app = typer.Typer()


def count_db_matches(league: str) -> tuple[int, str]:
    session = get_session()
    count = session.query(Match).filter(
        Match.league == league, Match.status == "FINISHED"
    ).count()
    last = session.query(Match).filter(
        Match.league == league, Match.status == "FINISHED"
    ).order_by(Match.date.desc()).first()
    session.close()
    return count, str(last.date) if last else "—"


def build_features(league: str) -> int:
    """Baut Feature-Matrix aus DB und speichert als CSV."""
    session = get_session()
    matches = session.query(Match).filter(
        Match.league == league, Match.status == "FINISHED"
    ).all()
    teams = {t.api_id: t.name for t in session.query(Team).all()}
    session.close()

    def res(h, a): return "H" if h > a else ("A" if h < a else "D")

    rows = [{
        "match_id":   m.api_id, "date": str(m.date),
        "league":     m.league, "season": m.season, "matchday": m.matchday,
        "home_team":  teams.get(m.home_team_id, f"ID:{m.home_team_id}"),
        "away_team":  teams.get(m.away_team_id, f"ID:{m.away_team_id}"),
        "home_goals": m.home_goals, "away_goals": m.away_goals,
        "result":     res(m.home_goals, m.away_goals),
    } for m in matches]

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning(f"Keine Spiele in DB für {league}")
        return 0

    features = FeatureBuilder(df).build_features()
    out = Path(f"data/processed/features_{league}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, index=False)
    return len(features)


def update_league(league: str, seasons: int = 2, check_only: bool = False):
    name   = config.SUPPORTED_LEAGUES.get(league, league)
    before, last_date = count_db_matches(league)

    log.info(f"{'─'*52}")
    log.info(f"{'CHECK' if check_only else 'UPDATE'}: {name} ({league})")
    log.info(f"DB aktuell: {before} Spiele | Letztes: {last_date}")

    if check_only:
        return

    collector = FootballDataCollector()
    collector.fetch_matches(league=league, seasons=seasons)
    collector.fetch_teams(league=league)

    after, _ = count_db_matches(league)
    new = after - before

    if new > 0:
        log.success(f"+{new} neue Spiele für {name}")
        n = build_features(league)
        log.success(f"Features: {n} Zeilen → data/processed/features_{league}.csv")
    else:
        log.info(f"Keine neuen Spiele für {name} — bereits aktuell")


@app.command()
def update(
    league:  str  = typer.Option("BL1", "--league",  "-l", help="Liga-Code"),
    all:     bool = typer.Option(False,  "--all",     "-a", help="Alle Ligen aktualisieren"),
    check:   bool = typer.Option(False,  "--check",   "-c", help="Nur Status anzeigen"),
    seasons: int  = typer.Option(2,      "--seasons", "-s", help="Anzahl Saisons"),
):
    leagues = list(config.SUPPORTED_LEAGUES.keys()) if all else [league]

    log.info(f"Football Predictor — {'Status' if check else 'Update'} — {datetime.now().strftime('%d.%m.%Y %H:%M')}")

    for lg in leagues:
        try:
            update_league(lg, seasons=seasons, check_only=check)
        except Exception as e:
            log.error(f"Fehler bei {lg}: {e}")

    log.success(f"Fertig um {datetime.now().strftime('%H:%M')}")

    if not check:
        log.info("Verfügbare Feature-Dateien:")
        for lg in config.SUPPORTED_LEAGUES:
            p = Path(f"data/processed/features_{lg}.csv")
            if p.exists():
                import pandas as pd
                n = len(pd.read_csv(p))
                log.info(f"  ✅ features_{lg}.csv — {n} Spiele")
            else:
                log.info(f"  ❌ features_{lg}.csv — nicht vorhanden (run: python scripts/update.py --league {lg})")


if __name__ == "__main__":
    app()