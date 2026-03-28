"""
Script: Fetch match data from football-data.org.

Usage:
    python scripts/fetch_data.py --league BL1 --seasons 5
    python scripts/fetch_data.py --league PL --seasons 3
    python scripts/fetch_data.py --league BL1 --upcoming
"""

import sys
from pathlib import Path
import typer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collectors.football_data_collector import FootballDataCollector
from src.utils.logger import get_logger

log = get_logger("fetch_data")
app = typer.Typer()


@app.command()
def fetch(
    league:   str = typer.Option("BL1",  "--league",   "-l", help="League code (BL1, PL, CL, PD, SA, FL1)"),
    seasons:  int = typer.Option(5,      "--seasons",  "-s", help="Number of past seasons to fetch"),
    upcoming: bool = typer.Option(False, "--upcoming", "-u", help="Fetch upcoming scheduled matches"),
    teams:    bool = typer.Option(False, "--teams",    "-t", help="Fetch & save team list"),
):
    collector = FootballDataCollector()

    if teams:
        collector.fetch_teams(league=league)

    if upcoming:
        df = collector.fetch_upcoming(league=league)
        print(df.to_string(index=False))
    else:
        df = collector.fetch_matches(league=league, seasons=seasons)
        log.info(f"Fetched {len(df)} finished matches")
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    app()
