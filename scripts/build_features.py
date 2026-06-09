"""
Script: Build features from raw match data and save to CSV + DB.

Usage:
    python scripts/build_features.py --league BL1
"""

import sys
from pathlib import Path
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import get_session, Match, Team
from src.features.feature_builder import FeatureBuilder
from src.utils.logger import get_logger

log = get_logger("build_features")
app = typer.Typer()


def load_matches_from_db(league: str = "BL1") -> pd.DataFrame:
    """Load finished matches from SQLite into a DataFrame."""
    session = get_session()

    matches = session.query(Match).filter(
        Match.league == league,
        Match.status == "FINISHED",
    ).all()

    # Resolve team IDs to names
    teams = {t.api_id: t.name for t in session.query(Team).all()}
    session.close()

    rows = []
    for m in matches:
        rows.append({
            "match_id":   m.api_id,
            "date":       str(m.date),
            "league":     m.league,
            "season":     m.season,
            "matchday":   m.matchday,
            "stage":      m.stage,
            "group_name": m.group_name,
            "home_team":  teams.get(m.home_team_id, f"ID:{m.home_team_id}"),
            "away_team":  teams.get(m.away_team_id, f"ID:{m.away_team_id}"),
            "home_goals": m.home_goals,
            "away_goals": m.away_goals,
            "result":     _result(m.home_goals, m.away_goals),
        })

    df = pd.DataFrame(rows)
    log.info(f"Loaded {len(df)} matches from DB for {league}")
    return df


def _result(h, a):
    if h > a: return "H"
    if h < a: return "A"
    return "D"


@app.command()
def build(
    league: str = typer.Option("BL1", "--league", "-l"),
    output: str = typer.Option("data/processed/features_BL1.csv", "--output", "-o"),
):
    log.info(f"Building features for {league}...")

    df_raw = load_matches_from_db(league)

    if len(df_raw) == 0:
        log.error("No matches found in DB. Run fetch_data.py first.")
        raise typer.Exit(1)

    # Teams without names = collector didn't fetch teams yet
    missing_names = df_raw[df_raw["home_team"].str.startswith("ID:")]["home_team"].unique()
    if len(missing_names) > 0:
        log.info(f"Note: {len(missing_names)} teams from older seasons have no name entry (relegated/promoted). This is expected.")

    fb       = FeatureBuilder(df_raw)
    features = fb.build_features()

    # Save processed features
    out_path = Path(output.replace("BL1", league))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)

    log.success(f"Saved {len(features)} rows → {out_path}")

    # Quick summary
    print("\n── Feature Matrix Summary ──────────────────────────────")
    print(f"  Rows:          {len(features)}")
    print(f"  Columns:       {features.shape[1]}")
    print(f"  Date range:    {features['date'].min()} → {features['date'].max()}")
    print(f"  Results:       H={( features['result']=='H').sum()}  "
          f"D={(features['result']=='D').sum()}  "
          f"A={(features['result']=='A').sum()}")
    print(f"\n── Sample Features (last 3 matches) ────────────────────")
    cols = ["date", "home_team", "away_team", "result",
            "home_form_ppg", "away_form_ppg",
            "home_goals_scored_avg", "away_goals_scored_avg",
            "position_diff", "h2h_home_win_rate"]
    print(features[cols].tail(3).to_string(index=False))


if __name__ == "__main__":
    app()