"""
WC CSV Importer — Importiert historische WM-Daten (1930–2014) aus WorldCupMatches.csv.

Normalisiert Stage-Namen auf API-Format (GROUP_STAGE, LAST_16, QUARTER_FINALS, …).
Generiert stabile synthetische api_ids für Nationalmannschaften (kein API-Key nötig).
Sicher wiederholbar (Upsert-Logik).

Usage:
    python scripts/import_wc_csv.py
    python scripts/import_wc_csv.py --csv data/WorldCupMatches.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re
import pandas as pd
from datetime import datetime
import typer
from src.utils.database import get_session, Match, Team
from src.utils.logger import get_logger

log = get_logger("import_wc_csv")
app = typer.Typer()

# ── Stage normalization ──────────────────────────────────────────────────────

# Maps CSV stage strings → API stage format
_STAGE_MAP = {
    # Group stage (any "group X" or "group Y" pattern)
    # handled dynamically below
    "round of 16":           "LAST_16",
    "round of 32":           "LAST_32",
    "quarter-finals":        "QUARTER_FINALS",
    "quarterfinals":         "QUARTER_FINALS",
    "semi-finals":           "SEMI_FINALS",
    "semifinals":            "SEMI_FINALS",
    "third place":           "THIRD_PLACE",
    "match for third place": "THIRD_PLACE",
    "final":                 "FINAL",
    "first round":           "GROUP_STAGE",
    "preliminary round":     "GROUP_STAGE",
}

# Group patterns that were used before letter-based format
_NUMBERED_GROUPS = {
    "group 1": "GROUP_A", "group 2": "GROUP_B", "group 3": "GROUP_C",
    "group 4": "GROUP_D", "group 5": "GROUP_E", "group 6": "GROUP_F",
    "group 7": "GROUP_G", "group 8": "GROUP_H",
    "pool 1": "GROUP_A", "pool 2": "GROUP_B", "pool 3": "GROUP_C", "pool 4": "GROUP_D",
}


def normalize_stage(raw: str) -> tuple[str, str | None]:
    """Returns (stage_api, group_name) from raw CSV Stage string."""
    if not raw or str(raw).strip() == "":
        return "GROUP_STAGE", None

    raw_clean = str(raw).strip().lower()

    # Letter-based groups: "Group A", "Group B", …
    m = re.match(r"group\s+([a-l])\b", raw_clean)
    if m:
        return "GROUP_STAGE", f"GROUP_{m.group(1).upper()}"

    # Numbered groups: "Group 1", …
    if raw_clean in _NUMBERED_GROUPS:
        return "GROUP_STAGE", _NUMBERED_GROUPS[raw_clean]

    # Fuzzy numbered group
    m2 = re.match(r"group\s+(\d+)", raw_clean)
    if m2:
        grp_letters = "ABCDEFGH"
        idx = int(m2.group(1)) - 1
        grp = f"GROUP_{grp_letters[idx]}" if idx < len(grp_letters) else "GROUP_A"
        return "GROUP_STAGE", grp

    # Exact lookup
    for key, val in _STAGE_MAP.items():
        if key in raw_clean:
            return val, None

    log.warning(f"Unknown stage: '{raw}' — defaulting to GROUP_STAGE")
    return "GROUP_STAGE", None


def _team_api_id(name: str) -> int:
    """Stable synthetic API id for a national team name (no real API id needed)."""
    return 900_000 + (abs(hash(name.strip().lower())) % 99_999)


def _parse_date(raw: str) -> "datetime.date | None":
    """Parses WC CSV datetime strings like '13 Jul 1930 - 15:00' or '17 June 1970'."""
    if not raw or str(raw).strip() == "":
        return None
    raw = str(raw).strip()
    # Remove time part: "13 Jul 1930 - 15:00" → "13 Jul 1930"
    raw = re.sub(r"\s*-\s*\d{2}:\d{2}.*$", "", raw).strip()
    for fmt in ("%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    log.warning(f"Could not parse date: '{raw}'")
    return None


def _assign_group_matchdays(df: pd.DataFrame) -> pd.DataFrame:
    """
    For group stage matches: assign matchday 1/2/3 within each group per year.
    For knockout matches: assign matchday based on stage.
    """
    KO_MATCHDAY = {
        "LAST_32":        4,
        "LAST_16":        4,   # pre-2026 WC has no LAST_32 so R16 = matchday 4
        "QUARTER_FINALS": 5,
        "SEMI_FINALS":    6,
        "THIRD_PLACE":    7,
        "FINAL":          8,
    }
    df = df.copy()
    df["matchday"] = None

    # Knockout: straightforward
    ko_mask = df["stage"] != "GROUP_STAGE"
    df.loc[ko_mask, "matchday"] = df.loc[ko_mask, "stage"].map(KO_MATCHDAY)

    # Group stage: number rounds per (year, group) by date order
    gs_mask = df["stage"] == "GROUP_STAGE"
    for (year, grp), sub in df[gs_mask].groupby(["season", "group_name"], dropna=False):
        sorted_dates = sorted(sub["date"].dropna().unique())
        date_to_md = {d: i + 1 for i, d in enumerate(sorted_dates)}
        for idx in sub.index:
            d = df.at[idx, "date"]
            df.at[idx, "matchday"] = date_to_md.get(d, 1)

    return df


@app.command()
def import_csv(
    csv_path: str = typer.Option("data/WorldCupMatches.csv", "--csv", "-f"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be imported, no DB writes"),
):
    path = Path(csv_path)
    if not path.exists():
        log.error(f"CSV nicht gefunden: {path}")
        raise typer.Exit(1)

    df = pd.read_csv(path)
    log.info(f"CSV geladen: {len(df)} Zeilen aus {path}")

    # Filter: nur Spiele mit echten Ergebnissen
    df = df[df["Home Team Name"].notna() & df["Away Team Name"].notna()].copy()
    df = df[df["Home Team Goals"].notna() & df["Away Team Goals"].notna()].copy()
    log.info(f"Nach Filter: {len(df)} abgeschlossene Spiele")

    # Parse stage + group
    stage_data = df["Stage"].apply(normalize_stage)
    df["stage"]      = stage_data.apply(lambda x: x[0])
    df["group_name"] = stage_data.apply(lambda x: x[1])

    # Parse date
    df["date"]   = df["Datetime"].apply(_parse_date)
    df["season"] = df["Year"].astype(int).astype(str)

    # Assign matchdays
    df = _assign_group_matchdays(df)

    # Home/away goals as int
    df["home_goals"] = df["Home Team Goals"].astype(int)
    df["away_goals"] = df["Away Team Goals"].astype(int)
    df["match_api_id"] = df["MatchID"].astype(int)

    if dry_run:
        log.info("DRY RUN — keine DB-Schreibvorgänge")
        print(df[["season", "date", "stage", "group_name", "matchday",
                   "Home Team Name", "home_goals", "away_goals", "Away Team Name"]].head(20).to_string())
        return

    session = get_session()
    new_teams = new_matches = skipped = 0

    for _, row in df.iterrows():
        home_name = str(row["Home Team Name"]).strip()
        away_name = str(row["Away Team Name"]).strip()

        # Upsert teams
        for name in (home_name, away_name):
            synthetic_id = _team_api_id(name)
            existing = session.query(Team).filter_by(api_id=synthetic_id).first()
            if not existing:
                session.add(Team(
                    api_id     = synthetic_id,
                    name       = name,
                    short_name = name[:3].upper(),
                    league     = "WC",
                    country    = name,
                ))
                new_teams += 1

        # Upsert match
        match_id = int(row["match_api_id"])
        existing_match = session.query(Match).filter_by(api_id=match_id).first()
        if existing_match:
            skipped += 1
            continue

        session.add(Match(
            api_id       = match_id,
            league       = "WC",
            season       = str(row["season"]),
            matchday     = int(row["matchday"]) if pd.notna(row["matchday"]) else None,
            date         = row["date"],
            home_team_id = _team_api_id(home_name),
            away_team_id = _team_api_id(away_name),
            home_goals   = row["home_goals"],
            away_goals   = row["away_goals"],
            status       = "FINISHED",
            stage        = row["stage"],
            group_name   = row["group_name"],
        ))
        new_matches += 1

    session.commit()
    session.close()

    log.success(
        f"Import abgeschlossen: {new_matches} neue Spiele, "
        f"{new_teams} neue Teams, {skipped} übersprungen (bereits vorhanden)"
    )


if __name__ == "__main__":
    app()
