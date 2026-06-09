"""
WC CSV Importer — Importiert historische WM-Daten (1930–2022) aus data/matches.csv.

Normalisiert Stage-Namen auf API-Format (GROUP_STAGE, LAST_16, QUARTER_FINALS, …).
Nutzt stabile team_ids aus CSV (T-30 → 830030) statt hash-basierter IDs.
Sicher wiederholbar (löscht vorhandene WC-Daten und importiert neu).

Usage:
    python scripts/import_wc_csv.py
    python scripts/import_wc_csv.py --csv data/matches.csv
    python scripts/import_wc_csv.py --dry-run
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

# Offset so synthetic WC team IDs never collide with football-data.org club IDs
_WC_TEAM_ID_OFFSET = 830_000

# ── Stage normalization ──────────────────────────────────────────────────────

_STAGE_MAP = {
    "group stage":          "GROUP_STAGE",
    "second group stage":   "GROUP_STAGE",   # 1974/1978 second group phase
    "first round":          "GROUP_STAGE",
    "preliminary round":    "GROUP_STAGE",
    "final round":          "GROUP_STAGE",   # 1950 WC had no knockout
    "round of 16":          "LAST_16",
    "round of 32":          "LAST_32",
    "quarter-finals":       "QUARTER_FINALS",
    "quarter-final":        "QUARTER_FINALS",
    "semi-finals":          "SEMI_FINALS",
    "semi-final":           "SEMI_FINALS",
    "third-place match":    "THIRD_PLACE",
    "match for third place":"THIRD_PLACE",
    "third place":          "THIRD_PLACE",
    "final":                "FINAL",
}

_KO_MATCHDAY = {
    "LAST_32":        4,
    "LAST_16":        4,
    "QUARTER_FINALS": 5,
    "SEMI_FINALS":    6,
    "THIRD_PLACE":    7,
    "FINAL":          8,
}

# Normalize "Group A" / "Group 1" → "GROUP_A"
_NUMBERED_GROUP = {str(i): c for i, c in enumerate("ABCDEFGH", 1)}


def normalize_stage(raw: str) -> str:
    """Returns stage API string from CSV stage_name."""
    if not raw or str(raw).strip().lower() == "not applicable":
        return "GROUP_STAGE"
    key = str(raw).strip().lower()
    return _STAGE_MAP.get(key, "GROUP_STAGE")


def normalize_group(raw: str) -> str | None:
    """Returns 'GROUP_A' format from 'Group A', 'Group 1', or None."""
    if not raw or str(raw).strip().lower() in ("not applicable", "nan", ""):
        return None
    raw = str(raw).strip()
    # "Group A" → "GROUP_A"
    m = re.match(r"[Gg]roup\s+([A-La-l])\b", raw)
    if m:
        return f"GROUP_{m.group(1).upper()}"
    # "Group 1" → "GROUP_A"
    m2 = re.match(r"[Gg]roup\s+(\d+)", raw)
    if m2:
        letter = _NUMBERED_GROUP.get(m2.group(1), "A")
        return f"GROUP_{letter}"
    # Lowercase typos like "Group c"
    m3 = re.match(r"[Gg]roup\s+([a-l])", raw, re.IGNORECASE)
    if m3:
        return f"GROUP_{m3.group(1).upper()}"
    return None


def _team_db_id(t_id: str) -> int:
    """Converts 'T-30' → stable integer DB id (830030)."""
    try:
        num = int(str(t_id).replace("T-", "").strip())
        return _WC_TEAM_ID_OFFSET + num
    except (ValueError, AttributeError):
        return _WC_TEAM_ID_OFFSET + (abs(hash(str(t_id))) % 9999)


def _match_db_id(key_id: int) -> int:
    """Converts CSV key_id (1-1248) to DB api_id in non-colliding range."""
    return 700_000 + int(key_id)


def _assign_group_matchdays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group stage: matchday 1/2/3 = round within (tournament, group) by date order.
    Knockout: fixed matchday from KO_MATCHDAY.
    """
    df = df.copy()
    df["matchday"] = None

    # Knockout
    ko_mask = df["stage"] != "GROUP_STAGE"
    df.loc[ko_mask, "matchday"] = df.loc[ko_mask, "stage"].map(_KO_MATCHDAY)

    # Group stage: rank unique dates per (season, group)
    gs_mask = df["stage"] == "GROUP_STAGE"
    for (season, grp), sub in df[gs_mask].groupby(["season", "group_name"], dropna=False):
        sorted_dates = sorted(sub["match_date"].dropna().unique())
        date_to_md = {d: i + 1 for i, d in enumerate(sorted_dates)}
        for idx in sub.index:
            d = df.at[idx, "match_date"]
            df.at[idx, "matchday"] = date_to_md.get(d, 1)

    return df


def _clear_existing_wc_data(session) -> int:
    """Löscht alle bestehenden WC-Matches und WC-Teams aus der DB."""
    deleted_matches = session.query(Match).filter(Match.league == "WC").delete()
    deleted_teams   = session.query(Team).filter(Team.league == "WC").delete()
    session.flush()
    return deleted_matches, deleted_teams


@app.command()
def import_csv(
    csv_path: str  = typer.Option("data/matches.csv", "--csv", "-f"),
    dry_run:  bool = typer.Option(False, "--dry-run", help="Zeigt was importiert würde, keine DB-Schreibvorgänge"),
    men_only: bool = typer.Option(True,  "--men-only/--all", help="Nur Männer-WM importieren"),
):
    path = Path(csv_path)
    if not path.exists():
        log.error(f"CSV nicht gefunden: {path}")
        raise typer.Exit(1)

    df = pd.read_csv(path)
    log.info(f"CSV geladen: {len(df)} Zeilen aus {path}")

    # Filter: nur Männer-WM
    if men_only:
        df = df[df["tournament_name"].str.contains("Men's", na=False)].copy()
        log.info(f"Nach Männer-WM-Filter: {len(df)} Spiele")

    # Filter: nur abgeschlossene Spiele mit Ergebnissen
    df = df[df["home_team_score"].notna() & df["away_team_score"].notna()].copy()
    log.info(f"Nach Ergebnis-Filter: {len(df)} Spiele")

    # Normalize stage + group
    df["stage"]      = df["stage_name"].apply(normalize_stage)
    df["group_norm"] = df["group_name"].apply(normalize_group)

    # Season from tournament_id: "WC-1930" → "1930"
    df["season"] = df["tournament_id"].str.extract(r"(\d{4})")[0]

    # Assign matchdays
    df = _assign_group_matchdays(df)

    # Result mapping
    def _result(row) -> str:
        if row["home_team_win"] == 1:
            return "H"
        elif row["away_team_win"] == 1:
            return "A"
        return "D"
    df["result_code"] = df.apply(_result, axis=1)

    if dry_run:
        log.info("DRY RUN — keine DB-Schreibvorgänge")
        print(df[["season", "match_date", "stage", "group_norm", "matchday",
                   "home_team_name", "home_team_score", "away_team_score",
                   "away_team_name", "result_code"]].head(25).to_string())
        print(f"\nStage distribution:\n{df['stage'].value_counts().to_string()}")
        print(f"\nGroup distribution:\n{df['group_norm'].value_counts().head(15).to_string()}")
        return

    session = get_session()

    # Lösche alte WC-Daten
    del_m, del_t = _clear_existing_wc_data(session)
    log.info(f"Gelöscht: {del_m} Matches, {del_t} Teams (WC)")

    new_teams = new_matches = 0

    for _, row in df.iterrows():
        home_name = str(row["home_team_name"]).strip()
        away_name = str(row["away_team_name"]).strip()
        home_t_id = str(row["home_team_id"]).strip()
        away_t_id = str(row["away_team_id"]).strip()
        home_db_id = _team_db_id(home_t_id)
        away_db_id = _team_db_id(away_t_id)

        # Upsert teams (by db_id)
        for db_id, name, code in [
            (home_db_id, home_name, str(row.get("home_team_code", "")).strip()),
            (away_db_id, away_name, str(row.get("away_team_code", "")).strip()),
        ]:
            existing = session.query(Team).filter_by(api_id=db_id).first()
            if not existing:
                session.add(Team(
                    api_id     = db_id,
                    name       = name,
                    short_name = code if code and code != "nan" else name[:3].upper(),
                    league     = "WC",
                    country    = name,
                ))
                new_teams += 1

        # Insert match
        match_db_id = _match_db_id(int(row["key_id"]))
        from datetime import date as _date
        raw_date = row["match_date"]
        parsed_date = datetime.strptime(str(raw_date), "%Y-%m-%d").date() if raw_date else None

        session.add(Match(
            api_id       = match_db_id,
            league       = "WC",
            season       = str(row["season"]),
            matchday     = int(row["matchday"]) if pd.notna(row["matchday"]) else None,
            date         = parsed_date,
            home_team_id = home_db_id,
            away_team_id = away_db_id,
            home_goals   = int(row["home_team_score"]),
            away_goals   = int(row["away_team_score"]),
            status       = "FINISHED",
            stage        = row["stage"],
            group_name   = row["group_norm"],
        ))
        new_matches += 1

    session.commit()
    session.close()

    log.success(
        f"Import abgeschlossen: {new_matches} neue Spiele, "
        f"{new_teams} neue Teams"
    )


if __name__ == "__main__":
    app()
