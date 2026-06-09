"""
WC Team Mapper — Speichert Transfermarkt-IDs für WC Nationalteams in der DB.

Nutzt hardcoded IDs für die wichtigsten Nationen (verifizieret von transfermarkt.com).
Für unbekannte Teams: optionaler dynamischer TM-Lookup.

Usage:
    python scripts/map_wc_teams.py              # Schreibt hardcoded IDs
    python scripts/map_wc_teams.py --dynamic    # + dynamische Suche für ungemappte
    python scripts/map_wc_teams.py --show       # Zeigt aktuellen Mapping-Status
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from src.utils.database import get_session, Team
from src.utils.logger import get_logger

log = get_logger("map_wc_teams")
app = typer.Typer()

# ── Hardcoded Transfermarkt national team IDs ────────────────────────────────
# Format: DB team name → TM club ID (from transfermarkt.com national team pages)
# Verified from transfermarkt.com URLs: /verein/{ID}/startseite
TM_NATIONAL_IDS: dict[str, str] = {
    # Top WC 2022 / 2026 nations
    "Germany":             "3262",
    "West Germany":        "3262",   # same team historically
    "France":              "3377",
    "Brazil":              "3439",
    "Argentina":           "3437",
    "Spain":               "3842",
    "England":             "3518",
    "Italy":               "3872",
    "Netherlands":         "3923",
    "Portugal":            "3947",
    "Belgium":             "3374",
    "Croatia":             "3553",
    "Uruguay":             "3504",
    "Mexico":              "3932",
    "United States":       "3505",
    "Morocco":             "3955",
    "Japan":               "3669",
    "South Korea":         "3384",
    "Senegal":             "3695",
    "Ghana":               "3629",
    "Cameroon":            "3393",
    "Nigeria":             "3651",
    "Algeria":             "3340",
    "Tunisia":             "3728",
    "Ivory Coast":         "3608",
    "Australia":           "3349",
    "Iran":                "3597",
    "Saudi Arabia":        "3683",
    "Qatar":               "3671",
    "Ecuador":             "3572",
    "Colombia":            "3501",
    "Chile":               "3456",
    "Peru":                "3661",
    "Paraguay":            "3657",
    "Bolivia":             "3370",
    "Costa Rica":          "3498",
    "Panama":              "3654",
    "Honduras":            "3593",
    "Canada":              "3395",
    "Poland":              "3662",
    "Czech Republic":      "3571",
    "Denmark":             "3567",
    "Sweden":              "3709",
    "Switzerland":         "3710",
    "Austria":             "3350",
    "Hungary":             "3594",
    "Romania":             "3672",
    "Bulgaria":            "3389",
    "Greece":              "3580",
    "Turkey":              "3729",
    "Russia":              "3674",
    "Ukraine":             "3730",
    "Serbia":              "3688",
    "Slovakia":            "3694",
    "Slovenia":            "3696",
    "Iceland":             "3596",
    "Norway":              "3648",
    "Wales":               "3742",
    "Scotland":            "3685",
    "Northern Ireland":    "3641",
    "Republic of Ireland": "3598",
    # Historic teams (map to modern equivalent or skip)
    "Soviet Union":        "3674",   # Russia as closest modern equivalent
    "Yugoslavia":          "3688",   # Serbia as closest modern equivalent
    "Czechoslovakia":      "3571",   # Czech Republic
}

# ── Sofascore national team IDs ──────────────────────────────────────────────
# Dynamic search already works via SofascoreCollector.get_team_id()
# These hardcoded IDs speed up lookups and avoid search failures
SOFASCORE_NATIONAL_IDS: dict[str, int] = {
    "Germany":         4711,
    "France":          4481,
    "Brazil":          4750,
    "Argentina":       3377,
    "Spain":           4698,
    "England":         4713,
    "Italy":           4707,
    "Netherlands":     4712,
    "Portugal":        4714,
    "Belgium":         4648,
    "Croatia":         4715,
    "Uruguay":         4752,
    "Mexico":          3809,
    "United States":   4397,
    "Morocco":         4754,
    "Japan":           4706,
    "South Korea":     4716,
    "Senegal":         4755,
    "Ghana":           4753,
    "Australia":       4401,
    "Poland":          4700,
    "Denmark":         4683,
    "Switzerland":     4702,
    "Serbia":          4503,
    "Costa Rica":      4766,
    "Ecuador":         4769,
    "Canada":          4656,
    "Qatar":           4771,
    "Cameroon":        4770,
    "Tunisia":         4751,
    "Saudi Arabia":    4757,
    "Iran":            4760,
    "Wales":           4704,
    "Norway":          4705,
    "Sweden":          4708,
    "Austria":         4703,
}


@app.command()
def map_teams(
    dynamic: bool = typer.Option(False, "--dynamic", help="Dynamischer TM-Lookup für ungemappte Teams"),
    show:    bool = typer.Option(False, "--show",    help="Zeigt Mapping-Status ohne zu schreiben"),
):
    session = get_session()
    wc_teams = session.query(Team).filter(Team.league == "WC").order_by(Team.name).all()

    if show:
        session.close()
        print(f"\n{'Team':<30} {'TM-ID':>8}  {'SS-ID (hardcoded)':>18}")
        print("-" * 62)
        for t in wc_teams:
            tm = t.transfermarkt_id or "—"
            ss = SOFASCORE_NATIONAL_IDS.get(t.name, "—")
            mapped = "✓" if t.transfermarkt_id else " "
            print(f"  {mapped} {t.name:<28} {tm:>8}  {str(ss):>18}")
        mapped_count = sum(1 for t in wc_teams if t.transfermarkt_id)
        print(f"\nGemappt: {mapped_count}/{len(wc_teams)}")
        return

    updated = 0
    for team in wc_teams:
        tm_id = TM_NATIONAL_IDS.get(team.name)
        if tm_id and team.transfermarkt_id != tm_id:
            team.transfermarkt_id = tm_id
            updated += 1
            log.info(f"TM: {team.name} → {tm_id}")

    if dynamic:
        # Versuche TM-Lookup für noch ungemappte Teams
        unmapped = [t for t in wc_teams if not t.transfermarkt_id]
        if unmapped:
            log.info(f"Dynamischer TM-Lookup für {len(unmapped)} ungemappte Teams...")
            try:
                from src.collectors.transfermarkt_collector import TransfermarktCollector
                collector = TransfermarktCollector()
                for team in unmapped:
                    try:
                        tm_id = collector.search_club(team.name, team.name)
                        if tm_id:
                            team.transfermarkt_id = tm_id
                            updated += 1
                            log.info(f"TM (dynamic): {team.name} → {tm_id}")
                        else:
                            log.warning(f"TM: kein Match für '{team.name}'")
                    except Exception as e:
                        log.warning(f"TM lookup fehlgeschlagen für {team.name}: {e}")
            except Exception as e:
                log.error(f"TransfermarktCollector nicht verfügbar: {e}")

    session.commit()
    session.close()

    log.success(f"Mapping abgeschlossen: {updated} Teams aktualisiert")

    # Status zeigen
    session2 = get_session()
    wc_teams2 = session2.query(Team).filter(Team.league == "WC").all()
    mapped_count = sum(1 for t in wc_teams2 if t.transfermarkt_id)
    session2.close()
    log.info(f"TM-Mapping Status: {mapped_count}/{len(wc_teams2)} Teams gemappt")


if __name__ == "__main__":
    app()
