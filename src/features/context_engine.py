"""
Context Engine — Motivation, Derby-Erkennung, CL Knockout-Dringlichkeit.

CL-Struktur (neues Format ab 2024/25):
  Matchday 1-8  → Ligaphase (Tabelle relevant, normaler Urgency-Betrieb)
  Matchday 9+   → Playoff/KO (kein Tabellenbezug, Boost skaliert mit Rundenfortschritt)
    Matchday  9 = Round of 16 Hinspiel   → Boost  +12%
    Matchday 10 = Round of 16 Rückspiel  → Boost  +16%
    Matchday 11 = Viertelfinale Hinspiel → Boost  +20%
    Matchday 12 = Viertelfinale Rückspiel→ Boost  +24%
    Matchday 13 = Halbfinale Hinspiel    → Boost  +28%
    Matchday 14 = Halbfinale Rückspiel   → Boost  +32%
    Matchday 15 = Finale                 → Boost  +40%

EL/UECL: gleiche Logik, KO ab Matchday 7.
"""

from src.utils.logger import get_logger
from src.features.derbies import is_derby

log = get_logger(__name__)

# Ligaphase-Grenze pro Wettbewerb (ab diesem Matchday = Knockout)
KO_STARTS_AT = {
    "CL":   9,   # Neues Format: 8 Ligaspiele, dann KO
    "EL":   7,   # Europa League: 6 Ligaspiele, dann KO
    "UECL": 7,   # Conference League: gleich wie EL
    "ECL":  7,
}

# CL-Stage-Labels für Logging/Anzeige (relativ zu KO-Start)
def _stage_label(matchday: int, ko_start: int) -> str:
    offset = matchday - ko_start  # 0=R16 H, 1=R16 R, 2=QF H, ...
    labels = [
        "Round of 16 — Hinspiel",
        "Round of 16 — Rückspiel",
        "Viertelfinale — Hinspiel",
        "Viertelfinale — Rückspiel",
        "Halbfinale — Hinspiel",
        "Halbfinale — Rückspiel",
        "Finale",
    ]
    return labels[min(offset, len(labels) - 1)]


def _knockout_boost(matchday: int, ko_start: int) -> tuple[float, float]:
    """
    Berechnet Motivationsboost und Urgency für eine KO-Runde.
    Formel: Basis 12% + 4% pro Runde. Finale: 40%.
    Je höher der Matchday, desto wichtiger das Spiel.
    """
    offset = matchday - ko_start  # 0-basiert
    if offset >= 6:
        # Finale
        return 0.40, 1.00
    boost   = 0.12 + offset * 0.04   # 0.12, 0.16, 0.20, 0.24, 0.28, 0.32
    urgency = 0.85 + offset * 0.03   # 0.85, 0.88, 0.91, 0.94, 0.97, 1.00
    return round(boost, 3), round(min(urgency, 1.0), 3)


class ContextEngine:
    def __init__(self):
        self.league_teams = {"BL1": 18, "PL": 20, "PD": 20, "SA": 20, "FL1": 18, "CL": 36}

    def calculate_context(
        self,
        home_team: str,
        away_team: str,
        league: str,
        matchday: int,
        standings: dict,
    ) -> dict:
        """Berechnet Derby, Tabellendruck und Knockout-Motivation."""

        derby = is_derby(home_team, away_team)
        ko_start = KO_STARTS_AT.get(league)

        # ── CL/EL Knockout Detection ──────────────────────────────────────────
        if ko_start and matchday and matchday >= ko_start:
            label = _stage_label(matchday, ko_start)
            ko_boost, urgency = _knockout_boost(matchday, ko_start)

            derby_extra = 0.05 if derby else 0.0
            motivation  = round(1.0 + ko_boost + derby_extra, 3)

            log.info(
                f"🏆 KO-Runde erkannt: {label} "
                f"(Matchday {matchday}, Boost +{ko_boost:.0%}, Motivation ×{motivation:.3f})"
            )
            if derby:
                log.info(f"🔥 DERBY im Knockout — extra +5%")

            return {
                "is_derby":        int(derby),
                "urgency":         urgency,
                "home_motivation": motivation,
                "away_motivation": motivation,
                "knockout_stage":  label,
                "is_knockout":     True,
            }

        # ── Ligaspielbetrieb ──────────────────────────────────────────────────
        if ko_start and matchday:
            log.info(f"{league} Ligaphase (Spieltag {matchday}/{ko_start - 1})")

        teams_count = self.league_teams.get(league, 20)
        total_matchdays = (teams_count - 1) * 2

        if not matchday or matchday <= 0:
            urgency = 0.5
        else:
            urgency = min(1.0, matchday / total_matchdays)

        home_boost = self._calc_motivation(home_team, urgency, standings, teams_count, derby)
        away_boost = self._calc_motivation(away_team, urgency, standings, teams_count, derby)

        if derby:
            log.info(f"🔥 DERBY: {home_team} vs {away_team}")

        return {
            "is_derby":        int(derby),
            "urgency":         round(urgency, 3),
            "home_motivation": home_boost,
            "away_motivation": away_boost,
            "knockout_stage":  None,
            "is_knockout":     False,
        }

    # ── Ligamotivation ────────────────────────────────────────────────────────

    def _calc_motivation(self, team, urgency, standings, teams_count, is_derby_match):
        boost = 1.0

        if is_derby_match:
            boost += 0.05

        team_stats = standings.get(team)
        if team_stats and urgency > 0.4:
            pos = team_stats.get("position", 10)
            pts = team_stats.get("points", 0)

            leader_pts = 0
            relegation_pts = 0
            for t_data in standings.values():
                p = t_data.get("position")
                if p == 1:
                    leader_pts = t_data.get("points", 0)
                elif p == teams_count - 2:
                    relegation_pts = t_data.get("points", 0)

            if pos == 1:
                log.info(f"🏆 Tabellenführer: {team}")
                boost += 0.05 * urgency
            elif pos <= 4 and (leader_pts - pts) <= 6:
                log.info(f"🏆 Titelrennen: {team} ({leader_pts - pts} Pkt Rückstand)")
                boost += 0.05 * urgency
            elif pos <= 7:
                log.info(f"🇪🇺 Europapokal-Kampf: {team} (Platz {pos})")
                boost += 0.02 * urgency

            if pos >= teams_count - 2:
                log.info(f"🆘 Abstiegsgefahr: {team} (Platz {pos})")
                boost += 0.08 * urgency
            elif (pts - relegation_pts) <= 6:
                log.info(f"⚔️ Abstiegspanik: {team} (nur {pts - relegation_pts} Pkt über dem Strich)")
                boost += 0.06 * urgency

        return round(boost, 3)