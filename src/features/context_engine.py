"""
Context Engine — Motivation, Derby-Erkennung, CL/WC Knockout-Dringlichkeit.

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

WC-Struktur (2026):
  Stage = GROUP_STAGE  → Gruppenphase (Druck steigt mit Matchday 1→3)
  Stage = LAST_32      → Boost +15%
  Stage = LAST_16      → Boost +20%
  Stage = QUARTER_FINALS → Boost +25%
  Stage = SEMI_FINALS  → Boost +30%
  Stage = THIRD_PLACE  → Boost +15%
  Stage = FINAL        → Boost +40%
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

# WC: stage-string → (boost, urgency, label)
WC_STAGE_CONFIG = {
    "LAST_32":        (0.15, 0.80, "Round of 32"),
    "LAST_16":        (0.20, 0.85, "Round of 16"),
    "QUARTER_FINALS": (0.25, 0.90, "Viertelfinale"),
    "SEMI_FINALS":    (0.30, 0.95, "Halbfinale"),
    "THIRD_PLACE":    (0.15, 0.80, "Spiel um Platz 3"),
    "FINAL":          (0.40, 1.00, "Finale"),
}

# WC Gruppenphase: Urgency je Matchday (1=Auftakt, 3=Alles-oder-Nichts)
WC_GROUP_URGENCY = {1: 0.35, 2: 0.65, 3: 1.0}


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
        stage: str = None,       # WC: stage string from API/DB
        group_name: str = None,  # WC: GROUP_A … GROUP_L
    ) -> dict:
        """Berechnet Derby, Tabellendruck und Knockout-Motivation."""

        derby = is_derby(home_team, away_team)

        # ── WC: stage-basierte Logik ──────────────────────────────────────────
        if league == "WC":
            return self._wc_context(home_team, away_team, derby, matchday, stage, group_name, standings)

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

    # ── WC Context ────────────────────────────────────────────────────────────

    def _wc_context(
        self,
        home_team: str,
        away_team: str,
        derby: bool,
        matchday: int,
        stage: str,
        group_name: str,
        standings: dict,
    ) -> dict:
        """WC-spezifische Motivations- und Urgency-Berechnung."""

        effective_stage = stage or "GROUP_STAGE"

        # ── Knockout Phase ────────────────────────────────────────────────────
        if effective_stage in WC_STAGE_CONFIG:
            boost, urgency, label = WC_STAGE_CONFIG[effective_stage]
            motivation = round(1.0 + boost, 3)
            log.info(f"🌍 WC KO-Runde: {label} (Boost +{boost:.0%})")
            return {
                "is_derby":        int(derby),
                "urgency":         urgency,
                "home_motivation": motivation,
                "away_motivation": motivation,
                "knockout_stage":  label,
                "is_knockout":     True,
                "wc_group":        None,
            }

        # ── Gruppenphase ──────────────────────────────────────────────────────
        # matchday 1/2/3 innerhalb der Gruppe
        md = matchday if matchday and 1 <= matchday <= 3 else 2
        urgency = WC_GROUP_URGENCY.get(md, 0.65)

        home_boost = self._wc_group_motivation(home_team, urgency, standings, group_name)
        away_boost = self._wc_group_motivation(away_team, urgency, standings, group_name)

        log.info(
            f"🌍 WC Gruppenphase {group_name or '?'} — MD{md} — "
            f"Urgency {urgency:.0%} — {home_team} ×{home_boost:.3f} / {away_team} ×{away_boost:.3f}"
        )

        return {
            "is_derby":        int(derby),
            "urgency":         urgency,
            "home_motivation": home_boost,
            "away_motivation": away_boost,
            "knockout_stage":  f"Gruppenphase {group_name or ''} — Spieltag {md}",
            "is_knockout":     False,
            "wc_group":        group_name,
        }

    def _wc_group_motivation(
        self,
        team: str,
        urgency: float,
        standings: dict,
        group_name: str,
    ) -> float:
        """Motivationsboost für WC Gruppenphase basierend auf Gruppenposition."""
        boost = 1.0
        stats = standings.get(team)
        if not stats or urgency < 0.5:
            return round(boost, 3)

        pos = stats.get("position", 2)
        # Platz 3/4 → Eliminierungsgefahr
        if pos == 4:
            boost += 0.10 * urgency   # letzter Platz, Druck max
        elif pos == 3:
            boost += 0.06 * urgency   # noch im Rennen
        elif pos == 1:
            boost += 0.03 * urgency   # Gruppenführer verteidigt Platz
        # Platz 2: kein Extra-Boost (komfortabel)

        return round(boost, 3)

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