from src.utils.logger import get_logger
from src.features.derbies import is_derby

log = get_logger(__name__)

class ContextEngine:
    def __init__(self):
        # Teams pro Liga (wichtig für die Berechnung der Relegationsplätze)
        self.league_teams = {"BL1": 18, "PL": 20, "PD": 20, "SA": 20, "FL1": 18, "CL": 36}

    def calculate_context(self, home_team: str, away_team: str, league: str, matchday: int, standings: dict) -> dict:
        """Berechnet Derby-Faktor und Tabellendruck (Motivation)."""
        derby = is_derby(home_team, away_team)
        teams_count = self.league_teams.get(league, 20)
        total_matchdays = (teams_count - 1) * 2
        
        # Zeitdruck-Faktor (0.0 am Anfang, 1.0 am 34. Spieltag)
        if not matchday or matchday <= 0:
            urgency = 0.5
        else:
            urgency = min(1.0, matchday / total_matchdays)

        home_boost = self._calc_motivation(home_team, urgency, standings, teams_count, derby)
        away_boost = self._calc_motivation(away_team, urgency, standings, teams_count, derby)
        
        if derby:
            log.info(f"🔥 DERBY DETEKTIERT: {home_team} vs {away_team}!")

        return {
            "is_derby": int(derby),
            "urgency": round(urgency, 3),
            "home_motivation": home_boost,
            "away_motivation": away_boost
        }

    def _calc_motivation(self, team: str, urgency: float, standings: dict, teams_count: int, is_derby: bool) -> float:
        boost = 1.0
        
        if is_derby:
            boost += 0.05
            
        team_stats = standings.get(team)
        # Wir senken die Urgency auf 0.4 (entspricht Spieltag 14+), 
        # weil Abstiegspanik oft schon in der Mitte der Saison beginnt!
        if team_stats and urgency > 0.4:
            pos = team_stats.get("position", 10)
            pts = team_stats.get("points", 0)
            
            # 1. Metriken aus der Tabelle scannen
            leader_pts = 0
            relegation_pts = 0
            
            for t_data in standings.values():
                p = t_data.get("position")
                if p == 1:
                    leader_pts = t_data.get("points", 0)
                # Relegationsplatz finden (Platz 16 in der Buli, 18 in der PL)
                elif p == teams_count - 2: 
                    relegation_pts = t_data.get("points", 0)
            
            # --- 2. DIE LOGIK ANWENDEN ---
            
            # A) Titelkampf
            if pos == 1:
                log.info(f"🏆 Titelkampf-Fokus für {team} (Tabellenführer)")
                boost += 0.05 * urgency
            elif pos <= 4 and (leader_pts - pts) <= 6:
                log.info(f"🏆 Heißes Titelrennen für {team} (Nur {leader_pts - pts} Pkt Rückstand auf P1)")
                boost += 0.05 * urgency
                
            # B) Europapokal
            elif pos <= 7:
                log.info(f"🇪🇺 Kampf um Europa für {team} (Platz {pos})")
                boost += 0.02 * urgency
                
            # C) ABSTIEGSKAMPF (DEINE IDEE: Basiert auf Punkten!)
            # Wenn das Team bereits auf einem Abstiegs/Relegationsplatz steht:
            if pos >= teams_count - 2:
                log.info(f"🆘 Akute Abstiegsgefahr für {team} (Platz {pos})")
                boost += 0.08 * urgency
            # Wenn das Team zwar oben steht (z.B. Platz 10), aber der Vorsprung winzig ist:
            elif (pts - relegation_pts) <= 6:
                log.info(f"⚔️ Abstiegskampf-Panik für {team} (Platz {pos}, aber nur {pts - relegation_pts} Pkt über dem Strich!)")
                # Wir geben ihnen denselben Boost, den sie kämpfen um ihr Leben!
                boost += 0.06 * urgency 
                
        return round(boost, 3)