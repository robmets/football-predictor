import asyncio
import difflib
import logging
from src.utils.logger import get_logger

# Importiere deine lokalen Sofascore-Skripte
from src.collectors.sofascore.search import search_team_on_sofascore
from src.collectors.sofascore.teams import get_team_events, get_team_players
from src.collectors.sofascore.matches import (
    get_match_lineups_and_ratings,
    get_predicted_lineups,
    parse_player_ratings,
)
from src.collectors.sofascore.players import get_player_stats_and_attributes

log = get_logger(__name__)

class SofascoreCollector:
    """
    Kapselt die asynchronen Sofascore-Funktionen in synchrone Aufrufe
    und übernimmt das automatische Mapping der Transfermarkt-Namen.
    """
    
    def get_team_id(self, team_name: str) -> int:
        """Sucht die Sofascore Team-ID über den Namen."""
        log.info(f"Mappe Team '{team_name}' zu Sofascore...")
        try:
            result = asyncio.run(search_team_on_sofascore(team_name))
            if result and "error" not in result:
                log.info(f"Sofascore ID für {team_name}: {result['sofascore_id']}")
                return result["sofascore_id"]
            return None
        except Exception as e:
            log.error(f"Sofascore Mapping-Fehler für {team_name}: {e}")
            return None

    def get_match_id(self, team_id: int, opponent_name: str) -> int:
        """Sucht im Kalender nach dem Spiel gegen den bestimmten Gegner."""
        try:
            result = asyncio.run(get_team_events(team_id))
            events = result.get("events", [])
            
            for event in events:
                home_name = event.get("home_team", {}).get("name", "")
                away_name = event.get("away_team", {}).get("name", "")
                
                # Mapping: Ist der Transfermarkt-Gegnername ähnlich dem Sofascore-Namen?
                if (difflib.SequenceMatcher(None, opponent_name.lower(), home_name.lower()).ratio() > 0.7 or 
                    difflib.SequenceMatcher(None, opponent_name.lower(), away_name.lower()).ratio() > 0.7):
                    return event.get("match_id")
            return None
        except Exception as e:
            log.error(f"Sofascore Match-Suche fehlgeschlagen: {e}")
            return None

    def get_match_ratings(self, match_id: int) -> dict:
        """Holt die Aufstellung und jagt sie direkt durch deinen Parser."""
        try:
            raw_data = asyncio.run(get_match_lineups_and_ratings(match_id))
            parsed_data = parse_player_ratings(raw_data)
            return parsed_data
        except Exception as e:
            log.error(f"Fehler beim Abrufen der Match-Ratings: {e}")
            return None

    def get_predicted_match_ratings(self, match_id: int) -> dict | None:
        """
        Holt voraussichtliche Aufstellung (Modus B).
        Funktioniert auch wenn Lineup noch nicht bestätigt.
        """
        try:
            return asyncio.run(get_predicted_lineups(match_id))
        except Exception as e:
            log.error(f"Fehler beim Abrufen der Predicted Lineups: {e}")
            return None

    def get_injured_player_impact(self, team_id: int, missing_player_names: list, match_starters: list = None) -> dict:
        """
        Berechnet Sofascore-Penalty für fehlende Spieler UND Team-Attribut-Rating.
        Rating wird IMMER berechnet, auch wenn keine Spieler fehlen.

        Returns dict mit:
          - penalty (float): Prozentpunkte Verletzungs-Penalty
          - team_avg_rating (float): Attribut-Rating 0-100 (50.0 wenn keine Daten)
        """
        impact_penalty = 0.0
        team_attribute_scores = []
        real_team_avg = 50.0  # Fallback: neutral

        try:
            # === STUFE 1 & 2: Voraussichtliche oder Live-Aufstellung ===
            if match_starters and len(match_starters) == 11:
                log.info("Nutze 11 Spieler aus der Match-Aufstellung für den Basis-Schnitt!")
                roster = match_starters
                core_players = roster
            else:
                # === STUFE 3: Fallback auf den Kader ===
                try:
                    team_data = asyncio.run(get_team_players(team_id))
                    roster = team_data.get("players", team_data.get("roster", []))
                except Exception as e:
                    log.error(f"Kader-Abruf fehlgeschlagen für Team {team_id}: {e}")
                    return {"penalty": 0.0, "team_avg_rating": 50.0}
                core_players = roster[:15]

            if not roster:
                log.warning(f"Keine Spielerdaten für Team {team_id} — neutrales Rating")
                return {"penalty": 0.0, "team_avg_rating": 50.0}

            # 2. DEN ECHTEN DURCHSCHNITT BERECHNEN
            team_attribute_scores = []
            
            for player in core_players: 
                player_dict = player.get("player", player) # FIX: Egal ob verschachtelt oder flach!
                p_id = player_dict.get("id")
                
                if p_id:
                    try:
                        stats = asyncio.run(get_player_stats_and_attributes(p_id))
                        attrs = stats.get("attributes", {})
                        vals = [v for v in attrs.values() if isinstance(v, (int, float))]
                        if vals:
                            team_attribute_scores.append(sum(vals) / len(vals))
                    except Exception:
                        pass
                        
            real_team_avg = sum(team_attribute_scores) / len(team_attribute_scores) if team_attribute_scores else 68.0
            log.info(f"Dynamisches Team-Attribut-Rating berechnet: {real_team_avg:.1f}")
            
            # 3. VERLETZTE SPIELER ANALYSIEREN
            for missing_name in missing_player_names:
                best_match = None
                highest_ratio = 0.0
                
                for player in roster:
                    player_dict = player.get("player", player)
                    p_name = player_dict.get("name", "")
                    ratio = difflib.SequenceMatcher(None, missing_name.lower(), p_name.lower()).ratio()
                    
                    if ratio > 0.75 and ratio > highest_ratio:
                        highest_ratio = ratio
                        best_match = player
                
                if best_match:
                    # FIX: Holt sich sicher den Namen und die ID!
                    player_dict = best_match.get("player", best_match)
                    p_id = player_dict.get("id")
                    p_name = player_dict.get("name", "Unbekannt")
                    
                    if p_id:
                        stats = asyncio.run(get_player_stats_and_attributes(p_id))
                        attrs = stats.get("attributes", {})
                        vals = [v for v in attrs.values() if isinstance(v, (int, float))]
                        
                        if vals:
                            player_avg = sum(vals) / len(vals)
                            
                            # DAS DUELL: Verletzter Spieler vs. aktuelles Team
                            if player_avg > real_team_avg:
                                diff = player_avg - real_team_avg
                                penalty = diff * 0.5  # diff=5 Punkte → 2.5 Prozentpunkte
                                impact_penalty += penalty
                                log.info(f"Sofascore Impact: {p_name} fehlt (Attribute Ø {player_avg:.1f} > Team-Ø {real_team_avg:.1f}) -> Penalty: -{penalty:.1%}")
                        else:
                            log.warning(f"Sofascore hat keine Attribute für {p_name} gefunden.")
                            
        except Exception as e:
            log.error(f"Fehler bei Impact-Berechnung: {e}")
            
        return {
            "penalty": round(impact_penalty, 4),
            "team_avg_rating": round(real_team_avg, 2) if team_attribute_scores else 50.0,
        }