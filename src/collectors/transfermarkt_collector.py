"""
Data Collector — Lokale Transfermarkt API (Microservice-frei!)
Holt Spielerprofile, Verletzungen und Statistiken DIREKT über die Python-Klassen.
"""

import difflib
import sys
import os
import logging
from src.utils.logger import get_logger

log = get_logger(__name__)

# ==========================================
# 1. Pfad-Magie: Transfermarkt-Ordner einbinden
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
tm_api_path = os.path.join(current_dir, "tm_api")

if tm_api_path not in sys.path:
    sys.path.insert(0, tm_api_path)

# ==========================================
# 2. Direkte Imports der KLASSEN
# ==========================================
from app.services.players.search import TransfermarktPlayerSearch
from app.services.players.profile import TransfermarktPlayerProfile
from app.services.players.injuries import TransfermarktPlayerInjuries
from app.services.players.absences import TransfermarktPlayerAbsences

# Für die Vereine (ich nutze hier die Standardnamen von felipeall)
from app.services.clubs.search import TransfermarktClubSearch
from app.services.clubs.players import TransfermarktClubPlayers


class TransfermarktCollector:
    
    def search_player(self, player_name: str) -> str:
        """Sucht nach einem Spieler und gibt die Transfermarkt-ID zurück."""
        log.info(f"Suche nach Spieler: {player_name}")
        
        try:
            # 1. Wir rufen die KLASSE auf und geben den Namen als Parameter mit
            tm_search = TransfermarktPlayerSearch(query=player_name)
            # 2. Wir führen die eigentliche Such-Methode der Klasse aus
            search_data = tm_search.search_players()
            
            # Die Daten stecken meistens im Feld "results"
            results = search_data.get("results", []) if isinstance(search_data, dict) else search_data
            
            if not results:
                log.warning(f"Kein Spieler gefunden für: {player_name}")
                return None
            
            return results[0]["id"]
        except Exception as e:
            log.error(f"Fehler bei Spielersuche: {e}")
            return None

    def get_player_profile(self, player_id: str) -> dict:
        """Holt das Profil (inkl. Marktwert) eines Spielers anhand seiner ID."""
        log.info(f"Hole Profil für Transfermarkt-ID: {player_id}")
        try:
            tm_profile = TransfermarktPlayerProfile(player_id=player_id)
            # HIER DIE METHODE ÄNDERN:
            return tm_profile.get_player_profile()
        except Exception as e:
            log.error(f"Fehler beim Profil-Abruf: {e}")
            return {}

    def get_player_injuries(self, player_id: str) -> list:
        """Holt medizinische Verletzungen UND Sperren (Rote Karten) und kombiniert sie."""
        injuries = []
        absences = []
        
        try:
            # Verletzungen über die Klasse abrufen
            tm_injuries = TransfermarktPlayerInjuries(player_id=player_id)
            inj_data = tm_injuries.get_player_injuries()
            injuries = inj_data.get("injuries", []) if isinstance(inj_data, dict) else []
        except Exception:
            pass 
            
        try:
            # Sperren über die Klasse abrufen
            tm_absences = TransfermarktPlayerAbsences(player_id=player_id)
            abs_data = tm_absences.get_player_absences()
            
            if isinstance(abs_data, dict):
                absences = abs_data.get("absences", abs_data.get("injuries", []))
        except Exception:
            pass
        
        return injuries + absences
    
    def search_club(self, search_term: str, original_name: str) -> str:
        """Sucht nach Vereinen und wählt mathematisch den ähnlichsten Namen aus."""
        log.info(f"Suche auf Transfermarkt nach: '{search_term}'")
        
        try:
            tm_club_search = TransfermarktClubSearch(query=search_term)
            search_data = tm_club_search.search_clubs()
            results = search_data.get("results", []) if isinstance(search_data, dict) else search_data
        except Exception as e:
            log.error(f"Fehler bei Vereinssuche: {e}")
            return None
            
        if not results:
            return None
        
        blacklist = [
            "u17", "u18", "u19", "u20", "u21", "u23",
            " ii", " 2", " b",          
            " b ",                       
            "junioren", "youth", "reserves", "women", "frauen",
            "amateur", "amateure", "academy", "under",
        ]

        valid_results = []
        for result in results:
            tm_name = result.get("name", "").lower().strip()
            is_blacklisted = any(bw in tm_name for bw in blacklist)
            ends_with_b = tm_name.endswith(" b")
            if not is_blacklisted and not ends_with_b:
                valid_results.append(result)
                
        if not valid_results:
            return None
            
        valid_results.sort(
            key=lambda x: difflib.SequenceMatcher(None, original_name.lower(), x["name"].lower()).ratio(), 
            reverse=True
        )
        
        best_match = valid_results[0]
        return best_match["id"]

    def get_club_players(self, club_id: str) -> list:
        """Holt den kompletten aktuellen Kader eines Vereins."""
        log.info(f"Hole Spieler für Transfermarkt-Club-ID: {club_id}")
        try:
            tm_club_players = TransfermarktClubPlayers(club_id=club_id)
            data = tm_club_players.get_club_players()
            players = data.get("players", []) if isinstance(data, dict) else data
            if not players:
                # Nationalteam-Kaderseiten haben anderes Markup als Vereinsseiten —
                # der Bibliotheks-Parser liefert dort eine leere Liste
                players = self._parse_squad_fallback(tm_club_players.page)
                if players:
                    log.info(f"Kader via Fallback-Parser: {len(players)} Spieler (Club-ID {club_id})")
            return players
        except Exception as e:
            log.error(f"Fehler beim Kader-Abruf: {e}")
            return []

    @staticmethod
    def _parse_squad_fallback(page) -> list:
        """
        Minimal-Parser für Kaderseiten, deren Markup der Bibliotheks-Parser
        nicht versteht (z.B. Nationalteams). Liefert id, name, marketValue —
        alles was die Verletzungsanalyse braucht.
        """
        players = []
        for row in page.xpath("//table[@class='items']//tbody/tr"):
            link = row.xpath(".//td[@class='hauptlink']/a[contains(@href, '/profil/spieler/')]")
            if not link:
                continue
            href = link[0].get("href", "")
            player_id = href.rstrip("/").split("/")[-1]
            name = "".join(link[0].itertext()).strip()
            mv = row.xpath(".//td[contains(@class, 'rechts')]//a[contains(@href, '/marktwertverlauf/')]/text()")
            players.append({
                "id": player_id,
                "name": name,
                "marketValue": mv[0].strip() if mv else "0",
            })
        return players

    def get_national_team_players(self, country_name: str) -> list:
        """
        Sucht Nationalteam-Spieler auf Transfermarkt via Vereinssuche (nation = club-like entity).
        Transfermarkt listet Nationalteams als spezielle Vereine (z.B. "Germany" club_id).
        Fallback: leere Liste wenn nicht gefunden.
        """
        log.info(f"Suche Nationalteam-Spieler für: {country_name}")
        try:
            # TM treat national teams like clubs — search with country name
            tm_id = self.search_club(country_name, country_name)

            if not tm_id:
                log.warning(f"Keine TM-ID für Nationalteam: {country_name}")
                return []

            return self.get_club_players(tm_id)
        except Exception as e:
            log.warning(f"National team player lookup failed for {country_name}: {e}")
            return []