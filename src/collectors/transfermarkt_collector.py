"""
Data Collector — Lokale Transfermarkt API (Microservice)
Holt Spielerprofile, Verletzungen und Statistiken über http://localhost:8000
"""

import requests
import difflib
from src.utils.logger import get_logger

log = get_logger(__name__)

class TransfermarktCollector:
    BASE_URL = "http://localhost:8000"

    def _get(self, endpoint: str) -> dict:
        """Hilfsfunktion für GET-Requests an die lokale API."""
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            log.error(f"Fehler bei Anfrage an {url}: {e}")
            return {}

    def search_player(self, player_name: str) -> str:
        """Sucht nach einem Spieler und gibt die Transfermarkt-ID zurück."""
        log.info(f"Suche nach Spieler: {player_name}")
        data = self._get(f"players/search/{player_name}")
        
        # Wir nehmen den ersten Treffer (kann später noch verfeinert werden)
        results = data.get("results", [])
        if not results:
            log.warning(f"Kein Spieler gefunden für: {player_name}")
            return None
        
        return results[0]["id"]

    def get_player_profile(self, player_id: str) -> dict:
        """Holt das Profil (inkl. Marktwert) eines Spielers anhand seiner ID."""
        log.info(f"Hole Profil für Transfermarkt-ID: {player_id}")
        return self._get(f"players/{player_id}/profile")

    def get_player_injuries(self, player_id: str) -> list:
        """Holt medizinische Verletzungen UND Sperren (Rote Karten) und kombiniert sie."""
        # 1. Medizinische Ausfälle holen
        inj_data = self._get(f"players/{player_id}/injuries")
        injuries = inj_data.get("injuries", []) if inj_data else []
        
        # 2. Sperren & Disziplinarische Ausfälle holen
        abs_data = self._get(f"players/{player_id}/absences")
        absences = abs_data.get("injuries", []) if abs_data else []
        
        # Beide Listen kombinieren
        return injuries + absences
    
    def search_club(self, search_term: str, original_name: str) -> str:
        """Sucht nach Vereinen und wählt mathematisch den ähnlichsten Namen aus."""
        log.info(f"Suche auf Transfermarkt nach: '{search_term}'")
        data = self._get(f"clubs/search/{search_term}")
        
        results = data.get("results", [])
        if not results:
            return None
        
        # Blacklist für ALLES was keine Profi-Herrenmannschaft ist
        blacklist = [
            "u17", "u18", "u19", "u20", "u21", "u23",
            " ii", " 2", " b",          # B-Teams / Reserven
            " b ",                       # " B " mitten im Namen
            "junioren", "youth", "reserves", "women", "frauen",
            "amateur", "amateure", "academy", "under",
        ]

        valid_results = []
        for result in results:
            tm_name = result.get("name", "").lower().strip()
            # KEIN Blacklist-Wort im Namen + Name endet nicht auf " b"
            is_blacklisted = any(bw in tm_name for bw in blacklist)
            ends_with_b = tm_name.endswith(" b")
            if not is_blacklisted and not ends_with_b:
                valid_results.append(result)
                
        if not valid_results:
            return None
            
        # MAGIE: Wir sortieren die restlichen Teams danach, wie ähnlich sie dem ORIGINAL-Datenbank-Namen sind!
        valid_results.sort(
            key=lambda x: difflib.SequenceMatcher(None, original_name.lower(), x["name"].lower()).ratio(), 
            reverse=True
        )
        
        best_match = valid_results[0]
        return best_match["id"]

    def get_club_players(self, club_id: str) -> list:
        """Holt den kompletten aktuellen Kader eines Vereins."""
        log.info(f"Hole Spieler für Transfermarkt-Club-ID: {club_id}")
        data = self._get(f"clubs/{club_id}/players")
        return data.get("players", [])