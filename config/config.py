"""
Central configuration loader.
Reads from environment variables / .env file.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env — suche in mehreren Orten
from pathlib import Path as _P
_env_paths = [
    _P(__file__).parent / ".env",           # config/.env
    _P(__file__).parent.parent / ".env",    # .env im Projektroot
]
for _ep in _env_paths:
    if _ep.exists():
        load_dotenv(dotenv_path=_ep, override=True)
        break


class Config:
    # --- API Keys ---
    FOOTBALL_DATA_API_KEY: str = os.getenv("FOOTBALL_DATA_API_KEY", "")
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
    ODDS_API_KEY: str = os.getenv("ODDS_API_KEY", "")

    # --- Database ---
    DB_PATH: Path = Path(os.getenv("DB_PATH", "data/football.db"))

    # --- App Settings ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DEFAULT_LEAGUE: str = os.getenv("DEFAULT_LEAGUE", "BL1")
    SIMULATION_RUNS: int = int(os.getenv("SIMULATION_RUNS", "10000"))

    # --- Leagues supported (football-data.org codes) ---
    SUPPORTED_LEAGUES: dict = {
        # Tier 1
        "BL1": "Bundesliga",
        "PL":  "Premier League",
        "PD":  "La Liga",
        "SA":  "Serie A",
        "FL1": "Ligue 1",
        # Tier 2
        "BL2": "2. Bundesliga",
        "ELC": "Championship",
        "SD":  "Segunda División",
        "SB":  "Serie B",
        "FL2": "Ligue 2",
        # Europa
        "CL":  "Champions League",
        "EL":  "Europa League",
        "UECL": "Conference League",
        # Turniere
        "WC":  "FIFA World Cup",
        "EC":  "European Championship",
    }

    # --- Turniere auf neutralem Platz (kein Heimvorteil im Modell) ---
    NEUTRAL_VENUE_LEAGUES: frozenset = frozenset({"WC", "EC"})

    # --- Dixon-Coles Zeitgewichtung: exp(-xi * Tage) ---
    # Klubfußball: Halbwertszeit ~230 Tage. Turniere finden nur alle 4 Jahre
    # statt — dort muss die Halbwertszeit ~9.5 Jahre betragen, sonst hat das
    # gesamte Datenset effektiv Gewicht ~0 und die Team-Stärken degenerieren.
    POISSON_TIME_DECAY: dict = {
        "default": 0.003,
        "WC": 0.0002,
        "EC": 0.0002,
    }


    # --- API Base URLs ---
    FOOTBALL_DATA_BASE_URL: str = "https://api.football-data.org/v4"
    OPENWEATHER_BASE_URL: str = "https://api.openweathermap.org/data/2.5"
    ODDS_API_BASE_URL: str = "https://api.the-odds-api.com/v4"

    def validate(self) -> None:
        """Raises if critical keys are missing."""
        missing = []
        if not self.FOOTBALL_DATA_API_KEY:
            missing.append("FOOTBALL_DATA_API_KEY")
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                f"→ Copy config/.env.example to config/.env and add your keys."
            )


config = Config()