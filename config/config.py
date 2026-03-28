"""
Central configuration loader.
Reads from environment variables / .env file.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env from config/ folder
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path)


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

    # --- Leagues supported by football-data.org ---
    SUPPORTED_LEAGUES: dict = {
        "BL1": "Bundesliga",
        "PL":  "Premier League",
        "CL":  "Champions League",
        "PD":  "La Liga",
        "SA":  "Serie A",
        "FL1": "Ligue 1",
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
