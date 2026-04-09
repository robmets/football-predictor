"""
Weather Collector — OpenWeatherMap API
Fetches current and forecast weather for stadium cities.

Weather affects match outcomes:
  - Heavy rain/wind → fewer goals (defensive play)
  - Cold temperatures → slower game
  - Extreme heat → fatigue effects

Usage:
    collector = WeatherCollector()
    weather = collector.get_match_weather("Hamburg", match_date="2025-04-05")
"""

import requests
from datetime import datetime, date, timedelta
from src.utils.logger import get_logger
from config.config import config

log = get_logger(__name__)

# Team → Stadion-Stadt (Bundesliga)
TEAM_CITY = {
    "FC Bayern München":          "Munich,DE",
    "Bayer 04 Leverkusen":        "Leverkusen,DE",
    "Borussia Dortmund":          "Dortmund,DE",
    "RB Leipzig":                 "Leipzig,DE",
    "VfB Stuttgart":              "Stuttgart,DE",
    "Eintracht Frankfurt":        "Frankfurt,DE",
    "SC Freiburg":                "Freiburg,DE",
    "1. FC Union Berlin":         "Berlin,DE",
    "1. FSV Mainz 05":            "Mainz,DE",
    "Borussia Mönchengladbach":   "Mönchengladbach,DE",
    "FC Augsburg":                "Augsburg,DE",
    "VfL Wolfsburg":              "Wolfsburg,DE",
    "SV Werder Bremen":           "Bremen,DE",
    "TSG 1899 Hoffenheim":        "Sinsheim,DE",
    "VfL Bochum 1848":            "Bochum,DE",
    "Hamburger SV":               "Hamburg,DE",
    "1. FC Heidenheim 1846":      "Heidenheim,DE",
    "FC St. Pauli 1910":          "Hamburg,DE",
    "Holstein Kiel":              "Kiel,DE",
    "1. FC Köln":                 "Cologne,DE",
    "Fortuna Düsseldorf":         "Düsseldorf,DE",
    "Hannover 96":                "Hannover,DE",
    "SpVgg Greuther Fürth":       "Fürth,DE",
    "SV Darmstadt 98":            "Darmstadt,DE",
    # Premier League
    "Arsenal FC":                 "London,GB",
    "Chelsea FC":                 "London,GB",
    "Liverpool FC":               "Liverpool,GB",
    "Manchester City FC":         "Manchester,GB",
    "Manchester United FC":       "Manchester,GB",
    "Tottenham Hotspur FC":       "London,GB",
    "Newcastle United FC":        "Newcastle,GB",
    "Aston Villa FC":             "Birmingham,GB",
    "West Ham United FC":         "London,GB",
    "Brighton & Hove Albion FC":  "Brighton,GB",
    "Fulham FC":                  "London,GB",
    "Wolverhampton Wanderers FC": "Wolverhampton,GB",
    "Everton FC":                 "Liverpool,GB",
    "Brentford FC":               "London,GB",
    "Crystal Palace FC":          "London,GB",
    "Nottingham Forest FC":       "Nottingham,GB",
    "AFC Bournemouth":            "Bournemouth,GB",
    "Leicester City FC":          "Leicester,GB",
    "Ipswich Town FC":            "Ipswich,GB",
    "Southampton FC":             "Southampton,GB",
    # La Liga
    "Real Madrid CF":             "Madrid,ES",
    "FC Barcelona":               "Barcelona,ES",
    "Club Atlético de Madrid":    "Madrid,ES",
    "Athletic Club":              "Bilbao,ES",
    "Real Sociedad de Fútbol":    "San Sebastian,ES",
    "Villarreal CF":              "Villarreal,ES",
    "Sevilla FC":                 "Seville,ES",
    "Real Betis Balompié":        "Seville,ES",
    "Rayo Vallecano":             "Madrid,ES",
    "Getafe CF":                  "Getafe,ES",
    "RC Celta de Vigo":           "Vigo,ES",
    "RCD Espanyol":               "Barcelona,ES",
    "Deportivo Alavés":           "Vitoria,ES",
    "RCD Mallorca":               "Palma,ES",
    "UD Las Palmas":              "Las Palmas,ES",
    "Valencia CF":                "Valencia,ES",
    "Girona FC":                  "Girona,ES",
    "CA Osasuna":                 "Pamplona,ES",
    "Real Valladolid CF":         "Valladolid,ES",
    "CD Leganés":                 "Leganes,ES",
    # Serie A
    "FC Internazionale Milano":   "Milan,IT",
    "AC Milan":                   "Milan,IT",
    "Juventus FC":                "Turin,IT",
    "SSC Napoli":                 "Naples,IT",
    "AS Roma":                    "Rome,IT",
    "SS Lazio":                   "Rome,IT",
    "ACF Fiorentina":             "Florence,IT",
    "Atalanta BC":                "Bergamo,IT",
    "Torino FC":                  "Turin,IT",
    "Bologna FC 1909":            "Bologna,IT",
    "US Sassuolo Calcio":         "Sassuolo,IT",
    "Genoa CFC":                  "Genoa,IT",
    "Hellas Verona FC":           "Verona,IT",
    "Udinese Calcio":             "Udine,IT",
    "Cagliari Calcio":            "Cagliari,IT",
    "Empoli FC":                  "Empoli,IT",
    "Venezia FC":                 "Venice,IT",
    "Como 1907":                  "Como,IT",
    "Parma Calcio 1913":          "Parma,IT",
    "US Lecce":                   "Lecce,IT",
    # Ligue 1
    "Paris Saint-Germain FC":     "Paris,FR",
    "Olympique de Marseille":     "Marseille,FR",
    "Olympique Lyonnais":         "Lyon,FR",
    "AS Monaco FC":               "Monaco,MC",
    "LOSC Lille":                 "Lille,FR",
    "Stade Rennais FC 1901":      "Rennes,FR",
    "OGC Nice":                   "Nice,FR",
    "RC Lens":                    "Lens,FR",
    "Montpellier HSC":            "Montpellier,FR",
    "Stade de Reims":             "Reims,FR",
    "RC Strasbourg Alsace":       "Strasbourg,FR",
    "Nantes FC":                  "Nantes,FR",
    "Toulouse FC":                "Toulouse,FR",
    "FC Nantes":                  "Nantes,FR",
    "Stade Brestois 29":          "Brest,FR",
    "Le Havre AC":                "Le Havre,FR",
    "Angers SCO":                 "Angers,FR",
    "AJ Auxerre":                 "Auxerre,FR",
    # 2. Bundesliga (BL2)
    "Hertha BSC":                 "Berlin,DE",
    "FC Schalke 04":              "Gelsenkirchen,DE",
    "Fortuna Düsseldorf":         "Düsseldorf,DE",
    "Hannover 96":                "Hannover,DE",
    "Karlsruher SC":              "Karlsruhe,DE",
    "SC Paderborn 07":            "Paderborn,DE",
    "1. FC Nürnberg":             "Nuremberg,DE",
    "1. FC Kaiserslautern":       "Kaiserslautern,DE",
    "Eintracht Braunschweig":     "Braunschweig,DE",
    "1. FC Magdeburg":            "Magdeburg,DE",
    
    # Championship (ELC)
    "Leeds United FC":            "Leeds,GB",
    "Sunderland AFC":             "Sunderland,GB",
    "Sheffield United FC":        "Sheffield,GB",
    "West Bromwich Albion FC":    "West Bromwich,GB",
    "Burnley FC":                 "Burnley,GB",
    "Middlesbrough FC":           "Middlesbrough,GB",
    "Norwich City FC":            "Norwich,GB",
    "Coventry City FC":           "Coventry,GB",
    "Watford FC":                 "Watford,GB",
    "Luton Town FC":              "Luton,GB",
    
    # Segunda División (SD)
    "UD Almería":                 "Almeria,ES",
    "Granada CF":                 "Granada,ES",
    "Cádiz CF":                   "Cadiz,ES",
    "Levante UD":                 "Valencia,ES",
    "Real Oviedo":                "Oviedo,ES",
    "Real Sporting de Gijón":     "Gijon,ES",
    "Elche CF":                   "Elche,ES",
    "Real Zaragoza":              "Zaragoza,ES",
    
    # Serie B (SB)
    "Frosinone Calcio":           "Frosinone,IT",
    "US Salernitana 1919":        "Salerno,IT",
    "UC Sampdoria":               "Genoa,IT",
    "Palermo FC":                 "Palermo,IT",
    "SSC Bari":                   "Bari,IT",
    "Brescia Calcio":             "Brescia,IT",
    "US Cremonese":               "Cremona,IT",
    
    # Ligue 2 (FL2)
    "FC Metz":                    "Metz,FR",
    "FC Lorient":                 "Lorient,FR",
    "Clermont Foot 63":           "Clermont-Ferrand,FR",
    "Paris FC":                   "Paris,FR",
    "En Avant Guingamp":          "Guingamp,FR",
    "SM Caen":                    "Caen,FR",
}

# Weather impact factors on expected goals
# Research shows ~5-8% goal reduction in heavy rain/wind
WEATHER_IMPACT = {
    "heavy_rain":    -0.08,   # Starkregen
    "rain":          -0.04,   # Regen
    "snow":          -0.10,   # Schnee
    "strong_wind":   -0.06,   # Starkwind (>50 km/h)
    "wind":          -0.02,   # Normaler Wind (30-50 km/h)
    "extreme_heat":  -0.05,   # Hitze >32°C
    "cold":          -0.02,   # Kälte <2°C
    "ideal":          0.0,    # Optimale Bedingungen
}


class WeatherCollector:
    """Fetches weather data and computes goal impact factor."""

    def __init__(self):
        self.api_key = config.OPENWEATHER_API_KEY
        self.base_url = config.OPENWEATHER_BASE_URL

        if not self.api_key:
            log.warning("OPENWEATHER_API_KEY not set — weather features disabled")

    def get_match_weather(
        self,
        home_team: str,
        match_date: str | date = None,
    ) -> dict:
        """
        Fetch weather for the home team's stadium city.

        Args:
            home_team:   Name of the home team (used to look up city)
            match_date:  Date of the match (YYYY-MM-DD or date object).
                         If None or today → current weather.
                         If future → forecast.

        Returns:
            {
                city, temperature_c, wind_kmh, precipitation_mm,
                condition, condition_code,
                goal_impact_factor,  # multiplier: 0.90 = 10% fewer goals
                description
            }
        """
        if not self.api_key:
            return self._no_data()

        city = TEAM_CITY.get(home_team)
        if not city:
            log.warning(f"No city mapping for '{home_team}' — skipping weather")
            return self._no_data()

        if match_date is None:
            match_date = date.today()
        elif isinstance(match_date, str):
            match_date = date.fromisoformat(match_date)

        days_ahead = (match_date - date.today()).days

        try:
            if days_ahead <= 0:
                data = self._fetch_current(city)
            elif days_ahead <= 5:
                data = self._fetch_forecast(city, days_ahead)
            else:
                log.info(f"Match too far ahead ({days_ahead}d) — using climatological average")
                return self._no_data(city=city)
        except Exception as e:
            log.error(f"Weather API error for {city}: {e}")
            return self._no_data(city=city)

        result = self._parse(data, city)
        log.info(
            f"Weather {city}: {result['temperature_c']:.0f}°C, "
            f"wind {result['wind_kmh']:.0f} km/h, "
            f"{result['condition']} → goal factor {result['goal_impact_factor']:+.0%}"
        )
        return result

    # ── Private ───────────────────────────────────────────────────────────────

    def _fetch_current(self, city: str) -> dict:
        resp = requests.get(
            f"{self.base_url}/weather",
            params={"q": city, "appid": self.api_key, "units": "metric"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def _fetch_forecast(self, city: str, days_ahead: int) -> dict:
        """Fetch 5-day/3-hour forecast and pick the closest entry to match time."""
        resp = requests.get(
            f"{self.base_url}/forecast",
            params={"q": city, "appid": self.api_key, "units": "metric"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        target = datetime.now() + timedelta(days=days_ahead)
        best = min(
            data["list"],
            key=lambda x: abs(datetime.fromtimestamp(x["dt"]) - target)
        )
        # Normalize to current-weather format
        return {
            "main":    best["main"],
            "wind":    best["wind"],
            "weather": best["weather"],
            "rain":    best.get("rain", {}),
            "snow":    best.get("snow", {}),
        }

    def _parse(self, data: dict, city: str) -> dict:
        temp_c    = data["main"]["temp"]
        wind_ms   = data["wind"]["speed"]
        wind_kmh  = wind_ms * 3.6
        rain_mm   = data.get("rain", {}).get("1h", 0) or data.get("rain", {}).get("3h", 0) or 0
        snow_mm   = data.get("snow", {}).get("1h", 0) or data.get("snow", {}).get("3h", 0) or 0
        condition = data["weather"][0]["main"].lower()
        code      = data["weather"][0]["id"]

        # Determine impact
        impact = 0.0
        label  = "ideal"

        if snow_mm > 0:
            impact = WEATHER_IMPACT["snow"]
            label  = "snow"
        elif rain_mm > 2.5:
            impact = WEATHER_IMPACT["heavy_rain"]
            label  = "heavy_rain"
        elif rain_mm > 0.5 or "rain" in condition:
            impact = WEATHER_IMPACT["rain"]
            label  = "rain"
        elif wind_kmh > 50:
            impact = WEATHER_IMPACT["strong_wind"]
            label  = "strong_wind"
        elif wind_kmh > 30:
            impact = WEATHER_IMPACT["wind"]
            label  = "wind"

        if temp_c > 32:
            impact += WEATHER_IMPACT["extreme_heat"]
            label   = "extreme_heat"
        elif temp_c < 2:
            impact += WEATHER_IMPACT["cold"]

        return {
            "city":               city,
            "temperature_c":      round(temp_c, 1),
            "wind_kmh":           round(wind_kmh, 1),
            "precipitation_mm":   round(rain_mm + snow_mm, 1),
            "condition":          label,
            "condition_raw":      condition,
            "goal_impact_factor": round(impact, 3),
            "description":        self._describe(label, temp_c, wind_kmh, rain_mm + snow_mm),
        }

    @staticmethod
    def _describe(label: str, temp: float, wind: float, precip: float) -> str:
        descriptions = {
            "snow":        f"Schneefall ({precip:.1f}mm) — deutlich weniger Tore erwartet",
            "heavy_rain":  f"Starkregen ({precip:.1f}mm) — weniger Tore erwartet",
            "rain":        f"Regen ({precip:.1f}mm) — leicht weniger Tore",
            "strong_wind": f"Starkwind ({wind:.0f} km/h) — erschwerte Bedingungen",
            "wind":        f"Windiges Wetter ({wind:.0f} km/h) — minimaler Einfluss",
            "extreme_heat":f"Hitze ({temp:.0f}°C) — Ermüdungseffekte möglich",
            "ideal":       f"Ideale Bedingungen ({temp:.0f}°C) — kein Wettereinfluss",
        }
        return descriptions.get(label, "Normales Wetter")

    @staticmethod
    def _no_data(city: str = "Unknown") -> dict:
        return {
            "city": city, "temperature_c": 12.0, "wind_kmh": 15.0,
            "precipitation_mm": 0.0, "condition": "unknown",
            "condition_raw": "unknown", "goal_impact_factor": 0.0,
            "description": "Keine Wetterdaten verfügbar — kein Einfluss berechnet",
        }