"""
Fetches weather data from the Open-Meteo API (no API key required).

Supports both historical race dates and future forecasts.
Data is cached locally to avoid redundant requests.
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Union

import requests

CACHE_DIR = Path(__file__).parent.parent / "data" / "raw"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "precipitation_hours",
]


def _cache_key(lat: float, lon: float, race_date: str) -> Path:
    lat_str = f"{lat:.4f}".replace(".", "_")
    lon_str = f"{lon:.4f}".replace(".", "_")
    return CACHE_DIR / f"weather_{lat_str}_{lon_str}_{race_date}.json"


def fetch_race_weather(
    lat: float,
    lon: float,
    race_date: Union[str, date],
) -> dict:
    """
    Fetch weather data for a specific race day and location.

    Args:
        lat: Circuit latitude
        lon: Circuit longitude
        race_date: Date of the race (YYYY-MM-DD string or date object)

    Returns:
        dict with keys:
            - temp_max_celsius: max air temperature
            - temp_min_celsius: min air temperature
            - rain_mm: total precipitation in mm
            - wind_speed_kmh: max wind speed
            - rain_hours: hours with precipitation (proxy for probability)
            - is_wet: bool (rain_mm > 1.0)
    """
    if isinstance(race_date, date):
        race_date = race_date.strftime("%Y-%m-%d")

    cache_file = _cache_key(lat, lon, race_date)
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    race_dt = datetime.strptime(race_date, "%Y-%m-%d").date()
    today = date.today()

    if race_dt < today - timedelta(days=2):
        # Historical data
        url = HISTORICAL_URL
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": race_date,
            "end_date": race_date,
            "daily": ",".join(WEATHER_VARIABLES),
            "timezone": "UTC",
        }
    else:
        # Forecast (up to 16 days ahead)
        url = FORECAST_URL
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ",".join(WEATHER_VARIABLES),
            "forecast_days": 16,
            "timezone": "UTC",
        }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
    except requests.RequestException as e:
        return _default_weather()

    daily = raw.get("daily", {})
    dates = daily.get("time", [])

    if race_date not in dates:
        return _default_weather()

    idx = dates.index(race_date)

    def safe_get(key: str, fallback=0.0):
        vals = daily.get(key, [])
        if idx < len(vals) and vals[idx] is not None:
            return vals[idx]
        return fallback

    result = {
        "temp_max_celsius": safe_get("temperature_2m_max", 20.0),
        "temp_min_celsius": safe_get("temperature_2m_min", 15.0),
        "rain_mm": safe_get("precipitation_sum", 0.0),
        "wind_speed_kmh": safe_get("wind_speed_10m_max", 10.0),
        "rain_hours": safe_get("precipitation_hours", 0.0),
        "is_wet": safe_get("precipitation_sum", 0.0) > 1.0,
    }

    with open(cache_file, "w") as f:
        json.dump(result, f)

    return result


def _default_weather() -> dict:
    """Return neutral weather defaults when data is unavailable."""
    return {
        "temp_max_celsius": 20.0,
        "temp_min_celsius": 15.0,
        "rain_mm": 0.0,
        "wind_speed_kmh": 10.0,
        "rain_hours": 0.0,
        "is_wet": False,
    }
