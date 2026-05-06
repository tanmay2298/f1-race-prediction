"""
Fetches F1 data from the Jolpica Ergast API and FastF1.

Jolpica is the community-maintained successor to the now-shutdown Ergast API.
All API responses are cached locally to data/raw/ to avoid redundant network calls.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

BASE_URL = "https://api.jolpi.ca/ergast/f1"
RAW_CACHE_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

FASTF1_CACHE_DIR = Path(__file__).parent.parent / "cache"
FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(key: str) -> Path:
    return RAW_CACHE_DIR / f"{key}.json"


def _jolpica_get(endpoint: str, cache_key: str, force_refresh: bool = False) -> dict:
    """Fetch from Jolpica API with local JSON caching."""
    path = _cache_path(cache_key)
    if path.exists() and not force_refresh:
        with open(path) as f:
            return json.load(f)

    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            with open(path, "w") as f:
                json.dump(data, f)
            return data
        except requests.RequestException as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return {}


def fetch_race_results(year: int, round_num: int) -> pd.DataFrame:
    """
    Returns a DataFrame with race result info for the given year/round.

    Columns: driver_id, constructor_id, grid, position, points, status, laps, fastest_lap_rank
    """
    data = _jolpica_get(
        f"{year}/{round_num}/results.json?limit=30",
        f"{year}_{round_num}_results"
    )
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return pd.DataFrame()

    rows = []
    race = races[0]
    circuit_id = race.get("Circuit", {}).get("circuitId", "")
    race_date = race.get("date", "")
    for result in race.get("Results", []):
        driver = result.get("Driver", {})
        constructor = result.get("Constructor", {})
        rows.append({
            "year": year,
            "round": round_num,
            "circuit_id": circuit_id,
            "race_date": race_date,
            "driver_id": driver.get("driverId", ""),
            "driver_code": driver.get("code", ""),
            "driver_nationality": driver.get("nationality", ""),
            "constructor_id": constructor.get("constructorId", ""),
            "grid": int(result.get("grid", 0) or 0),
            "position": int(result.get("position", 0) or 0) if result.get("status") == "Finished" or "Lap" in result.get("status", "") else None,
            "points": float(result.get("points", 0) or 0),
            "status": result.get("status", ""),
            "laps": int(result.get("laps", 0) or 0),
            "fastest_lap_rank": int(result.get("FastestLap", {}).get("rank", 0) or 0),
            "winner": 1 if result.get("position") == "1" else 0,
        })
    return pd.DataFrame(rows)


def fetch_qualifying(year: int, round_num: int) -> pd.DataFrame:
    """
    Returns qualifying results for the given year/round.

    Columns: driver_id, constructor_id, qualifying_position, q1, q2, q3
    """
    data = _jolpica_get(
        f"{year}/{round_num}/qualifying.json?limit=30",
        f"{year}_{round_num}_qualifying"
    )
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return pd.DataFrame()

    rows = []
    for result in races[0].get("QualifyingResults", []):
        driver = result.get("Driver", {})
        constructor = result.get("Constructor", {})
        rows.append({
            "driver_id": driver.get("driverId", ""),
            "constructor_id": constructor.get("constructorId", ""),
            "qualifying_position": int(result.get("position", 0) or 0),
            "q1": result.get("Q1", ""),
            "q2": result.get("Q2", ""),
            "q3": result.get("Q3", ""),
        })
    return pd.DataFrame(rows)


def fetch_driver_standings(year: int, round_num: int) -> pd.DataFrame:
    """
    Returns driver championship standings after the given round.

    Columns: driver_id, driver_standings_pos, driver_points, driver_wins
    """
    data = _jolpica_get(
        f"{year}/{round_num}/driverstandings.json",
        f"{year}_{round_num}_driverstandings"
    )
    standings_list = (
        data.get("MRData", {})
        .get("StandingsTable", {})
        .get("StandingsLists", [])
    )
    if not standings_list:
        return pd.DataFrame()

    rows = []
    for entry in standings_list[0].get("DriverStandings", []):
        rows.append({
            "driver_id": entry.get("Driver", {}).get("driverId", ""),
            "driver_standings_pos": int(entry.get("position", 0) or 0),
            "driver_points": float(entry.get("points", 0) or 0),
            "driver_wins": int(entry.get("wins", 0) or 0),
        })
    return pd.DataFrame(rows)


def fetch_constructor_standings(year: int, round_num: int) -> pd.DataFrame:
    """
    Returns constructor championship standings after the given round.

    Columns: constructor_id, constructor_standings_pos, constructor_points
    """
    data = _jolpica_get(
        f"{year}/{round_num}/constructorstandings.json",
        f"{year}_{round_num}_constructorstandings"
    )
    standings_list = (
        data.get("MRData", {})
        .get("StandingsTable", {})
        .get("StandingsLists", [])
    )
    if not standings_list:
        return pd.DataFrame()

    rows = []
    for entry in standings_list[0].get("ConstructorStandings", []):
        rows.append({
            "constructor_id": entry.get("Constructor", {}).get("constructorId", ""),
            "constructor_standings_pos": int(entry.get("position", 0) or 0),
            "constructor_points": float(entry.get("points", 0) or 0),
        })
    return pd.DataFrame(rows)


def fetch_final_driver_standings(year: int) -> pd.DataFrame:
    """
    Returns the final driver championship standings for an entire season
    (no round number — Jolpica returns the last available round).

    Useful for pre-season predictions where no current-year standings exist.
    """
    data = _jolpica_get(
        f"{year}/driverstandings.json",
        f"{year}_final_driverstandings"
    )
    standings_list = (
        data.get("MRData", {})
        .get("StandingsTable", {})
        .get("StandingsLists", [])
    )
    if not standings_list:
        return pd.DataFrame()

    rows = []
    for entry in standings_list[0].get("DriverStandings", []):
        rows.append({
            "driver_id": entry.get("Driver", {}).get("driverId", ""),
            "driver_standings_pos": int(entry.get("position", 0) or 0),
            "driver_points": float(entry.get("points", 0) or 0),
            "driver_wins": int(entry.get("wins", 0) or 0),
        })
    return pd.DataFrame(rows)


def fetch_final_constructor_standings(year: int) -> pd.DataFrame:
    """
    Returns the final constructor championship standings for an entire season.
    """
    data = _jolpica_get(
        f"{year}/constructorstandings.json",
        f"{year}_final_constructorstandings"
    )
    standings_list = (
        data.get("MRData", {})
        .get("StandingsTable", {})
        .get("StandingsLists", [])
    )
    if not standings_list:
        return pd.DataFrame()

    rows = []
    for entry in standings_list[0].get("ConstructorStandings", []):
        rows.append({
            "constructor_id": entry.get("Constructor", {}).get("constructorId", ""),
            "constructor_standings_pos": int(entry.get("position", 0) or 0),
            "constructor_points": float(entry.get("points", 0) or 0),
        })
    return pd.DataFrame(rows)


def fetch_season_schedule(year: int) -> pd.DataFrame:
    """
    Returns the race schedule for the given season.

    Columns: round, race_name, circuit_id, circuit_name, country, locality, lat, lon, date
    """
    data = _jolpica_get(f"{year}.json", f"{year}_schedule")
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    rows = []
    for race in races:
        circuit = race.get("Circuit", {})
        location = circuit.get("Location", {})
        rows.append({
            "round": int(race.get("round", 0)),
            "race_name": race.get("raceName", ""),
            "circuit_id": circuit.get("circuitId", ""),
            "circuit_name": circuit.get("circuitName", ""),
            "country": location.get("country", ""),
            "locality": location.get("locality", ""),
            "lat": float(location.get("lat", 0) or 0),
            "lon": float(location.get("long", 0) or 0),
            "date": race.get("date", ""),
            "time": race.get("time", ""),
        })
    return pd.DataFrame(rows)


def fetch_all_season_results(year: int) -> pd.DataFrame:
    """Fetch all race results for an entire season at once (more efficient)."""
    data = _jolpica_get(
        f"{year}/results.json?limit=500",
        f"{year}_all_results"
    )
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    all_rows = []
    for race in races:
        round_num = int(race.get("round", 0))
        circuit_id = race.get("Circuit", {}).get("circuitId", "")
        race_date = race.get("date", "")
        for result in race.get("Results", []):
            driver = result.get("Driver", {})
            constructor = result.get("Constructor", {})
            all_rows.append({
                "year": year,
                "round": round_num,
                "circuit_id": circuit_id,
                "race_date": race_date,
                "driver_id": driver.get("driverId", ""),
                "driver_code": driver.get("code", ""),
                "driver_nationality": driver.get("nationality", ""),
                "constructor_id": constructor.get("constructorId", ""),
                "grid": int(result.get("grid", 0) or 0),
                "position": int(result.get("position", 0) or 0) if result.get("position", "0").isdigit() else None,
                "points": float(result.get("points", 0) or 0),
                "status": result.get("status", ""),
                "laps": int(result.get("laps", 0) or 0),
                "winner": 1 if result.get("position") == "1" else 0,
            })
    return pd.DataFrame(all_rows)


def fetch_all_season_qualifying(year: int) -> pd.DataFrame:
    """Fetch all qualifying results for an entire season at once."""
    data = _jolpica_get(
        f"{year}/qualifying.json?limit=500",
        f"{year}_all_qualifying"
    )
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    rows = []
    for race in races:
        round_num = int(race.get("round", 0))
        for result in race.get("QualifyingResults", []):
            driver = result.get("Driver", {})
            constructor = result.get("Constructor", {})
            rows.append({
                "year": year,
                "round": round_num,
                "driver_id": driver.get("driverId", ""),
                "constructor_id": constructor.get("constructorId", ""),
                "qualifying_position": int(result.get("position", 0) or 0),
            })
    return pd.DataFrame(rows)


def fetch_all_season_sprint_results(year: int) -> pd.DataFrame:
    """Fetch all sprint race results for an entire season. Returns empty DataFrame if no sprints."""
    data = _jolpica_get(
        f"{year}/sprint.json?limit=500",
        f"{year}_all_sprint"
    )
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    all_rows = []
    for race in races:
        round_num = int(race.get("round", 0))
        circuit_id = race.get("Circuit", {}).get("circuitId", "")
        for result in race.get("SprintResults", []):
            driver = result.get("Driver", {})
            pos_raw = result.get("position", "")
            all_rows.append({
                "year": year,
                "round": round_num,
                "circuit_id": circuit_id,
                "driver_id": driver.get("driverId", ""),
                "sprint_grid": int(result.get("grid", 0) or 0),
                "sprint_position": int(pos_raw) if str(pos_raw).isdigit() else None,
                "sprint_points": float(result.get("points", 0) or 0),
                "sprint_status": result.get("status", ""),
            })
    return pd.DataFrame(all_rows)


def fetch_sprint_results(year: int, round_num: int) -> pd.DataFrame:
    """Fetch sprint race results for a specific race. Returns empty DataFrame if no sprint."""
    data = _jolpica_get(
        f"{year}/{round_num}/sprint.json?limit=30",
        f"{year}_{round_num}_sprint"
    )
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return pd.DataFrame()
    rows = []
    for result in races[0].get("SprintResults", []):
        driver = result.get("Driver", {})
        pos_raw = result.get("position", "")
        rows.append({
            "driver_id": driver.get("driverId", ""),
            "sprint_grid": int(result.get("grid", 0) or 0),
            "sprint_position": int(pos_raw) if str(pos_raw).isdigit() else None,
            "sprint_points": float(result.get("points", 0) or 0),
            "sprint_status": result.get("status", ""),
        })
    return pd.DataFrame(rows)


def fetch_qualifying_fastf1(year: int, round_num: int) -> Optional[pd.DataFrame]:
    """
    Try to get qualifying results via FastF1 (works well for 2018+).
    Returns None if FastF1 is unavailable or data cannot be loaded.

    Columns: driver_id (using abbreviation), qualifying_position, grid_position
    """
    try:
        import logging
        import fastf1

        # Suppress FastF1's verbose request/warning logs — only show errors
        logging.getLogger("fastf1").setLevel(logging.ERROR)
        logging.getLogger("fastf1.req").setLevel(logging.ERROR)
        logging.getLogger("fastf1._api").setLevel(logging.ERROR)
        logging.getLogger("fastf1.core").setLevel(logging.ERROR)
        logging.getLogger("fastf1.logger").setLevel(logging.ERROR)

        fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))
        session = fastf1.get_session(year, round_num, "Q")
        session.load(telemetry=False, weather=False, messages=False)
        results = session.results[["Abbreviation", "Position", "DriverId"]].copy()
        results.columns = ["driver_code", "qualifying_position", "driver_id"]
        results["qualifying_position"] = results["qualifying_position"].astype(int)
        return results
    except Exception:
        return None
