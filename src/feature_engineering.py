"""
Builds per-driver feature rows for each race, assembling data from
Jolpica (race results, standings), OpenMeteo (weather), and optionally
NewsAPI (sentiment) for live predictions.

Two main entry points:
  - build_training_dataset(start_year, end_year) → full historical DataFrame for model training
  - build_race_features(year, round_num, ...)    → feature rows for a single upcoming race
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_fetcher import (
    fetch_all_season_qualifying,
    fetch_all_season_results,
    fetch_all_season_sprint_results,
    fetch_constructor_standings,
    fetch_driver_standings,
    fetch_final_constructor_standings,
    fetch_final_driver_standings,
    fetch_qualifying,
    fetch_race_results,
    fetch_season_schedule,
    fetch_sprint_results,
)
from src.weather_fetcher import fetch_race_weather

# Grid-position → historical win-rate lookup built from full dataset
# Keys: 1..20+, Values: fraction of races won from that grid spot
GRID_WIN_RATES: dict[int, float] = {}


def _compute_grid_win_rates(all_results: pd.DataFrame) -> dict[int, float]:
    """Pre-compute win rate for each starting grid position from historical data."""
    if all_results.empty:
        return {}
    rates = {}
    for grid_pos in range(1, 21):
        sub = all_results[all_results["grid"] == grid_pos]
        if len(sub) == 0:
            rates[grid_pos] = 0.0
        else:
            rates[grid_pos] = sub["winner"].sum() / len(sub)
    return rates


def _circuit_safety_car_rate(circuit_id: str, all_results: pd.DataFrame) -> float:
    """
    Approximate safety car rate by looking at % of races at a circuit where the
    laps leader completed is < expected (gap between leader laps and total race laps).
    This is a rough proxy; a more accurate approach would require lap-by-lap data.
    Falls back to league-wide average if not enough data.
    """
    # We don't have direct safety car flags from Ergast, so return a reasonable default
    # based on circuit type. This could be enhanced with FastF1 lap data.
    KNOWN_HIGH_SC_CIRCUITS = {
        "monaco", "baku", "singapore", "jeddah", "melbourne", "las_vegas"
    }
    KNOWN_LOW_SC_CIRCUITS = {
        "spa", "monza", "silverstone", "suzuka", "bahrain"
    }
    circuit_lower = circuit_id.lower()
    for c in KNOWN_HIGH_SC_CIRCUITS:
        if c in circuit_lower:
            return 0.65
    for c in KNOWN_LOW_SC_CIRCUITS:
        if c in circuit_lower:
            return 0.25
    return 0.40  # league-wide average


def _driver_nationality_country_map() -> dict[str, str]:
    """Map from Ergast nationality string to country name (for home race detection)."""
    return {
        "British": "UK",
        "German": "Germany",
        "Dutch": "Netherlands",
        "Spanish": "Spain",
        "Finnish": "Finland",
        "Australian": "Australia",
        "Brazilian": "Brazil",
        "Mexican": "Mexico",
        "French": "France",
        "Italian": "Italy",
        "Canadian": "Canada",
        "Japanese": "Japan",
        "Chinese": "China",
        "Thai": "Thailand",
        "Monegasque": "Monaco",
        "Danish": "Denmark",
        "Austrian": "Austria",
        "American": "USA",
        "New Zealander": "New Zealand",
        "Argentine": "Argentina",
        "Polish": "Poland",
        "Russian": "Russia",
        "Swiss": "Switzerland",
        "Belgian": "Belgium",
    }


def _circuit_country_map() -> dict[str, str]:
    """Map circuit_id → country name for home-race detection."""
    return {
        "silverstone": "UK",
        "monza": "Italy",
        "spa": "Belgium",
        "monaco": "Monaco",
        "hungaroring": "Hungary",
        "hockenheimring": "Germany",
        "nurburgring": "Germany",
        "bahrain": "Bahrain",
        "albert_park": "Australia",
        "sepang": "Malaysia",
        "shanghai": "China",
        "catalunya": "Spain",
        "istanbul": "Turkey",
        "suzuka": "Japan",
        "interlagos": "Brazil",
        "yas_marina": "UAE",
        "americas": "USA",
        "rodriguez": "Mexico",
        "marina_bay": "Singapore",
        "red_bull_ring": "Austria",
        "baku": "Azerbaijan",
        "sochi": "Russia",
        "portimao": "Portugal",
        "mugello": "Italy",
        "imola": "Italy",
        "ricard": "France",
        "zandvoort": "Netherlands",
        "jeddah": "Saudi Arabia",
        "miami": "USA",
        "las_vegas": "USA",
        "losail": "Qatar",
        "villeneuve": "Canada",
        "shanghai": "China",
    }


def build_training_dataset(
    start_year: int = 2010,
    end_year: int = 2026,
    end_round: Optional[int] = None,
    include_weather: bool = True,
) -> pd.DataFrame:
    """
    Build the full historical training dataset.

    For each race from start_year to end_year:
      - Fetch race results and qualifying
      - Compute driver/team circuit stats and recent form
      - Optionally fetch historical weather from OpenMeteo
      - Assemble one feature row per driver per race

    Returns a DataFrame ready for model training.
    """
    end_label = f"{end_year} R{end_round}" if end_round else str(end_year)
    print(f"Building training dataset for {start_year}–{end_label}...")
    all_results_list = []
    all_qualifying_list = []
    all_sprint_list = []
    schedules = {}

    for year in tqdm(range(start_year, end_year + 1), desc="Fetching seasons"):
        results = fetch_all_season_results(year)
        qualifying = fetch_all_season_qualifying(year)
        schedule = fetch_season_schedule(year)
        try:
            sprints = fetch_all_season_sprint_results(year)
        except Exception:
            sprints = pd.DataFrame()

        if not results.empty:
            all_results_list.append(results)
        if not qualifying.empty:
            all_qualifying_list.append(qualifying)
        if not sprints.empty:
            all_sprint_list.append(sprints)
        schedules[year] = schedule

    if not all_results_list:
        raise ValueError("No race results fetched. Check your internet connection.")

    all_results = pd.concat(all_results_list, ignore_index=True)
    all_qualifying = pd.concat(all_qualifying_list, ignore_index=True) if all_qualifying_list else pd.DataFrame()
    all_sprints = pd.concat(all_sprint_list, ignore_index=True) if all_sprint_list else pd.DataFrame()

    # Pre-compute grid win rates from the full dataset
    global GRID_WIN_RATES
    GRID_WIN_RATES = _compute_grid_win_rates(all_results)

    nationality_map = _driver_nationality_country_map()
    circuit_country_map = _circuit_country_map()

    feature_rows = []
    for year in tqdm(range(start_year, end_year + 1), desc="Engineering features"):
        schedule = schedules.get(year, pd.DataFrame())
        year_results = all_results[all_results["year"] == year]

        for _, race_row in schedule.iterrows():
            round_num = int(race_row["round"])

            # Skip rounds beyond the cutoff in the final year
            if end_round is not None and year == end_year and round_num > end_round:
                continue

            circuit_id = race_row["circuit_id"]
            race_date = race_row["date"]
            lat = race_row["lat"]
            lon = race_row["lon"]
            country = race_row["country"]
            total_rounds = len(schedule)

            # Historical results before this race (for circuit stats + recent form)
            prior_results = all_results[
                (all_results["year"] < year) |
                ((all_results["year"] == year) & (all_results["round"] < round_num))
            ]
            circuit_history = prior_results[prior_results["circuit_id"] == circuit_id]

            # Get results for this specific race
            this_race = all_results[
                (all_results["year"] == year) & (all_results["round"] == round_num)
            ]
            if this_race.empty:
                continue

            # Get qualifying for this race
            if not all_qualifying.empty:
                this_qual = all_qualifying[
                    (all_qualifying["year"] == year) & (all_qualifying["round"] == round_num)
                ] if "year" in all_qualifying.columns else pd.DataFrame()
            else:
                this_qual = pd.DataFrame()

            # Sprint weekend detection and same-weekend sprint results
            if not all_sprints.empty:
                this_sprint = all_sprints[
                    (all_sprints["year"] == year) & (all_sprints["round"] == round_num)
                ]
                is_sprint_weekend = 1 if not this_sprint.empty else 0
            else:
                this_sprint = pd.DataFrame()
                is_sprint_weekend = 0

            # Get standings from previous round (or use defaults for round 1)
            if round_num > 1:
                try:
                    driver_standings = fetch_driver_standings(year, round_num - 1)
                    constructor_standings = fetch_constructor_standings(year, round_num - 1)
                except Exception:
                    driver_standings = pd.DataFrame()
                    constructor_standings = pd.DataFrame()
            else:
                driver_standings = pd.DataFrame()
                constructor_standings = pd.DataFrame()

            # Weather for race day
            weather = {}
            if include_weather and lat and lon and race_date:
                try:
                    weather = fetch_race_weather(lat, lon, race_date)
                except Exception:
                    weather = {}

            circuit_sc_rate = _circuit_safety_car_rate(circuit_id, prior_results)

            for _, result in this_race.iterrows():
                driver_id = result["driver_id"]
                constructor_id = result["constructor_id"]
                grid_pos = result["grid"]
                driver_nationality = result.get("driver_nationality", "")

                # Circuit stats for this driver
                driver_circuit = circuit_history[circuit_history["driver_id"] == driver_id]
                circuit_starts = len(driver_circuit)
                circuit_wins = driver_circuit["winner"].sum()
                circuit_podiums = (driver_circuit["position"] <= 3).sum() if "position" in driver_circuit.columns else 0
                circuit_win_rate = circuit_wins / circuit_starts if circuit_starts > 0 else 0.0
                circuit_podium_rate = circuit_podiums / circuit_starts if circuit_starts > 0 else 0.0

                # Recent form: last 5 races before this one
                driver_prior = prior_results[
                    (prior_results["driver_id"] == driver_id) &
                    (prior_results["year"] >= year - 2)
                ].tail(5)
                recent_points_5 = driver_prior["points"].sum() if not driver_prior.empty else 0.0
                recent_wins_5 = driver_prior["winner"].sum() if not driver_prior.empty else 0

                # Standings
                drv_standing_row = (
                    driver_standings[driver_standings["driver_id"] == driver_id]
                    if not driver_standings.empty else pd.DataFrame()
                )
                driver_standings_pos = int(drv_standing_row["driver_standings_pos"].values[0]) if not drv_standing_row.empty else 10
                driver_season_wins = int(drv_standing_row["driver_wins"].values[0]) if not drv_standing_row.empty else 0

                con_standing_row = (
                    constructor_standings[constructor_standings["constructor_id"] == constructor_id]
                    if not constructor_standings.empty else pd.DataFrame()
                )
                constructor_standings_pos = int(con_standing_row["constructor_standings_pos"].values[0]) if not con_standing_row.empty else 5

                # Grid position win rate (historical)
                grid_pos_win_rate = GRID_WIN_RATES.get(min(grid_pos, 20), 0.0) if grid_pos > 0 else 0.0

                # Grid penalty detection: compare qualifying_position vs grid
                qual_pos = grid_pos  # default: assume no penalty
                if not this_qual.empty and "driver_id" in this_qual.columns:
                    qual_row = this_qual[this_qual["driver_id"] == driver_id]
                    if not qual_row.empty:
                        qual_pos = int(qual_row["qualifying_position"].values[0])

                has_grid_penalty = int(grid_pos > qual_pos + 1) if qual_pos > 0 else 0
                grid_penalty_positions = max(0, grid_pos - qual_pos) if qual_pos > 0 else 0

                # Home race check
                driver_country = nationality_map.get(driver_nationality, "")
                circuit_country = circuit_country_map.get(circuit_id, country)
                is_home_race = int(driver_country.lower() == circuit_country.lower() and driver_country != "")

                # Season progress
                season_round_pct = round_num / max(total_rounds, 1)

                # Sprint features: same-weekend sprint result (Saturday before Sunday race)
                # sprint_pos_score: inverted position so higher = better (P1→22, P22→1, no sprint→0)
                sprint_row = this_sprint[this_sprint["driver_id"] == driver_id] if not this_sprint.empty else pd.DataFrame()
                if not sprint_row.empty:
                    pos_val = sprint_row["sprint_position"].iloc[0]
                    raw_pos = int(pos_val) if pd.notna(pos_val) else 0
                    sprint_pos_score = (23 - raw_pos) if raw_pos > 0 else 0
                    sprint_pts = float(sprint_row["sprint_points"].iloc[0])
                else:
                    sprint_pos_score = 0
                    sprint_pts = 0.0

                # Recent sprint form: sum of last 3 sprint race points before this race
                if not all_sprints.empty:
                    prior_sprints = all_sprints[
                        (all_sprints["year"] < year) |
                        ((all_sprints["year"] == year) & (all_sprints["round"] < round_num))
                    ]
                    drv_prior_sprints = prior_sprints[prior_sprints["driver_id"] == driver_id].tail(3)
                    recent_sprint_pts_3 = float(drv_prior_sprints["sprint_points"].sum()) if not drv_prior_sprints.empty else 0.0
                else:
                    recent_sprint_pts_3 = 0.0

                feature_rows.append({
                    # Identifiers (not features)
                    "year": year,
                    "round": round_num,
                    "circuit_id": circuit_id,
                    "race_date": race_date,
                    "driver_id": driver_id,
                    "constructor_id": constructor_id,
                    # Features
                    "grid_position": grid_pos,
                    "grid_pos_win_rate": round(grid_pos_win_rate, 4),
                    "driver_circuit_win_rate": round(circuit_win_rate, 4),
                    "driver_circuit_podium_rate": round(circuit_podium_rate, 4),
                    "driver_circuit_starts": circuit_starts,
                    "driver_recent_points_5": recent_points_5,
                    "driver_recent_wins_5": int(recent_wins_5),
                    "driver_standings_pos": driver_standings_pos,
                    "driver_season_wins": driver_season_wins,
                    "constructor_standings_pos": constructor_standings_pos,
                    "is_home_race": is_home_race,
                    "has_grid_penalty": has_grid_penalty,
                    "grid_penalty_positions": grid_penalty_positions,
                    "circuit_safety_car_rate": circuit_sc_rate,
                    "season_round_pct": round(season_round_pct, 3),
                    # Weather (defaults to 0/neutral if unavailable)
                    "rain_mm": weather.get("rain_mm", 0.0),
                    "temp_celsius": weather.get("temp_max_celsius", 20.0),
                    "wind_speed_kmh": weather.get("wind_speed_kmh", 10.0),
                    "is_wet_race": int(weather.get("is_wet", False)),
                    # Sprint format features
                    "is_sprint_weekend": is_sprint_weekend,
                    "sprint_pos_score": sprint_pos_score,
                    "sprint_points": sprint_pts,
                    "driver_recent_sprint_pts_3": round(recent_sprint_pts_3, 1),
                    # Sentiment (live-only, default 0.0 for training)
                    "driver_news_sentiment": 0.0,
                    "team_news_sentiment": 0.0,
                    "team_upgrade_flag": 0,
                    # Target
                    "winner": result["winner"],
                })

    df = pd.DataFrame(feature_rows)
    # Filter out rows with missing grid position (pit lane starts, DNS, etc.)
    df = df[df["grid_position"] > 0].reset_index(drop=True)
    print(f"Training dataset: {len(df)} rows, {df['winner'].sum()} winners")
    return df


def build_race_features(
    year: int,
    round_num: int,
    use_news: bool = True,
    all_history: Optional[pd.DataFrame] = None,
    known_drivers: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build feature rows for all drivers in an upcoming or recent race.

    If qualifying results are available (via Jolpica or FastF1), they are used.
    If not, historical average grid positions are used as a proxy.

    Args:
        year: Season year
        round_num: Race round number
        use_news: Whether to fetch live news sentiment (requires NEWSAPI_KEY)
        all_history: Pre-loaded historical results DataFrame (to avoid re-fetching).
                     If None, fetches history for years 2010 to year-1.
        known_drivers: Optional dict mapping driver_id → constructor_id. When provided,
                       this is used as the authoritative driver list (e.g. for pre-season
                       predictions where the full grid is known but no API data exists yet).
                       Qualifying grid positions are still applied on top if available.

    Returns:
        DataFrame with one row per driver, all feature columns, no `winner` column.
    """
    from src.data_fetcher import fetch_qualifying_fastf1

    schedule = fetch_season_schedule(year)
    if schedule.empty:
        raise ValueError(f"No schedule found for {year}")

    race_info = schedule[schedule["round"] == round_num]
    if race_info.empty:
        raise ValueError(f"Round {round_num} not found in {year} schedule")

    race_info = race_info.iloc[0]
    circuit_id = race_info["circuit_id"]
    race_date = race_info["date"]
    lat = race_info["lat"]
    lon = race_info["lon"]
    country = race_info["country"]
    total_rounds = len(schedule)

    # Try to get qualifying results
    qual_df = fetch_qualifying_fastf1(year, round_num)
    if qual_df is None or qual_df.empty:
        try:
            qual_df = fetch_qualifying(year, round_num)
        except Exception:
            qual_df = pd.DataFrame()

    # Get actual race results if available (for backtesting)
    try:
        race_results = fetch_race_results(year, round_num)
    except Exception:
        race_results = pd.DataFrame()

    # Get sprint results if this is a sprint weekend (sprint runs Saturday, main race Sunday)
    try:
        sprint_df = fetch_sprint_results(year, round_num)
    except Exception:
        sprint_df = pd.DataFrame()
    is_sprint_weekend = 1 if not sprint_df.empty else 0

    # Build sprint history for recent form (covers last few seasons of sprint weekends)
    sprint_hist_list = []
    for hist_year in range(max(2021, year - 3), year + 1):
        try:
            ys = fetch_all_season_sprint_results(hist_year)
            if not ys.empty:
                if hist_year == year:
                    ys = ys[ys["round"] < round_num]
                sprint_hist_list.append(ys)
        except Exception:
            pass
    all_sprint_history = pd.concat(sprint_hist_list, ignore_index=True) if sprint_hist_list else pd.DataFrame()

    # Load historical data for context
    if all_history is None:
        print("Loading historical data for feature context...")
        hist_list = []
        for hist_year in range(2010, year):
            try:
                df = fetch_all_season_results(hist_year)
                hist_list.append(df)
            except Exception:
                continue
        if year == year and round_num > 1:
            try:
                curr_season = fetch_all_season_results(year)
                curr_before = curr_season[curr_season["round"] < round_num]
                hist_list.append(curr_before)
            except Exception:
                pass
        all_history = pd.concat(hist_list, ignore_index=True) if hist_list else pd.DataFrame()

    global GRID_WIN_RATES
    if not GRID_WIN_RATES and not all_history.empty:
        GRID_WIN_RATES = _compute_grid_win_rates(all_history)

    circuit_history = all_history[all_history["circuit_id"] == circuit_id] if not all_history.empty else pd.DataFrame()

    # Get standings from the most recent completed round before this race.
    # Walk backwards to handle cancelled rounds (e.g. if Round 5 was cancelled,
    # try Round 4, then Round 3, etc.).
    driver_standings = pd.DataFrame()
    constructor_standings = pd.DataFrame()
    for prev_round in range(round_num - 1, 0, -1):
        try:
            ds = fetch_driver_standings(year, prev_round)
            cs = fetch_constructor_standings(year, prev_round)
            if not ds.empty:
                driver_standings = ds
                constructor_standings = cs
                break
        except Exception:
            continue
    if driver_standings.empty:
        try:
            driver_standings = fetch_final_driver_standings(year - 1)
            constructor_standings = fetch_final_constructor_standings(year - 1)
            if not driver_standings.empty:
                print(f"Note: No {year} standings yet — using {year - 1} final standings as context.")
        except Exception:
            pass

    # Weather
    weather = {}
    if lat and lon and race_date:
        try:
            weather = fetch_race_weather(lat, lon, race_date)
        except Exception:
            weather = {}

    nationality_map = _driver_nationality_country_map()
    circuit_country_map = _circuit_country_map()
    circuit_sc_rate = _circuit_safety_car_rate(circuit_id, all_history)

    # Pre-build a constructor_lookup from history so every fallback branch can use it
    constructor_lookup: dict[str, str] = {}
    if not all_history.empty and "constructor_id" in all_history.columns:
        latest_entries = (
            all_history.sort_values(["year", "round"])
            .drop_duplicates(subset="driver_id", keep="last")
            [["driver_id", "constructor_id"]]
        )
        constructor_lookup = dict(zip(latest_entries["driver_id"], latest_entries["constructor_id"]))

    # Determine the driver list: from qualifying > race results > standings > prev-season proxy
    # If known_drivers is provided, it overrides all fallbacks for the driver list and teams.
    drivers = []
    if known_drivers:
        # Authoritative 2026 (or current-season) grid override.
        # Grid positions are filled from qualifying/results if they exist.
        qual_pos_map: dict[str, int] = {}
        if not qual_df.empty and "driver_id" in qual_df.columns:
            for _, row in qual_df.iterrows():
                qual_pos_map[row["driver_id"]] = int(row.get("qualifying_position", 0))
        elif not race_results.empty:
            for _, row in race_results.iterrows():
                qual_pos_map[row["driver_id"]] = int(row.get("grid", 0))

        for drv, constructor_id in known_drivers.items():
            grid_pos = qual_pos_map.get(drv, 0)
            drivers.append({
                "driver_id": drv,
                "constructor_id": constructor_id,
                "grid_position": grid_pos,
                "qualifying_position": grid_pos,
            })
        if not qual_pos_map:
            print(
                f"\nNote: No {year} Round {round_num} qualifying or results found yet.\n"
                f"Grid positions are unknown (pre-qualifying prediction).\n"
            )
    elif not qual_df.empty and "driver_id" in qual_df.columns:
        for _, row in qual_df.iterrows():
            drv = row["driver_id"]
            drivers.append({
                "driver_id": drv,
                "constructor_id": row.get("constructor_id", "") or constructor_lookup.get(drv, ""),
                "grid_position": int(row.get("qualifying_position", 0)),
                "qualifying_position": int(row.get("qualifying_position", 0)),
            })
    elif not race_results.empty:
        for _, row in race_results.iterrows():
            drv = row["driver_id"]
            drivers.append({
                "driver_id": drv,
                "constructor_id": row.get("constructor_id", "") or constructor_lookup.get(drv, ""),
                "grid_position": int(row.get("grid", 0)),
                "qualifying_position": int(row.get("grid", 0)),
            })
    elif not driver_standings.empty:
        # Fall back to standings (may be current year or prev-season proxy)
        for _, row in driver_standings.iterrows():
            drv = row["driver_id"]
            drivers.append({
                "driver_id": drv,
                "constructor_id": constructor_lookup.get(drv, ""),
                "grid_position": 0,
                "qualifying_position": 0,
            })

    # Fallback 4: previous season's final standings as driver list proxy
    # Triggered when it's the very first round of a new season and no data exists yet.
    if not drivers:
        try:
            prev_standings = fetch_final_driver_standings(year - 1)
            if not prev_standings.empty:
                print(
                    f"\nNote: No {year} Round {round_num} qualifying or results found yet.\n"
                    f"Using {year - 1} final championship standings as driver list proxy.\n"
                    f"Grid positions are unknown (pre-qualifying prediction).\n"
                )
                for _, row in prev_standings.iterrows():
                    drv = row["driver_id"]
                    drivers.append({
                        "driver_id": drv,
                        "constructor_id": constructor_lookup.get(drv, ""),
                        "grid_position": 0,
                        "qualifying_position": 0,
                    })
        except Exception:
            pass

    if not drivers:
        raise ValueError(
            f"No driver data found for {year} round {round_num}. "
            f"Check that the season schedule exists and the API is reachable."
        )

    # Fetch news sentiment if requested
    news_driver_sentiments: dict[str, float] = {}
    news_team_sentiments: dict[str, float] = {}
    upgrade_flags: dict[str, bool] = {}

    if use_news:
        print("Fetching news sentiment...")
        from src.news_fetcher import fetch_driver_sentiment, fetch_team_sentiment, fetch_upgrade_flag
        seen_teams: set[str] = set()
        for d in drivers:
            driver_id = d["driver_id"]
            driver_name = driver_id.replace("_", " ").title()
            news_driver_sentiments[driver_id] = fetch_driver_sentiment(driver_name)

            team_id = d["constructor_id"]
            if team_id and team_id not in seen_teams:
                team_display = team_id.replace("_", " ").title()
                news_team_sentiments[team_id] = fetch_team_sentiment(team_display)
                upgrade_flags[team_id] = fetch_upgrade_flag(team_display)
                seen_teams.add(team_id)

    feature_rows = []
    for d in drivers:
        driver_id = d["driver_id"]
        constructor_id = d["constructor_id"]
        grid_pos = d["grid_position"]
        qual_pos = d["qualifying_position"]

        # Get driver nationality from history
        drv_hist = all_history[all_history["driver_id"] == driver_id] if not all_history.empty else pd.DataFrame()
        driver_nationality = drv_hist["driver_nationality"].iloc[-1] if not drv_hist.empty and "driver_nationality" in drv_hist.columns else ""

        # Circuit stats
        driver_circuit = circuit_history[circuit_history["driver_id"] == driver_id] if not circuit_history.empty else pd.DataFrame()
        circuit_starts = len(driver_circuit)
        circuit_wins = int(driver_circuit["winner"].sum()) if not driver_circuit.empty else 0
        circuit_podiums = int((driver_circuit["position"] <= 3).sum()) if not driver_circuit.empty and "position" in driver_circuit.columns else 0
        circuit_win_rate = circuit_wins / circuit_starts if circuit_starts > 0 else 0.0
        circuit_podium_rate = circuit_podiums / circuit_starts if circuit_starts > 0 else 0.0

        # Recent form
        if not all_history.empty:
            driver_recent = all_history[all_history["driver_id"] == driver_id].tail(5)
            recent_points_5 = float(driver_recent["points"].sum())
            recent_wins_5 = int(driver_recent["winner"].sum())
        else:
            recent_points_5, recent_wins_5 = 0.0, 0

        # Standings
        drv_standing_row = (
            driver_standings[driver_standings["driver_id"] == driver_id]
            if not driver_standings.empty else pd.DataFrame()
        )
        driver_standings_pos = int(drv_standing_row["driver_standings_pos"].values[0]) if not drv_standing_row.empty else 10
        driver_season_wins = int(drv_standing_row["driver_wins"].values[0]) if not drv_standing_row.empty else 0

        con_standing_row = (
            constructor_standings[constructor_standings["constructor_id"] == constructor_id]
            if not constructor_standings.empty else pd.DataFrame()
        )
        constructor_standings_pos = int(con_standing_row["constructor_standings_pos"].values[0]) if not con_standing_row.empty else 5

        grid_pos_win_rate = GRID_WIN_RATES.get(min(max(grid_pos, 1), 20), 0.0) if grid_pos > 0 else 0.0
        has_grid_penalty = int(grid_pos > qual_pos + 1) if qual_pos > 0 and grid_pos > 0 else 0
        grid_penalty_positions = max(0, grid_pos - qual_pos) if qual_pos > 0 and grid_pos > 0 else 0

        driver_country = nationality_map.get(driver_nationality, "")
        circuit_country_name = circuit_country_map.get(circuit_id, country)
        is_home_race = int(driver_country.lower() == circuit_country_name.lower() and driver_country != "")

        season_round_pct = round_num / max(total_rounds, 1)

        # Sprint features for this race
        # sprint_pos_score: inverted position so higher = better (P1→22, P22→1, no sprint→0)
        sprint_row = sprint_df[sprint_df["driver_id"] == driver_id] if not sprint_df.empty else pd.DataFrame()
        if not sprint_row.empty:
            pos_val = sprint_row["sprint_position"].iloc[0]
            raw_pos = int(pos_val) if pd.notna(pos_val) else 0
            sprint_pos_score = (23 - raw_pos) if raw_pos > 0 else 0
            sprint_pts = float(sprint_row["sprint_points"].iloc[0])
        else:
            sprint_pos_score = 0
            sprint_pts = 0.0

        if not all_sprint_history.empty:
            drv_sprint_hist = all_sprint_history[all_sprint_history["driver_id"] == driver_id].tail(3)
            recent_sprint_pts_3 = float(drv_sprint_hist["sprint_points"].sum()) if not drv_sprint_hist.empty else 0.0
        else:
            recent_sprint_pts_3 = 0.0

        feature_rows.append({
            "year": year,
            "round": round_num,
            "circuit_id": circuit_id,
            "race_date": race_date,
            "driver_id": driver_id,
            "constructor_id": constructor_id,
            "grid_position": grid_pos,
            "grid_pos_win_rate": round(grid_pos_win_rate, 4),
            "driver_circuit_win_rate": round(circuit_win_rate, 4),
            "driver_circuit_podium_rate": round(circuit_podium_rate, 4),
            "driver_circuit_starts": circuit_starts,
            "driver_recent_points_5": recent_points_5,
            "driver_recent_wins_5": recent_wins_5,
            "driver_standings_pos": driver_standings_pos,
            "driver_season_wins": driver_season_wins,
            "constructor_standings_pos": constructor_standings_pos,
            "is_home_race": is_home_race,
            "has_grid_penalty": has_grid_penalty,
            "grid_penalty_positions": grid_penalty_positions,
            "circuit_safety_car_rate": circuit_sc_rate,
            "season_round_pct": round(season_round_pct, 3),
            "rain_mm": weather.get("rain_mm", 0.0),
            "temp_celsius": weather.get("temp_max_celsius", 20.0),
            "wind_speed_kmh": weather.get("wind_speed_kmh", 10.0),
            "is_wet_race": int(weather.get("is_wet", False)),
            "is_sprint_weekend": is_sprint_weekend,
            "sprint_pos_score": sprint_pos_score,
            "sprint_points": sprint_pts,
            "driver_recent_sprint_pts_3": round(recent_sprint_pts_3, 1),
            "driver_news_sentiment": news_driver_sentiments.get(driver_id, 0.0),
            "team_news_sentiment": news_team_sentiments.get(constructor_id, 0.0),
            "team_upgrade_flag": int(upgrade_flags.get(constructor_id, False)),
        })

    return pd.DataFrame(feature_rows)


FEATURE_COLS = [
    "grid_position",
    "grid_pos_win_rate",
    "driver_circuit_win_rate",
    "driver_circuit_podium_rate",
    "driver_circuit_starts",
    "driver_recent_points_5",
    "driver_recent_wins_5",
    "driver_standings_pos",
    "driver_season_wins",
    "constructor_standings_pos",
    "is_home_race",
    "has_grid_penalty",
    "grid_penalty_positions",
    "circuit_safety_car_rate",
    "season_round_pct",
    "rain_mm",
    "temp_celsius",
    "wind_speed_kmh",
    "is_wet_race",
    # Sprint format features (0 for non-sprint weekends, populated from Saturday sprint data)
    "is_sprint_weekend",
    "sprint_pos_score",
    "sprint_points",
    "driver_recent_sprint_pts_3",
    # NOTE: driver_news_sentiment / team_news_sentiment / team_upgrade_flag are intentionally
    # excluded from FEATURE_COLS.  Historical news data is unavailable for training (NewsAPI
    # only keeps ~30 days), so these columns are always 0.0 in the training dataset and the
    # models cannot learn meaningful weights from them.  They are applied as a post-hoc
    # multiplicative adjustment in ensemble.py instead.
]
