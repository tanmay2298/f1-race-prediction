#!/usr/bin/env python3
"""
Predict the F1 race winner for a given race.

Loads both trained models and outputs a ranked probability table
for all drivers in the specified race.

Usage:
    python predict.py --year 2026 --round 4
    python predict.py --year 2024 --round 6 --no-news         # backtest without news
    python predict.py --year 2026 --round 8 --no-qualifying   # pre-qualifying prediction
    python predict.py --year 2025 --round 1                   # full grid shown by default
    python predict.py --year 2025 --round 1 --top 5           # show only top 5 drivers

Environment variables:
    NEWSAPI_KEY   - Required for live news sentiment (optional, skips if not set)
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

MODEL_DIR = Path(__file__).parent / "models"
LOGISTIC_MODEL_PATH = MODEL_DIR / "logistic_model.pkl"
XGB_MODEL_PATH = MODEL_DIR / "xgb_model.pkl"

# Driver ID → full display name mapping
# Includes current 2026 grid + historical drivers for backtesting
DRIVER_NAMES = {
    # 2026 grid
    "norris": "Lando Norris",           # McLaren
    "piastri": "Oscar Piastri",         # McLaren
    "leclerc": "Charles Leclerc",       # Ferrari
    "hamilton": "Lewis Hamilton",       # Ferrari
    "max_verstappen": "Max Verstappen", # Red Bull
    "hadjar": "Isack Hadjar",           # Red Bull
    "russell": "George Russell",        # Mercedes
    "antonelli": "Kimi Antonelli",      # Mercedes
    "alonso": "Fernando Alonso",        # Aston Martin
    "stroll": "Lance Stroll",           # Aston Martin
    "gasly": "Pierre Gasly",            # Alpine
    "colapinto": "Franco Colapinto",    # Alpine
    "sainz": "Carlos Sainz",            # Williams
    "albon": "Alexander Albon",         # Williams
    "ocon": "Esteban Ocon",             # Haas
    "bearman": "Oliver Bearman",        # Haas
    "hulkenberg": "Nico Hulkenberg",    # Audi (formerly Sauber)
    "bortoleto": "Gabriel Bortoleto",   # Audi (formerly Sauber)
    "lawson": "Liam Lawson",            # Racing Bulls
    "arvid_lindblad": "Arvid Lindblad", # Racing Bulls (2026 rookie)
    "bottas": "Valtteri Bottas",        # Cadillac (new team)
    "perez": "Sergio Perez",            # Cadillac (new team)
    # Historical drivers (kept for backtesting)
    "tsunoda": "Yuki Tsunoda",
    "doohan": "Jack Doohan",
    "sargeant": "Logan Sargeant",
    "zhou": "Guanyu Zhou",
    "ricciardo": "Daniel Ricciardo",
    "kevin_magnussen": "Kevin Magnussen",
    "sainz_jr": "Carlos Sainz Jr.",
}

# Hardcoded 2026 driver → constructor_id map used to fill in blank team names
# for drivers with no historical data (rookies, late-season call-ups, new teams).
DRIVER_TEAM_2026 = {
    "norris": "mclaren",
    "piastri": "mclaren",
    "leclerc": "ferrari",
    "hamilton": "ferrari",
    "max_verstappen": "red_bull",
    "hadjar": "red_bull",
    "russell": "mercedes",
    "antonelli": "mercedes",
    "alonso": "aston_martin",
    "stroll": "aston_martin",
    "gasly": "alpine",
    "colapinto": "alpine",
    "sainz": "williams",
    "albon": "williams",
    "ocon": "haas",
    "bearman": "haas",
    "hulkenberg": "audi",
    "bortoleto": "audi",
    "lawson": "rb",
    "arvid_lindblad": "rb",
    "bottas": "cadillac",
    "perez": "cadillac",
}

CONSTRUCTOR_NAMES = {
    # 2026 constructors
    "mclaren": "McLaren",
    "ferrari": "Ferrari",
    "red_bull": "Red Bull",
    "mercedes": "Mercedes",
    "aston_martin": "Aston Martin",
    "alpine": "Alpine",
    "williams": "Williams",
    "haas": "Haas",
    "audi": "Audi",              # rebranded from Sauber/Kick Sauber
    "rb": "Racing Bulls",
    "cadillac": "Cadillac",      # new 2026 team (GM)
    # Historical aliases (kept for backtesting)
    "sauber": "Kick Sauber",
    "alphatauri": "AlphaTauri",
    "alfa": "Alfa Romeo",
    "toro_rosso": "Toro Rosso",
    "force_india": "Force India",
    "racing_point": "Racing Point",
    "renault": "Renault",
    "lotus_f1": "Lotus",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict F1 race winner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--year", type=int, required=True, help="Season year (e.g., 2026)")
    parser.add_argument("--round", type=int, required=True, dest="round_num", help="Race round number (e.g., 4)")
    parser.add_argument("--no-news", action="store_true", help="Skip news sentiment fetching")
    parser.add_argument("--no-qualifying", action="store_true",
                        help="Skip qualifying results — use historical grid position averages")
    parser.add_argument("--top", type=int, default=0, metavar="N",
                        help="Show top N drivers only (default: 0 = show full grid)")
    parser.add_argument("--show-all", action="store_true",
                        help="(Deprecated alias) Same as --top 0: show full grid")
    parser.add_argument("--logistic-weight", type=float, default=0.5,
                        help="Weight for logistic regression model (0-1)")
    return parser.parse_args()


def _check_models():
    missing = []
    if not LOGISTIC_MODEL_PATH.exists():
        missing.append("models/logistic_model.pkl")
    if not XGB_MODEL_PATH.exists():
        missing.append("models/xgb_model.pkl")
    if missing:
        print("ERROR: Trained models not found. Run training first:")
        print("  python train.py")
        print(f"\nMissing: {', '.join(missing)}")
        sys.exit(1)


def _fmt_sentiment(val: float) -> str:
    """Format a VADER sentiment score as a signed 2-decimal string, e.g. '+0.42' or '-0.08'."""
    return f"{val:+.2f}"


def _format_table(predictions: pd.DataFrame, race_name: str, top_n: int = 0, year: int = 0) -> str:
    """Format the ranked predictions table.

    Args:
        top_n: How many drivers to show. 0 (default) = show entire grid.
    """
    n = len(predictions) if top_n <= 0 else min(top_n, len(predictions))
    rows = predictions.head(n)

    # Detect whether live news data is present — if every value is 0.0 (backtest /
    # --no-news run) we skip the sentiment columns to keep the layout compact.
    has_news = (
        predictions["driver_news_sentiment"].abs().sum() > 0
        or predictions["team_news_sentiment"].abs().sum() > 0
        or predictions["team_upgrade_flag"].sum() > 0
    )

    W = 95 if has_news else 70  # table width

    if has_news:
        header = (
            f"{'Rank':<5} {'Driver':<22} {'Team':<18} {'Grid':<6}"
            f" {'Win Prob':>10}  {'Logit':>7}  {'XGB':>7}"
            f"  {'DrvNews':>7}  {'TmNews':>7}  {'U':<1}"
        )
    else:
        header = (
            f"{'Rank':<5} {'Driver':<22} {'Team':<18} {'Grid':<6}"
            f" {'Win Prob':>10}  {'Logit':>7}  {'XGB':>7}"
        )

    lines = [
        "",
        f"{'=' * W}",
        f"  F1 RACE WINNER PREDICTION — {race_name.upper()}",
        f"{'=' * W}",
        header,
        f"{'-' * W}",
    ]

    for _, row in rows.iterrows():
        driver_id = row["driver_id"]
        # For 2026+ predictions, always prefer the known 2026 team mapping so that
        # drivers like Hulkenberg/Bortoleto show "Audi" instead of historical "Kick Sauber".
        if year >= 2026 and driver_id in DRIVER_TEAM_2026:
            constructor_id = DRIVER_TEAM_2026[driver_id]
        else:
            constructor_id = row["constructor_id"] or DRIVER_TEAM_2026.get(driver_id, "")
        driver_name = DRIVER_NAMES.get(driver_id, driver_id.replace("_", " ").title())
        team_name = CONSTRUCTOR_NAMES.get(constructor_id, constructor_id.replace("_", " ").title())
        grid = int(row["grid_position"]) if row["grid_position"] > 0 else "?"
        win_prob = f"{row['final_prob_pct']:.1f}%"
        logit = f"{row['logistic_prob_pct']:.1f}%"
        xgb = f"{row['xgb_prob_pct']:.1f}%"

        base = (
            f"{int(row['rank']):<5} {driver_name:<22} {team_name:<18} {str(grid):<6}"
            f" {win_prob:>10}  {logit:>7}  {xgb:>7}"
        )

        if has_news:
            drv_sent = _fmt_sentiment(float(row.get("driver_news_sentiment", 0.0)))
            tm_sent  = _fmt_sentiment(float(row.get("team_news_sentiment", 0.0)))
            upgrade  = "U" if bool(row.get("team_upgrade_flag", 0)) else ""
            line = f"{base}  {drv_sent:>7}  {tm_sent:>7}  {upgrade:<1}"
        else:
            line = base

        lines.append(line)

    total = len(predictions)
    shown = len(rows)

    if has_news:
        footer_note = "  DrvNews = driver sentiment (–1 to +1)   TmNews = team sentiment   U = upgrade reported"
    else:
        footer_note = "  News sentiment hidden (run without --no-news to show DrvNews / TmNews columns)"

    if shown < total:
        footer_note += f"   (showing {shown}/{total} — use --top {total} or omit --top to see all)"

    lines.extend([
        f"{'-' * W}",
        footer_note,
        f"{'=' * W}",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    _check_models()

    use_news = not args.no_news and bool(os.environ.get("NEWSAPI_KEY"))
    if not args.no_news and not os.environ.get("NEWSAPI_KEY"):
        print("Note: NEWSAPI_KEY not set — skipping news sentiment. Set it for richer predictions.")

    print(f"\nPredicting winner for {args.year} Round {args.round_num}...")

    # Get race name from schedule
    from src.data_fetcher import fetch_season_schedule
    schedule = fetch_season_schedule(args.year)
    race_info = schedule[schedule["round"] == args.round_num]
    race_name = race_info["race_name"].values[0] if not race_info.empty else f"Round {args.round_num}"
    print(f"Race: {race_name}")

    # Build features
    # For 2026+ predictions, pass the known 2026 grid as the authoritative driver list.
    # This ensures rookies (Lindblad) and returning drivers (Bottas, Perez at Cadillac)
    # are included even if they don't appear in the prior season's standings.
    from src.feature_engineering import build_race_features
    known_drivers = DRIVER_TEAM_2026 if args.year >= 2026 else None
    features_df = build_race_features(
        year=args.year,
        round_num=args.round_num,
        use_news=use_news,
        known_drivers=known_drivers,
    )

    if features_df.empty:
        print("ERROR: Could not build features. No driver data found for this race.")
        sys.exit(1)

    print(f"Built features for {len(features_df)} drivers")

    # Run predictions
    from src import statistical_model, ml_model
    from src.ensemble import combine

    logistic_probs = statistical_model.predict(features_df, model_path=LOGISTIC_MODEL_PATH)
    xgb_probs = ml_model.predict(features_df, model_path=XGB_MODEL_PATH)

    xgb_weight = 1.0 - args.logistic_weight
    predictions = combine(
        features_df,
        logistic_probs,
        xgb_probs,
        logistic_weight=args.logistic_weight,
        xgb_weight=xgb_weight,
        apply_news_adjustment=use_news,
    )

    # Print results
    # --show-all is a legacy alias for --top 0 (full grid)
    top_n = 0 if (args.top <= 0 or args.show_all) else args.top
    table = _format_table(predictions, race_name, top_n=top_n, year=args.year)
    print(table)

    # Weather summary
    if not features_df.empty:
        rain = features_df["rain_mm"].iloc[0]
        temp = features_df["temp_celsius"].iloc[0]
        wind = features_df["wind_speed_kmh"].iloc[0]
        wet = bool(features_df["is_wet_race"].iloc[0])
        print(f"Weather: {temp:.0f}°C, Wind {wind:.0f} km/h, Rain {rain:.1f}mm {'[WET RACE]' if wet else ''}")
        print()


if __name__ == "__main__":
    main()
