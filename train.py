#!/usr/bin/env python3
"""
Train the F1 race winner prediction models.

Fetches historical race data (Jolpica Ergast API + OpenMeteo weather),
engineers features, then trains both models:
  1. Logistic Regression  → models/logistic_model.pkl
  2. XGBoost              → models/xgb_model.pkl

Optionally runs leave-one-season-out cross-validation to report held-out metrics.

Usage:
    python train.py
    python train.py --start-year 2012 --end-year 2024
    python train.py --no-weather --no-cv
    python train.py --start-year 2015 --end-year 2024 --cv-years 2022 2023 2024
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DATA_PATH = PROCESSED_DIR / "training_data.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train F1 race winner prediction models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start-year", type=int, default=2010, help="First season to include in training data")
    parser.add_argument("--end-year", type=int, default=2026, help="Last season to include in training data")
    parser.add_argument("--end-round", type=int, default=None,
                        help="Last round in end-year to include (e.g. --end-round 4 for Japanese GP). Default: all completed rounds.")
    parser.add_argument("--no-weather", action="store_true", help="Skip fetching weather data (faster)")
    parser.add_argument("--no-cv", action="store_true", help="Skip cross-validation")
    parser.add_argument("--cv-years", type=int, nargs="+", default=None,
                        help="Specific years to use as CV holdout (default: last 3 years)")
    parser.add_argument("--use-cached", action="store_true",
                        help="Load training data from data/processed/training_data.csv if it exists")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("F1 Race Winner Prediction — Model Training")
    print("=" * 60)

    # --- Step 1: Build or load training dataset ---
    if args.use_cached and TRAINING_DATA_PATH.exists():
        print(f"\nLoading cached training data from {TRAINING_DATA_PATH}...")
        features_df = pd.read_csv(TRAINING_DATA_PATH)
        print(f"Loaded {len(features_df)} rows covering years {features_df['year'].min()}–{features_df['year'].max()}")
    else:
        end_label = f"{args.end_year} R{args.end_round}" if args.end_round else str(args.end_year)
        print(f"\nStep 1/3: Building training dataset ({args.start_year}–{end_label})...")
        from src.feature_engineering import build_training_dataset
        features_df = build_training_dataset(
            start_year=args.start_year,
            end_year=args.end_year,
            end_round=args.end_round,
            include_weather=not args.no_weather,
        )
        features_df.to_csv(TRAINING_DATA_PATH, index=False)
        print(f"Training data saved to {TRAINING_DATA_PATH}")

    print(f"\nDataset summary:")
    print(f"  Rows:   {len(features_df)}")
    print(f"  Races:  {features_df.groupby(['year', 'round']).ngroups}")
    print(f"  Years:  {features_df['year'].min()}–{features_df['year'].max()}")
    print(f"  Winners: {int(features_df['winner'].sum())}")

    # --- Step 2: Train models ---
    print("\nStep 2/3: Training models...")

    from src import statistical_model, ml_model

    print("\n--- Logistic Regression ---")
    logistic_pipeline = statistical_model.train(features_df)

    print("\n--- XGBoost ---")
    xgb_model = ml_model.train(features_df)

    # --- Step 3: Evaluate ---
    if not args.no_cv:
        print("\nStep 3/3: Cross-validation...")
        all_years = sorted(features_df["year"].unique())
        cv_years = args.cv_years or [y for y in all_years if y >= all_years[-3]]
        print(f"CV holdout years: {cv_years}")

        print("\n[Logistic Regression — held-out evaluation]")
        logistic_cv_data = features_df[features_df["year"].isin(cv_years)]
        logistic_metrics = statistical_model.evaluate(logistic_cv_data, pipeline=logistic_pipeline)
        print(f"  Accuracy:        {logistic_metrics['accuracy']:.4f}")
        print(f"  ROC-AUC:         {logistic_metrics['roc_auc']:.4f}")
        print(f"  Top-3 Accuracy:  {logistic_metrics['top3_accuracy']:.4f}")
        print(f"  Races evaluated: {logistic_metrics['total_races']}")

        print("\n[XGBoost — leave-one-season-out CV]")
        xgb_cv_metrics = ml_model.cross_validate_seasons(features_df, holdout_years=cv_years)
        if xgb_cv_metrics:
            print(f"  Mean ROC-AUC:        {xgb_cv_metrics['mean_roc_auc']:.4f}")
            print(f"  Mean Top-3 Accuracy: {xgb_cv_metrics['mean_top3_accuracy']:.4f}")
            print(f"  Mean Winner Rank:    {xgb_cv_metrics['mean_winner_rank']:.2f}")
    else:
        print("\nStep 3/3: Skipped (--no-cv)")

    print("\n" + "=" * 60)
    print("Training complete. Models saved to models/")
    print("  models/logistic_model.pkl")
    print("  models/xgb_model.pkl")
    print("\nNext step: python predict.py --year 2026 --round 5")
    print("=" * 60)


if __name__ == "__main__":
    main()
