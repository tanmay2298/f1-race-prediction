"""
Ensemble: Combines Logistic Regression and XGBoost model predictions.

Default weighting is 50/50. Live prediction features (news sentiment,
upgrade flags) are applied as multiplicative adjustments on top of the
combined base probability, then renormalized across the full driver field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.feature_engineering import FEATURE_COLS


def combine(
    features_df: pd.DataFrame,
    logistic_probs: pd.DataFrame,
    xgb_probs: pd.DataFrame,
    logistic_weight: float = 0.5,
    xgb_weight: float = 0.5,
    apply_news_adjustment: bool = True,
    news_sentiment_multiplier: float = 0.10,
    upgrade_multiplier: float = 0.05,
) -> pd.DataFrame:
    """
    Combine logistic and XGBoost win probabilities into a final ranked prediction.

    Args:
        features_df:              Full feature DataFrame (contains sentiment, upgrade flag).
        logistic_probs:           Output of statistical_model.predict() — must have 'logistic_prob'.
        xgb_probs:                Output of ml_model.predict() — must have 'xgb_prob'.
        logistic_weight:          Weight for logistic model (default 0.5).
        xgb_weight:               Weight for XGBoost model (default 0.5).
        apply_news_adjustment:    Whether to apply news sentiment / upgrade multipliers.
        news_sentiment_multiplier: Fraction by which sentiment shifts probability.
                                   E.g., 0.10 means sentiment=+1 → +10% probability boost.
        upgrade_multiplier:       Fraction added when team has confirmed upgrade news.

    Returns:
        DataFrame sorted by final win probability (descending), with columns:
        rank, driver_id, constructor_id, grid_position,
        logistic_prob, xgb_prob, base_prob, adjustment_factor, final_prob (%)
    """
    # Merge on driver_id
    merged = (
        features_df[["driver_id", "constructor_id", "grid_position",
                     "driver_news_sentiment", "team_news_sentiment", "team_upgrade_flag"]]
        .merge(logistic_probs[["driver_id", "logistic_prob"]], on="driver_id", how="left")
        .merge(xgb_probs[["driver_id", "xgb_prob"]], on="driver_id", how="left")
    )

    merged["logistic_prob"] = merged["logistic_prob"].fillna(0.0)
    merged["xgb_prob"] = merged["xgb_prob"].fillna(0.0)

    # Weighted base probability
    merged["base_prob"] = (
        logistic_weight * merged["logistic_prob"] +
        xgb_weight * merged["xgb_prob"]
    )

    # Renormalise base probability
    base_total = merged["base_prob"].sum()
    if base_total > 0:
        merged["base_prob"] = merged["base_prob"] / base_total

    # Apply news/upgrade adjustments (live predictions only)
    if apply_news_adjustment:
        driver_sentiment = merged["driver_news_sentiment"].fillna(0.0)
        team_sentiment = merged["team_news_sentiment"].fillna(0.0)
        upgrade_flag = merged["team_upgrade_flag"].fillna(0).astype(float)

        # Combined sentiment: average of driver and team sentiment
        combined_sentiment = (driver_sentiment + team_sentiment) / 2.0
        adjustment = (
            1.0
            + news_sentiment_multiplier * combined_sentiment
            + upgrade_multiplier * upgrade_flag
        )
        # Clip to avoid negative probabilities
        adjustment = adjustment.clip(lower=0.5, upper=2.0)
        merged["adjustment_factor"] = adjustment
        merged["adjusted_prob"] = merged["base_prob"] * adjustment
    else:
        merged["adjustment_factor"] = 1.0
        merged["adjusted_prob"] = merged["base_prob"]

    # Renormalise final probabilities
    final_total = merged["adjusted_prob"].sum()
    if final_total > 0:
        merged["final_prob"] = merged["adjusted_prob"] / final_total
    else:
        merged["final_prob"] = merged["adjusted_prob"]

    # Sort and rank
    merged = merged.sort_values("final_prob", ascending=False).reset_index(drop=True)
    merged.index = merged.index + 1
    merged.index.name = "rank"
    merged = merged.reset_index()

    # Convert to percentage for display
    merged["final_prob_pct"] = (merged["final_prob"] * 100).round(2)
    merged["logistic_prob_pct"] = (merged["logistic_prob"] * 100).round(2)
    merged["xgb_prob_pct"] = (merged["xgb_prob"] * 100).round(2)

    return merged[[
        "rank",
        "driver_id",
        "constructor_id",
        "grid_position",
        "logistic_prob_pct",
        "xgb_prob_pct",
        "final_prob_pct",
        "driver_news_sentiment",
        "team_news_sentiment",
        "team_upgrade_flag",
        "adjustment_factor",
    ]]


def predict_from_features(
    features_df: pd.DataFrame,
    logistic_model_path: Optional[Path] = None,
    xgb_model_path: Optional[Path] = None,
    apply_news_adjustment: bool = True,
) -> pd.DataFrame:
    """
    Convenience wrapper: loads both models and returns ensemble predictions.

    Args:
        features_df:          Feature DataFrame for one race (from build_race_features).
        logistic_model_path:  Path to logistic model pickle (uses default if None).
        xgb_model_path:       Path to XGBoost model pickle (uses default if None).
        apply_news_adjustment: Whether to apply live news/upgrade multipliers.

    Returns:
        Ranked DataFrame from combine().
    """
    from src import statistical_model, ml_model

    logistic_probs = statistical_model.predict(features_df, model_path=logistic_model_path or statistical_model.DEFAULT_MODEL_PATH)
    xgb_probs = ml_model.predict(features_df, model_path=xgb_model_path or ml_model.DEFAULT_MODEL_PATH)

    return combine(
        features_df,
        logistic_probs,
        xgb_probs,
        apply_news_adjustment=apply_news_adjustment,
    )
