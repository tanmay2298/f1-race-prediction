"""
ML Model: XGBoost classifier trained on historical F1 race data.

XGBoost handles non-linear feature interactions well (e.g., the interaction
between grid_position and circuit_safety_car_rate — a pole-sitter is
more vulnerable at street circuits). It uses scale_pos_weight to handle
the strong class imbalance (~5% positive class, 1 winner per ~20 drivers).

Feature importances are printed after training via SHAP-like gain scores.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier

from src.feature_engineering import FEATURE_COLS

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
DEFAULT_MODEL_PATH = MODEL_DIR / "xgb_model.pkl"


def _compute_scale_pos_weight(y: pd.Series) -> float:
    """Compute scale_pos_weight = (negative samples) / (positive samples)."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float(n_neg / n_pos) if n_pos > 0 else 10.0


def train(
    features_df: pd.DataFrame,
    model_path: Path = DEFAULT_MODEL_PATH,
    print_importances: bool = True,
) -> XGBClassifier:
    """
    Train an XGBoost classifier on the provided historical race data.

    Uses scale_pos_weight to compensate for the fact that only ~5% of
    driver-race rows have winner=1.

    Args:
        features_df: Training DataFrame with FEATURE_COLS + 'winner' column.
        model_path:  Where to save the trained model (pickle).
        print_importances: If True, print top feature importances after training.

    Returns:
        Fitted XGBClassifier.
    """
    df = features_df.dropna(subset=FEATURE_COLS + ["winner"]).copy()
    X = df[FEATURE_COLS].astype(float)
    y = df["winner"].astype(int)

    spw = _compute_scale_pos_weight(y)
    print(f"scale_pos_weight = {spw:.2f}  ({int(y.sum())} winners / {len(y)} rows)")

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=1.0,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X, y)

    if print_importances:
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        print("\nXGBoost — Feature Importances (gain):")
        print(imp_df.head(15).to_string(index=False))

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nXGBoost model saved to {model_path}")

    return model


def predict(
    features_df: pd.DataFrame,
    model_path: Path = DEFAULT_MODEL_PATH,
    model: XGBClassifier = None,
) -> pd.DataFrame:
    """
    Predict win probabilities for each driver in features_df.

    Win probabilities are normalised to sum to 1.0 across the race field.

    Args:
        features_df: DataFrame with FEATURE_COLS (no 'winner' required).
        model_path:  Path to saved pickle (used if model is None).
        model:       Pre-loaded XGBClassifier (skips disk load if provided).

    Returns:
        DataFrame with columns: driver_id, constructor_id, grid_position,
        xgb_raw_prob, xgb_prob (normalised).
    """
    if model is None:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    df = features_df.copy()
    X = df[FEATURE_COLS].fillna(0).astype(float)

    raw_probs = model.predict_proba(X)[:, 1]

    total = raw_probs.sum()
    norm_probs = raw_probs / total if total > 0 else raw_probs

    result = df[["driver_id", "constructor_id", "grid_position"]].copy()
    result["xgb_raw_prob"] = raw_probs
    result["xgb_prob"] = norm_probs
    return result.reset_index(drop=True)


def evaluate(
    features_df: pd.DataFrame,
    model_path: Path = DEFAULT_MODEL_PATH,
    model: XGBClassifier = None,
    holdout_years: list[int] = None,
) -> dict:
    """
    Evaluate the XGBoost model on labelled data.

    If holdout_years is provided, evaluation is restricted to those years
    (leave-one-season-out style). Otherwise evaluates on full dataset.

    Returns dict with accuracy, roc_auc, top3_accuracy, and winner_rank_avg.
    """
    if model is None:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    df = features_df.dropna(subset=FEATURE_COLS + ["winner"]).copy()
    if holdout_years:
        df = df[df["year"].isin(holdout_years)]

    if df.empty:
        return {"error": "No data for evaluation"}

    X = df[FEATURE_COLS].astype(float)
    y = df["winner"].astype(int)

    raw_probs = model.predict_proba(X)[:, 1]
    preds = (raw_probs >= 0.5).astype(int)

    df = df.copy()
    df["prob"] = raw_probs

    # Top-3 accuracy: did actual winner appear in model's top-3 predictions?
    top3_correct = 0
    total_races = 0
    rank_sum = 0
    for (year, round_num), group in df.groupby(["year", "round"]):
        actual_winner_mask = group["winner"] == 1
        if not actual_winner_mask.any():
            continue
        ranked = group.sort_values("prob", ascending=False).reset_index(drop=True)
        top3_predicted = ranked.head(3)["driver_id"].values
        actual_winner = group[actual_winner_mask]["driver_id"].values[0]
        top3_correct += int(actual_winner in top3_predicted)
        rank = ranked[ranked["driver_id"] == actual_winner].index[0] + 1 if actual_winner in ranked["driver_id"].values else len(ranked)
        rank_sum += rank
        total_races += 1

    metrics = {
        "accuracy": round(accuracy_score(y, preds), 4),
        "roc_auc": round(roc_auc_score(y, raw_probs), 4),
        "top3_accuracy": round(top3_correct / total_races, 4) if total_races > 0 else 0.0,
        "winner_avg_predicted_rank": round(rank_sum / total_races, 2) if total_races > 0 else 0.0,
        "total_races": total_races,
    }
    return metrics


def cross_validate_seasons(
    features_df: pd.DataFrame,
    holdout_years: list[int] = None,
) -> dict:
    """
    Leave-one-season-out cross-validation.

    For each holdout year: train on all other years, evaluate on holdout.
    Returns averaged metrics across all holdout years.
    """
    if holdout_years is None:
        all_years = sorted(features_df["year"].unique())
        holdout_years = [y for y in all_years if y >= all_years[-4]]

    all_metrics = []
    for holdout_year in holdout_years:
        train_df = features_df[features_df["year"] != holdout_year]
        test_df = features_df[features_df["year"] == holdout_year]

        if train_df.empty or test_df.empty:
            continue

        print(f"CV fold: train on all except {holdout_year}, test on {holdout_year}")
        model = train(train_df, model_path=MODEL_DIR / f"xgb_cv_{holdout_year}.pkl", print_importances=False)
        metrics = evaluate(test_df, model=model)
        metrics["holdout_year"] = holdout_year
        all_metrics.append(metrics)
        print(f"  {holdout_year}: ROC-AUC={metrics['roc_auc']}, Top-3={metrics['top3_accuracy']}")

    if not all_metrics:
        return {}

    avg = {
        "mean_roc_auc": round(np.mean([m["roc_auc"] for m in all_metrics]), 4),
        "mean_top3_accuracy": round(np.mean([m["top3_accuracy"] for m in all_metrics]), 4),
        "mean_winner_rank": round(np.mean([m["winner_avg_predicted_rank"] for m in all_metrics]), 2),
        "folds": all_metrics,
    }
    return avg
