"""
Statistical Model: Logistic Regression trained on historical F1 race data.

Although called "statistical", this model learns optimal feature weights
from data using sklearn's LogisticRegression. It is more interpretable
than XGBoost (coefficients map directly to feature importance) while
still being data-driven.

Coefficients printed after training give insight into which features
drive win probability — e.g., grid_position should have a strong
negative coefficient (lower grid pos = higher win chance).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

from src.feature_engineering import FEATURE_COLS

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
DEFAULT_MODEL_PATH = MODEL_DIR / "logistic_model.pkl"


def train(
    features_df: pd.DataFrame,
    model_path: Path = DEFAULT_MODEL_PATH,
    print_coefficients: bool = True,
) -> Pipeline:
    """
    Train a Logistic Regression model on the provided training data.

    The model is wrapped in a sklearn Pipeline with StandardScaler so
    that the logistic regression receives normalised features (important
    for interpretable coefficients and convergence).

    Args:
        features_df: Training DataFrame with FEATURE_COLS + 'winner' column.
        model_path:  Where to save the trained pipeline (pickle).
        print_coefficients: If True, print feature importances after training.

    Returns:
        Fitted sklearn Pipeline (scaler + logistic regression).
    """
    df = features_df.dropna(subset=FEATURE_COLS + ["winner"]).copy()
    X = df[FEATURE_COLS].astype(float)
    y = df["winner"].astype(int)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            C=0.5,
            random_state=42,
        )),
    ])

    pipeline.fit(X, y)

    if print_coefficients:
        coef = pipeline.named_steps["clf"].coef_[0]
        coef_df = pd.DataFrame({
            "feature": FEATURE_COLS,
            "coefficient": coef,
        }).sort_values("coefficient", ascending=False)
        print("\nLogistic Regression — Feature Coefficients:")
        print(coef_df.to_string(index=False))

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nLogistic model saved to {model_path}")

    return pipeline


def predict(
    features_df: pd.DataFrame,
    model_path: Path = DEFAULT_MODEL_PATH,
    pipeline: Pipeline = None,
) -> pd.DataFrame:
    """
    Predict win probabilities for each driver in features_df.

    Win probabilities are normalised so they sum to 1.0 across all drivers
    in the race (reflects relative likelihood, not absolute probability).

    Args:
        features_df: DataFrame with FEATURE_COLS (no 'winner' required).
        model_path:  Path to saved pickle (used if pipeline is None).
        pipeline:    Pre-loaded pipeline (skips disk load if provided).

    Returns:
        DataFrame with columns: driver_id, constructor_id, grid_position,
        logistic_raw_prob, logistic_prob (normalised).
    """
    if pipeline is None:
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)

    df = features_df.copy()
    X = df[FEATURE_COLS].fillna(0).astype(float)

    raw_probs = pipeline.predict_proba(X)[:, 1]

    # Normalise so all driver probabilities sum to 1
    total = raw_probs.sum()
    norm_probs = raw_probs / total if total > 0 else raw_probs

    result = df[["driver_id", "constructor_id", "grid_position"]].copy()
    result["logistic_raw_prob"] = raw_probs
    result["logistic_prob"] = norm_probs
    return result.reset_index(drop=True)


def evaluate(
    features_df: pd.DataFrame,
    model_path: Path = DEFAULT_MODEL_PATH,
    pipeline: Pipeline = None,
) -> dict:
    """
    Evaluate the logistic model on labelled data.

    Returns dict with accuracy, roc_auc, and top3_accuracy metrics.
    """
    if pipeline is None:
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)

    df = features_df.dropna(subset=FEATURE_COLS + ["winner"]).copy()
    X = df[FEATURE_COLS].astype(float)
    y = df["winner"].astype(int)

    raw_probs = pipeline.predict_proba(X)[:, 1]
    preds = (raw_probs >= 0.5).astype(int)

    # Top-3 accuracy: for each race, did the actual winner appear in top-3 by prob?
    df["prob"] = raw_probs
    top3_correct = 0
    total_races = 0
    for (year, round_num), group in df.groupby(["year", "round"]):
        actual_winner_mask = group["winner"] == 1
        if not actual_winner_mask.any():
            continue
        top3_predicted = group.nlargest(3, "prob")["driver_id"].values
        actual_winner = group[actual_winner_mask]["driver_id"].values[0]
        top3_correct += int(actual_winner in top3_predicted)
        total_races += 1

    metrics = {
        "accuracy": round(accuracy_score(y, preds), 4),
        "roc_auc": round(roc_auc_score(y, raw_probs), 4),
        "top3_accuracy": round(top3_correct / total_races, 4) if total_races > 0 else 0.0,
        "total_races": total_races,
    }
    return metrics
