<img src="assets/banner.svg" alt="F1 Race Winner Prediction" width="100%"/>

> **Predict the winner of a Formula 1 Grand Prix** using a Logistic Regression + XGBoost ensemble trained on historical race data from 2010 to 2026.

---

## Technical Design

The full approach — feature rationale, model parameters, ensemble logic, data sources, limitations, and future work — is documented in the PDF:

**[📄 F1_Prediction_Technical_Doc.pdf](F1_Prediction_Technical_Doc.pdf)**

---

## How It Works

```
Qualifying results (FastF1 / Jolpica)
       +
Historical race data (2010 → present)          ──► Feature vector (23 features)
       +                                               │
Weather forecast (Open-Meteo)                         ├──► Logistic Regression ──┐
       +                                               │                          ├──► 50/50 ensemble
Live news sentiment (NewsAPI + VADER)                  └──► XGBoost              ──┘
                                                                                   │
                                                                          News multiplier (±10%)
                                                                                   │
                                                                       Ranked win-probability list
```

---

## Quick Start

```bash
# 1. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Train (fetches & caches all data automatically)
python train.py --start-year 2010 --end-year 2026 --end-round 3

# 3. Predict an upcoming race
python predict.py --year 2026 --round 5

# 4. Backtest a past race (skip live news)
python predict.py --year 2025 --round 10 --no-news

# 5. Pre-qualifying prediction (no grid positions yet)
python predict.py --year 2026 --round 6 --no-qualifying
```

> Set `NEWSAPI_KEY` in your environment for live news sentiment (free at [newsapi.org](https://newsapi.org)). The system falls back gracefully to 0.0 sentiment if the key is absent.

---

## Features (23 total)

| Category | Features |
|---|---|
| **Grid** | `grid_position`, `grid_pos_win_rate` |
| **Circuit** | `driver_circuit_win_rate`, `driver_circuit_podium_rate`, `driver_circuit_starts`, `circuit_safety_car_rate` |
| **Recent form** | `driver_recent_points_5`, `driver_recent_wins_5` |
| **Championship** | `driver_standings_pos`, `driver_season_wins`, `constructor_standings_pos` |
| **Race context** | `is_home_race`, `has_grid_penalty`, `grid_penalty_positions`, `season_round_pct` |
| **Weather** | `rain_mm`, `temp_celsius`, `wind_speed_kmh`, `is_wet_race` |
| **Sprint** | `is_sprint_weekend`, `sprint_pos_score`, `sprint_points`, `driver_recent_sprint_pts_3` |
| **Live only** | `driver_news_sentiment`, `team_news_sentiment`, `team_upgrade_flag` *(post-hoc multiplier, not trained)* |

---

## Data Sources

| Source | Provides | Cost |
|---|---|---|
| [Jolpica API](https://api.jolpi.ca/ergast/f1/) | Race results, qualifying, standings, schedule | Free |
| [FastF1](https://docs.fastf1.dev/) | Current-season qualifying (2018+) | Free |
| [Open-Meteo](https://open-meteo.com/) | Historical & forecast weather | Free |
| [NewsAPI](https://newsapi.org/) | Live news articles for sentiment | Free (500 req/day) |

All API responses are cached locally — re-running after the first fetch is near-instant.

---

## Models

### Logistic Regression (`src/statistical_model.py`)
sklearn `Pipeline` with `StandardScaler` → `LogisticRegression(C=0.5, class_weight='balanced')`. Interpretable via printed feature coefficients. Normalised probabilities summed to 1.0 across the field.

### XGBoost (`src/ml_model.py`)
`XGBClassifier` with `n_estimators=400`, `max_depth=5`, `learning_rate=0.05`, auto-computed `scale_pos_weight ≈ 19` to handle the ~5% positive-class rate. Captures non-linear feature interactions (e.g. grid position × safety car rate).

### Ensemble (`src/ensemble.py`)
`final_prob = 0.5 × logistic + 0.5 × xgb`, renormalised, then adjusted by news/upgrade multipliers (±10% / +5%), clipped to `[0.5×, 2.0×]` and renormalised again.

---

## Project Structure

```
├── train.py                   # CLI: fetch data, build features, train both models
├── predict.py                 # CLI: predict winner for a given year/round
├── requirements.txt
├── src/
│   ├── data_fetcher.py        # Jolpica API + FastF1 wrappers (cached)
│   ├── weather_fetcher.py     # Open-Meteo historical/forecast
│   ├── news_fetcher.py        # NewsAPI + VADER sentiment
│   ├── feature_engineering.py # Builds 23-feature rows per driver/race
│   ├── statistical_model.py   # Logistic Regression pipeline
│   ├── ml_model.py            # XGBoost classifier
│   └── ensemble.py            # Combine + news adjustments
├── models/                    # Saved model pickles (gitignored)
├── data/                      # Raw cache + processed CSV (gitignored)
└── assets/
    └── banner.svg
```

---

## Known Limitations

- **Regulation eras** — 2026 rules are a major reset; pre-2026 stats are less predictive for the current season
- **Safety car rate** — heuristic per circuit, not real SC frequency from telemetry
- **No tyre strategy** — compound choice and pit windows are unmodelled
- **No DNF/reliability** — mechanical failures in practice are not penalised
- **Uncalibrated probabilities** — model outputs are relative rankings, not calibrated win probabilities

See the [technical document](F1_Prediction_Technical_Doc.pdf) for the full critique and planned improvements.
