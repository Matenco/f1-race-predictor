"""
Central configuration for the F1 prediction pipeline.

All paths, feature lists, and tunable parameters live here so individual modules
stay focused on logic.
"""

from datetime import datetime
from pathlib import Path

# --- Project paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CACHE_DIR = PROJECT_ROOT / "cache"
MODEL_DIR = PROJECT_ROOT / "model"
FIGURES_DIR = PROJECT_ROOT / "figures"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

for d in (DATA_DIR, CACHE_DIR, MODEL_DIR, FIGURES_DIR, PREDICTIONS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- File paths ---
HISTORICAL_FILE = DATA_DIR / "f1_historical.csv"
FEATURES_FILE = DATA_DIR / "f1_features.csv"
MODEL_FILE = MODEL_DIR / "xgb_f1_final.json"
MODEL_META_FILE = MODEL_DIR / "model_metadata.json"
OPTUNA_BEST_PARAMS_FILE = MODEL_DIR / "optuna_best_params.json"
PREDICTION_HISTORY_FILE = MODEL_DIR / "prediction_history.json"
VALIDATION_HISTORY_FILE = MODEL_DIR / "validation_history.json"

# --- Data extraction ---
# Default: 2022 (start of current technical era) to current year
SEASONS: list[int] = list(range(2022, datetime.now().year + 1))

# --- Feature engineering ---
WEIGHT_HALF_LIFE_DAYS = 365     # Time-decay half-life for sample weights
NEW_ERA_WEIGHT_BOOST = 1.5      # Multiplier for races in the 2026+ regulation era
NEW_ERA_YEAR = 2026             # First year of the new regulation era

# Feature column list — single source of truth used by training and prediction
FEATURE_COLS: list[str] = [
    # Driver form
    "driver_avg_pos_last3",
    "driver_avg_pos_last5",
    "driver_best_pos_last5",
    "driver_std_pos_last5",
    "driver_finish_rate_last10",
    "driver_pos_trend",
    # Team strength
    "team_avg_pos_last3",
    "team_best_pos_season",
    # Grid
    "GridPosition",
    "grid_vs_driver_form",
    # Qualifying-derived
    "teammate_pos_delta_last5",
    "quali_time_delta_s",
    # Circuit history (parameterised — works for any track)
    "driver_track_avg",
    "driver_track_count",
    "driver_similar_circuits_avg",
    # Context
    "is_rain",
    "season_race_number",
    "regulation_era",
]

TARGET_COL = "Position"
RANDOM_STATE = 42

# --- Training ---
N_OPTUNA_TRIALS = 50            # Set to 0 to skip tuning and load saved best params
MIN_TRAIN_RACES = 40            # Minimum number of past races before a race becomes part of validation
USE_SAVED_OPTUNA_PARAMS = False  # If True and saved params exist, skip new tuning

# Fixed XGBoost params (not tuned)
XGB_FIXED_PARAMS = {
    "objective":    "multi:softprob",
    "random_state": RANDOM_STATE,
    "verbosity":    0,
    "tree_method":  "hist",
}

# Default XGBoost params (used as fallback if Optuna is skipped)
XGB_DEFAULT_PARAMS = {
    "max_depth":        5,
    "learning_rate":    0.08,
    "n_estimators":     300,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
}

# --- Prediction ---
N_SIMULATIONS = 10_000          # Monte Carlo iterations
TOP_N_COMBOS = 10               # Most-likely combinations to display

# --- Scoring rule (used by Hungarian assignment + validation) ---
SCORE_EXACT_HIT = 2             # Points for predicting exact position
SCORE_TOP5_HIT = 1              # Points for getting driver in top 5 but wrong position
