"""
Train the XGBoost model with Optuna-tuned hyperparameters and time-series CV.

Pipeline:
1. Load engineered features
2. Optuna tuning on 2022–2024 (held-out 2025/2026 for validation)
3. Time-series cross-validation over 2025–2026
4. Compare against grid and form baselines
5. Train final model on all available data
6. Save model + metadata + best params to model/

Outputs:
    model/xgb_f1_final.json
    model/model_metadata.json
    model/optuna_best_params.json
    figures/09_model_comparison.png
    figures/10_feature_importance.png
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from scipy.optimize import linear_sum_assignment

from . import config

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# =============================================================================
#  Scoring rule (Hungarian assignment optimises this)
# =============================================================================
def compute_scoring(pred_top5: list, actual_positions: dict) -> int:
    """Score a top-5 prediction.

    +2 for an exact-position hit, +1 for an in-top-5 miss, 0 otherwise.
    Mirrors common F1 prediction-game rules.
    """
    actual_top5_drivers = {
        abbr for abbr, pos in actual_positions.items() if pos <= 5
    }
    score = 0
    for pred_pos, driver in pred_top5:
        actual_pos = actual_positions.get(driver)
        if actual_pos is None:
            continue
        if actual_pos == pred_pos:
            score += config.SCORE_EXACT_HIT
        elif driver in actual_top5_drivers:
            score += config.SCORE_TOP5_HIT
    return score


def hungarian_optimal_assignment(prob_matrix: np.ndarray,
                                 drivers: list) -> list[tuple[int, str]]:
    """Find the assignment of 5 drivers to 5 positions that maximises EV.

    The expected-value matrix is built so that the Hungarian algorithm
    (linear_sum_assignment, O(n³)) finds the global optimum for the scoring
    rule above. Without this, picking the top-5 by P(top5) is a greedy
    suboptimal heuristic.

    EV[i][j] = P(driver i finishes at position j+1) + P(driver i in top 5).
    The first term rewards exact-position hits; the second rewards just
    being in the top 5.
    """
    n_drivers = prob_matrix.shape[0]
    p_top5 = prob_matrix[:, :5].sum(axis=1)

    ev_matrix = np.zeros((n_drivers, 5))
    for j in range(5):
        ev_matrix[:, j] = prob_matrix[:, j] + p_top5

    # linear_sum_assignment minimises, so negate to maximise
    row_ind, col_ind = linear_sum_assignment(-ev_matrix)
    assignments = [(c + 1, drivers[r]) for r, c in zip(row_ind, col_ind, strict=False)]
    return sorted(assignments, key=lambda x: x[0])


def baseline_grid_top5(race_df: pd.DataFrame) -> list:
    """Baseline: predict top 5 = grid positions 1–5."""
    grid_sorted = race_df.dropna(subset=["GridPosition"]).sort_values("GridPosition")
    top5 = grid_sorted.head(5)
    return [(int(r["GridPosition"]), r["Abbreviation"]) for _, r in top5.iterrows()]


def baseline_form_top5(race_df: pd.DataFrame) -> list:
    """Baseline: predict top 5 = drivers with the best last-5-race average."""
    form_sorted = race_df.dropna(subset=["driver_avg_pos_last5"]).sort_values(
        "driver_avg_pos_last5"
    )
    top5 = form_sorted.head(5)
    return [(i + 1, r["Abbreviation"]) for i, (_, r) in enumerate(top5.iterrows())]


# =============================================================================
#  Optuna tuning
# =============================================================================
def _build_optuna_objective(df: pd.DataFrame, num_classes: int,
                            tune_val_races: list[int]):
    """Build the Optuna objective: minimise mean RMSE across tuning races."""
    def _objective(trial: optuna.Trial) -> float:
        params = {
            **config.XGB_FIXED_PARAMS,
            "num_class":        num_classes,
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        rmse_scores = []
        for idx in tune_val_races:
            tr = df[df["race_idx"] < idx]
            te = df[df["race_idx"] == idx]
            if len(te) == 0 or len(tr) < 100:
                continue
            X_tr, y_tr = tr[config.FEATURE_COLS], tr["target_class"]
            X_te = te[config.FEATURE_COLS]
            w_tr = tr["sample_weight"]

            m = xgb.XGBClassifier(**params)
            m.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)

            pred_pos = m.predict_proba(X_te).argmax(axis=1) + 1
            actual = te[config.TARGET_COL].astype(float).values
            rmse_scores.append(np.sqrt(np.mean((pred_pos - actual) ** 2)))
        return float(np.mean(rmse_scores)) if rmse_scores else float("inf")
    return _objective


def run_optuna(df: pd.DataFrame, num_classes: int) -> dict:
    """Run Optuna tuning on 2022–2024 data and return the best params dict.

    Reuses saved best params from disk if config.USE_SAVED_OPTUNA_PARAMS is
    True and a saved file exists — avoids re-tuning on every pipeline run.
    """
    if config.USE_SAVED_OPTUNA_PARAMS and config.OPTUNA_BEST_PARAMS_FILE.exists():
        logger.info("Loading saved Optuna best params from %s", config.OPTUNA_BEST_PARAMS_FILE)
        with open(config.OPTUNA_BEST_PARAMS_FILE) as f:
            return json.load(f)

    if config.N_OPTUNA_TRIALS <= 0:
        logger.info("Optuna skipped (N_OPTUNA_TRIALS=0); using defaults.")
        return dict(config.XGB_DEFAULT_PARAMS)

    logger.info("Running Optuna with %d trials...", config.N_OPTUNA_TRIALS)

    # Tuning races: hold out 2025/2026 entirely
    tune_race_idxs = sorted(df[df["Year"] <= 2024]["race_idx"].unique())
    tune_val_races = tune_race_idxs[config.MIN_TRAIN_RACES:]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
    )
    study.optimize(
        _build_optuna_objective(df, num_classes, tune_val_races),
        n_trials=config.N_OPTUNA_TRIALS,
        show_progress_bar=True,
    )

    logger.info("Best RMSE (tuning 2022–2024): %.4f", study.best_value)
    logger.info("Best params:")
    for k, v in study.best_params.items():
        logger.info("  %s: %s", k, v)

    # Persist for future runs
    with open(config.OPTUNA_BEST_PARAMS_FILE, "w") as f:
        json.dump(study.best_params, f, indent=2)
    logger.info("Saved best params to %s", config.OPTUNA_BEST_PARAMS_FILE)

    return dict(study.best_params)


# =============================================================================
#  Time-series cross-validation
# =============================================================================
def run_validation(df: pd.DataFrame, xgb_params: dict, num_classes: int) -> pd.DataFrame:
    """Walk-forward validation over 2025–2026: train on all earlier races,
    test on the next one, accumulate scores.
    """
    val_races = sorted(df[df["Year"].isin([2025, 2026])]["race_idx"].unique())
    results = []
    for val_race_idx in val_races:
        if val_race_idx < config.MIN_TRAIN_RACES:
            continue
        train_df = df[df["race_idx"] < val_race_idx]
        test_df = df[df["race_idx"] == val_race_idx]
        if len(test_df) == 0 or len(train_df) < 100:
            continue

        X_tr, y_tr = train_df[config.FEATURE_COLS], train_df["target_class"]
        w_tr = train_df["sample_weight"]
        X_te = test_df[config.FEATURE_COLS]

        params = {**config.XGB_FIXED_PARAMS, "num_class": num_classes, **xgb_params}
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)

        prob_matrix = model.predict_proba(X_te)
        drivers = test_df["Abbreviation"].tolist()
        actual_positions = dict(
            zip(test_df["Abbreviation"], test_df[config.TARGET_COL].astype(int), strict=False)
        )

        # XGBoost + Hungarian
        try:
            xgb_top5 = hungarian_optimal_assignment(prob_matrix, drivers)
            xgb_score = compute_scoring(xgb_top5, actual_positions)
        except Exception:
            xgb_top5, xgb_score = [], 0

        grid_score = compute_scoring(baseline_grid_top5(test_df), actual_positions)
        form_score = compute_scoring(baseline_form_top5(test_df), actual_positions)

        race_info = test_df.iloc[0]
        pred_positions = prob_matrix.argmax(axis=1) + 1
        actual_arr = test_df[config.TARGET_COL].astype(float).values
        rmse = float(np.sqrt(np.mean((pred_positions - actual_arr) ** 2)))

        results.append({
            "Year": int(race_info["Year"]),
            "Round": int(race_info["Round"]),
            "GP": race_info["GP"],
            "XGBoost_Hungarian": xgb_score,
            "Baseline_Grid": grid_score,
            "Baseline_Form": form_score,
            "RMSE": rmse,
        })
        logger.info(
            "  [%d R%2d] %-35s | XGB+H: %d/10  Grid: %d/10  Form: %d/10  RMSE: %.2f",
            int(race_info["Year"]), int(race_info["Round"]), str(race_info["GP"])[:35],
            xgb_score, grid_score, form_score, rmse,
        )
    return pd.DataFrame(results)


def plot_validation_results(results_df: pd.DataFrame) -> None:
    """Save a comparison plot: boxplot + rolling-average lines."""
    if len(results_df) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    data_to_plot = [
        results_df["XGBoost_Hungarian"],
        results_df["Baseline_Grid"],
        results_df["Baseline_Form"],
    ]
    bp = ax.boxplot(
        data_to_plot,
        tick_labels=["XGBoost\n+ Hungarian", "Baseline\n(Grid)", "Baseline\n(Form)"],
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], ["#3671C6", "#E8002D", "#FF8000"], strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Points (0–10)")
    ax.set_title("Per-race score distribution", fontsize=12, weight="bold")
    ax.set_ylim(-0.5, 10.5)

    ax = axes[1]
    window = 5
    ax.plot(results_df["XGBoost_Hungarian"].rolling(window, min_periods=1).mean(),
            label="XGBoost + Hungarian", color="#3671C6", linewidth=2)
    ax.plot(results_df["Baseline_Grid"].rolling(window, min_periods=1).mean(),
            label="Baseline (Grid)", color="#E8002D", linewidth=2, linestyle="--")
    ax.plot(results_df["Baseline_Form"].rolling(window, min_periods=1).mean(),
            label="Baseline (Form)", color="#FF8000", linewidth=2, linestyle=":")
    ax.set_xlabel("Race (chronological)")
    ax.set_ylabel(f"{window}-race rolling mean")
    ax.set_title("Score trend over time", fontsize=12, weight="bold")
    ax.legend()
    ax.set_ylim(0, 10)

    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "09_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model: xgb.XGBClassifier) -> None:
    """Save a horizontal bar chart of feature importances."""
    importance = model.feature_importances_
    feat_imp = pd.Series(importance, index=config.FEATURE_COLS).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    feat_imp.plot(kind="barh", ax=ax, color="#3671C6", edgecolor="black")
    ax.set_title("Feature importance — final XGBoost model", fontsize=13, weight="bold")
    ax.set_xlabel("Importance (gain)")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "10_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
#  Main
# =============================================================================
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=" * 70)
    logger.info("MODEL TRAINING")
    logger.info("=" * 70)

    df = pd.read_csv(config.FEATURES_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    df = df.dropna(subset=[config.TARGET_COL]).copy()

    max_pos = int(df[config.TARGET_COL].max())
    df["target_class"] = (df[config.TARGET_COL].astype(int) - 1).clip(0, max_pos - 1)
    num_classes = max_pos

    # Build chronological race index
    df["race_id"] = df["Year"].astype(str) + "_R" + df["Round"].astype(str)
    race_order = (
        df.groupby("race_id")["Date"].first().sort_values().reset_index()
    )
    race_order["race_idx"] = range(len(race_order))
    df = df.merge(race_order[["race_id", "race_idx"]], on="race_id", how="left")

    logger.info("Rows: %d, classes: %d, features: %d",
                len(df), num_classes, len(config.FEATURE_COLS))

    # --- Optuna tuning ---
    best_params = run_optuna(df, num_classes)

    # --- Time-series CV ---
    logger.info("\n%s", "=" * 70)
    logger.info("TIME-SERIES VALIDATION")
    logger.info("=" * 70)
    results_df = run_validation(df, best_params, num_classes)

    if len(results_df) > 0:
        logger.info("\nValidation summary (%d races):", len(results_df))
        for method in ("XGBoost_Hungarian", "Baseline_Grid", "Baseline_Form"):
            logger.info("  %-25s mean %.2f ± %.2f",
                        method, results_df[method].mean(), results_df[method].std())
        logger.info("  %-25s mean %.2f ± %.2f",
                    "RMSE", results_df["RMSE"].mean(), results_df["RMSE"].std())
        plot_validation_results(results_df)

        # Persist per-race validation results so the visualiser can plot the
        # full time series (not just summary stats)
        results_df.to_json(config.VALIDATION_HISTORY_FILE, orient="records", indent=2)
        logger.info("Saved per-race validation history: %s", config.VALIDATION_HISTORY_FILE)

    # --- Final model on all data ---
    logger.info("\n%s", "=" * 70)
    logger.info("TRAINING FINAL MODEL (all data)")
    logger.info("=" * 70)

    X_all = df[config.FEATURE_COLS]
    y_all = df["target_class"]
    w_all = df["sample_weight"]

    final_params = {**config.XGB_FIXED_PARAMS, "num_class": num_classes, **best_params}
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_all, y_all, sample_weight=w_all, verbose=False)

    final_model.save_model(str(config.MODEL_FILE))
    plot_feature_importance(final_model)

    # --- Save metadata ---
    metadata = {
        "trained_at": datetime.utcnow().isoformat(),
        "feature_cols": config.FEATURE_COLS,
        "target_col": config.TARGET_COL,
        "num_classes": num_classes,
        "n_training_rows": len(df),
        "training_seasons": sorted(df["Year"].unique().tolist()),
        "xgb_params": final_params,
        "validation_summary": (
            {
                "n_races": int(len(results_df)),
                "xgboost_hungarian_mean": float(results_df["XGBoost_Hungarian"].mean()),
                "xgboost_hungarian_std": float(results_df["XGBoost_Hungarian"].std()),
                "baseline_grid_mean": float(results_df["Baseline_Grid"].mean()),
                "baseline_form_mean": float(results_df["Baseline_Form"].mean()),
                "rmse_mean": float(results_df["RMSE"].mean()),
            }
            if len(results_df) > 0 else None
        ),
    }
    with open(config.MODEL_META_FILE, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Saved model:    %s", config.MODEL_FILE)
    logger.info("Saved metadata: %s", config.MODEL_META_FILE)

    # Top-10 feature importances
    feat_imp = pd.Series(
        final_model.feature_importances_, index=config.FEATURE_COLS
    ).sort_values(ascending=False)
    logger.info("\nTop-10 features by importance:")
    for feat, imp in feat_imp.head(10).items():
        logger.info("  %-32s %.4f", feat, imp)


if __name__ == "__main__":
    main()
