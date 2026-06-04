"""
Train the XGBoost position, top-5, and ranker models with time-series CV.

Pipeline:
1. Load engineered features
2. Optuna tuning for the position model on pre-validation races
3. Time-series cross-validation over 2025–2026
4. Compare ranker, diagnostic model variants, grid baseline, and form baseline
5. Train final model bundle on all available data
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


def compute_ranking_metrics(pred_top5: list, actual_positions: dict) -> dict:
    """Ranking metrics for a predicted top 5."""
    actual_top5 = {abbr: pos for abbr, pos in actual_positions.items() if pos <= 5}
    exact_hits = 0
    top5_hits = 0
    dcg = 0.0
    for rank, (pred_pos, driver) in enumerate(pred_top5, start=1):
        actual_pos = actual_positions.get(driver)
        if actual_pos is None:
            continue
        if actual_pos == pred_pos:
            exact_hits += 1
        if driver in actual_top5:
            top5_hits += 1
            relevance = 6 - actual_pos
            dcg += relevance / np.log2(rank + 1)

    ideal_relevance = [5, 4, 3, 2, 1]
    idcg = sum(rel / np.log2(rank + 1) for rank, rel in enumerate(ideal_relevance, start=1))
    return {
        "exact_hits": exact_hits,
        "top5_hits": top5_hits,
        "precision_at_5": top5_hits / 5,
        "ndcg_at_5": dcg / idcg if idcg else 0.0,
    }


def hungarian_optimal_assignment(prob_matrix: np.ndarray,
                                 drivers: list,
                                 p_top5_override: np.ndarray | None = None) -> list[tuple[int, str]]:
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
    if p_top5_override is not None:
        p_top5 = np.asarray(p_top5_override, dtype=float)

    ev_matrix = np.zeros((n_drivers, 5))
    for j in range(5):
        ev_matrix[:, j] = prob_matrix[:, j] + p_top5

    # linear_sum_assignment minimises, so negate to maximise
    row_ind, col_ind = linear_sum_assignment(-ev_matrix)
    assignments = [(c + 1, drivers[r]) for r, c in zip(row_ind, col_ind, strict=False)]
    return sorted(assignments, key=lambda x: x[0])


def baseline_grid_top5(race_df: pd.DataFrame) -> list:
    """Baseline: predict top 5 = grid positions 1–5, falling back to QualiPosition."""
    grid_col = "GridPosition"
    if race_df["GridPosition"].isna().all() and "QualiPosition" in race_df.columns:
        grid_col = "QualiPosition"
    grid_df = race_df.dropna(subset=[grid_col]).copy()
    if len(grid_df) == 0:
        return []
    grid_df["_grid_order"] = pd.to_numeric(grid_df[grid_col], errors="coerce")
    positive = grid_df["_grid_order"][grid_df["_grid_order"] > 0]
    if len(positive) > 0:
        grid_df.loc[grid_df["_grid_order"] <= 0, "_grid_order"] = positive.max() + 1
    tie_cols = ["_grid_order"]
    if "QualiPosition" in grid_df.columns:
        tie_cols.append("QualiPosition")
    grid_sorted = grid_df.sort_values(tie_cols, na_position="last")
    top5 = grid_sorted.head(5)
    return [(i + 1, r["Abbreviation"]) for i, (_, r) in enumerate(top5.iterrows())]


def baseline_form_top5(race_df: pd.DataFrame) -> list:
    """Baseline: predict top 5 = drivers with the best last-5-race average."""
    form_sorted = race_df.dropna(subset=["driver_avg_pos_last5"]).sort_values(
        "driver_avg_pos_last5"
    )
    top5 = form_sorted.head(5)
    return [(i + 1, r["Abbreviation"]) for i, (_, r) in enumerate(top5.iterrows())]


def compute_training_weights(df: pd.DataFrame, target_date: pd.Timestamp) -> pd.Series:
    """Time-decay weights relative to the race currently being predicted.

    Feature engineering stores default weights for the final production run.
    Walk-forward validation needs this dynamic version so a 2025 validation
    race is weighted as if we were standing in 2025, not after the latest race
    in the full dataset.
    """
    race_dates = pd.to_datetime(df["Date"], errors="coerce")
    days_ago = (target_date - race_dates).dt.days.clip(lower=0)
    weights = 0.5 ** (days_ago / config.WEIGHT_HALF_LIFE_DAYS)
    if pd.Timestamp(target_date).year >= config.NEW_ERA_YEAR:
        weights = weights * np.where(
            df["Year"] >= config.NEW_ERA_YEAR, config.NEW_ERA_WEIGHT_BOOST, 1.0,
        )
    weights = pd.Series(weights, index=df.index, dtype=float)
    max_weight = weights.max()
    if max_weight > 0:
        weights = weights / max_weight
    return weights.fillna(1.0)


# =============================================================================
#  Optuna tuning
# =============================================================================
def _build_optuna_objective(df: pd.DataFrame, num_classes: int,
                            tune_val_races: list[int]):
    """Build the Optuna objective: maximise mean game score across tuning races.

    Uses the same +2/+1 scoring rule and Hungarian assignment as the real
    evaluation, so hyperparameters are tuned for what actually matters rather
    than minimising a generic RMSE.
    """
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
        game_scores = []
        for idx in tune_val_races:
            tr = df[df["race_idx"] < idx]
            te = df[df["race_idx"] == idx]
            if len(te) == 0 or len(tr) < 100:
                continue
            X_tr, y_tr = tr[config.FEATURE_COLS], tr["target_class"]
            X_te = te[config.FEATURE_COLS]
            w_tr = compute_training_weights(tr, pd.to_datetime(te["Date"].iloc[0]))

            m = xgb.XGBClassifier(**params)
            m.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)

            prob_matrix = m.predict_proba(X_te)
            drivers = te["Abbreviation"].tolist()
            actual_positions = dict(
                zip(te["Abbreviation"], te[config.TARGET_COL].astype(int), strict=False)
            )
            try:
                top5 = hungarian_optimal_assignment(prob_matrix, drivers)
                score = compute_scoring(top5, actual_positions)
            except Exception:
                score = 0
            game_scores.append(score)
        return float(np.mean(game_scores)) if game_scores else 0.0
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
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
    )
    study.optimize(
        _build_optuna_objective(df, num_classes, tune_val_races),
        n_trials=config.N_OPTUNA_TRIALS,
        show_progress_bar=True,
    )

    logger.info("Best game score (tuning 2022–2024): %.4f", study.best_value)
    logger.info("Best params:")
    for k, v in study.best_params.items():
        logger.info("  %s: %s", k, v)

    # Persist for future runs
    with open(config.OPTUNA_BEST_PARAMS_FILE, "w") as f:
        json.dump(study.best_params, f, indent=2)
    logger.info("Saved best params to %s", config.OPTUNA_BEST_PARAMS_FILE)

    return dict(study.best_params)


# =============================================================================
#  Model helpers
# =============================================================================
def _top5_model_params(xgb_params: dict) -> dict:
    """Convert the shared tree params into a binary top-5 classifier config."""
    params = {**config.XGB_FIXED_PARAMS, **xgb_params}
    params.pop("num_class", None)
    params["objective"] = "binary:logistic"
    params["eval_metric"] = "logloss"
    return params


def _blended_top5_probability(prob_matrix: np.ndarray, top5_prob: np.ndarray) -> np.ndarray:
    """Blend multiclass P(top5) with the dedicated top-5 classifier."""
    multi_top5 = prob_matrix[:, :5].sum(axis=1)
    w = config.TOP5_BLEND_WEIGHT
    return (1.0 - w) * multi_top5 + w * top5_prob


def _ranker_relevance(position: pd.Series) -> np.ndarray:
    """Relevance labels for learning to rank: 5 for P1 down to 1 for P5."""
    pos = pd.to_numeric(position, errors="coerce")
    return np.where(pos <= 5, 6 - pos, 0).astype(float)


def _fit_ranker(train_df: pd.DataFrame,
                sample_weight: pd.Series | np.ndarray | None = None) -> xgb.XGBRanker:
    """Fit an XGBoost ranker grouped by race."""
    rank_train = train_df.sort_values(["race_idx", "GridPosition"]).copy()
    group = rank_train.groupby("race_idx").size().to_numpy()
    group_weight = None
    if sample_weight is not None:
        if isinstance(sample_weight, pd.Series):
            row_weight = sample_weight.reindex(rank_train.index)
        else:
            row_weight = pd.Series(sample_weight, index=train_df.index).reindex(rank_train.index)
        group_weight = (
            pd.DataFrame({"race_idx": rank_train["race_idx"], "weight": row_weight.to_numpy()})
            .groupby("race_idx", sort=False)["weight"]
            .mean()
            .to_numpy()
        )
    model = xgb.XGBRanker(**config.XGB_RANKER_PARAMS)
    model.fit(
        rank_train[config.FEATURE_COLS],
        _ranker_relevance(rank_train[config.TARGET_COL]),
        group=group,
        sample_weight=group_weight,
        verbose=False,
    )
    return model


def ranker_grid_top5_assignment(race_df: pd.DataFrame,
                                ranker_scores: np.ndarray,
                                selection_scores: np.ndarray | None = None,
                                candidate_pool: int | None = None,
                                ranker_weight: float | None = None,
                                selection_weight: float | None = None) -> list[tuple[int, str]]:
    """Blend grid order with ranker score over a small grid candidate pool.

    With ``candidate_pool=5`` this only reorders the grid baseline's top five.
    Larger pools allow controlled swaps from P6/P7 when the learned score is
    strong enough to overcome the grid-rank penalty.
    """
    by_driver = dict(zip(race_df["Abbreviation"], ranker_scores, strict=False))
    grid_col = "GridPosition"
    if race_df["GridPosition"].isna().all() and "QualiPosition" in race_df.columns:
        grid_col = "QualiPosition"
    grid_df = race_df.dropna(subset=[grid_col]).copy()
    if len(grid_df) == 0:
        return []
    grid_df["_grid_order"] = pd.to_numeric(grid_df[grid_col], errors="coerce")
    positive = grid_df["_grid_order"][grid_df["_grid_order"] > 0]
    if len(positive) > 0:
        grid_df.loc[grid_df["_grid_order"] <= 0, "_grid_order"] = positive.max() + 1
    tie_cols = ["_grid_order"]
    if "QualiPosition" in grid_df.columns:
        tie_cols.append("QualiPosition")
    grid_sorted = grid_df.sort_values(tie_cols, na_position="last")
    pool_size = candidate_pool if candidate_pool is not None else config.RANKER_CANDIDATE_POOL
    grid_drivers = grid_sorted.head(max(5, int(pool_size)))["Abbreviation"].tolist()
    grid_rank = {driver: rank for rank, driver in enumerate(grid_drivers, start=1)}
    score_values = np.array([by_driver.get(driver, 0.0) for driver in grid_drivers], dtype=float)

    def _zscore(values: np.ndarray) -> np.ndarray:
        std = float(values.std())
        if std > 1e-9:
            return (values - float(values.mean())) / std
        return np.zeros_like(values)

    score_values = _zscore(score_values)
    if selection_scores is not None:
        by_selection = dict(zip(race_df["Abbreviation"], selection_scores, strict=False))
        selection_values = _zscore(
            np.array([by_selection.get(driver, 0.0) for driver in grid_drivers], dtype=float)
        )
    else:
        selection_values = np.zeros_like(score_values)
    normalized_ranker = dict(zip(grid_drivers, score_values, strict=False))
    normalized_selection = dict(zip(grid_drivers, selection_values, strict=False))
    rw = config.RANKER_GRID_BLEND_WEIGHT if ranker_weight is None else ranker_weight
    sw = config.RANKER_TOP5_BLEND_WEIGHT if selection_weight is None else selection_weight
    ordered = sorted(
        grid_drivers,
        key=lambda driver: (
            -grid_rank[driver]
            + rw * normalized_ranker[driver]
            + sw * normalized_selection[driver]
        ),
        reverse=True,
    )[:5]
    return [(i + 1, driver) for i, driver in enumerate(ordered)]


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
        y_top5_tr = (train_df[config.TARGET_COL] <= 5).astype(int)
        w_tr = compute_training_weights(train_df, pd.to_datetime(test_df["Date"].iloc[0]))
        X_te = test_df[config.FEATURE_COLS]

        params = {**config.XGB_FIXED_PARAMS, "num_class": num_classes, **xgb_params}
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)

        prob_matrix = model.predict_proba(X_te)
        top5_model = xgb.XGBClassifier(**_top5_model_params(xgb_params))
        top5_model.fit(X_tr, y_top5_tr, sample_weight=w_tr, verbose=False)
        top5_prob = top5_model.predict_proba(X_te)[:, 1]
        blended_top5 = _blended_top5_probability(prob_matrix, top5_prob)
        ranker = _fit_ranker(train_df, sample_weight=w_tr)
        ranker_scores = ranker.predict(X_te)

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

        try:
            ensemble_top5 = hungarian_optimal_assignment(
                prob_matrix, drivers, p_top5_override=blended_top5,
            )
            ensemble_score = compute_scoring(ensemble_top5, actual_positions)
        except Exception:
            ensemble_top5, ensemble_score = [], 0

        grid_top5 = baseline_grid_top5(test_df)
        ranker_top5 = ranker_grid_top5_assignment(
            test_df, ranker_scores, selection_scores=blended_top5,
        )
        form_top5 = baseline_form_top5(test_df)
        ranker_score = compute_scoring(ranker_top5, actual_positions)
        grid_score = compute_scoring(grid_top5, actual_positions)
        form_score = compute_scoring(form_top5, actual_positions)

        ranker_metrics = compute_ranking_metrics(ranker_top5, actual_positions)
        ensemble_metrics = compute_ranking_metrics(ensemble_top5, actual_positions)
        grid_metrics = compute_ranking_metrics(grid_top5, actual_positions)
        form_metrics = compute_ranking_metrics(form_top5, actual_positions)

        race_info = test_df.iloc[0]
        pred_positions = prob_matrix.argmax(axis=1) + 1
        actual_arr = test_df[config.TARGET_COL].astype(float).values
        rmse = float(np.sqrt(np.mean((pred_positions - actual_arr) ** 2)))

        results.append({
            "Year": int(race_info["Year"]),
            "Round": int(race_info["Round"]),
            "GP": race_info["GP"],
            "XGBoost_Ranker_GridTop5": ranker_score,
            "XGBoost_Hungarian": xgb_score,
            "XGBoost_Top5_Ensemble": ensemble_score,
            "Baseline_Grid": grid_score,
            "Baseline_Form": form_score,
            "Ranker_Top5_Hits": ranker_metrics["top5_hits"],
            "Ensemble_Top5_Hits": ensemble_metrics["top5_hits"],
            "Grid_Top5_Hits": grid_metrics["top5_hits"],
            "Form_Top5_Hits": form_metrics["top5_hits"],
            "Ranker_NDCG@5": ranker_metrics["ndcg_at_5"],
            "Ensemble_NDCG@5": ensemble_metrics["ndcg_at_5"],
            "Grid_NDCG@5": grid_metrics["ndcg_at_5"],
            "Form_NDCG@5": form_metrics["ndcg_at_5"],
            "RMSE": rmse,
        })
        logger.info(
            "  [%d R%2d] %-35s | Ranker: %d/10  XGB+H: %d/10  Top5: %d/10  Grid: %d/10  Form: %d/10  RMSE: %.2f",
            int(race_info["Year"]), int(race_info["Round"]), str(race_info["GP"])[:35],
            ranker_score, xgb_score, ensemble_score, grid_score, form_score, rmse,
        )
    return pd.DataFrame(results)


def plot_validation_results(results_df: pd.DataFrame) -> None:
    """Save a comparison plot: boxplot + rolling-average lines."""
    if len(results_df) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    data_to_plot = [
        results_df["XGBoost_Ranker_GridTop5"],
        results_df["XGBoost_Top5_Ensemble"],
        results_df["XGBoost_Hungarian"],
        results_df["Baseline_Grid"],
        results_df["Baseline_Form"],
    ]
    bp = ax.boxplot(
        data_to_plot,
        tick_labels=[
            "XGB\nRanker", "XGB\nTop5", "XGB\nPosition", "Baseline\n(Grid)", "Baseline\n(Form)",
        ],
        patch_artist=True,
    )
    for patch, color in zip(
        bp["boxes"], ["#27f4d2", "#64C4FF", "#3671C6", "#E8002D", "#FF8000"], strict=False,
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Points (0–10)")
    ax.set_title("Per-race score distribution", fontsize=12, weight="bold")
    ax.set_ylim(-0.5, 10.5)

    ax = axes[1]
    window = 5
    ax.plot(results_df["XGBoost_Ranker_GridTop5"].rolling(window, min_periods=1).mean(),
            label="XGBoost ranker (grid top5)", color="#27f4d2", linewidth=3)
    ax.plot(results_df["XGBoost_Top5_Ensemble"].rolling(window, min_periods=1).mean(),
            label="XGBoost Top5 ensemble", color="#64C4FF", linewidth=2)
    ax.plot(results_df["XGBoost_Hungarian"].rolling(window, min_periods=1).mean(),
            label="XGBoost position", color="#3671C6", linewidth=2, linestyle="-.")
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
        for method in ("XGBoost_Ranker_GridTop5", "XGBoost_Top5_Ensemble", "XGBoost_Hungarian",
                       "Baseline_Grid", "Baseline_Form"):
            logger.info("  %-25s mean %.2f ± %.2f",
                        method, results_df[method].mean(), results_df[method].std())
        for method in ("Ranker_Top5_Hits", "Ensemble_Top5_Hits",
                       "Grid_Top5_Hits", "Form_Top5_Hits"):
            logger.info("  %-25s mean %.2f", method, results_df[method].mean())
        for method in ("Ranker_NDCG@5", "Ensemble_NDCG@5", "Grid_NDCG@5", "Form_NDCG@5"):
            logger.info("  %-25s mean %.3f", method, results_df[method].mean())
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

    y_top5_all = (df[config.TARGET_COL] <= 5).astype(int)
    top5_model = xgb.XGBClassifier(**_top5_model_params(best_params))
    top5_model.fit(X_all, y_top5_all, sample_weight=w_all, verbose=False)
    ranker_model = _fit_ranker(df, sample_weight=w_all)

    final_model.save_model(str(config.MODEL_FILE))
    top5_model.save_model(str(config.TOP5_MODEL_FILE))
    ranker_model.save_model(str(config.RANKER_MODEL_FILE))
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
                "xgboost_ranker_grid_top5_mean": float(
                    results_df["XGBoost_Ranker_GridTop5"].mean()
                ),
                "xgboost_ranker_grid_top5_std": float(
                    results_df["XGBoost_Ranker_GridTop5"].std()
                ),
                "xgboost_top5_ensemble_mean": float(
                    results_df["XGBoost_Top5_Ensemble"].mean()
                ),
                "xgboost_top5_ensemble_std": float(
                    results_df["XGBoost_Top5_Ensemble"].std()
                ),
                "xgboost_hungarian_mean": float(results_df["XGBoost_Hungarian"].mean()),
                "xgboost_hungarian_std": float(results_df["XGBoost_Hungarian"].std()),
                "baseline_grid_mean": float(results_df["Baseline_Grid"].mean()),
                "baseline_form_mean": float(results_df["Baseline_Form"].mean()),
                "xgboost_ranker_top5_hits_mean": float(results_df["Ranker_Top5_Hits"].mean()),
                "xgboost_top5_hits_mean": float(results_df["Ensemble_Top5_Hits"].mean()),
                "baseline_grid_top5_hits_mean": float(results_df["Grid_Top5_Hits"].mean()),
                "xgboost_ranker_ndcg_at_5_mean": float(results_df["Ranker_NDCG@5"].mean()),
                "xgboost_ndcg_at_5_mean": float(results_df["Ensemble_NDCG@5"].mean()),
                "baseline_grid_ndcg_at_5_mean": float(results_df["Grid_NDCG@5"].mean()),
                "rmse_mean": float(results_df["RMSE"].mean()),
            }
            if len(results_df) > 0 else None
        ),
    }
    with open(config.MODEL_META_FILE, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Saved model:    %s", config.MODEL_FILE)
    logger.info("Saved top5:     %s", config.TOP5_MODEL_FILE)
    logger.info("Saved ranker:   %s", config.RANKER_MODEL_FILE)
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
