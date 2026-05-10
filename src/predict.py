"""
Predict the top 5 of the next upcoming Grand Prix.

End-to-end prediction:
1. Detect the next race from the F1 calendar
2. Build a feature row for every active driver, computed as of "now"
   (correctly — no shift-based heuristics)
3. Run the trained XGBoost model to get the probability matrix
4. Apply Hungarian assignment for the optimal top 5
5. Run Monte Carlo simulation for win/podium/top5 probabilities
6. Return everything as a dataclass for downstream visualisation

Bug fix vs the previous Miami-specific version:
The old code took shift-based feature values and applied a heuristic
`(2 * old_avg3 + last_pos) / 3` to "update" them. That formula is
mathematically wrong because the shifted value already excluded the most
recent race for a different reason. The fix here computes rolling
statistics directly from the raw historical positions, which gives the
correct "as-of-now" value with no fudging.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config
from .circuit_metadata import get_similar_circuits
from .next_race import NextRace, get_next_race
from .train_model import compute_scoring, hungarian_optimal_assignment

logger = logging.getLogger(__name__)


# =============================================================================
#  Data classes for prediction output
# =============================================================================
@dataclass
class DriverPrediction:
    abbreviation: str
    full_name: str
    team: str
    grid_position: float
    p_top5: float
    p_win: float
    p_podium: float
    prob_per_position: list[float]   # P(P1), P(P2), ..., P(P5)


@dataclass
class LastRaceRecap:
    """Self-evaluation: how well did the previous prediction actually do?

    Populated when (a) a prediction was saved before the previous race and
    (b) actual race results are now available from FastF1. Drives the
    "Last race scorecard" section at the top of the HTML report — concrete
    proof of model quality, not just abstract validation metrics.
    """
    race_name: str
    race_date: str
    predicted_top5: list[tuple[int, str]]
    actual_top5: list[tuple[int, str]]
    score: int                   # Out of 10 — see compute_scoring
    grid_baseline_score: int
    grid_baseline_top5: list[tuple[int, str]]
    exact_hits: list[str]        # Drivers predicted at the EXACT correct position
    in_top5_hits: list[str]      # Predicted in top 5 but at wrong position
    misses: list[str]            # Predicted to be in top 5, were not


@dataclass
class RacePrediction:
    next_race: NextRace
    mode: str                        # "blind" or "informed"
    predicted_top5: list[tuple[int, str]]   # [(position, driver_abbr), ...]
    sixth_driver: str | None      # First reserve if a top-5 driver fails
    drivers: list[DriverPrediction]
    most_likely_combos: list[tuple[tuple, float]]  # [((d1, d2, d3, d4, d5), freq), ...]
    n_simulations: int
    model_validation_summary: dict = field(default_factory=dict)
    last_race_recap: LastRaceRecap | None = None
    validation_history: list[dict] = field(default_factory=list)


# =============================================================================
#  Build feature row for one driver, as of the upcoming race
# =============================================================================
def _compute_driver_features(driver_history: pd.DataFrame,
                             target_circuit: str) -> dict:
    """Compute the feature dict for one driver at the moment of the next race.

    Uses the driver's full known history (no shift) — the resulting values
    represent the driver's state going INTO the upcoming race.
    """
    history = driver_history.sort_values("Date").copy()
    finished_pos = history.dropna(subset=["Position"])["Position"]

    last3 = finished_pos.tail(3)
    last5 = finished_pos.tail(5)
    last10_finished = history.tail(10)["Finished"]

    avg3 = last3.mean() if len(last3) > 0 else np.nan
    avg5 = last5.mean() if len(last5) > 0 else np.nan

    features = {
        "driver_avg_pos_last3":      avg3,
        "driver_avg_pos_last5":      avg5,
        "driver_best_pos_last5":     last5.min() if len(last5) > 0 else np.nan,
        "driver_std_pos_last5":      last5.std() if len(last5) > 1 else np.nan,
        "driver_finish_rate_last10": last10_finished.mean() if len(last10_finished) > 0 else np.nan,
        "driver_pos_trend":          (avg3 - avg5) if (pd.notna(avg3) and pd.notna(avg5)) else np.nan,
    }

    # --- Per-track history ---
    track_hist = history[history["Circuit"] == target_circuit].dropna(subset=["Position"])
    features["driver_track_avg"] = track_hist["Position"].mean() if len(track_hist) > 0 else np.nan
    features["driver_track_count"] = float(len(track_hist))

    # --- Similar-circuits average ---
    similar = get_similar_circuits(target_circuit)
    if similar:
        sim_hist = history[history["Circuit"].isin(similar)].dropna(subset=["Position"])
        features["driver_similar_circuits_avg"] = (
            sim_hist["Position"].mean() if len(sim_hist) > 0 else np.nan
        )
    else:
        features["driver_similar_circuits_avg"] = np.nan

    return features


def _compute_team_features(team_history: pd.DataFrame) -> dict:
    """Compute team-level features from a team's full race history."""
    # Average team finishing position per race, sorted by date
    team_per_race = (
        team_history.groupby(["Year", "Round", "Date"])["Position"]
        .mean()
        .reset_index()
        .sort_values("Date")
    )
    last3 = team_per_race["Position"].tail(3)
    return {
        "team_avg_pos_last3":   last3.mean() if len(last3) > 0 else np.nan,
        "team_best_pos_season": team_per_race["Position"].min() if len(team_per_race) > 0 else np.nan,
    }


def _compute_teammate_delta_last5(driver: str, team: str,
                                   features_df: pd.DataFrame) -> float:
    """Average position delta vs teammate over the driver's last 5 races.

    Reuses the value already computed in feature_engineering — that column
    captures the leakage-free rolling delta. We just take the most recent
    non-null value for this driver.
    """
    driver_rows = features_df[features_df["Abbreviation"] == driver]
    last_valid = driver_rows["teammate_pos_delta_last5"].dropna()
    return float(last_valid.iloc[-1]) if len(last_valid) > 0 else np.nan


# =============================================================================
#  Build the full prediction-time feature matrix
# =============================================================================
def build_prediction_features(features_df: pd.DataFrame,
                              next_race: NextRace,
                              grid_positions: dict[str, int] | None = None,
                              is_rain: int = 0) -> pd.DataFrame:
    """Build one row per active driver with all 18 features filled.

    Parameters
    ----------
    features_df : DataFrame
        The full engineered feature table (output of feature_engineering).
    next_race : NextRace
        Auto-detected next race metadata.
    grid_positions : dict, optional
        Map of driver abbreviation → grid position. If None, grid is
        estimated from each driver's recent form (blind mode).
    is_rain : int
        1 if rain is forecast for race day; 0 otherwise.
    """
    # Identify the active driver pool — drivers from the most recent season
    most_recent_year = int(features_df["Year"].max())
    active_drivers = features_df[features_df["Year"] == most_recent_year]["Abbreviation"].unique()

    rows = []
    for driver in active_drivers:
        driver_hist = features_df[features_df["Abbreviation"] == driver].sort_values("Date")
        if len(driver_hist) == 0:
            continue
        latest = driver_hist.iloc[-1]
        team = latest.get("TeamName", "Unknown")

        row: dict = {
            "Abbreviation": driver,
            "FullName":     latest.get("FullName", driver),
            "TeamName":     team,
        }

        # Driver form features (correctly recomputed from raw history)
        row.update(_compute_driver_features(driver_hist, next_race.circuit))

        # Team features
        team_hist = features_df[features_df["TeamName"] == team]
        row.update(_compute_team_features(team_hist))

        # Teammate delta (reuse already-computed rolling value)
        row["teammate_pos_delta_last5"] = _compute_teammate_delta_last5(
            driver, team, features_df,
        )

        # Quali time delta — only known after qualifying. Use 0.0 in blind mode
        # (model treats this as "missing/unknown")
        if grid_positions is not None and driver in grid_positions:
            row["GridPosition"] = float(grid_positions[driver])
            # In informed mode the quali delta is best supplied by the caller;
            # leaving as NaN lets XGBoost's native NaN handling take over.
            row["quali_time_delta_s"] = np.nan
        else:
            # Blind mode: estimate grid from recent form
            row["GridPosition"] = float(row.get("driver_avg_pos_last3") or 11.0)
            row["quali_time_delta_s"] = np.nan

        # Grid vs form
        avg5 = row.get("driver_avg_pos_last5")
        row["grid_vs_driver_form"] = (
            row["GridPosition"] - avg5 if pd.notna(avg5) else np.nan
        )

        # Context features
        row["is_rain"] = is_rain
        row["season_race_number"] = float(next_race.round_number)
        row["regulation_era"] = 1 if next_race.year >= config.NEW_ERA_YEAR else 0

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
#  Monte Carlo simulation
# =============================================================================
def monte_carlo_simulation(prob_matrix: np.ndarray, drivers: list,
                           n_simulations: int = config.N_SIMULATIONS,
                           seed: int = config.RANDOM_STATE) -> dict:
    """Run n_simulations races by sampling each driver's outcome from the
    model's probability matrix, ranking, and aggregating frequencies.

    For each simulation, each driver samples a target position from their
    own row of `prob_matrix`. Tie-breaking uses small uniform noise so the
    final ranking is well-defined and respects the categorical sample order.
    """
    from collections import Counter

    rng = np.random.default_rng(seed)
    n_drivers, n_positions = prob_matrix.shape

    # Defensive normalisation (XGBoost predict_proba should already sum to 1)
    probs = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)

    # Sample target position for each (simulation, driver) pair
    target = np.zeros((n_simulations, n_drivers), dtype=np.int32)
    for i in range(n_drivers):
        target[:, i] = rng.choice(n_positions, size=n_simulations, p=probs[i])

    noise = rng.uniform(0.0, 0.99, size=(n_simulations, n_drivers))
    scores = target.astype(float) + noise   # lower = better
    rankings = np.argsort(scores, axis=1)

    win_counts = np.zeros(n_drivers, dtype=np.int32)
    podium_counts = np.zeros(n_drivers, dtype=np.int32)
    top5_counts = np.zeros(n_drivers, dtype=np.int32)
    top5_combos = []

    for sim in range(n_simulations):
        order = rankings[sim]
        win_counts[order[0]] += 1
        for k in range(min(3, n_drivers)):
            podium_counts[order[k]] += 1
        for k in range(min(5, n_drivers)):
            top5_counts[order[k]] += 1
        top5_combos.append(tuple(drivers[order[k]] for k in range(min(5, n_drivers))))

    combo_counter = Counter(top5_combos)

    return {
        "win_prob":     win_counts / n_simulations,
        "podium_prob":  podium_counts / n_simulations,
        "top5_prob":    top5_counts / n_simulations,
        "top_combos":   combo_counter.most_common(config.TOP_N_COMBOS),
        "n_simulations": n_simulations,
    }


# =============================================================================
#  Main prediction function
# =============================================================================
def predict_next_race(grid_positions: dict[str, int] | None = None,
                      is_rain: int = 0) -> RacePrediction:
    """End-to-end prediction for the next race.

    Parameters
    ----------
    grid_positions : dict, optional
        Pass actual grid positions to switch to "informed" mode (after
        qualifying). Otherwise predicts in blind mode (grid estimated from form).
    is_rain : int
        Rain forecast flag.
    """
    # 1. Detect the next race
    next_race = get_next_race()
    logger.info("Next race: %s", next_race)

    # 2. Load engineered features and trained model
    if not config.FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Features file missing: {config.FEATURES_FILE}\n"
            "Run feature_engineering first."
        )
    if not config.MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file missing: {config.MODEL_FILE}\n"
            "Run train_model first."
        )

    features_df = pd.read_csv(config.FEATURES_FILE)
    features_df["Date"] = pd.to_datetime(features_df["Date"], errors="coerce")
    features_df["Position"] = pd.to_numeric(features_df["Position"], errors="coerce")

    model = xgb.XGBClassifier()
    model.load_model(str(config.MODEL_FILE))

    # 3. If no grid given but the API reports quali done, try fetching it
    mode = "informed" if grid_positions is not None else (
        "informed" if next_race.has_quali_results else "blind"
    )
    if grid_positions is None and next_race.has_quali_results:
        grid_positions = _fetch_quali_results(next_race)
        if grid_positions:
            logger.info("Loaded %d grid positions from F1 API", len(grid_positions))
        else:
            mode = "blind"

    # 4. Build feature matrix for the upcoming race
    next_race_df = build_prediction_features(
        features_df, next_race,
        grid_positions=grid_positions, is_rain=is_rain,
    )
    logger.info("Built features for %d active drivers", len(next_race_df))

    # 5. Run model
    X_pred = next_race_df[config.FEATURE_COLS]
    drivers_list = next_race_df["Abbreviation"].tolist()
    prob_matrix = model.predict_proba(X_pred)

    # 6. Hungarian optimal assignment
    predicted_top5 = hungarian_optimal_assignment(prob_matrix, drivers_list)

    # 7. Monte Carlo simulation
    mc = monte_carlo_simulation(prob_matrix, drivers_list)

    # 8. Build per-driver result objects
    p_top5 = prob_matrix[:, :5].sum(axis=1)
    driver_results: list[DriverPrediction] = []
    for i, drv in enumerate(drivers_list):
        meta = next_race_df.iloc[i]
        driver_results.append(DriverPrediction(
            abbreviation=drv,
            full_name=str(meta.get("FullName", drv)),
            team=str(meta.get("TeamName", "Unknown")),
            grid_position=float(meta.get("GridPosition", np.nan)),
            p_top5=float(p_top5[i]),
            p_win=float(mc["win_prob"][i]),
            p_podium=float(mc["podium_prob"][i]),
            prob_per_position=[float(prob_matrix[i, j]) for j in range(5)],
        ))

    # 9. First reserve (highest P(top5) outside the chosen 5)
    top5_set = {d for _, d in predicted_top5}
    sixth = max(
        ((d for d in driver_results if d.abbreviation not in top5_set)),
        key=lambda d: d.p_top5,
        default=None,
    )

    # 10. Validation summary from saved metadata (for context)
    val_summary = {}
    if config.MODEL_META_FILE.exists():
        with open(config.MODEL_META_FILE) as f:
            meta = json.load(f)
            val_summary = meta.get("validation_summary") or {}

    # 11. Last race recap — how did our previous prediction actually do?
    #     Done BEFORE saving the current prediction so we don't try to recap
    #     a race that hasn't happened yet.
    last_recap = _build_last_race_recap(next_race)
    if last_recap:
        logger.info("Last race recap: %s — %d/10 (grid baseline: %d/10)",
                    last_recap.race_name, last_recap.score,
                    last_recap.grid_baseline_score)
    else:
        logger.info("No prior prediction with available results — recap skipped")

    # 12. Persist this prediction to history (so the NEXT run can recap it)
    pred = RacePrediction(
        next_race=next_race,
        mode=mode,
        predicted_top5=predicted_top5,
        sixth_driver=sixth.abbreviation if sixth else None,
        drivers=driver_results,
        most_likely_combos=mc["top_combos"],
        n_simulations=mc["n_simulations"],
        model_validation_summary=val_summary,
        last_race_recap=last_recap,
        validation_history=_load_validation_history(),
    )
    try:
        _save_prediction_to_history(pred)
    except Exception as exc:
        logger.warning("Could not save prediction history: %s", exc)
    return pred


def _fetch_quali_results(next_race: NextRace) -> dict[str, int] | None:
    """Pull grid positions from the F1 API if qualifying has happened."""
    try:
        import fastf1
        quali = fastf1.get_session(next_race.year, next_race.round_number, "Q")
        quali.load(laps=False, telemetry=False, weather=False, messages=False)
        results = quali.results
        if "Position" not in results.columns:
            return None
        out: dict[str, int] = {}
        for _, row in results.iterrows():
            if pd.notna(row.get("Position")):
                out[str(row["Abbreviation"])] = int(row["Position"])
        return out if out else None
    except Exception as exc:
        logger.warning("Could not fetch quali: %s", exc)
        return None


# =============================================================================
#  Prediction history persistence + last-race recap
# =============================================================================
def _load_prediction_history() -> list[dict]:
    """Load saved predictions from disk; return [] if file is missing."""
    if not config.PREDICTION_HISTORY_FILE.exists():
        return []
    try:
        with open(config.PREDICTION_HISTORY_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load prediction history: %s", exc)
        return []


def _save_prediction_to_history(pred: RacePrediction) -> None:
    """Append the current prediction to the history file (deduplicated by race)."""
    history = _load_prediction_history()
    race_key = (int(pred.next_race.year), int(pred.next_race.round_number))
    # Replace existing entry for the same race (re-running before race day)
    history = [
        h for h in history
        if (h["race"]["year"], h["race"]["round"]) != race_key
    ]
    history.append({
        "predicted_at": datetime.utcnow().isoformat() + "Z",
        "race": {
            "name": str(pred.next_race.name),
            "year": int(pred.next_race.year),
            "round": int(pred.next_race.round_number),
            "circuit": str(pred.next_race.circuit),
            "date": pred.next_race.date.strftime("%Y-%m-%d"),
        },
        "mode": pred.mode,
        "predicted_top5": [[int(p), str(d)] for p, d in pred.predicted_top5],
    })
    history.sort(key=lambda h: (h["race"]["year"], h["race"]["round"]))
    config.PREDICTION_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.PREDICTION_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved prediction to history: %s", config.PREDICTION_HISTORY_FILE)


def _fetch_actual_race_results(year: int, round_number: int) -> dict[str, int] | None:
    """Try to pull actual finishing positions from FastF1 for a completed race.

    Returns ``{driver_abbr: finish_position}`` or ``None`` if the race
    hasn't been run yet or the API call fails.
    """
    try:
        import fastf1
        fastf1.Cache.enable_cache(str(config.CACHE_DIR))
        race = fastf1.get_session(year, round_number, "R")
        race.load(laps=False, telemetry=False, weather=False, messages=False)
        results = race.results
        if results is None or len(results) == 0:
            return None
        if "Position" not in results.columns:
            return None
        out: dict[str, int] = {}
        for _, row in results.iterrows():
            pos = row.get("Position")
            abbr = row.get("Abbreviation")
            if pd.notna(pos) and pd.notna(abbr):
                out[str(abbr)] = int(pos)
        return out if out else None
    except Exception as exc:
        logger.info("Actual results not yet available for %d R%d (%s)",
                    year, round_number, type(exc).__name__)
        return None


def _build_last_race_recap(current_race: NextRace) -> LastRaceRecap | None:
    """Find the most recent prediction whose race has been run, score it.

    Iterates through the saved prediction history newest-first, skipping the
    current upcoming race, and returns a recap for the first race that has
    actual results available from the F1 API. Returns ``None`` if no such
    prior prediction exists yet (cold start).
    """
    history = _load_prediction_history()
    if not history:
        return None

    current_key = (current_race.year, current_race.round_number)

    # Check predictions newest-first
    for entry in reversed(history):
        race = entry["race"]
        if (race["year"], race["round"]) == current_key:
            continue   # this is the upcoming race — skip
        actual = _fetch_actual_race_results(race["year"], race["round"])
        if actual is None:
            continue   # not run yet, or API hiccup; try next-newest

        predicted = [(int(p), str(d)) for p, d in entry["predicted_top5"]]
        score = compute_scoring(predicted, actual)

        # Compute grid baseline for the same race
        grid_baseline_top5: list[tuple[int, str]] = []
        try:
            import fastf1
            grid_session = fastf1.get_session(race["year"], race["round"], "R")
            grid_session.load(laps=False, telemetry=False, weather=False, messages=False)
            grid_results = grid_session.results.dropna(subset=["GridPosition"])
            grid_results = grid_results.sort_values("GridPosition").head(5)
            grid_baseline_top5 = [
                (int(r["GridPosition"]), str(r["Abbreviation"]))
                for _, r in grid_results.iterrows()
            ]
        except Exception:
            grid_baseline_top5 = []
        grid_score = compute_scoring(grid_baseline_top5, actual) if grid_baseline_top5 else 0

        # Categorise predictions: exact hits, in-top-5 misses, complete misses
        actual_top5_set = {d for d, p in actual.items() if p <= 5}
        exact_hits = [d for pos, d in predicted if actual.get(d) == pos]
        in_top5_hits = [
            d for pos, d in predicted
            if d in actual_top5_set and actual.get(d) != pos
        ]
        misses = [d for _, d in predicted if d not in actual_top5_set]

        # Sort actual top 5 by position for display
        actual_sorted = sorted(
            ((d, p) for d, p in actual.items() if p <= 5),
            key=lambda x: x[1],
        )
        actual_top5 = [(int(p), str(d)) for d, p in actual_sorted]

        return LastRaceRecap(
            race_name=race["name"],
            race_date=race["date"],
            predicted_top5=predicted,
            actual_top5=actual_top5,
            score=int(score),
            grid_baseline_score=int(grid_score),
            grid_baseline_top5=grid_baseline_top5,
            exact_hits=exact_hits,
            in_top5_hits=in_top5_hits,
            misses=misses,
        )

    return None


def _load_validation_history() -> list[dict]:
    """Load per-race validation results saved by train_model."""
    if not config.VALIDATION_HISTORY_FILE.exists():
        return []
    try:
        with open(config.VALIDATION_HISTORY_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def main() -> None:
    """CLI: run prediction and pretty-print to stdout."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pred = predict_next_race()
    print()
    print("=" * 70)
    print(f" PREDICTION — {pred.next_race.name} ({pred.mode.upper()} MODE)")
    print("=" * 70)
    print(f"  Date: {pred.next_race.date.strftime('%Y-%m-%d')}")
    print(f"  Circuit: {pred.next_race.circuit}\n")

    by_abbr = {d.abbreviation: d for d in pred.drivers}
    print(f"  {'Pos':>4s}  {'Driver':>6s}  {'Team':>22s}  {'P(top5)':>7s}  {'P(win)':>7s}")
    print("  " + "-" * 60)
    for pos, drv in pred.predicted_top5:
        d = by_abbr[drv]
        print(f"  P{pos:>2d}   {drv:>6s}  {d.team:>22s}  {d.p_top5:>7.1%}  {d.p_win:>7.1%}")
    if pred.sixth_driver:
        d6 = by_abbr[pred.sixth_driver]
        print(f"\n  First reserve: {pred.sixth_driver} ({d6.team}) — P(top5) = {d6.p_top5:.1%}")


if __name__ == "__main__":
    main()
