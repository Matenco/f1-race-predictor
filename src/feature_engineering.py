"""
Feature engineering pipeline.

Builds 18 leakage-free features from raw race results. Every per-driver and
per-team statistic is computed only from races strictly *before* the current
one (groupby + shift(1) + rolling/expanding).

The previous version hardcoded everything to Miami. This version is
parameterised by circuit through `circuit_metadata.CIRCUIT_FAMILIES`, so the
same engineered features work for predicting any race.

Input:  data/processed/f1_historical.csv
Output: data/processed/f1_features.csv
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config
from .circuit_metadata import CIRCUIT_FAMILIES

logger = logging.getLogger(__name__)


# =============================================================================
#  Helpers: leakage-free rolling and expanding statistics per driver
# =============================================================================
def _driver_rolling(df: pd.DataFrame, col: str, window: int,
                    func: str = "mean") -> pd.Series:
    """Per-driver rolling statistic with a one-step shift to prevent leakage.

    The shift(1) ensures the rolling window for race N contains only races
    1…N-1 — race N's own outcome is never used to predict race N.
    """
    def _roll(x: pd.Series) -> pd.Series:
        shifted = x.shift(1)
        r = shifted.rolling(window=window, min_periods=1)
        return getattr(r, func)()
    return df.groupby("Abbreviation")[col].transform(_roll)


def _driver_expanding(df: pd.DataFrame, col: str, func: str = "mean") -> pd.Series:
    """Per-driver expanding statistic with shift(1)."""
    def _expand(x: pd.Series) -> pd.Series:
        shifted = x.shift(1)
        e = shifted.expanding(min_periods=1)
        if func == "count":
            return shifted.expanding().count()
        return getattr(e, func)()
    return df.groupby("Abbreviation")[col].transform(_expand)


# =============================================================================
#  Feature blocks
# =============================================================================
def _add_driver_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Driver recent form: rolling 3/5-race statistics on finishing position."""
    df["driver_avg_pos_last3"] = _driver_rolling(df, "Position", window=3, func="mean")
    df["driver_avg_pos_last5"] = _driver_rolling(df, "Position", window=5, func="mean")
    df["driver_best_pos_last5"] = _driver_rolling(df, "Position", window=5, func="min")
    df["driver_std_pos_last5"] = _driver_rolling(df, "Position", window=5, func="std")
    df["driver_finish_rate_last10"] = _driver_rolling(df, "Finished", window=10, func="mean")

    # Trend: negative = improving (last 3 better than last 5)
    df["driver_pos_trend"] = df["driver_avg_pos_last3"] - df["driver_avg_pos_last5"]
    return df


def _add_team_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Team strength: rolling team-average finishing position."""
    # Build a per-(team, race) timeline
    team_timeline = (
        df.groupby(["TeamName", "Year", "Round", "Date"])["Position"]
        .mean()
        .reset_index()
        .sort_values(["TeamName", "Date"])
    )
    team_timeline["team_avg_pos_last3"] = (
        team_timeline.groupby("TeamName")["Position"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    team_timeline["team_best_pos_season"] = (
        team_timeline.groupby("TeamName")["Position"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).min())
    )

    df = df.merge(
        team_timeline[["TeamName", "Year", "Round",
                       "team_avg_pos_last3", "team_best_pos_season"]],
        on=["TeamName", "Year", "Round"],
        how="left",
    )
    return df


def _add_grid_features(df: pd.DataFrame) -> pd.DataFrame:
    """Grid-derived features: GridPosition is already in the data, just augment."""
    df["grid_vs_driver_form"] = df["GridPosition"] - df["driver_avg_pos_last5"]
    return df


def _add_teammate_and_quali_features(df: pd.DataFrame) -> pd.DataFrame:
    """Teammate position delta (rolling) and gap to pole (per race)."""
    # Teammate delta: difference vs same-team partner, averaged over last 5 races
    pairs = df[["Year", "Round", "TeamName", "Abbreviation", "Position"]].copy()
    partner = pairs.merge(
        pairs.rename(columns={"Abbreviation": "_partner", "Position": "_partner_pos"}),
        on=["Year", "Round", "TeamName"],
    )
    partner = partner[partner["Abbreviation"] != partner["_partner"]]
    partner_avg = (
        partner.groupby(["Year", "Round", "Abbreviation"])["_partner_pos"]
        .mean().reset_index()
        .rename(columns={"_partner_pos": "_teammate_pos"})
    )
    df = df.merge(partner_avg, on=["Year", "Round", "Abbreviation"], how="left")
    df["_pos_delta_vs_teammate"] = df["Position"] - df["_teammate_pos"]

    df["teammate_pos_delta_last5"] = _driver_rolling(
        df, "_pos_delta_vs_teammate", window=5, func="mean",
    )
    df.drop(columns=["_teammate_pos", "_pos_delta_vs_teammate"], inplace=True)

    # Quali time delta vs pole sitter (in seconds)
    if "QualiTime_s" in df.columns:
        pole = (
            df[df["QualiTime_s"].notna()]
            .groupby(["Year", "Round"])["QualiTime_s"].min()
            .reset_index()
            .rename(columns={"QualiTime_s": "_pole_time_s"})
        )
        df = df.merge(pole, on=["Year", "Round"], how="left")
        df["quali_time_delta_s"] = df["QualiTime_s"] - df["_pole_time_s"]
        df.drop(columns=["_pole_time_s"], inplace=True)
    else:
        df["quali_time_delta_s"] = np.nan
    return df


def _add_circuit_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-driver history at this exact track and at stylistically similar tracks.

    `driver_track_avg`: expanding mean of the driver's finishing position at
    the *same* circuit, computed over all earlier visits (with shift(1)).
    `driver_track_count`: how many times the driver has raced at this circuit.
    `driver_similar_circuits_avg`: same idea but across the family of similar
    circuits defined in `circuit_metadata.CIRCUIT_FAMILIES`.

    This is what generalises the pipeline beyond Miami: every track gets its
    own feature value, computed from that track's family.
    """
    df = df.sort_values(["Abbreviation", "Date"]).reset_index(drop=True)

    # --- Per-track expanding mean and count ---
    df["driver_track_avg"] = (
        df.groupby(["Abbreviation", "Circuit"])["Position"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )
    df["driver_track_count"] = (
        df.groupby(["Abbreviation", "Circuit"])["Position"]
        .transform(lambda x: x.shift(1).expanding().count())
    )

    # --- Similar-circuits average ---
    # For each row, look up the family for that row's circuit, then compute
    # the driver's average finish at those circuits in earlier races.
    df["driver_similar_circuits_avg"] = np.nan

    # Sort by date for the lookup loop
    df_sorted = df.sort_values("Date")

    # Pre-index by driver for fast lookup.
    # Note: ``dict(groupby_obj)`` looks equivalent but doesn't work — the
    # pandas GroupBy object yields (key, sub_df) tuples, but dict() consumes
    # them differently and fails on later access. Keep the comprehension.
    by_driver = {abbr: g for abbr, g in df_sorted.groupby("Abbreviation")}  # noqa: C416

    # Iterate over rows whose circuit has a family defined.
    # We avoid the O(n²) trap by only computing for circuits we care about.
    for circuit_name, similar_list in CIRCUIT_FAMILIES.items():
        mask = df["Circuit"] == circuit_name
        if not mask.any():
            continue
        for idx in df.index[mask]:
            row = df.loc[idx]
            driver_history = by_driver.get(row["Abbreviation"])
            if driver_history is None:
                continue
            past = driver_history[
                (driver_history["Date"] < row["Date"])
                & (driver_history["Circuit"].isin(similar_list))
                & (driver_history["Position"].notna())
            ]
            if len(past) > 0:
                df.at[idx, "driver_similar_circuits_avg"] = past["Position"].mean()

    return df


def _add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Race context: rain flag, season race number, regulation era."""
    df["is_rain"] = df["Rainfall"].fillna(False).astype(int)
    df["season_race_number"] = df["Round"]
    df["regulation_era"] = (df["Year"] >= config.NEW_ERA_YEAR).astype(int)
    return df


def _add_sample_weights(df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """Time-decay sample weights with a regulation-era boost.

    Weight = 0.5 ** (days_to_target / half_life), then × NEW_ERA_WEIGHT_BOOST
    for races in the new era. Normalised to [0, 1].

    `target_date` is the date of the race we ultimately want to predict —
    older races count less, more recent ones count more.
    """
    days_ago = (target_date - df["Date"]).dt.days.clip(lower=0)
    weights = 0.5 ** (days_ago / config.WEIGHT_HALF_LIFE_DAYS)
    weights = weights * np.where(
        df["Year"] >= config.NEW_ERA_YEAR, config.NEW_ERA_WEIGHT_BOOST, 1.0,
    )
    df["sample_weight"] = weights / weights.max()
    return df


# =============================================================================
#  Public API
# =============================================================================
def build_features(historical_df: pd.DataFrame,
                   target_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Add all engineered feature columns to the historical dataframe.

    Parameters
    ----------
    historical_df : DataFrame
        Output of ``extract_data.main()`` — one row per (driver, race).
    target_date : Timestamp, optional
        Date of the race to be predicted. Used only for time-decay sample
        weights. Defaults to the latest race date in the data + 7 days.
    """
    df = historical_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    df["GridPosition"] = pd.to_numeric(df["GridPosition"], errors="coerce")
    df["QualiPosition"] = pd.to_numeric(df["QualiPosition"], errors="coerce")
    df["GridPosition"] = df["GridPosition"].fillna(df["QualiPosition"])
    df = df.sort_values(["Date", "Round", "Position"]).reset_index(drop=True)

    df = _add_driver_form_features(df)
    df = _add_team_form_features(df)
    df = _add_grid_features(df)
    df = _add_teammate_and_quali_features(df)
    df = _add_circuit_history_features(df)
    df = _add_context_features(df)

    if target_date is None:
        target_date = df["Date"].max() + pd.Timedelta(days=7)
    df = _add_sample_weights(df, target_date)

    df = df.sort_values(["Date", "Round", "Position"]).reset_index(drop=True)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=" * 70)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 70)

    df_raw = pd.read_csv(config.HISTORICAL_FILE)
    logger.info("Loaded %d rows from %s", len(df_raw), config.HISTORICAL_FILE)

    df_features = build_features(df_raw)
    df_features.to_csv(config.FEATURES_FILE, index=False)

    # Coverage report
    logger.info("\nFeature coverage:")
    for col in config.FEATURE_COLS:
        if col in df_features.columns:
            non_null = df_features[col].notna().sum()
            pct = non_null / len(df_features) * 100
            logger.info("  %-32s %6d non-null (%5.1f%%)", col, non_null, pct)
        else:
            logger.warning("  %-32s MISSING from output!", col)

    logger.info("\nWritten to: %s", config.FEATURES_FILE)
    logger.info("Rows: %d, columns: %d", len(df_features), df_features.shape[1])


if __name__ == "__main__":
    main()
