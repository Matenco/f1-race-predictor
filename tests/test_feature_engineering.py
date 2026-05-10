"""Tests for feature engineering — leakage detection and shape sanity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.feature_engineering import build_features


@pytest.fixture
def features(synthetic_history: pd.DataFrame) -> pd.DataFrame:
    return build_features(synthetic_history)


def test_all_feature_cols_present(features: pd.DataFrame):
    """Every column in config.FEATURE_COLS must exist in the output."""
    missing = [c for c in config.FEATURE_COLS if c not in features.columns]
    assert not missing, f"Missing feature columns: {missing}"


def test_no_leakage_first_race_per_driver_is_nan(features: pd.DataFrame):
    """For each driver, the first chronological race must have NaN rolling
    features — there is no past to compute from. If shift(1) was forgotten,
    the rolling values would be filled with the current race's own outcome.

    Note: ``groupby().first()`` skips NaN per column; we want the literal
    first row, so use ``drop_duplicates`` instead.
    """
    first_rows = (
        features.sort_values(["Abbreviation", "Date"])
        .drop_duplicates("Abbreviation", keep="first")
    )
    for col in ["driver_avg_pos_last3", "driver_avg_pos_last5",
                "driver_best_pos_last5"]:
        assert first_rows[col].isna().all(), (
            f"Leakage: {col} is filled on a driver's very first race "
            f"(non-null on {first_rows[col].notna().sum()} drivers)"
        )


def test_no_leakage_track_features(features: pd.DataFrame):
    """Per-track features must be NaN on a driver's first visit to a track."""
    first_visits = (
        features.sort_values(["Abbreviation", "Circuit", "Date"])
        .drop_duplicates(["Abbreviation", "Circuit"], keep="first")
    )
    assert first_visits["driver_track_avg"].isna().all(), (
        "Leakage: driver_track_avg is filled on a driver's first visit to "
        f"the track ({first_visits['driver_track_avg'].notna().sum()} cases)"
    )


def test_rolling_values_within_reasonable_bounds(features: pd.DataFrame):
    """Sanity: rolling positions should be in [1, ~25]."""
    valid = features["driver_avg_pos_last5"].dropna()
    assert (valid >= 1.0).all()
    assert (valid <= 25.0).all()


def test_finish_rate_is_a_proportion(features: pd.DataFrame):
    """driver_finish_rate_last10 must be in [0, 1]."""
    valid = features["driver_finish_rate_last10"].dropna()
    assert (valid >= 0.0).all()
    assert (valid <= 1.0).all()


def test_sample_weights_sum_positive(features: pd.DataFrame):
    """Sample weights must be positive and normalised to <= 1."""
    w = features["sample_weight"]
    assert (w > 0).all()
    assert w.max() <= 1.0 + 1e-9


def test_regulation_era_flag_is_correct(features: pd.DataFrame):
    """regulation_era should be 1 only for years >= NEW_ERA_YEAR."""
    year_to_era = features.groupby("Year")["regulation_era"].first()
    for year, era in year_to_era.items():
        expected = 1 if year >= config.NEW_ERA_YEAR else 0
        assert era == expected, f"Year {year}: era={era}, expected {expected}"


def test_position_trend_makes_sense(features: pd.DataFrame):
    """driver_pos_trend = avg3 - avg5; verify."""
    valid = features.dropna(subset=["driver_avg_pos_last3", "driver_avg_pos_last5",
                                     "driver_pos_trend"])
    diff = valid["driver_avg_pos_last3"] - valid["driver_avg_pos_last5"]
    assert np.allclose(valid["driver_pos_trend"], diff)
