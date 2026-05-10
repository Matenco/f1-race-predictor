"""Tests for prediction-time logic.

Particularly important: the bug fix for `build_prediction_features`.
The previous version applied a heuristic `(2 * old_avg3 + last_pos) / 3`
to "update" shifted features, which is mathematically wrong. The fix
recomputes from raw history. We verify the fix by checking that the
rolling average for the upcoming race equals the actual mean of the
driver's last 3 finishes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.feature_engineering import build_features
from src.next_race import NextRace
from src.predict import (
    _compute_driver_features,
    build_prediction_features,
    monte_carlo_simulation,
)
from src.train_model import (
    baseline_grid_top5,
    compute_scoring,
    hungarian_optimal_assignment,
)


# =============================================================================
#  Hungarian algorithm
# =============================================================================
def test_hungarian_returns_5_unique_drivers():
    """Output is exactly 5 (position, driver) pairs with unique drivers."""
    n_drivers = 20
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(20), size=n_drivers)
    drivers = [f"D{i:02d}" for i in range(n_drivers)]

    top5 = hungarian_optimal_assignment(probs, drivers)
    assert len(top5) == 5
    positions = [p for p, _ in top5]
    selected = [d for _, d in top5]
    assert positions == [1, 2, 3, 4, 5]
    assert len(set(selected)) == 5


def test_hungarian_picks_high_probability_driver_for_p1():
    """If one driver has overwhelmingly high P(P1), they should be assigned P1."""
    probs = np.full((10, 20), 0.05)
    probs[3, 0] = 0.95   # driver 3 has 95% P(P1)
    probs[3, 1:] = 0.05 / 19
    probs = probs / probs.sum(axis=1, keepdims=True)
    drivers = [f"D{i:02d}" for i in range(10)]

    top5 = hungarian_optimal_assignment(probs, drivers)
    assert dict(top5)[1] == "D03"


def test_hungarian_beats_or_matches_greedy_on_scoring():
    """On many random matrices, Hungarian's score should be >= greedy-by-P(top5)."""
    rng = np.random.default_rng(1)
    n_better = 0
    n_total = 30
    for _ in range(n_total):
        n_drivers = 20
        probs = rng.dirichlet(np.ones(20), size=n_drivers)
        drivers = [f"D{i:02d}" for i in range(n_drivers)]
        # Sample one true outcome
        order = rng.permutation(n_drivers)
        actual = {drivers[i]: int(np.where(order == i)[0][0]) + 1 for i in range(n_drivers)}

        # Hungarian
        h_top5 = hungarian_optimal_assignment(probs, drivers)
        h_score = compute_scoring(h_top5, actual)

        # Greedy: top 5 by P(top5), positions 1..5 in P(top5) order
        p_top5 = probs[:, :5].sum(axis=1)
        greedy_idx = np.argsort(p_top5)[::-1][:5]
        greedy_top5 = [(i + 1, drivers[idx]) for i, idx in enumerate(greedy_idx)]
        g_score = compute_scoring(greedy_top5, actual)

        if h_score >= g_score:
            n_better += 1
    # On average Hungarian should match or beat greedy in most random samples.
    # Allow a small fraction of ties or losses (sampled outcome is noisy).
    assert n_better >= int(n_total * 0.7), (
        f"Hungarian only matched/beat greedy in {n_better}/{n_total}"
    )


# =============================================================================
#  Scoring rule
# =============================================================================
def test_scoring_exact_hits():
    """All 5 exact hits = 10 points."""
    pred = [(1, "A"), (2, "B"), (3, "C"), (4, "D"), (5, "E")]
    actual = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    assert compute_scoring(pred, actual) == 10


def test_scoring_in_top5_wrong_position():
    """All 5 in top 5 but at wrong positions = 5 points."""
    pred = [(1, "A"), (2, "B"), (3, "C"), (4, "D"), (5, "E")]
    actual = {"A": 5, "B": 1, "C": 2, "D": 3, "E": 4}
    assert compute_scoring(pred, actual) == 5


def test_scoring_complete_miss():
    """No predicted drivers in actual top 5 = 0 points."""
    pred = [(1, "A"), (2, "B"), (3, "C"), (4, "D"), (5, "E")]
    actual = {"A": 6, "B": 7, "C": 8, "D": 9, "E": 10,
              "X": 1, "Y": 2, "Z": 3, "W": 4, "V": 5}
    assert compute_scoring(pred, actual) == 0


# =============================================================================
#  Monte Carlo simulation
# =============================================================================
def test_monte_carlo_win_probs_sum_to_one():
    """Across all drivers, P(win) must sum to ~1 (one driver wins each sim)."""
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(20), size=20)
    drivers = [f"D{i:02d}" for i in range(20)]
    mc = monte_carlo_simulation(probs, drivers, n_simulations=2000, seed=0)
    assert abs(mc["win_prob"].sum() - 1.0) < 0.01


def test_monte_carlo_top5_probs_sum_to_five():
    """Across all drivers, P(top5) must sum to ~5 (5 drivers in top 5 each sim)."""
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(20), size=20)
    drivers = [f"D{i:02d}" for i in range(20)]
    mc = monte_carlo_simulation(probs, drivers, n_simulations=2000, seed=0)
    assert abs(mc["top5_prob"].sum() - 5.0) < 0.05


def test_monte_carlo_high_p1_driver_wins_most(synthetic_history: pd.DataFrame):
    """A driver with overwhelmingly concentrated P(P1) probability should win
    the majority of simulations.

    Note: even with P(P1)=0.98 for one driver, ~5%-uniform other drivers
    will collide on position 0 in some sims, so the dominant driver doesn't
    win 98% of the time — they win ~80%+ depending on tie-breaking. We test
    a defensible threshold (>50%) that is way above uniform (1/20 = 5%).
    """
    n = 20
    probs = np.full((n, 20), 1.0 / 20)
    # Driver 5 has 98% P(P1), 0.1% on each other position
    probs[5, :] = 0.001
    probs[5, 0] = 1.0 - 0.001 * 19
    probs = probs / probs.sum(axis=1, keepdims=True)
    drivers = [f"D{i:02d}" for i in range(n)]
    mc = monte_carlo_simulation(probs, drivers, n_simulations=2000, seed=42)
    win_leader_idx = int(np.argmax(mc["win_prob"]))
    assert win_leader_idx == 5
    assert mc["win_prob"][5] > 0.50, (
        f"Dominant driver only won {mc['win_prob'][5]:.1%} of simulations"
    )


# =============================================================================
#  Bug fix: prediction-time feature computation is mathematically correct
# =============================================================================
def test_compute_driver_features_uses_actual_recent_finishes(synthetic_history):
    """The driver_avg_pos_last3 for the upcoming race must equal the mean
    of the driver's actual last 3 finishing positions — NOT the old buggy
    `(2 * shifted_avg + last_pos) / 3` heuristic.
    """
    df = build_features(synthetic_history)
    # Pick a driver with a long history
    driver = "AAA"
    driver_history = df[df["Abbreviation"] == driver].sort_values("Date")
    feats = _compute_driver_features(driver_history, target_circuit="Miami")

    actual_last3 = (
        driver_history.dropna(subset=["Position"])["Position"].tail(3).mean()
    )
    assert np.isclose(feats["driver_avg_pos_last3"], actual_last3), (
        f"avg3 mismatch: got {feats['driver_avg_pos_last3']}, "
        f"expected {actual_last3}"
    )


def test_compute_driver_features_track_avg_uses_only_target_track(synthetic_history):
    """driver_track_avg should equal the driver's actual mean finish at
    the target circuit — and NO other circuits should leak in.
    """
    df = build_features(synthetic_history)
    driver = "BBA"
    target = "Singapore"
    driver_history = df[df["Abbreviation"] == driver].sort_values("Date")
    feats = _compute_driver_features(driver_history, target_circuit=target)

    track_history = driver_history[driver_history["Circuit"] == target].dropna(subset=["Position"])
    if len(track_history) > 0:
        expected = track_history["Position"].mean()
        assert np.isclose(feats["driver_track_avg"], expected)
        assert feats["driver_track_count"] == float(len(track_history))


def test_build_prediction_features_yields_one_row_per_driver(synthetic_history):
    """Output should have one row per active driver (most recent season)."""
    df = build_features(synthetic_history)
    next_race = NextRace(
        year=2024, round_number=8, name="Test GP",
        circuit="Miami", country="TBD",
        date=pd.Timestamp("2024-08-01"),
        has_quali_results=False,
    )
    next_race_df = build_prediction_features(df, next_race)
    n_active = df[df["Year"] == df["Year"].max()]["Abbreviation"].nunique()
    assert len(next_race_df) == n_active

    # All FEATURE_COLS must be present (XGBoost will reject otherwise)
    for col in config.FEATURE_COLS:
        assert col in next_race_df.columns


def test_build_prediction_features_blind_mode_estimates_grid(synthetic_history):
    """In blind mode (no grid_positions), GridPosition should be filled
    from form, not left NaN."""
    df = build_features(synthetic_history)
    next_race = NextRace(
        year=2024, round_number=8, name="Test GP",
        circuit="Miami", country="TBD",
        date=pd.Timestamp("2024-08-01"),
        has_quali_results=False,
    )
    next_race_df = build_prediction_features(df, next_race, grid_positions=None)
    assert next_race_df["GridPosition"].notna().all()


def test_build_prediction_features_informed_mode_uses_provided_grid(synthetic_history):
    """In informed mode, the supplied grid positions should appear verbatim."""
    df = build_features(synthetic_history)
    next_race = NextRace(
        year=2024, round_number=8, name="Test GP",
        circuit="Miami", country="TBD",
        date=pd.Timestamp("2024-08-01"),
        has_quali_results=True,
    )
    grid = {"AAA": 1, "AAB": 2, "BBA": 3}
    next_race_df = build_prediction_features(df, next_race, grid_positions=grid)
    for drv, pos in grid.items():
        row = next_race_df[next_race_df["Abbreviation"] == drv]
        if len(row) > 0:
            assert row["GridPosition"].iloc[0] == pos


# =============================================================================
#  Baselines (sanity)
# =============================================================================
def test_baseline_grid_top5_picks_grid_1_through_5():
    df = pd.DataFrame({
        "Abbreviation": [f"D{i}" for i in range(10)],
        "GridPosition": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })
    top5 = baseline_grid_top5(df)
    assert top5 == [(1, "D0"), (2, "D1"), (3, "D2"), (4, "D3"), (5, "D4")]
