"""
Integration test: synthetic data through the full pipeline.

Creates a small fake historical dataset (no FastF1 API call), runs feature
engineering, trains a model, and generates a prediction + HTML.

The point is to catch runtime bugs in the refactored pipeline before
you run it on real data.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.circuit_metadata import CIRCUIT_FAMILIES
from src.next_race import NextRace


def make_synthetic_data(n_seasons: int = 4, races_per_season: int = 22,
                        n_drivers: int = 20, seed: int = 42) -> pd.DataFrame:
    """Build a believable synthetic historical dataset."""
    rng = np.random.default_rng(seed)

    teams_drivers = {
        "Red Bull Racing":    ["VER", "PER"],
        "Ferrari":            ["LEC", "SAI"],
        "Mercedes":           ["HAM", "RUS"],
        "McLaren":            ["NOR", "PIA"],
        "Aston Martin":       ["ALO", "STR"],
        "Alpine":             ["GAS", "OCO"],
        "Williams":           ["ALB", "SAR"],
        "RB":                 ["TSU", "RIC"],
        "Kick Sauber":        ["BOT", "ZHO"],
        "Haas F1 Team":       ["MAG", "HUL"],
    }

    # Team strength priors (lower = better)
    team_strength = {
        "Red Bull Racing": 2.5, "Ferrari": 4, "Mercedes": 5, "McLaren": 3.5,
        "Aston Martin": 8, "Alpine": 11, "Williams": 13, "RB": 12,
        "Kick Sauber": 16, "Haas F1 Team": 14,
    }

    circuit_pool = list(CIRCUIT_FAMILIES.keys())[:races_per_season]
    rows = []
    for season_idx in range(n_seasons):
        year = 2022 + season_idx
        for r in range(1, races_per_season + 1):
            circuit = circuit_pool[(r - 1) % len(circuit_pool)]
            race_date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=14 * r)
            grid_assignments = []
            for team, drivers in teams_drivers.items():
                base = team_strength[team]
                for drv in drivers:
                    quali_perf = base + rng.normal(0, 1.5)
                    grid_assignments.append((drv, team, quali_perf))
            grid_assignments.sort(key=lambda x: x[2])
            for grid_pos, (drv, team, quali) in enumerate(grid_assignments, 1):
                race_perf = quali + rng.normal(0, 2.0)
                # Re-rank for finishing position
                # We'll rank everyone after the loop below
                rows.append({
                    "DriverNumber": int(grid_pos),
                    "Abbreviation": drv,
                    "FullName": drv,
                    "TeamName": team,
                    "GridPosition": float(grid_pos),
                    "_race_perf": race_perf,
                    "Status": "Finished" if rng.random() > 0.05 else "Engine",
                    "Points": 0.0,
                    "QualiPosition": float(grid_pos),
                    "QualiTime_s": 80.0 + quali / 5.0,
                    "AvgAirTemp": 25.0,
                    "AvgTrackTemp": 35.0,
                    "Humidity": 50.0,
                    "Rainfall": False,
                    "Year": year,
                    "Round": r,
                    "GP": f"{circuit} Grand Prix",
                    "Circuit": circuit,
                    "Country": "TBD",
                    "Date": race_date,
                })
            # Rank within this race to get finish positions
            race_rows = [r_ for r_ in rows if r_["Year"] == year and r_["Round"] == r]
            race_rows.sort(key=lambda r_: r_["_race_perf"])
            for finish_pos, r_ in enumerate(race_rows, 1):
                if r_["Status"] == "Finished":
                    r_["Position"] = float(finish_pos)
                    r_["Finished"] = True
                else:
                    r_["Position"] = np.nan
                    r_["Finished"] = False
                del r_["_race_perf"]

    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print(" INTEGRATION TEST")
    print("=" * 70)

    # 1. Synthetic data
    print("\n[1/5] Generating synthetic historical data...")
    df = make_synthetic_data(n_seasons=4, races_per_season=20, n_drivers=20)
    df.to_csv(config.HISTORICAL_FILE, index=False)
    print(f"      Wrote {len(df)} rows to {config.HISTORICAL_FILE}")
    print(f"      Drivers: {df['Abbreviation'].nunique()}, "
          f"races: {df.groupby(['Year', 'Round']).ngroups}")

    # 2. Feature engineering
    print("\n[2/5] Running feature engineering...")
    from src.feature_engineering import main as fe_main
    fe_main()
    feat_df = pd.read_csv(config.FEATURES_FILE)

    # Check all FEATURE_COLS are present
    missing = [c for c in config.FEATURE_COLS if c not in feat_df.columns]
    assert not missing, f"Missing feature cols: {missing}"
    print(f"      All {len(config.FEATURE_COLS)} feature cols present.")

    # 3. Train model (skip Optuna for speed)
    print("\n[3/5] Training model (Optuna disabled for speed)...")
    config.N_OPTUNA_TRIALS = 0
    config.USE_SAVED_OPTUNA_PARAMS = False
    # Make sure no stale Optuna params interfere
    if config.OPTUNA_BEST_PARAMS_FILE.exists():
        config.OPTUNA_BEST_PARAMS_FILE.unlink()
    from src.train_model import main as train_main
    train_main()
    assert config.MODEL_FILE.exists(), "Model file not saved"
    assert config.MODEL_META_FILE.exists(), "Model metadata not saved"
    print("      Model + metadata saved.")

    # 4. Predict — mock get_next_race so we don't hit FastF1
    print("\n[4/5] Running prediction (mocked next race)...")
    fake_next_race = NextRace(
        year=2026,
        round_number=8,
        name="Canadian Grand Prix",
        circuit="Montréal",
        country="Canada",
        date=pd.Timestamp("2026-06-14"),
        has_quali_results=False,
    )
    with patch("src.predict.get_next_race", return_value=fake_next_race):
        from src.predict import predict_next_race
        pred = predict_next_race(is_rain=0)

    print("      Predicted top 5:")
    for pos, drv in pred.predicted_top5:
        d = next(d for d in pred.drivers if d.abbreviation == drv)
        print(f"        P{pos}  {drv:>4s}  {d.team:>22s}  P(top5)={d.p_top5:.0%}")

    # Sanity checks on prediction output
    assert len(pred.predicted_top5) == 5
    assert pred.sixth_driver is not None
    assert pred.n_simulations > 0
    assert all(0 <= d.p_top5 <= 1 for d in pred.drivers)
    # Win probabilities should sum to ~1 across all drivers
    total_p_win = sum(d.p_win for d in pred.drivers)
    assert abs(total_p_win - 1.0) < 0.01, f"Win probs don't sum to 1: {total_p_win}"
    print(f"      Sanity checks passed (win probs sum = {total_p_win:.3f})")

    # 5. Render HTML
    print("\n[5/5] Rendering interactive HTML report...")
    from src.visualizer import render_html
    html_path = render_html(pred)
    assert html_path.exists()
    html_content = html_path.read_text()
    # Spot-check key elements are in the HTML
    assert "Canadian Grand Prix" in html_content
    assert "Plotly" in html_content or "plotly" in html_content
    assert pred.predicted_top5[0][1] in html_content  # winner abbrev
    assert "Hungarian" in html_content
    assert "How well does this model actually work" in html_content
    print(f"      HTML written ({len(html_content):,} bytes)")
    print(f"      Path: {html_path}")

    # 6. Verify recap mechanism: simulate a previous prediction by injecting
    #    one into history, mock actual race results, run again — recap should
    #    appear in the new HTML.
    print("\n[6/6] Verifying last-race recap mechanism...")
    from src.predict import _build_last_race_recap, _save_prediction_to_history

    # Save the current prediction to history first
    _save_prediction_to_history(pred)

    # Now create a synthetic "previous race" prediction in history
    history = json.loads(config.PREDICTION_HISTORY_FILE.read_text())
    # Inject a hypothetical prior prediction for round 7 (before our round 8)
    history.insert(0, {
        "predicted_at": "2026-06-01T00:00:00Z",
        "race": {
            "name": "Spanish Grand Prix",
            "year": 2026, "round": 7,
            "circuit": "Barcelona",
            "date": "2026-06-07",
        },
        "mode": "informed",
        "predicted_top5": [[1, "AAA"], [2, "BBA"], [3, "GGA"], [4, "DDA"], [5, "EEA"]],
    })
    config.PREDICTION_HISTORY_FILE.write_text(json.dumps(history, indent=2))

    # Mock _fetch_actual_race_results so we don't need FastF1
    fake_actual = {"AAA": 1, "GGA": 2, "BBA": 3, "FFF": 4, "DDA": 6, "EEA": 7,
                    "ZZZ": 5}
    with patch("src.predict._fetch_actual_race_results", return_value=fake_actual):
        # Also mock fastf1 inside _build_last_race_recap (for grid baseline)
        recap = _build_last_race_recap(fake_next_race)
    assert recap is not None, "Recap should have been built"
    assert recap.race_name == "Spanish Grand Prix"
    # AAA was predicted P1 and finished P1 → exact hit
    assert "AAA" in recap.exact_hits
    # GGA was predicted P3 but finished P2 → in-top-5 hit
    assert "GGA" in recap.in_top5_hits
    # EEA was predicted P5 but finished P7 → miss
    assert "EEA" in recap.misses
    # Score: AAA (+2 exact) + BBA (P2, actual P3, +1) + GGA (P3, actual P2, +1)
    # + DDA (P4, actual P6, 0) + EEA (P5, actual P7, 0) = 4
    assert recap.score == 4, f"Expected score 4, got {recap.score}"
    print(f"      Recap built: {recap.race_name}, score {recap.score}/10")
    print(f"      Exact hits: {recap.exact_hits}")
    print(f"      Partial hits: {recap.in_top5_hits}")
    print(f"      Misses: {recap.misses}")

    # Show validation summary
    with open(config.MODEL_META_FILE) as f:
        meta = json.load(f)
    val = meta.get("validation_summary")
    if val:
        print("\n      Model validation (synthetic data):")
        print(f"        XGBoost+Hungarian: {val['xgboost_hungarian_mean']:.2f} ± {val['xgboost_hungarian_std']:.2f}")
        print(f"        Grid baseline:     {val['baseline_grid_mean']:.2f}")
        print(f"        Form baseline:     {val['baseline_form_mean']:.2f}")

    print("\n" + "=" * 70)
    print(" ALL INTEGRATION TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
