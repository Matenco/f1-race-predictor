"""Generate a sample HTML with the new "last race scorecard" section
populated, so the user can see all the visual elements at once.

NOTE: this generates synthetic data — driver names AAA/BBA/etc are fake
abbreviations from the test fixture, NOT the real grid. The real pipeline
on FastF1 data will show VER, LEC, ANT, NOR, etc.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.next_race import NextRace


def main():
    # Run the integration test pipeline first (training + features)
    sys.path.insert(0, str(PROJECT_ROOT / "tests"))
    from tests.test_integration import make_synthetic_data

    print("[1/4] Generating synthetic data...")
    df = make_synthetic_data(n_seasons=4, races_per_season=20, n_drivers=20)
    df.to_csv(config.HISTORICAL_FILE, index=False)

    print("[2/4] Running feature engineering and training...")
    config.N_OPTUNA_TRIALS = 0
    config.USE_SAVED_OPTUNA_PARAMS = False
    if config.OPTUNA_BEST_PARAMS_FILE.exists():
        config.OPTUNA_BEST_PARAMS_FILE.unlink()
    from src.feature_engineering import main as fe_main
    from src.train_model import main as train_main
    fe_main()
    train_main()

    print("[3/4] Pre-populating prediction history with a 'previous' prediction...")
    # Inject a hypothetical prior race prediction
    history = [{
        "predicted_at": "2026-06-01T00:00:00Z",
        "race": {
            "name": "Spanish Grand Prix",
            "year": 2026, "round": 7,
            "circuit": "Barcelona",
            "date": "2026-06-07",
        },
        "mode": "informed",
        "predicted_top5": [[1, "AAA"], [2, "BBA"], [3, "GGA"], [4, "DDA"], [5, "EEA"]],
    }]
    config.PREDICTION_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.PREDICTION_HISTORY_FILE.write_text(json.dumps(history, indent=2))

    print("[4/4] Running prediction with mocked next race + mocked actual results...")
    fake_next_race = NextRace(
        year=2026, round_number=8,
        name="Canadian Grand Prix",
        circuit="Montréal",
        country="Canada",
        date=pd.Timestamp("2026-06-14"),
        has_quali_results=False,
    )
    # Mock both the next-race detector AND the historical actuals fetch
    fake_actual = {
        "AAA": 1, "GGA": 2, "BBA": 3, "FFF": 4, "ZZA": 5,
        "DDA": 6, "EEA": 7, "ZZB": 8, "TTA": 9, "CCC": 10,
    }
    with patch("src.predict.get_next_race", return_value=fake_next_race), \
         patch("src.predict._fetch_actual_race_results", return_value=fake_actual):
        from src.predict import predict_next_race
        from src.visualizer import render_html
        pred = predict_next_race(is_rain=0)
        html_path = render_html(pred)

    print(f"\nSample HTML: {html_path}")
    print(f"Size: {html_path.stat().st_size:,} bytes")
    print(f"\nLast race recap: {pred.last_race_recap.score}/10 vs grid {pred.last_race_recap.grid_baseline_score}/10")
    print(f"Validation history points: {len(pred.validation_history)}")


if __name__ == "__main__":
    main()
