"""
F1 Race Predictor — main entry point.

Runs the full pipeline end-to-end:
    1. extract historical data    (skipped if cache exists, unless --refresh)
    2. feature engineering
    3. model training              (skipped if model exists, unless --retrain)
    4. predict next race
    5. generate interactive HTML report

Usage:
    python pipeline.py                  # full run with sensible caching
    python pipeline.py --refresh        # re-pull all historical data
    python pipeline.py --retrain        # re-train the model from scratch
    python pipeline.py --skip-eda       # skip the EDA figures
    python pipeline.py --rain           # mark next race as wet

After the run finishes, the HTML report path is printed and (on most desktops)
opened in the default browser.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import webbrowser
from pathlib import Path

from src import config


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )


def _step(title: str) -> None:
    print()
    print("#" * 72)
    print(f"# {title}")
    print("#" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="F1 Race Predictor end-to-end pipeline")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-extract historical data from FastF1 (slow)")
    parser.add_argument("--retrain", action="store_true",
                        help="Re-train the model from scratch")
    parser.add_argument("--skip-eda", action="store_true",
                        help="Skip the EDA figure generation")
    parser.add_argument("--rain", action="store_true",
                        help="Set rain flag for the next race")
    parser.add_argument("--no-open", action="store_true",
                        help="Do not auto-open the HTML report")
    args = parser.parse_args()

    _setup_logging()

    # 1. Extract historical data
    if args.refresh or not config.HISTORICAL_FILE.exists():
        _step("STEP 1/5: Extract historical data from FastF1")
        from src.extract_data import main as extract_main
        extract_main()

        # If extract bailed out (rate limit, network error, etc.) the
        # downstream steps will fail in confusing ways. Stop cleanly here.
        if not config.HISTORICAL_FILE.exists():
            print("\n# Extraction did not produce any data. Aborting pipeline.")
            print("# (See the message above for what to do next.)")
            return 1
    else:
        print(f"# Skipping data extraction (cache: {config.HISTORICAL_FILE}); use --refresh to redo")

    # 2. Feature engineering — always re-run, it's fast
    _step("STEP 2/5: Feature engineering")
    from src.feature_engineering import main as fe_main
    fe_main()

    # 3. EDA (optional)
    if not args.skip_eda:
        _step("STEP 3/5: Exploratory data analysis")
        from src.eda import main as eda_main
        try:
            eda_main()
        except Exception as exc:
            print(f"  (EDA failed: {exc} — continuing)")

    # 4. Train (skip if model exists unless --retrain)
    if args.retrain or not config.MODEL_FILE.exists():
        _step("STEP 4/5: Train model")
        from src.train_model import main as train_main
        train_main()
    else:
        print(f"# Skipping training (model: {config.MODEL_FILE}); use --retrain to redo")

    # 5. Predict + render HTML
    _step("STEP 5/5: Predict next race and generate HTML report")
    from src.predict import predict_next_race
    from src.visualizer import render_html

    pred = predict_next_race(is_rain=1 if args.rain else 0)
    html_path = render_html(pred)

    print()
    print("=" * 72)
    print(" DONE")
    print("=" * 72)
    print(f"  Race:        {pred.next_race.name}")
    print(f"  Date:        {pred.next_race.date.strftime('%Y-%m-%d')}")
    print(f"  Mode:        {pred.mode.upper()}")
    print(f"  Top 5:       {' -> '.join(d for _, d in pred.predicted_top5)}")
    print(f"  Report:      {html_path}")
    print()

    if not args.no_open:
        with contextlib.suppress(Exception):
            webbrowser.open(f"file://{Path(html_path).resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
