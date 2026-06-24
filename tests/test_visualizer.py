"""Tests for HTML report rendering."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.next_race import NextRace
from src.predict import DriverPrediction, RacePrediction
from src.visualizer import render_html


def test_render_html_shows_prediction_timestamp(tmp_path):
    drivers = [
        DriverPrediction(
            abbreviation=abbr,
            full_name=abbr,
            team="Test Team",
            grid_position=float(i),
            p_top5=0.9 - i * 0.05,
            p_win=0.2,
            p_podium=0.4,
            prob_per_position=[0.2, 0.2, 0.2, 0.2, 0.1],
        )
        for i, abbr in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"], 1)
    ]
    pred = RacePrediction(
        next_race=NextRace(
            year=2026,
            round_number=8,
            name="Austrian Grand Prix",
            circuit="Spielberg",
            country="Austria",
            date=pd.Timestamp("2026-06-28"),
            has_quali_results=False,
        ),
        mode="blind",
        predicted_top5=[(i, d.abbreviation) for i, d in enumerate(drivers, 1)],
        sixth_driver=None,
        drivers=drivers,
        most_likely_combos=[(tuple(d.abbreviation for d in drivers), 1)],
        n_simulations=1,
        predicted_at=datetime(2026, 6, 24, 20, 30),
    )

    html_path = render_html(pred, tmp_path / "prediction.html")
    html = html_path.read_text()

    assert "Prediction made: 2026-06-24 20:30 UTC" in html
