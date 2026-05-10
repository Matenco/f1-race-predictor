"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src/ importable from tests/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def synthetic_history() -> pd.DataFrame:
    """A small but realistic historical dataset for testing.

    20 drivers, 4 teams, 3 seasons of 12 races each. Drivers have
    consistent team affiliations and per-team strength priors. Used by the
    feature-engineering and prediction tests.
    """
    rng = np.random.default_rng(42)
    teams_drivers = {
        "Alpha":   ["AAA", "AAB"],
        "Beta":    ["BBA", "BBB"],
        "Gamma":   ["GGA", "GGB"],
        "Delta":   ["DDA", "DDB"],
        "Epsilon": ["EEA", "EEB"],
        "Zeta":    ["ZZA", "ZZB"],
        "Eta":     ["HHA", "HHB"],
        "Theta":   ["TTA", "TTB"],
        "Iota":    ["IIA", "IIB"],
        "Kappa":   ["KKA", "KKB"],
    }
    team_strength = {t: 2.0 + 1.5 * i for i, t in enumerate(teams_drivers)}

    circuits = ["Miami", "Monaco", "Singapore", "Baku", "Jeddah",
                "Las Vegas", "Melbourne", "Barcelona", "Hungaroring",
                "Suzuka", "Monza", "Spa-Francorchamps"]

    rows = []
    for season_idx in range(3):
        year = 2022 + season_idx
        for r in range(1, 13):
            circuit = circuits[(r - 1) % len(circuits)]
            race_date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=21 * r)
            grid_assignments = []
            for team, drivers in teams_drivers.items():
                for drv in drivers:
                    perf = team_strength[team] + rng.normal(0, 1.2)
                    grid_assignments.append((drv, team, perf))
            grid_assignments.sort(key=lambda x: x[2])
            new_rows = []
            for grid_pos, (drv, team, quali) in enumerate(grid_assignments, 1):
                race_perf = quali + rng.normal(0, 1.5)
                new_rows.append({
                    "Abbreviation": drv,
                    "FullName": drv,
                    "TeamName": team,
                    "GridPosition": float(grid_pos),
                    "_perf": race_perf,
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
            new_rows.sort(key=lambda r_: r_["_perf"])
            for finish_pos, r_ in enumerate(new_rows, 1):
                if r_["Status"] == "Finished":
                    r_["Position"] = float(finish_pos)
                    r_["Finished"] = True
                else:
                    r_["Position"] = np.nan
                    r_["Finished"] = False
                del r_["_perf"]
            rows.extend(new_rows)

    return pd.DataFrame(rows)
