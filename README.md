# F1 Race Predictor

ML pipeline that predicts the **top 5 finishers of the next Formula 1 Grand Prix** using XGBoost ranking models, validated with walk-forward historical backtests.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

> One command, automatic next-race detection, interactive HTML output. No hardcoded race names, no manual config — runs today, runs in 6 months.

---

## What it does

```bash
python pipeline.py
```

1. Pulls historical race data (2022–current) from the official F1 API via FastF1
2. Engineers 18 leakage-free features (driver form, team strength, circuit history, qualifying delta, weather)
3. Trains XGBoost models with time-series cross-validation
4. Auto-detects the next upcoming Grand Prix from the F1 calendar
5. Runs Monte Carlo simulation (10 000 iterations) on the model's probability matrix
6. Generates an interactive standalone HTML report with Plotly

Open the HTML in any browser. No server, no dependencies at view-time.

## Why it's interesting

**Grid-aware model instead of naive grid copy.** The starting grid is a very strong baseline, so the production model starts from the grid's P1-P7 candidates and blends a top-5 probability model with a lightly weighted `rank:ndcg` XGBoost ranker. This allows controlled P6/P7 swaps without turning the prediction into a free-form guess.

**Position and ranker models for diagnostics.** The pipeline still trains a multiclass XGBoost position model and an XGBoost ranker. They are reported in validation for transparency, but the deployed selection is driven mainly by the dedicated top-5 model because that combination gives the best walk-forward points score.

**Time-decay sample weights with regulation-era boost.** Older races count less (half-life: 365 days), and the new 2026 regulation era gets a 1.5× weight multiplier so the model doesn't anchor on obsolete car physics.

**Leakage-free rolling features.** Every per-driver and per-team rolling statistic uses `groupby().shift(1).rolling()` so the feature for race N is computed only from races 1…N-1. Walk-forward validation recomputes time-decay sample weights relative to the race being predicted, so a 2025 backtest is weighted as if the model were standing in 2025.

**Generalises to any track.** Circuit-history features are parameterised through a `CIRCUIT_FAMILIES` mapping (Miami → [Jeddah, Las Vegas, Baku, Melbourne, Singapore]), so the same model predicts Monaco, Monza, or any new venue.

## Validation results

Time-series cross-validation across 28 held-out races in 2025–2026, after filling the missing 2024–2025 races and filtering incomplete FastF1 result pulls. Each race is held out, and the model is trained only on earlier races.

| Method                         | Mean points/race | Std  | Notes |
|--------------------------------|------------------|------|-------|
| **XGBoost top-5/ranker + grid P1-P7** | **5.86** | 1.88 | deployed strategy |
| XGBoost position + Hungarian   | 5.25             | 1.78 | old production approach |
| XGBoost top-5 ensemble         | 5.39             | 1.75 | tested, not deployed |
| Baseline: starting grid        | 5.75             | 1.90 | strongest points baseline |
| Baseline: recent form          | 4.25             | 1.14 | secondary baseline |

On the full 96-race dataset, the deployed strategy beats the grid baseline in the project-game points metric (5.86 vs 5.75), improves mean top-5 set hits (4.00 vs 3.93), and improves ranking quality (`NDCG@5`: 0.859 vs 0.853). Race-by-race, it wins 4 validation races, ties 23, and loses 1.

## Pipeline architecture

```
extract_data.py        → raw race results, qualifying, weather (FastF1 API)
feature_engineering.py → 18 features, leakage-free, parameterised by circuit
train_model.py         → time-series CV + position/top5/ranker models + metadata
next_race.py           → auto-detects next upcoming GP from FastF1 calendar
predict.py             → builds features for the upcoming race, runs model + Monte Carlo
visualizer.py          → standalone HTML report (Plotly)
pipeline.py            → orchestrator: runs everything end-to-end
```

## Output preview

The HTML report contains:

- **Top 5 prediction card** with predicted positions and per-driver `P(top 5)` confidence
- **Probability heatmap** showing `P(position j | driver i)` for the predicted top 5 drivers
- **Monte Carlo bar chart** comparing `P(win)` / `P(podium)` / `P(top 5)` across drivers
- **Most likely top-5 combinations** table (five most frequent permutations from 10k simulations)
- **Feature importance** plot from the trained XGBoost model

All charts are interactive: hover for exact values, zoom, export as PNG.

## Tech stack

- **Data**: FastF1 (official F1 timing data API)
- **ML**: XGBoost, scikit-learn, scipy.optimize.linear_sum_assignment (Hungarian)
- **Tuning**: Optuna with TPE sampler
- **Visualisation**: Plotly (interactive standalone HTML)
- **Data wrangling**: pandas, numpy

## Quick start

```bash
git clone https://github.com/Matenco/f1-race-predictor.git
cd f1-race-predictor
pip install -e .
python pipeline.py
```

First run can take a while because FastF1/Jolpica rate limits historical result pulls. Re-running is faster because FastF1 caches downloaded sessions.

### Configuration

Edit `src/config.py` to adjust:

- `SEASONS` — which seasons to pull (default: 2022–current)
- `N_OPTUNA_TRIALS` — set to 0 to skip tuning and use saved best params
- `N_SIMULATIONS` — Monte Carlo iteration count (default: 10 000)
- `WEIGHT_HALF_LIFE` — sample weight decay (default: 365 days)

### Skipping the heavy steps

```bash
python -m src.predict          # uses cached features + saved model
python -m src.next_race        # just shows what the next race is
```

## Project structure

```
f1-race-predictor/
├── README.md
├── .gitignore
├── pipeline.py              ← main entry point
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── circuit_metadata.py
│   ├── extract_data.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── next_race.py
│   ├── train_model.py
│   ├── predict.py
│   └── visualizer.py
├── data/
│   └── processed/           ← f1_historical.csv, f1_features.csv (gitignored)
├── model/                   ← saved XGBoost model + metadata (gitignored)
├── predictions/             ← generated HTML reports (gitignored)
└── figures/                 ← static PNG charts from EDA + training
```

## Limitations and honest caveats

- **Historical pulls can be incomplete under API rate limits**. The pipeline now filters incomplete races instead of training on rows with missing race results. If a refresh hits rate limits, rerun later to add more completed races.
- **2026 regulations are new**. The model has only a handful of races' worth of 2026 data. Predictions for the first 2026 races lean heavily on historical priors and may underweight recent pace shifts. Sample weights mitigate but don't eliminate this.
- **Driver-team changes mid-season** (substitutions, contract moves) are handled by using the latest known team affiliation but feature continuity is imperfect.
- **Wet-race specifics** are coarse: a single binary `is_rain` flag, no tyre-strategy modelling.
- **No safety-car or DNF causality model**. The probability matrix implicitly captures DNF rates per driver, but doesn't condition on race incidents.

This is a portfolio project, not a betting tool.

## License

MIT — see `LICENSE`.

## Acknowledgements

- [FastF1](https://github.com/theOehrly/Fast-F1) for the F1 timing data API
- F1's open data ecosystem
