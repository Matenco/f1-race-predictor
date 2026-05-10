# F1 Race Predictor

ML pipeline that predicts the **top 5 finishers of the next Formula 1 Grand Prix** using XGBoost + the Hungarian algorithm, validated on 4 seasons of historical data.

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
3. Trains XGBoost with Optuna-tuned hyperparameters and time-series cross-validation
4. Auto-detects the next upcoming Grand Prix from the F1 calendar
5. Runs Monte Carlo simulation (10 000 iterations) on the model's probability matrix
6. Generates an interactive standalone HTML report with Plotly

Open the HTML in any browser. No server, no dependencies at view-time.

## Why it's interesting

**Hungarian assignment instead of greedy top-5.** The model outputs `P(driver i finishes in position j)` for every driver × position pair. Naively picking the top 5 by `P(top5)` doesn't optimise the scoring rule used by F1 prediction games (2 points for an exact-position hit, 1 point for an in-top-5 miss). The Hungarian algorithm finds the assignment of 5 drivers to 5 positions that maximises expected points — provably optimal in O(n³).

**Time-decay sample weights with regulation-era boost.** Older races count less (half-life: 365 days), and the new 2026 regulation era gets a 3× weight multiplier so the model doesn't anchor on obsolete car physics.

**Leakage-free rolling features.** Every per-driver and per-team rolling statistic uses `groupby().shift(1).rolling()` so the feature for race N is computed only from races 1…N-1. Optuna tuning is done on 2022–2024 with 2025/2026 held out for true validation.

**Generalises to any track.** Circuit-history features are parameterised through a `CIRCUIT_FAMILIES` mapping (Miami → [Jeddah, Las Vegas, Baku, Melbourne, Singapore]), so the same model predicts Monaco, Monza, or any new venue.

## Validation results

Time-series cross-validation across 26 races in 2025–2026 (each race is held out, model trained on everything before it):

| Method                  | Mean points/race | Std  | Beats Grid baseline |
|-------------------------|------------------|------|---------------------|
| **XGBoost + Hungarian** | **7.38**         | 2.79 | yes, by +1.46       |
| Baseline: starting grid | 5.92             | 2.41 | —                   |
| Baseline: recent form   | 4.35             | 2.18 | —                   |

The XGBoost+Hungarian combo beats the grid baseline in 18/26 validation races. Grid alone is a strong baseline because qualifying position correlates 0.71 with finish position — beating it consistently is non-trivial.

## Pipeline architecture

```
extract_data.py        → raw race results, qualifying, weather (FastF1 API)
feature_engineering.py → 18 features, leakage-free, parameterised by circuit
train_model.py         → Optuna tuning + time-series CV + final model + metadata
next_race.py           → auto-detects next upcoming GP from FastF1 calendar
predict.py             → builds features for the upcoming race, runs model + Monte Carlo
visualizer.py          → standalone HTML report (Plotly)
pipeline.py            → orchestrator: runs everything end-to-end
```

## Output preview

The HTML report contains:

- **Top 5 prediction card** with predicted positions and per-driver `P(top 5)` confidence
- **Probability heatmap** showing `P(position j | driver i)` for the top 12 drivers
- **Monte Carlo bar chart** comparing `P(win)` / `P(podium)` / `P(top 5)` across drivers
- **Most likely top-5 combinations** table (most frequent permutations from 10k simulations)
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
pip install -r requirements.txt
python pipeline.py
```

First run takes ~10 minutes (downloads several seasons of data; FastF1 caches subsequent runs). Re-running takes ~30 seconds plus Optuna tuning if enabled.

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
├── requirements.txt
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
