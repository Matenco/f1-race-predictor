# Model Card — F1 Top 5 Predictor

A short summary of the model, its training data, intended use, and known
limitations. Written in the spirit of [Mitchell et al. 2019, "Model Cards for
Model Reporting"](https://arxiv.org/abs/1810.03993) — the document model risk
practitioners look for when assessing third-party ML systems.

## Model details

- **Model type**: XGBoost ranker (`rank:pairwise`) for the deployed top-5
  ordering, plus diagnostic XGBoost classifiers for finishing position and
  top-5 membership
- **Output**: the production prediction starts from grid candidates P1-P7
  and selects the final top five with a blend of grid order, dedicated top-5
  probability, and a lightly weighted `rank:ndcg` ranker. This is deliberately
  grid-aware because the starting grid is the strongest baseline available
  after qualifying.
- **Hyperparameters**: the multiclass position model uses saved Optuna params
  in `model/optuna_best_params.json`; the ranker uses conservative fixed
  parameters stored in `src/config.py`
- **Training framework**: XGBoost 2.x on CPU (`tree_method=hist`)
- **License**: MIT

## Intended use

- **Primary**: predicting the top 5 of an upcoming Formula 1 Grand Prix for
  prediction-game scoring (2 points exact-position, 1 point in-top-5 miss)
- **Secondary**: portfolio / educational reference for time-series feature
  engineering, leakage-safe rolling statistics, and learning-to-rank for a
  strong sports baseline

This is **not** designed for, and should **not** be used for**:**

- Real-money betting (no calibration guarantees, no drift monitoring)
- Predicting individual race events (DNFs, safety cars, weather changes)
- Predicting pole position, fastest lap, championship outcomes, or any
  target other than top-5 finishing order

## Training data

- **Source**: [FastF1](https://github.com/theOehrly/Fast-F1) — official F1
  timing data API
- **Seasons**: 2022 to current (4–5 seasons depending on when the pipeline
  is run)
- **Race count**: typically 80–100 races, covering every Grand Prix in the
  current technical regulation era (2022–2025) and the new 2026 regulations
  as they accumulate
- **Driver count**: ~25 unique drivers across the period
- **Per-race features** include: race results, qualifying results and
  times, weather (air temp, track temp, humidity, rain flag), grid position,
  finish status

## Features (18)

Three categories, all leakage-free (per-driver and per-team statistics use
`groupby + shift(1) + rolling/expanding`):

| Category          | Features |
|-------------------|----------|
| Driver form       | `driver_avg_pos_last3`, `driver_avg_pos_last5`, `driver_best_pos_last5`, `driver_std_pos_last5`, `driver_finish_rate_last10`, `driver_pos_trend` |
| Team strength     | `team_avg_pos_last3`, `team_best_pos_season` |
| Grid / qualifying | `GridPosition`, `grid_vs_driver_form`, `teammate_pos_delta_last5`, `quali_time_delta_s` |
| Circuit history   | `driver_track_avg`, `driver_track_count`, `driver_similar_circuits_avg` |
| Context           | `is_rain`, `season_race_number`, `regulation_era` |

`driver_similar_circuits_avg` uses the per-circuit family map in
`src/circuit_metadata.py` — for example, Miami's family is
`[Jeddah, Las Vegas, Baku, Melbourne, Singapore]` (street/hybrid layouts
with limited overtaking).

## Sample weighting

Time decay with regulation-era boost:

```
weight ∝ 0.5 ** (days_to_target / 365)   × 1.5 if year ≥ 2026 else 1.0
```

Older races contribute less; the new 2026 regulation era gets a 1.5× boost
to mitigate the small-sample problem early in the season. Weights are
normalised to `[0, 1]`.

## Validation methodology

**Walk-forward time-series cross-validation.** For each race in the
2025–2026 validation window:

1. Train on every race chronologically before it
2. Predict the top 5 using the ranker-grid production model
3. Score against actual finish order
4. Compare against two baselines: the starting grid as-is, and recent-form
   ranking

Optuna hyperparameter tuning runs only on 2022–2024 data, holding out
2025/2026 entirely so the validation is genuinely out-of-sample.

## Performance summary

Last evaluated on real data — see `model/model_metadata.json` for current
numbers. Current walk-forward validation uses 28 held-out races in 2025–2026
after filling missing 2024–2025 races and filtering incomplete FastF1 pulls.

| Method                         | Mean pts/race | Std  | Notes |
|--------------------------------|---------------|------|-------|
| XGBoost top-5/ranker + grid P1-P7 | 5.86       | 1.88 | deployed strategy |
| XGBoost position + Hungarian   | 5.25          | 1.78 | previous approach |
| XGBoost top-5 ensemble         | 5.39          | 1.75 | tested, not deployed |
| Baseline: grid                 | 5.75          | 1.90 | strongest points baseline |
| Baseline: recent form          | 4.25          | 1.14 | secondary baseline |

The deployed strategy beats the grid baseline in the project-game points
metric (5.86 vs 5.75), improves mean top-5 set hits (4.00 vs 3.93), and
improves ranking quality (`NDCG@5`: 0.859 vs 0.853).

## Known limitations and biases

- **Cold-start for new regulations.** With only a handful of 2026 races
  available, predictions for early-2026 events lean on 2022–2025 priors
  and may underweight pace shifts. Sample weights mitigate but don't
  eliminate this.
- **Incomplete API pulls are filtered, not imputed.** FastF1/Jolpica can
  return driver/session metadata before official result data is available or
  while rate-limited. The pipeline drops races with too few classified
  finishers; rerunning after the API limit clears should add more races.
- **The deployed model is grid-aware.** It can promote P6/P7 into the final
  top five, but only through a controlled blend that keeps grid order as the
  dominant prior.
- **DNF causality is implicit.** Per-driver DNF rates are captured via
  `driver_finish_rate_last10`, but the model doesn't condition on
  race-day incidents (safety cars, mechanical failures, weather changes
  mid-race). A driver who DNFs through no fault of their own still gets
  a finish-rate hit.
- **Driver-team mid-season changes.** When a driver swaps teams or a
  reserve driver subs in, the per-driver rolling features carry over
  but team features reset. The model handles this gracefully (XGBoost's
  native NaN support), but predictions for substitute drivers should be
  treated with extra scepticism.
- **Wet-race granularity is coarse.** A binary `is_rain` flag — no
  intra-race weather modelling, no tyre-strategy input.
- **Calibration is not guaranteed.** The model outputs probabilities,
  but they are not formally calibrated (no Platt scaling, no isotonic
  regression). Treat exact percentages as relative confidence, not
  absolute.

## Update cadence

- **Data refresh and retraining**: weekly via
  `.github/workflows/weekly-retrain.yml`
- **Prediction refresh**: every 6 hours via
  `.github/workflows/update-prediction.yml`
- **Hyperparameter retuning**: manual; saved Optuna params are reused
  unless `USE_SAVED_OPTUNA_PARAMS = False` in `src/config.py`.

## Contact

Open an issue on the GitHub repository.
