"""
Exploratory data analysis on the historical F1 dataset.

Generates 8 figures into figures/. Run after extract_data.py.
This is independent of the prediction pipeline
it's purely descriptive.

Bug fix vs the previous version: the .agg(name=("col")) call was malformed
— ``("mean")`` is not a valid named aggregation tuple. Replaced with the
keyword-argument form ``.agg(name="mean")``.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


def _save(fig_path_name: str) -> None:
    plt.savefig(config.FIGURES_DIR / fig_path_name)
    plt.close()
    logger.info("  -> %s", fig_path_name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = pd.read_csv(config.HISTORICAL_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    logger.info("Loaded %d rows from %s", len(df), config.HISTORICAL_FILE)
    logger.info("Date range: %s to %s",
                df["Date"].min().date(), df["Date"].max().date())

    finished_only = df[df["Finished"] & df["Position"].notna()].copy()

    # --- 1. Data coverage ---
    season_summary = df.groupby("Year").agg(
        races=("Round", "nunique"),
        drivers=("Abbreviation", "nunique"),
        rows=("Round", "count"),
    )
    logger.info("\n%s", season_summary)

    fig, ax = plt.subplots(figsize=(9, 5))
    season_summary["races"].plot(kind="bar", ax=ax, color="#3671C6", edgecolor="black")
    ax.set_title("Races in dataset per season", fontsize=13, weight="bold")
    ax.set_xlabel("Season")
    ax.set_ylabel("Number of races")
    ax.set_xticklabels(season_summary.index, rotation=0)
    for i, v in enumerate(season_summary["races"]):
        ax.text(i, v + 0.3, str(v), ha="center", fontweight="bold")
    _save("01_data_coverage.png")

    # --- 2. Grid -> Finish delta ---
    valid_races = df[
        df["Finished"] & df["Position"].notna()
        & df["GridPosition"].notna() & (df["GridPosition"] > 0)
    ].copy()
    valid_races["Position_Delta"] = valid_races["GridPosition"] - valid_races["Position"]

    fig, ax = plt.subplots(figsize=(12, 6))
    bins = np.arange(int(valid_races["Position_Delta"].min()) - 0.5,
                     int(valid_races["Position_Delta"].max()) + 1.5, 1)
    sns.histplot(data=valid_races, x="Position_Delta", bins=bins,
                 color="#7B8A99", edgecolor="black", alpha=0.85, ax=ax)
    ax.axvline(x=0, color="#E8002D", linestyle="--", linewidth=2, label="No change")
    ax.set_title("Grid -> Finish position delta", fontsize=14, weight="bold")
    ax.set_xlabel("Position change (negative = lost places, positive = gained)")
    ax.set_ylabel("Count")
    ax.legend()
    _save("02_grid_finish_delta.png")

    # --- 3. Grid vs finish correlation ---
    grid_analysis = df.dropna(subset=["GridPosition", "Position"]).copy()
    grid_analysis = grid_analysis[grid_analysis["GridPosition"] > 0]
    correlation = grid_analysis["GridPosition"].corr(grid_analysis["Position"])
    logger.info("Grid x Finish Pearson correlation: %.3f", correlation)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    ax.scatter(grid_analysis["GridPosition"], grid_analysis["Position"],
               alpha=0.15, s=30, color="#3671C6")
    ax.plot([1, 20], [1, 20], "r--", lw=2, label="Grid = Finish")
    ax.set_title(f"Grid x Finish (r = {correlation:.3f})", fontsize=12, weight="bold")
    ax.set_xlabel("Grid position")
    ax.set_ylabel("Finish position")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.legend()

    ax = axes[1]
    grid_analysis["Top5"] = grid_analysis["Position"] <= 5
    prob_top5 = grid_analysis.groupby("GridPosition")["Top5"].mean() * 100
    ax.bar(prob_top5.index, prob_top5.values, color="#27F4D2", edgecolor="black")
    ax.set_title("P(top 5 finish) by grid position", fontsize=12, weight="bold")
    ax.set_xlabel("Grid position")
    ax.set_ylabel("P(top 5) [%]")
    _save("03_grid_vs_finish.png")

    # --- 4. DNF analysis ---
    # FIX: the old code used .agg(ukonczone=("mean"),) which is malformed.
    # Use the keyword-argument form instead.
    dnf_by_season = df.groupby("Year")["Finished"].agg(
        finish_rate="mean",
        total="count",
    )
    dnf_by_season["finish_pct"] = (dnf_by_season["finish_rate"] * 100).round(1)
    logger.info("\nDNF rates by season:\n%s", dnf_by_season[["total", "finish_pct"]])

    dnf_reasons = df[~df["Finished"]]["Status"].value_counts().head(10)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    dnf_by_season["finish_pct"].plot(kind="bar", ax=ax, color="#229971", edgecolor="black")
    ax.set_title("Finish percentage per season", fontsize=12, weight="bold")
    ax.set_xlabel("Season")
    ax.set_ylabel("% finished")
    ax.set_ylim(0, 100)
    ax.set_xticklabels(dnf_by_season.index, rotation=0)
    for i, v in enumerate(dnf_by_season["finish_pct"]):
        ax.text(i, v + 1, f"{v}%", ha="center", fontweight="bold")

    ax = axes[1]
    dnf_reasons.head(8).plot(kind="barh", ax=ax, color="#E8002D", edgecolor="black")
    ax.set_title("Top 8 DNF reasons", fontsize=12, weight="bold")
    ax.set_xlabel("Count")
    ax.invert_yaxis()
    _save("04_dnf_analysis.png")

    # --- 5. Team performance heatmap ---
    team_performance = (
        finished_only.groupby(["Year", "TeamName"])["Position"]
        .mean().unstack().round(2)
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(team_performance.T, annot=True, fmt=".1f", cmap="RdYlGn_r",
                cbar_kws={"label": "Mean finish position"}, ax=ax, linewidths=0.5)
    ax.set_title("Mean finish position: team x season", fontsize=13, weight="bold")
    ax.set_xlabel("Season")
    ax.set_ylabel("Team")
    _save("05_team_performance.png")

    # --- 6. Weather ---
    weather_per_race = df.groupby(["Year", "Round"]).agg(
        AvgAirTemp=("AvgAirTemp", "first"),
        AvgTrackTemp=("AvgTrackTemp", "first"),
        Rainfall=("Rainfall", "first"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.scatter(weather_per_race["AvgAirTemp"], weather_per_race["AvgTrackTemp"],
               c=weather_per_race["Year"], cmap="viridis",
               s=60, alpha=0.7, edgecolor="black")
    ax.set_title("Air vs track temperature", fontsize=12, weight="bold")
    ax.set_xlabel("Avg air temp [C]")
    ax.set_ylabel("Avg track temp [C]")

    ax = axes[1]
    rain_counts = weather_per_race.groupby("Year")["Rainfall"].sum()
    rain_counts.plot(kind="bar", ax=ax, color="#64C4FF", edgecolor="black")
    ax.set_title("Wet races per season", fontsize=12, weight="bold")
    ax.set_xlabel("Season")
    ax.set_ylabel("Wet races")
    ax.set_xticklabels(rain_counts.index, rotation=0)
    _save("06_weather.png")

    # --- 7. Correlation matrix ---
    numeric_cols = ["GridPosition", "QualiPosition", "Position", "Points",
                    "AvgAirTemp", "AvgTrackTemp", "Humidity"]
    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation matrix of numeric features",
                 fontsize=13, weight="bold")
    _save("07_correlation_matrix.png")

    logger.info("\nEDA complete. Figures saved to %s", config.FIGURES_DIR)


if __name__ == "__main__":
    main()
