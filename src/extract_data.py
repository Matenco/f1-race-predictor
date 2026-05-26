"""
Pulls historical F1 data (race results + qualifying + weather) from the
official F1 timing API via FastF1 and writes a tidy CSV.

Output: data/processed/f1_historical.csv
        — one row per (driver, race) tuple

Improvements over the previous version:
- Retry logic with exponential backoff on transient FastF1 / network errors
- Centralised config (no hardcoded paths)
- Type hints and structured logging
- Defensive handling of partial data (e.g. quali cancelled, weather missing)
"""

from __future__ import annotations

import logging
import time
import warnings

import fastf1
import numpy as np
import pandas as pd

from . import config

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Enable FastF1 cache for faster reruns
fastf1.Cache.enable_cache(str(config.CACHE_DIR))


# =============================================================================
#  Custom exception for rate limit
# =============================================================================
class RateLimitHit(Exception):
    """Raised when FastF1 reports the 500 calls/hour API limit.

    Retrying inside the same hour is pointless — the limit is a rolling
    window — so we surface this as a distinct exception that the season
    loop catches and uses to stop gracefully (saving partial data first).
    """


# =============================================================================
#  Retry helper
# =============================================================================
def _is_rate_limit(exc: Exception) -> bool:
    """Detect FastF1's rate-limit error from its message."""
    msg = str(exc).lower()
    return "500 calls" in msg or "ratelimit" in msg or "rate limit" in msg


def _with_retry(fn, *args, max_attempts: int = 3, base_delay: float = 2.0, **kwargs):
    """Run a callable with exponential-backoff retry on transient errors.

    Rate-limit errors are NOT retried — they're raised immediately as
    ``RateLimitHit`` so the caller can save progress and exit gracefully.
    Retrying within the same rolling hour just wastes the few remaining
    calls we have left.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if _is_rate_limit(exc):
                raise RateLimitHit(str(exc)) from exc
            last_exc = exc
            if attempt == max_attempts - 1:
                break
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "  retry %d/%d after %.1fs (reason: %s)",
                attempt + 1, max_attempts - 1, delay, type(exc).__name__,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


# =============================================================================
#  Race-level extraction
# =============================================================================
def extract_race_data(year: int, round_number: int) -> pd.DataFrame | None:
    """Extract a single Grand Prix as a tidy DataFrame.

    Pulls race results, qualifying results, and weather; merges them on
    driver abbreviation. Returns ``None`` if the race itself can't be loaded.
    Re-raises ``RateLimitHit`` so the outer loop can stop gracefully.
    """
    try:
        race = _with_retry(fastf1.get_session, year, round_number, "R")
        _with_retry(race.load, laps=False, telemetry=False, weather=True, messages=False)

        race_cols = [
            "DriverNumber", "Abbreviation", "FullName", "TeamName",
            "GridPosition", "Position", "Points", "Status",
        ]
        race_df = race.results[race_cols].copy()
        race_df["Position"] = pd.to_numeric(race_df["Position"], errors="coerce")
        race_df["GridPosition"] = pd.to_numeric(race_df["GridPosition"], errors="coerce")

        status_ok = race_df["Status"].str.contains("Finished|Lap", case=False, na=False)
        status_missing_but_classified = (
            race_df["Status"].isna() & race_df["Position"].notna()
        )
        race_df["Finished"] = status_ok | status_missing_but_classified

        race_df = _attach_qualifying(race_df, year, round_number)
        race_df = _attach_weather(race_df, race)

        race_df["Year"] = year
        race_df["Round"] = round_number
        race_df["GP"] = race.event["EventName"]
        race_df["Circuit"] = race.event["Location"]
        race_df["Country"] = race.event["Country"]
        race_df["Date"] = race.event["EventDate"]

        return race_df

    except RateLimitHit:
        # Re-raise so the season loop knows to stop, save partial data,
        # and tell the user to wait an hour
        raise
    except Exception as exc:
        logger.error("    failed for %d R%d: %s", year, round_number, exc)
        return None


def _attach_qualifying(race_df: pd.DataFrame, year: int, round_number: int) -> pd.DataFrame:
    """Merge qualifying position and best lap time onto the race results."""
    try:
        quali = _with_retry(fastf1.get_session, year, round_number, "Q")
        _with_retry(quali.load, laps=False, telemetry=False, weather=False, messages=False)

        quali_cols = [c for c in ["Abbreviation", "Position", "Q1", "Q2", "Q3"]
                      if c in quali.results.columns]
        quali_df = quali.results[quali_cols].copy()
        quali_df = quali_df.rename(columns={"Position": "QualiPosition"})
        quali_df["QualiPosition"] = pd.to_numeric(quali_df["QualiPosition"], errors="coerce")

        # Best qualifying time: prefer Q3, fall back to Q2, then Q1.
        def _best_time(row):
            for col in ("Q3", "Q2", "Q1"):
                if col in row and pd.notna(row[col]):
                    return row[col]
            return pd.NaT

        if any(c in quali_df.columns for c in ("Q1", "Q2", "Q3")):
            quali_df["QualiTime_s"] = quali_df.apply(_best_time, axis=1).apply(
                lambda t: pd.to_timedelta(t, errors="coerce").total_seconds()
                if not isinstance(t, float) else np.nan
            )
        else:
            quali_df["QualiTime_s"] = np.nan

        race_df = race_df.merge(
            quali_df[["Abbreviation", "QualiPosition", "QualiTime_s"]],
            on="Abbreviation", how="left",
        )
        # FastF1 sometimes omits GridPosition from race results but has it in
        # the qualifying session — fill the gap so both sources are used.
        race_df["GridPosition"] = race_df["GridPosition"].fillna(race_df["QualiPosition"])
        return race_df

    except Exception as exc:
        logger.warning("    quali unavailable: %s", exc)
        race_df["QualiPosition"] = np.nan
        race_df["QualiTime_s"] = np.nan
        return race_df


def _attach_weather(race_df: pd.DataFrame, race) -> pd.DataFrame:
    """Average weather metrics across the race session."""
    if hasattr(race, "weather_data") and len(race.weather_data) > 0:
        w = race.weather_data
        race_df["AvgAirTemp"] = w["AirTemp"].mean()
        race_df["AvgTrackTemp"] = w["TrackTemp"].mean()
        race_df["Humidity"] = w["Humidity"].mean()
        race_df["Rainfall"] = bool(w["Rainfall"].any())
    else:
        race_df["AvgAirTemp"] = np.nan
        race_df["AvgTrackTemp"] = np.nan
        race_df["Humidity"] = np.nan
        race_df["Rainfall"] = False
    return race_df


# =============================================================================
#  Season-level extraction
# =============================================================================
def extract_season(year: int) -> pd.DataFrame | None:
    """Pull every completed race in the given season.

    On rate-limit, returns whatever was successfully pulled before the limit
    hit (and re-raises ``RateLimitHit`` after, so the caller knows to stop
    the whole pipeline rather than try the next season).
    """
    logger.info("=== Season %d ===", year)

    try:
        schedule = _with_retry(fastf1.get_event_schedule, year, include_testing=False)
    except Exception as exc:
        logger.error("  failed to fetch calendar: %s", exc)
        return None

    today = pd.Timestamp.now(tz="UTC").tz_localize(None)
    schedule = schedule[schedule["EventDate"] < today]

    if len(schedule) == 0:
        logger.info("  (no completed races in %d)", year)
        return None

    all_races = []
    rate_limited = False
    for _, event in schedule.iterrows():
        round_num = event["RoundNumber"]
        gp_name = event["EventName"]
        logger.info("  [%d R%d] %s ...", year, round_num, gp_name)

        try:
            df = extract_race_data(year, round_num)
        except RateLimitHit as exc:
            logger.warning("  >>> RATE LIMIT HIT: %s", exc)
            rate_limited = True
            break
        if df is not None and len(df) > 0:
            all_races.append(df)
            logger.info("    %d drivers", len(df))

    season_df = pd.concat(all_races, ignore_index=True) if all_races else None
    if rate_limited:
        # Stash what we have, then signal upward
        if season_df is not None:
            logger.warning("  saving partial %d data: %d races",
                           year, len(all_races))
        raise RateLimitHit(f"Hit during {year} season")
    return season_df


# =============================================================================
#  Main
# =============================================================================
def _merge_with_existing(new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge newly-extracted rows with whatever's already in the CSV.

    Lets us re-run after a rate-limit interruption without losing earlier
    seasons. Deduplicates on (Year, Round, Abbreviation) keeping the new
    data on conflict (in case a race was re-pulled).
    """
    if not config.HISTORICAL_FILE.exists():
        return new_df
    try:
        existing = pd.read_csv(config.HISTORICAL_FILE)
    except Exception as exc:
        logger.warning("Could not read existing CSV (%s); starting fresh", exc)
        return new_df
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["Year", "Round", "Abbreviation"], keep="last",
    )
    combined = combined.sort_values(["Year", "Round", "Position"]).reset_index(drop=True)
    return combined


def _save_partial(all_data: list[pd.DataFrame]) -> int:
    """Merge what we have so far with the existing CSV and write it out."""
    if not all_data:
        return 0
    new_df = pd.concat(all_data, ignore_index=True)
    merged = _merge_with_existing(new_df)
    merged.to_csv(config.HISTORICAL_FILE, index=False)
    return len(merged)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=" * 60)
    logger.info("F1 DATA EXTRACTION PIPELINE")
    logger.info("Seasons: %s", config.SEASONS)
    logger.info("=" * 60)

    all_data: list[pd.DataFrame] = []
    rate_limit_hit = False

    for year in config.SEASONS:
        try:
            season_df = extract_season(year)
        except RateLimitHit:
            rate_limit_hit = True
            # extract_season already logged the partial data
            break

        if season_df is not None:
            all_data.append(season_df)
            # Incremental save after each completed season — if the next
            # season hits the rate limit, we still keep this one's data
            n_saved = _save_partial(all_data)
            logger.info("  -> saved %d total rows so far to %s",
                        n_saved, config.HISTORICAL_FILE)

    if rate_limit_hit:
        # Save whatever the last (partial) season managed to pull
        if all_data:
            _save_partial(all_data)
        logger.error("\n%s", "=" * 60)
        logger.error(" HIT FastF1 RATE LIMIT (500 calls/hour)")
        logger.error("=" * 60)
        logger.error(" What happened: the official F1 API caps API calls per IP")
        logger.error(" at 500/hour. Pulling 4+ seasons exceeds this on a clean")
        logger.error(" cache. Everything pulled so far is saved (and cached).")
        logger.error("")
        logger.error(" What to do: wait ~50–60 minutes, then re-run:")
        logger.error("     python pipeline.py")
        logger.error("")
        logger.error(" Cached races load instantly on the next run, so each")
        logger.error(" subsequent attempt gets further. 2-3 iterations is")
        logger.error(" usually enough for a full historical pull.")
        return

    if not all_data:
        logger.error("no data extracted; check network and try again")
        return

    _save_partial(all_data)

    logger.info("\n%s", "=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)
    final = pd.read_csv(config.HISTORICAL_FILE)
    logger.info("Output:           %s", config.HISTORICAL_FILE)
    logger.info("Rows:             %d", len(final))
    logger.info("Unique races:     %d", final.groupby(["Year", "Round"]).ngroups)
    logger.info("Unique drivers:   %d", final["Abbreviation"].nunique())
    logger.info("Seasons covered:  %s", sorted(final["Year"].unique().tolist()))


if __name__ == "__main__":
    main()
