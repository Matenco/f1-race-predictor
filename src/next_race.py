"""
Auto-detect the next upcoming Formula 1 Grand Prix from the official calendar.

Returns metadata needed by the prediction pipeline: year, round number,
event name, circuit name, scheduled date, and whether qualifying has already
been held (which determines blind vs informed prediction mode).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import fastf1
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

fastf1.Cache.enable_cache(str(config.CACHE_DIR))


@dataclass
class NextRace:
    """Metadata for the next upcoming Grand Prix."""
    year: int
    round_number: int
    name: str           # e.g. "Canadian Grand Prix"
    circuit: str        # e.g. "Montréal" — matches the `Circuit` field in historical data
    country: str
    date: pd.Timestamp
    has_quali_results: bool   # True if qualifying session is loadable

    def __str__(self) -> str:
        return (
            f"{self.name} ({self.country}) — "
            f"{self.year} round {self.round_number}, "
            f"{self.date.strftime('%Y-%m-%d')}"
        )


def get_next_race(reference_date: pd.Timestamp | None = None) -> NextRace:
    """Return metadata for the next race after ``reference_date``.

    Looks at the current season first; if the season is over, looks at the
    next year. Raises ``RuntimeError`` if no upcoming race can be found.
    """
    if reference_date is None:
        reference_date = pd.Timestamp.now(tz="UTC").tz_localize(None)
    else:
        # Strip timezone for consistent comparison with FastF1 dates
        if reference_date.tz is not None:
            reference_date = reference_date.tz_localize(None)

    year = reference_date.year

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    schedule["EventDate"] = pd.to_datetime(schedule["EventDate"])
    reference_date_only = reference_date.normalize()
    upcoming = schedule[schedule["EventDate"] >= reference_date_only].sort_values("EventDate")

    # If the season has ended, look at next year
    if len(upcoming) == 0:
        logger.info("Season %d is over — looking at %d", year, year + 1)
        year += 1
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            schedule["EventDate"] = pd.to_datetime(schedule["EventDate"])
            upcoming = schedule.sort_values("EventDate")
        except Exception as exc:
            raise RuntimeError(f"Could not fetch {year} schedule: {exc}") from exc

    if len(upcoming) == 0:
        raise RuntimeError("No upcoming races found in the F1 calendar.")

    event = upcoming.iloc[0]

    # Has qualifying happened yet? Try to load the Q session.
    has_quali = _check_quali_available(year, int(event["RoundNumber"]))

    return NextRace(
        year=year,
        round_number=int(event["RoundNumber"]),
        name=str(event["EventName"]),
        circuit=str(event["Location"]),
        country=str(event["Country"]),
        date=pd.Timestamp(event["EventDate"]),
        has_quali_results=has_quali,
    )


def _check_quali_available(year: int, round_number: int) -> bool:
    """Best-effort check for whether qualifying results are already in the API."""
    try:
        quali = fastf1.get_session(year, round_number, "Q")
        quali.load(laps=False, telemetry=False, weather=False, messages=False)
        return len(quali.results) > 0 and quali.results["Position"].notna().any()
    except Exception:
        return False


def main() -> None:
    """CLI: print the next race info."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    nr = get_next_race()
    print()
    print("=" * 70)
    print(" NEXT RACE")
    print("=" * 70)
    print(f"  Event:    {nr.name}")
    print(f"  Country:  {nr.country}")
    print(f"  Circuit:  {nr.circuit}")
    print(f"  Round:    {nr.round_number} of {nr.year}")
    print(f"  Date:     {nr.date.strftime('%A, %B %d, %Y')}")
    print(f"  Quali:    {'AVAILABLE (informed mode)' if nr.has_quali_results else 'NOT YET (blind mode)'}")
    print()


if __name__ == "__main__":
    main()
