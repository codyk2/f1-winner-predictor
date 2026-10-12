"""Phase 1 - download every session once and flatten it into six raw tables.

    python -m f1pred.ingest                 # all seasons 2019-2025
    python -m f1pred.ingest 2019            # a single season
    python -m f1pred.ingest 2026 --through 17 --allow-missing   # a season in progress

Writes ``data/raw/{events,race,quali,sprint,laps,weather}_{season}.parquet``.
Nothing after this stage calls the API: every later stage reads these files.

Two outside limits shape this script:

* The results API (Jolpica) rejects anything faster than about one request
  every two seconds, so results sessions are fetched one at a time.
* FastF1 caps each process at 500 uncached requests an hour, so each season
  runs in its own short-lived worker and waits if the limiter still trips.

Responses for finished seasons are cached forever, so a rebuild reads purely
from ``data/cache`` and never downloads a session twice.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date

import fastf1
import pandas as pd
from fastf1.ergast import Ergast
from fastf1.exceptions import RateLimitExceededError

from f1pred.config import ALL_SEASONS, CACHE, RAW

RESULT_TABLE = {"Race": "race", "Qualifying": "quali", "Sprint": "sprint"}
TABLES = ("events", "race", "quali", "sprint", "laps", "weather")

RESULT_COLS = [
    "DriverNumber", "Abbreviation", "DriverId", "FullName", "TeamName", "TeamId",
    "GridPosition", "Position", "ClassifiedPosition", "Status", "Points", "Laps",
    "Q1", "Q2", "Q3",
]
LAP_COLS = [
    "Driver", "DriverNumber", "Team", "LapNumber", "LapTime", "Stint", "Compound",
    "TyreLife", "FreshTyre", "TrackStatus", "IsAccurate", "PitInTime", "PitOutTime",
]
WEATHER_COLS = ["Time", "AirTemp", "TrackTemp", "Rainfall", "Humidity", "WindSpeed"]
STRING_COLS = [
    "DriverNumber", "Abbreviation", "DriverId", "FullName", "TeamName", "TeamId",
    "ClassifiedPosition", "Status", "Driver", "Team", "Compound", "TrackStatus",
]

MISSING_TEXT = ["", "nan", "NaN", "None", "NaT"]  # FastF1 stringifies some gaps

RATE_LIMIT_WAIT = 300   # seconds between attempts while FastF1's limiter is closed
API_PAUSE = 4           # seconds to back off after the results API refuses a request
API_MAX_WAIT = 900      # give up on one results session after this many seconds
PRACTICE_RETRIES = 1    # extra attempts for a practice session with no timing data


@dataclass
class SeasonReport:
    season: int
    skips: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def missing_results(self) -> list[str]:
        """Skipped Race or Qualifying sessions: these make the season unusable."""
        return [s for s in self.skips if " Race:" in s or " Qualifying:" in s]


def _log(year: int, msg: str) -> None:
    print(f"[{year}] {msg}", flush=True)


def enable_cache(year: int | None = None) -> None:
    """Point FastF1 at ``data/cache``; keep finished seasons' responses forever."""
    CACHE.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))
    fastf1.set_log_level("ERROR")
    if year is not None and year < date.today().year:
        # A finished season never changes. FastF1 would otherwise re-request
        # results after ten minutes, which makes rebuilds slow and hammers an
        # API that tolerates one request every two seconds.
        session = getattr(fastf1.Cache, "_requests_session_cached", None)
        if session is not None:
            session.settings.cache_control = False
            session.settings.expire_after = -1  # requests-cache: never expire
    # Several workers share one HTTP cache; WAL keeps their writes from blocking.
    try:
        with sqlite3.connect(CACHE / "fastf1_http_cache.sqlite", timeout=60) as con:
            con.execute("pragma journal_mode=wal")
    except sqlite3.Error:
        pass


def load(year: int, rnd: int, name: str, laps: bool) -> fastf1.core.Session:
    """Load one session, waiting out FastF1's rate limiter instead of failing."""
    while True:
        try:
            s = fastf1.get_session(year, rnd, name)
            s.load(laps=laps, telemetry=False, weather=True, messages=False)
            return s
        except RateLimitExceededError as exc:
            _log(year, f"FastF1 rate limit hit ({exc}); waiting {RATE_LIMIT_WAIT}s, "
                       f"then resuming at R{rnd} {name}")
            time.sleep(RATE_LIMIT_WAIT)


def tag(df: pd.DataFrame, year: int, rnd: int, name: str) -> pd.DataFrame:
    return pd.DataFrame(df).assign(season=year, round=rnd, session=name)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Give every text column one dtype and one way of saying "missing"."""
    for col in STRING_COLS:
        if col in df:
            df[col] = df[col].astype("string").replace(MISSING_TEXT, pd.NA)
    return df


def _results(s: fastf1.core.Session) -> pd.DataFrame:
    return _clean(pd.DataFrame(s.results).reindex(columns=RESULT_COLS).reset_index(drop=True))


def _weather(s: fastf1.core.Session, year: int, rnd: int, name: str) -> pd.DataFrame:
    try:
        w = pd.DataFrame(s.weather_data).reindex(columns=WEATHER_COLS)
    except Exception:  # weather feed missing for this session
        w = pd.DataFrame(columns=WEATHER_COLS)
    return tag(w, year, rnd, name)


def _results_api_state(year: int, rnd: int, name: str) -> str:
    """Ask the results API directly: 'ok', 'empty' (no such result) or 'busy'."""
    api = Ergast()
    call = {"Race": api.get_race_results, "Qualifying": api.get_qualifying_results,
            "Sprint": api.get_sprint_results}[name]
    try:
        response = call(season=year, round=rnd)
    except RateLimitExceededError:
        raise
    except Exception:  # HTTP 429 or a network error
        return "busy"
    return "ok" if len(response.content) else "empty"


def _fetch_results(year: int, rnd: int, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Results and weather for a Race, Qualifying or Sprint session.

    FastF1 swallows a refused results request and returns a frame with no
    positions. Saving that would silently drop the race, so check for it, wait
    for the API and load again.
    """
    waited = 0
    for _ in range(3):  # a refused request is retried inside the while loop
        s = load(year, rnd, name, laps=False)
        res = _results(s)
        if len(res) and res["DriverId"].notna().any() and res["Position"].notna().any():
            return tag(res, year, rnd, name), _weather(s, year, rnd, name)
        while True:
            time.sleep(API_PAUSE)
            waited += API_PAUSE
            state = _results_api_state(year, rnd, name)
            if state == "empty":
                raise LookupError("no result data")
            if state == "ok":
                break  # now cached, so the next load has it
            if waited >= API_MAX_WAIT:
                raise LookupError(f"results API unavailable after {waited}s")
    raise LookupError("results incomplete")


def _fetch_practice(year: int, rnd: int, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Laps (keyed on DriverId) and weather for one practice session."""
    last: Exception = LookupError("no data")
    for attempt in range(1 + PRACTICE_RETRIES):
        if attempt:
            time.sleep(API_PAUSE)
        try:
            s = load(year, rnd, name, laps=True)
            laps = pd.DataFrame(s.laps).reindex(columns=LAP_COLS)
            ids = _results(s)[["DriverNumber", "DriverId"]].dropna().drop_duplicates("DriverNumber")
        except RateLimitExceededError:
            raise
        except Exception as exc:  # no timing data for this session
            last = exc
            continue
        if laps.empty:
            last = LookupError("no laps")
            continue
        laps = _clean(laps).merge(ids, on="DriverNumber", how="left")
        return tag(laps, year, rnd, name), _weather(s, year, rnd, name)
    raise last


def _save(parts: list[pd.DataFrame], table: str, year: int) -> int:
    """Write one raw table atomically; returns the row count.

    A session that was saved before and could not be fetched this time is kept,
    so a failed request during a later run never shrinks a season on disk.
    """
    parts = [p for p in parts if not p.empty]
    path = RAW / f"{table}_{year}.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        keys = ["round", "session"] if "session" in old else ["round"]
        fetched = (pd.concat([p[keys] for p in parts]).drop_duplicates() if parts
                   else pd.DataFrame(columns=keys))
        kept = old.merge(fetched, on=keys, how="left", indicator=True)
        kept = kept[kept["_merge"] == "left_only"].drop(columns="_merge")
        if len(kept):
            parts.append(kept)
    if not parts:
        return 0
    df = pd.concat(parts, ignore_index=True).sort_values(
        ["round", "session"] if "session" in parts[0] else ["round"], kind="stable")
    df = df.reset_index(drop=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return len(df)


def load_tables(seasons: Iterable[int]) -> dict[str, pd.DataFrame]:
    """Read the raw tables for the given seasons by exact file name.

    Exact names matter: a synced folder (iCloud, Dropbox) can drop a conflict
    copy such as ``events_2019 2.parquet`` next to the real file, and a
    wildcard would read both.
    """
    seasons = list(seasons)
    tables = {}
    for table in TABLES:
        files = [RAW / f"{table}_{year}.parquet" for year in seasons]
        files = [f for f in files if f.exists()]  # no sprint table before 2021
        if not files:
            raise FileNotFoundError(f"no {table}_<season>.parquet in {RAW}; run `make data`")
        tables[table] = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return tables


def _weekends(year: int, through: int | None) -> list[tuple[pd.Series, list[str]]]:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    if through is not None:
        schedule = schedule[schedule["RoundNumber"] <= through]
    return [(ev, [str(ev[f"Session{i}"]) for i in range(1, 6)])
            for _, ev in schedule.iterrows()]


def ingest_results(year: int, through: int | None = None) -> SeasonReport:
    """Events plus Race, Qualifying and Sprint results for one season."""
    enable_cache(year)
    report = SeasonReport(year)
    tables: dict[str, list[pd.DataFrame]] = {k: [] for k in TABLES}
    for ev, names in _weekends(year, through):
        rnd = int(ev["RoundNumber"])
        tables["events"].append(pd.DataFrame([{
            "season": year, "round": rnd, "event": ev["EventName"],
            "location": ev["Location"], "country": ev["Country"],
            "date": ev["EventDate"], "format": ev["EventFormat"],
            **{f"session{i}": n for i, n in enumerate(names, start=1)},
        }]))
        for name in names:
            if name not in RESULT_TABLE:
                continue
            try:
                res, weather = _fetch_results(year, rnd, name)
                tables[RESULT_TABLE[name]].append(res)
                if name == "Qualifying":
                    tables["weather"].append(weather)
            except Exception as exc:  # missing session, no data
                report.skips.append(f"skip {year} R{rnd} {name}: {exc}")
    for table in ("events", "race", "quali", "sprint"):
        report.counts[table] = _save(tables[table], table, year)
    _save(tables["weather"], "_quali_weather", year)
    _log(year, "results done: " + ", ".join(f"{t}={n}" for t, n in report.counts.items()))
    return report


def ingest_practice(year: int, through: int | None = None) -> SeasonReport:
    """Practice laps and weather for one season (the bulk of the download)."""
    enable_cache(year)
    report = SeasonReport(year)
    laps_parts: list[pd.DataFrame] = []
    quali_weather = RAW / f"_quali_weather_{year}.parquet"
    weather_parts = [pd.read_parquet(quali_weather)] if quali_weather.exists() else []
    for ev, names in _weekends(year, through):
        rnd = int(ev["RoundNumber"])
        for name in names:
            if not name.startswith("Practice"):
                continue
            try:
                laps, weather = _fetch_practice(year, rnd, name)
                laps_parts.append(laps)
                weather_parts.append(weather)
            except Exception as exc:  # cancelled session, no timing data
                report.skips.append(f"skip {year} R{rnd} {name}: {exc}")
    report.counts["laps"] = _save(laps_parts, "laps", year)
    report.counts["weather"] = _save(weather_parts, "weather", year)
    quali_weather.unlink(missing_ok=True)
    _log(year, "practice done: " + ", ".join(f"{t}={n}" for t, n in report.counts.items()))
    return report


def _in_fresh_process(fn, year: int, through: int | None) -> SeasonReport:
    """Run one season in its own process so FastF1's request budget starts at zero."""
    with ProcessPoolExecutor(max_workers=1) as pool:
        return pool.submit(fn, year, through).result()


def ingest(seasons: list[int], through: int | None = None) -> list[SeasonReport]:
    """Download the given seasons and write their raw tables."""
    enable_cache()
    # Results one season at a time: the results API refuses parallel clients.
    results = [_in_fresh_process(ingest_results, year, through) for year in seasons]
    # Practice timing comes from F1's own servers; one worker per season.
    with ProcessPoolExecutor(max_workers=len(seasons)) as pool:
        practice = list(pool.map(ingest_practice, seasons, [through] * len(seasons)))
    for res, prac in zip(results, practice, strict=True):
        res.skips += prac.skips
        res.counts.update(prac.counts)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("seasons", nargs="*", type=int, default=list(ALL_SEASONS))
    parser.add_argument("--through", type=int, help="only rounds up to this one")
    parser.add_argument("--allow-missing", action="store_true",
                        help="do not fail when a Race or Qualifying session is missing")
    args = parser.parse_args(argv)

    reports = ingest(args.seasons, args.through)

    print("\nSkipped sessions")
    for rep in reports:
        for line in rep.skips:
            print("  " + line)
    print("\nSummary")
    for rep in reports:
        print(f"  {rep.season}: " + ", ".join(f"{t}={n}" for t, n in rep.counts.items())
              + f", skipped={len(rep.skips)}")
    missing = [s for rep in reports for s in rep.missing_results]
    if missing and not args.allow_missing:
        print("\nRace or Qualifying sessions are missing; rerun this command:", file=sys.stderr)
        for line in missing:
            print("  " + line, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
