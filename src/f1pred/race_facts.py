"""Phase 6 - what happened in the races the model missed.

    python -m f1pred.race_facts        # after `make test-2025`

For every 2025 race where the pick lost, pull the facts needed to label the
miss: where the pick and the winner started, where they were after lap 1, how
the pick's race ended, pit stops, safety cars, red flags, rain and penalties.
Writes ``reports/test_2025/miss_facts.csv``.

This reads race laps and race-control messages. That is fine here and only
here: it explains results after the fact and feeds no feature.
"""

from __future__ import annotations

import fastf1
import pandas as pd

from f1pred.config import REPORTS, TEST_SEASON
from f1pred.ingest import enable_cache, load

TEST_DIR = REPORTS / "test_2025"
TRACK_STATUS = {"4": "safety car", "5": "red flag", "6": "virtual safety car"}


def _lap1_position(laps: pd.DataFrame, code: str) -> float:
    first = laps[(laps["Driver"] == code) & (laps["LapNumber"] == 1)]
    return float(first["Position"].iloc[0]) if len(first) and first["Position"].notna().any() \
        else float("nan")


def _ranges(laps: list[int]) -> str:
    """[1, 2, 3, 7] -> '1-3, 7'."""
    spans, start = [], None
    for prev, lap in zip([None, *laps], laps, strict=False):
        if prev is None or lap != prev + 1:
            if start is not None:
                spans.append((start, prev))
            start = lap
    spans.append((start, laps[-1]))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in spans)


def _neutralisations(laps: pd.DataFrame) -> str:
    """Race laps (any driver's count) touched by a safety car, VSC or red flag."""
    out = []
    status = laps.dropna(subset=["TrackStatus"])
    for code, name in TRACK_STATUS.items():
        hit = sorted(status.loc[status["TrackStatus"].str.contains(code), "LapNumber"]
                     .dropna().astype(int).unique())
        if hit:
            out.append(f"{name} laps {_ranges(hit)}")
    return "; ".join(out) or "none"


def _pit_laps(laps: pd.DataFrame, code: str) -> str:
    stops = laps[(laps["Driver"] == code) & laps["PitInTime"].notna()]
    return ", ".join(str(int(n)) for n in stops["LapNumber"])


def facts(season: int, rnd: int, pick: str, winner: str) -> dict:
    s = fastf1.get_session(season, rnd, "Race")
    s.load(laps=True, telemetry=False, weather=True, messages=True)
    res = s.results.set_index("Abbreviation")
    laps = pd.DataFrame(s.laps)
    number = str(res.loc[pick, "DriverNumber"])
    msgs = s.race_control_messages
    penalties = msgs[msgs["Message"].str.contains("PENALTY", na=False)
                     & msgs["Message"].str.contains(rf"CAR {number} \(", na=False)]["Message"]
    led = laps[laps["Position"] == 1].groupby("Driver")["LapNumber"].count()
    return {
        "round": rnd, "event": s.event["EventName"],
        "pick": pick, "pick_grid": res.loc[pick, "GridPosition"],
        "pick_lap1": _lap1_position(laps, pick), "pick_finish": res.loc[pick, "Position"],
        "pick_status": res.loc[pick, "Status"], "pick_laps_led": int(led.get(pick, 0)),
        "pick_pit_laps": _pit_laps(laps, pick),
        "pick_penalties": " | ".join(penalties) or "none",
        "winner": winner, "winner_grid": res.loc[winner, "GridPosition"],
        "winner_lap1": _lap1_position(laps, winner), "winner_laps_led": int(led.get(winner, 0)),
        "winner_pit_laps": _pit_laps(laps, winner),
        "race_laps": int(laps["LapNumber"].max()),
        "neutralisations": _neutralisations(laps),
        "rain": bool(pd.DataFrame(s.weather_data)["Rainfall"].any()),
    }


def main() -> None:
    races = pd.read_csv(TEST_DIR / "per_race.csv")
    misses = races[races["hit"] == 0]
    enable_cache(TEST_SEASON)
    rows = []
    for r in misses.itertuples():
        load(TEST_SEASON, r.round, "Race", laps=False)  # waits out the rate limiter if needed
        rows.append(facts(TEST_SEASON, r.round, r.pick, r.winner))
    out = pd.DataFrame(rows)
    out.to_csv(TEST_DIR / "miss_facts.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.max_colwidth", 90):
        for row in out.to_dict("records"):
            print("\n".join(f"  {k}: {v}" for k, v in row.items()), "\n")


if __name__ == "__main__":
    main()
