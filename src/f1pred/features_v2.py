"""Version 2 features: the v1 table plus rebuilt pace features and more history.

    python -m f1pred.features_v2    # data/raw -> data/processed/features_v2.parquet

``features.py`` is frozen at the v1.0 tag and is not edited. This module calls
it and adds columns. What is new, and why:

* ``quali_seg_gap_pct``, ``quali_margin_pct``. v1 measured the qualifying gap
  against the fastest time of the whole session, which is not the pole lap
  when the track slows before Q3. Here each driver is compared with the best
  time in the last segment they ran, and the margin says by how much the
  segment's fastest driver led it.
* ``longrun_gap_v2``. v1's long-run gap used the single quickest median on a
  compound as its benchmark, which is often a short push run. Here a long run
  needs six laps after the slow-lap filter, the benchmark is the field median
  on that compound, and the session is chosen by how many drivers ran long.
* Pace-based form: how a driver and a car have qualified lately, and how this
  weekend compares with that.
* Circuit stickiness, shrunk towards the overall rate so that four races at a
  circuit do not read as certainty.
* Season context: the round number and a hand-set flag for seasons with new
  technical regulations, so the model can learn that form goes stale.
* History. Race and qualifying results for 2014-2018 feed every history
  window, and can also be used as training rows.

Every addition obeys the lights-out rule and runs through the same three
leakage tests as v1 (``tests/test_v2.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1pred.config import PROCESSED, RACE, TEST_SEASON
from f1pred.features import (
    FEATURE_GROUPS,
    FEATURES,
    ID_COLS,
    KEYS,
    OUTCOME_COLS,
    _seconds,
    attach_driver_ids,
    build_features,
)
from f1pred.ingest import load_tables

FEATURES_V2_PATH = PROCESSED / "features_v2.parquet"
HISTORY_SEASONS = tuple(range(2014, 2019))            # results only: no timing feed
V2_SEASONS = (*HISTORY_SEASONS, *range(2019, TEST_SEASON + 1))   # 2014-2025
NEW_REGULATIONS = {2014, 2017, 2022, 2026}            # hand-set: major rule changes

V1_FROZEN = [f for g in ("grid", "practice", "sprint", "form") for f in FEATURE_GROUPS[g]]
V2_GROUPS: dict[str, list[str]] = {
    "quali": ["quali_seg_gap_pct", "quali_margin_pct"],
    "pace_form": ["drv_quali_gap_avg_5", "quali_gap_vs_form", "team_quali_gap",
                  "team_quali_gap_avg_5", "team_gap_vs_form"],
    "practice": ["longrun_gap_v2"],
    "sticky": ["track_pole_win_eb", "track_grid_corr_eb"],
    "season": ["season_round", "new_regs"],
}
V2_FEATURES: list[str] = [f for group in V2_GROUPS.values() for f in group]

# Team and venue names from before 2019 that v1's maps do not know.
OLD_TEAMS = {"force_india": "racing_point", "lotus_f1": "renault"}
OLD_LOCATIONS = {"Abu Dhabi": "Yas Island", "Spa": "Spa-Francorchamps",
                 "Montreal": "Montréal", "Sao Paulo": "São Paulo"}

GAP_CAP = 5.0          # % ; wet-dry sessions produce gaps far beyond this
LONGRUN_MIN_LAPS = 6   # laps left in a run after the slow-lap filter
LONGRUN_MIN_DRIVERS = 4
SHRINK = 4.0           # prior weight, in races, for circuit statistics
DRY_OR_WET = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
SEGMENTS = ("Q3", "Q2", "Q1")  # latest first


def normalise_history(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Map pre-2019 team ids and venue names onto the ones v1 already handles."""
    out = dict(raw)
    for name in ("race", "quali", "sprint"):
        out[name] = raw[name].assign(TeamId=raw[name]["TeamId"].replace(OLD_TEAMS))
    out["events"] = raw["events"].assign(location=raw["events"]["location"].replace(OLD_LOCATIONS))
    return out


def _prior(group: pd.core.groupby.SeriesGroupBy, window: int) -> pd.Series:
    """Mean over the previous ``window`` rows of each group, current row excluded."""
    return group.transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())


# ----------------------------------------------------------------- qualifying

def quali_v2(quali: pd.DataFrame) -> pd.DataFrame:
    """Gap to the best time anyone set in the driver's last qualifying segment,
    and the margin to the next best in it (negative for the segment's fastest)."""
    quali = quali.dropna(subset=["DriverId"]).reset_index(drop=True)
    q = quali[RACE + ["DriverId"]].copy()
    times = pd.concat([_seconds(quali[seg]).rename(seg) for seg in SEGMENTS], axis=1)
    race = [quali["season"], quali["round"]]
    best = times.groupby(race).transform("min")                       # fastest per segment
    second = times.groupby(race).transform(
        lambda s: s.nsmallest(2).iloc[-1] if s.notna().sum() >= 2 else np.nan)
    last = times.notna().to_numpy().argmax(axis=1)                    # latest segment run
    rows = np.arange(len(q))
    own, seg_best, seg_second = (frame.to_numpy()[rows, last] for frame in (times, best, second))
    gap = (own / seg_best - 1) * 100                                  # NaN with no time at all
    q["quali_seg_gap_pct"] = gap
    q["quali_margin_pct"] = np.where(gap == 0, -(seg_second / seg_best - 1) * 100, gap)
    return q[RACE + ["DriverId", "quali_seg_gap_pct", "quali_margin_pct"]]


def pace_form(df: pd.DataFrame) -> pd.DataFrame:
    """How the driver and the car have qualified lately, and this weekend against that."""
    df = df.sort_values(RACE + ["grid_pos", "DriverId"]).reset_index(drop=True)
    gap = df["quali_seg_gap_pct"].clip(upper=GAP_CAP)
    df["drv_quali_gap_avg_5"] = _prior(gap.groupby(df["DriverId"], sort=False), 5)
    df["quali_gap_vs_form"] = gap - df["drv_quali_gap_avg_5"]

    team = (df.assign(gap=gap).groupby(RACE + ["team_lineage"], as_index=False)["gap"].min()
              .rename(columns={"gap": "team_quali_gap"}).sort_values(RACE))
    team["team_quali_gap_avg_5"] = _prior(
        team.groupby("team_lineage", sort=False)["team_quali_gap"], 5)
    df = df.merge(team, on=RACE + ["team_lineage"], how="left", validate="many_to_one")
    df["team_gap_vs_form"] = df["team_quali_gap"] - df["team_quali_gap_avg_5"]
    return df


# ------------------------------------------------------------------- practice

def longrun_v2(laps: pd.DataFrame) -> pd.DataFrame:
    """Long-run pace per driver per weekend, in % against the field median.

    A run is one stint on one compound. Laps slower than 1.03 times the run's
    median are dropped as traffic or cool-down, and the run counts only if six
    laps remain. A driver's pace on a compound is the median of those laps; the
    gap is to the median driver on the same compound in the same session,
    which a single low-fuel run cannot move. Several compounds are averaged by
    laps. The weekend's session is FP2 or FP1, whichever more drivers ran long
    in (FP2 on a tie); a driver with no long run there falls back to the other.
    """
    cols = RACE + ["DriverId", "longrun_gap_v2"]
    laps = laps[laps["session"].isin(["Practice 1", "Practice 2"])]
    clean = laps[
        laps["IsAccurate"].fillna(False).astype(bool)
        & laps["PitInTime"].isna() & laps["PitOutTime"].isna()
        & (laps["TrackStatus"] == "1") & laps["LapTime"].notna()
        & laps["Compound"].isin(DRY_OR_WET)
    ].copy()
    if clean.empty:
        return pd.DataFrame(columns=cols)
    clean["sec"] = _seconds(clean["LapTime"])
    run = KEYS + ["DriverId", "Stint", "Compound"]
    clean = clean[clean["sec"] < 1.03 * clean.groupby(run)["sec"].transform("median")]
    clean = clean[clean.groupby(run)["sec"].transform("size") >= LONGRUN_MIN_LAPS]

    pace = (clean.groupby(KEYS + ["DriverId", "Compound"])["sec"]
                 .agg(sec="median", laps="size").reset_index())
    by_compound = pace.groupby(KEYS + ["Compound"])["sec"]
    pace = pace[by_compound.transform("size") >= LONGRUN_MIN_DRIVERS].copy()
    pace["gap"] = (pace["sec"] / pace.groupby(KEYS + ["Compound"])["sec"].transform("median")
                   - 1) * 100
    pace["weighted"] = pace["gap"] * pace["laps"]
    drv = pace.groupby(KEYS + ["DriverId"])[["weighted", "laps"]].sum().reset_index()
    drv["gap"] = drv["weighted"] / drv["laps"]

    # main session of the weekend: most drivers with a long run, FP2 on a tie
    count = drv.groupby(KEYS)["DriverId"].transform("size")
    drv["order"] = -count + (drv["session"] != "Practice 2") * 0.5
    drv = drv.sort_values(RACE + ["DriverId", "order"]).drop_duplicates(RACE + ["DriverId"])
    return drv.rename(columns={"gap": "longrun_gap_v2"})[cols]


# -------------------------------------------------------------------- circuit

def circuit_stickiness(df: pd.DataFrame) -> pd.DataFrame:
    """How much the grid has decided earlier races at this circuit, shrunk
    towards the average over all earlier races so far."""
    df = df.sort_values(RACE + ["grid_pos", "DriverId"]).reset_index(drop=True)
    finishers = df[(df["classified"] == 1) & df["grid_pos"].notna()]
    corr = pd.DataFrame(
        [{"season": season, "round": rnd,
          "grid_corr": g["grid_pos"].corr(g["finish_pos"], method="spearman")
          if len(g) >= 5 else np.nan}
         for (season, rnd), g in finishers.groupby(RACE)],
        columns=RACE + ["grid_corr"])  # the columns exist even when no race has finishers yet
    races = (df.assign(pole_won=((df["grid_pos"] == 1) & (df["won"] == 1)).astype(float))
               .groupby(RACE + ["circuit"], as_index=False)["pole_won"].max()
               .merge(corr, on=RACE, how="left").sort_values(RACE)
               .reset_index(drop=True))
    by_circuit = races.groupby("circuit", sort=False)
    for stat, out in (("pole_won", "track_pole_win_eb"), ("grid_corr", "track_grid_corr_eb")):
        overall = races[stat].shift(1).expanding().mean()            # all earlier races
        seen = by_circuit[stat].transform(lambda s: s.shift(1).notna().cumsum())
        total = by_circuit[stat].transform(lambda s: s.shift(1).fillna(0).cumsum())
        races[out] = (total + SHRINK * overall) / (seen + SHRINK)
    keep = RACE + ["circuit", "track_pole_win_eb", "track_grid_corr_eb"]
    return df.merge(races[keep], on=RACE + ["circuit"], how="left", validate="many_to_one")


# ---------------------------------------------------------------------- build

def build_features_v2(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Raw tables -> the v1 table plus the v2 features."""
    raw = normalise_history(raw)
    for table, allowed in (("laps", {"Practice 1", "Practice 2", "Practice 3"}),
                           ("weather", {"Practice 1", "Practice 2", "Practice 3", "Qualifying"})):
        if not set(raw[table]["session"]) <= allowed:  # a race session here would be a leak
            raise ValueError(f"{table} holds sessions other than {sorted(allowed)}")
    df = build_features(raw)
    weekend = RACE + ["DriverId"]

    df = df.merge(quali_v2(raw["quali"]), on=weekend, how="left", validate="one_to_one")
    df = pace_form(df)

    laps = attach_driver_ids(raw["laps"], raw["race"], raw["quali"], raw["sprint"])
    df = df.merge(longrun_v2(laps), on=weekend, how="left", validate="one_to_one")

    df = circuit_stickiness(df)
    df["season_round"] = df["round"].astype(float)
    df["new_regs"] = df["season"].isin(NEW_REGULATIONS).astype(float)

    df[V2_FEATURES] = df[V2_FEATURES].astype(float)
    out = df[ID_COLS + FEATURES + V2_FEATURES + OUTCOME_COLS]
    return out.sort_values(RACE + ["grid_pos", "DriverId"]).reset_index(drop=True)


def load_features_v2() -> pd.DataFrame:
    """The v2 feature table, 2014-2025. 2026 is never in this file."""
    return pd.read_parquet(FEATURES_V2_PATH)


def main() -> None:
    features = build_features_v2(load_tables(V2_SEASONS))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_V2_PATH, index=False)
    print(f"wrote {FEATURES_V2_PATH.name}: {len(features)} rows, "
          f"{features.groupby(RACE).ngroups} races, seasons {features['season'].min()}-"
          f"{features['season'].max()}, {len(V2_FEATURES)} new features")
    recent = features[features["season"] >= 2019]
    print("non-missing % (2019 on):")
    print(recent[V2_FEATURES].notna().mean().mul(100).round(1).to_string())


if __name__ == "__main__":
    main()
