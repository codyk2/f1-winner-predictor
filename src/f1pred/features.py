"""Phase 3 - build the driver-race feature table.

    python -m f1pred.features     # data/raw/*.parquet -> data/processed/features.parquet

One row per driver per race, the target ``won`` and 19 features. Every feature
passes the Saturday-night test: it is computed from practice, qualifying, the
sprint, the starting grid or earlier races, never from the race itself.
``tests/test_leakage.py`` enforces that.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from f1pred.config import ALL_SEASONS, FEATURES_PATH, PROCESSED, RACE, RAW, TEST_SEASON
from f1pred.ingest import TABLES

KEYS = ["season", "round", "session"]  # one session

# Feature groups, in the order the ablation adds them.
FEATURE_GROUPS: dict[str, list[str]] = {
    "grid": ["grid_pos", "quali_pos", "quali_gap_pct", "teammate_quali_delta"],
    "practice": ["longrun_gap_pct", "shortrun_gap_pct", "has_longrun"],
    "sprint": ["sprint_pos"],
    "form": ["drv_avg_finish_5", "drv_wins_5", "drv_podiums_5", "team_points_5",
             "champ_rank", "dnf_rate_10"],
    "circuit": ["drv_track_avg_finish", "track_pole_win_rate", "street_circuit"],
    "weather": ["quali_rain", "fp_track_temp"],
}
FEATURES: list[str] = [f for group in FEATURE_GROUPS.values() for f in group]

ID_COLS = ["season", "round", "event", "circuit", "date", "format", "DriverId",
           "Abbreviation", "FullName", "TeamId", "team_lineage"]
OUTCOME_COLS = ["won", "finish_pos", "classified", "Points"]

# Constructors that changed name keep one id, otherwise team form resets at each
# rename. Keys are Ergast constructor ids.
TEAM_LINEAGE = {
    "toro_rosso": "racing_bulls", "alphatauri": "racing_bulls", "rb": "racing_bulls",
    "racing_point": "aston_martin",
    "renault": "alpine",
    "alfa": "sauber", "audi": "sauber",
}

# The schedule spells some venues differently from year to year.
CIRCUIT_ALIASES = {
    "miami_gardens": "miami",
    "monte_carlo": "monaco",
    "marina_bay": "singapore",
    "yas_island": "yas_marina",
}
# Hand-labelled: temporary circuits laid out on public roads or in a park.
STREET_CIRCUITS = {
    "melbourne", "jeddah", "miami", "monaco", "baku", "singapore", "las_vegas",
    "sochi", "montreal", "madrid",
}

LONGRUN_MIN_LAPS = 5       # clean laps on one set of tyres
LONGRUN_MIN_DRIVERS = 3    # drivers needed on a compound for a gap to mean anything
LONGRUN_SESSIONS = ("Practice 2", "Practice 1")  # preferred first


# --------------------------------------------------------------------- loading

def load_raw(seasons: tuple[int, ...] | None = None,
             raw_dir: Path = RAW) -> dict[str, pd.DataFrame]:
    """Read the six raw tables for the given seasons (default: all on disk)."""
    raw: dict[str, pd.DataFrame] = {}
    for table in TABLES:
        files = sorted(raw_dir.glob(f"{table}_*.parquet"))
        if seasons is not None:
            files = [f for f in files if int(f.stem.rsplit("_", 1)[1]) in seasons]
        if not files:
            raise FileNotFoundError(f"no {table}_*.parquet in {raw_dir}; run `make data`")
        raw[table] = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return raw


def load_features(include_test: bool = False, path: Path = FEATURES_PATH) -> pd.DataFrame:
    """The feature table. The locked 2025 season is excluded unless asked for."""
    df = pd.read_parquet(path)
    return df if include_test else df[df["season"] < TEST_SEASON].reset_index(drop=True)


# --------------------------------------------------------------------- helpers

def circuit_id(location: str, event: str = "") -> str:
    """One stable id per circuit, whatever the schedule calls the venue."""
    ascii_name = unicodedata.normalize("NFKD", location).encode("ascii", "ignore").decode()
    slug = "_".join(ascii_name.lower().replace("-", " ").split())
    slug = CIRCUIT_ALIASES.get(slug, slug)
    if slug == "sakhir" and event == "Sakhir Grand Prix":
        return "sakhir_outer"  # 2020's second Bahrain race used the short outer layout
    return slug


def _seconds(td: pd.Series) -> pd.Series:
    return pd.to_timedelta(td).dt.total_seconds()


def _shifted(group: pd.core.groupby.SeriesGroupBy, window: int, how: str) -> pd.Series:
    """Rolling statistic over the previous ``window`` rows, excluding the current one."""
    return group.transform(
        lambda s: getattr(s.shift(1).rolling(window, min_periods=1), how)())


# ---------------------------------------------------------------------- pieces

def base_table(race: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """One row per driver per race with the target and the starting grid."""
    df = race.merge(events[["season", "round", "event", "location", "date", "format"]],
                    on=RACE, how="left", validate="many_to_one")
    df["circuit"] = [circuit_id(loc, ev)
                     for loc, ev in zip(df["location"], df["event"], strict=True)]
    df["team_lineage"] = df["TeamId"].map(lambda t: TEAM_LINEAGE.get(t, t))

    df["finish_pos"] = pd.to_numeric(df["Position"], errors="coerce")
    # ClassifiedPosition is a number for classified finishers and a letter
    # (R, D, E, W, F, N) otherwise. DNFs still get a Position.
    df["classified"] = pd.to_numeric(df["ClassifiedPosition"], errors="coerce").notna().astype(int)
    df["won"] = (df["finish_pos"] == 1).astype(int)
    df["podium"] = (df["finish_pos"] <= 3).astype(int)
    df["dnf"] = 1 - df["classified"]
    df["Points"] = pd.to_numeric(df["Points"], errors="coerce").fillna(0.0)

    # GridPosition 0 means a pit-lane start: put it behind the last grid slot
    # (21 with a 20-car field).
    grid = pd.to_numeric(df["GridPosition"], errors="coerce")
    behind_grid = df.groupby(RACE)["DriverId"].transform("size").clip(lower=20) + 1
    df["grid_pos"] = grid.where(grid != 0, behind_grid).astype(float)

    df["street_circuit"] = df["circuit"].isin(STREET_CIRCUITS).astype(int)
    return df.sort_values(RACE + ["grid_pos", "DriverId"]).reset_index(drop=True)


def quali_features(quali: pd.DataFrame) -> pd.DataFrame:
    """Qualifying position, gap to pole and gap to the team-mate."""
    quali = quali.dropna(subset=["DriverId"])  # entered but took no part
    q = quali[RACE + ["DriverId", "TeamId", "Position"]].copy()
    q["quali_pos"] = pd.to_numeric(q["Position"], errors="coerce")
    q["best"] = pd.concat([_seconds(quali[c]) for c in ("Q1", "Q2", "Q3")], axis=1).min(axis=1)
    pole = q.groupby(RACE)["best"].transform("min")
    q["quali_gap_pct"] = (q["best"] / pole - 1) * 100

    team = q.groupby(RACE + ["TeamId"])["best"]
    # With two cars per team, the team-mate's time is the team total minus your own.
    mate = (team.transform("sum") - q["best"]).where(team.transform("count") == 2)
    q["teammate_quali_delta"] = (q["best"] / mate - 1) * 100
    return q[RACE + ["DriverId", "quali_pos", "quali_gap_pct", "teammate_quali_delta"]]


def attach_driver_ids(laps: pd.DataFrame, *results: pd.DataFrame) -> pd.DataFrame:
    """Fill DriverId on practice laps from the same weekend's entry lists.

    Ingestion already merges each session's own results; this covers weekends
    where that lookup was unavailable (for example a race not yet run).
    """
    entries = pd.concat([r[RACE + ["DriverNumber", "DriverId"]] for r in results if len(r)])
    entries = entries.dropna().drop_duplicates(RACE + ["DriverNumber"])
    laps = laps.merge(entries.rename(columns={"DriverId": "_id"}),
                      on=RACE + ["DriverNumber"], how="left")
    laps["DriverId"] = laps["DriverId"].fillna(laps["_id"])
    # Practice-only drivers have no Ergast id; keep them as rivals under their code.
    laps["DriverId"] = laps["DriverId"].fillna("fp_only_" + laps["Driver"].astype("string"))
    return laps.drop(columns="_id")


def longrun_pace(laps: pd.DataFrame, min_laps: int = LONGRUN_MIN_LAPS) -> pd.DataFrame:
    """Median long-run lap per driver, as % gap to the best driver on the same compound."""
    clean = laps[
        laps["IsAccurate"].fillna(False).astype(bool)
        & laps["PitInTime"].isna() & laps["PitOutTime"].isna()
        & (laps["TrackStatus"] == "1")
        & laps["LapTime"].notna()
        & laps["Compound"].isin(["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"])
    ].copy()
    clean["sec"] = _seconds(clean["LapTime"])
    run = KEYS + ["DriverId", "Stint", "Compound"]
    clean = clean[clean.groupby(run)["sec"].transform("size") >= min_laps]
    # drop cool-down and traffic laps inside a run
    clean = clean[clean["sec"] < 1.03 * clean.groupby(run)["sec"].transform("median")]
    pace = (clean.groupby(KEYS + ["DriverId", "Compound"])["sec"]
                 .agg(longrun_sec="median", longrun_laps="size").reset_index())
    by_compound = pace.groupby(KEYS + ["Compound"])["longrun_sec"]
    pace["longrun_gap_pct"] = (pace["longrun_sec"] / by_compound.transform("min") - 1) * 100
    pace["compound_drivers"] = by_compound.transform("size")
    return pace


def practice_features(laps: pd.DataFrame) -> pd.DataFrame:
    """Long-run and short-run pace per driver per weekend."""
    cols = RACE + ["DriverId", "longrun_gap_pct", "shortrun_gap_pct", "has_longrun"]
    if laps.empty:
        return pd.DataFrame(columns=cols)

    # Long runs: FP2 where it produced any, otherwise FP1 (sprint weekends).
    pace = longrun_pace(laps[laps["session"].isin(LONGRUN_SESSIONS)])
    order = {name: i for i, name in enumerate(LONGRUN_SESSIONS)}
    pace["pref"] = pace["session"].map(order)
    pace = pace[pace["pref"] == pace.groupby(RACE)["pref"].transform("min")]
    # A gap on a compound almost nobody ran says nothing; a driver with runs on
    # several compounds gets the lap-weighted mean of the gaps.
    comparable = pace[pace["compound_drivers"] >= LONGRUN_MIN_DRIVERS].copy()
    comparable["weighted"] = comparable["longrun_gap_pct"] * comparable["longrun_laps"]
    sums = comparable.groupby(RACE + ["DriverId"])[["weighted", "longrun_laps"]].sum()
    longrun = (sums["weighted"] / sums["longrun_laps"]).rename("longrun_gap_pct").reset_index()
    has_run = pace[RACE + ["DriverId"]].drop_duplicates().assign(has_longrun=1)

    # Short runs: best accurate lap against the session best, taking the
    # driver's best showing across the weekend's practice sessions.
    fast = laps[laps["IsAccurate"].fillna(False).astype(bool) & laps["LapTime"].notna()].copy()
    fast["sec"] = _seconds(fast["LapTime"])
    best = fast.groupby(KEYS + ["DriverId"])["sec"].min().reset_index()
    best["gap"] = (best["sec"] / best.groupby(KEYS)["sec"].transform("min") - 1) * 100
    shortrun = (best.groupby(RACE + ["DriverId"])["gap"].min()
                    .rename("shortrun_gap_pct").reset_index())

    out = shortrun.merge(longrun, on=RACE + ["DriverId"], how="outer")
    out = out.merge(has_run, on=RACE + ["DriverId"], how="outer")
    return out[cols]


def weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    """Rain during qualifying and mean practice track temperature, per weekend."""
    w = weather.copy()
    w["Rainfall"] = w["Rainfall"].fillna(False).astype(bool)
    rain = (w[w["session"] == "Qualifying"].groupby(RACE)["Rainfall"].any()
            .astype(int).rename("quali_rain"))
    temp = (w[w["session"].str.startswith("Practice")].groupby(RACE)["TrackTemp"].mean()
            .rename("fp_track_temp"))
    return pd.concat([rain, temp], axis=1).reset_index()


def championship_rank(df: pd.DataFrame, sprint: pd.DataFrame) -> pd.Series:
    """Championship position before each race.

    Points from earlier races plus every sprint up to and including this
    weekend's (the sprint is run before the race). Empty until someone has scored.
    """
    out = pd.Series(np.nan, index=df.index, name="champ_rank")
    sprint_pts = sprint.assign(Points=pd.to_numeric(sprint["Points"], errors="coerce").fillna(0.0))
    for season, grp in df.groupby("season"):
        race_pts = grp.pivot_table(index="round", columns="DriverId", values="Points",
                                   aggfunc="sum")
        entered = race_pts.notna().cummax()          # has appeared by this round
        before = race_pts.fillna(0.0).cumsum().shift(1, fill_value=0.0)
        sp = sprint_pts[sprint_pts["season"] == season]
        if len(sp):
            sp = (sp.pivot_table(index="round", columns="DriverId", values="Points", aggfunc="sum")
                    .reindex(index=race_pts.index, columns=race_pts.columns).fillna(0.0).cumsum())
            before = before + sp
        ranks = before.where(entered).rank(axis=1, ascending=False, method="min")
        ranks[before.max(axis=1) <= 0] = np.nan
        lookup = ranks.stack().rename("champ_rank")
        idx = pd.MultiIndex.from_frame(grp[["round", "DriverId"]])
        out.loc[grp.index] = lookup.reindex(idx).to_numpy()
    return out


def history_features(df: pd.DataFrame, sprint: pd.DataFrame) -> pd.DataFrame:
    """Form and circuit history. Every window is shifted by one race first,
    so a race never sees its own result."""
    df = df.sort_values(RACE + ["grid_pos", "DriverId"]).reset_index(drop=True)
    by_driver = df.groupby("DriverId", sort=False)

    df["drv_avg_finish_5"] = _shifted(by_driver["finish_pos"], 5, "mean")
    df["drv_wins_5"] = _shifted(by_driver["won"], 5, "sum")
    df["drv_podiums_5"] = _shifted(by_driver["podium"], 5, "sum")
    df["dnf_rate_10"] = _shifted(by_driver["dnf"], 10, "mean")

    team = (df.groupby(RACE + ["team_lineage"], as_index=False)["Points"].sum()
              .sort_values(RACE))
    team["team_points_5"] = _shifted(team.groupby("team_lineage", sort=False)["Points"], 5, "sum")
    df = df.merge(team.drop(columns="Points"), on=RACE + ["team_lineage"], how="left",
                  validate="many_to_one")

    df["champ_rank"] = championship_rank(df, sprint)

    df["drv_track_avg_finish"] = df.groupby(["DriverId", "circuit"], sort=False)["finish_pos"] \
        .transform(lambda s: s.shift(1).expanding().mean())

    # Pole conversion at this circuit over earlier races only.
    races = (df.assign(pole_won=((df["grid_pos"] == 1) & (df["won"] == 1)).astype(int))
               .groupby(RACE + ["circuit"], as_index=False)["pole_won"].max()
               .sort_values(RACE))
    races["track_pole_win_rate"] = races.groupby("circuit", sort=False)["pole_won"] \
        .transform(lambda s: s.shift(1).expanding().mean())
    df = df.merge(races.drop(columns="pole_won"), on=RACE + ["circuit"], how="left",
                  validate="many_to_one")
    return df


# ----------------------------------------------------------------------- build

def build_features(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Raw tables -> one row per driver per race with features and the target."""
    df = base_table(raw["race"], raw["events"])
    weekend = RACE + ["DriverId"]

    df = df.merge(quali_features(raw["quali"]), on=weekend, how="left", validate="one_to_one")

    laps = attach_driver_ids(raw["laps"], raw["race"], raw["quali"], raw["sprint"])
    df = df.merge(practice_features(laps), on=weekend, how="left", validate="one_to_one")
    df["has_longrun"] = df["has_longrun"].fillna(0).astype(int)

    sprint = raw["sprint"].dropna(subset=["DriverId"])
    sprint_pos = sprint.assign(sprint_pos=pd.to_numeric(sprint["Position"], errors="coerce"))
    df = df.merge(sprint_pos[weekend + ["sprint_pos"]], on=weekend, how="left",
                  validate="one_to_one")

    df = df.merge(weather_features(raw["weather"]), on=RACE, how="left", validate="many_to_one")
    df = history_features(df, sprint)

    df[FEATURES] = df[FEATURES].astype(float)
    out = df[ID_COLS + FEATURES + OUTCOME_COLS]
    return out.sort_values(RACE + ["grid_pos", "DriverId"]).reset_index(drop=True)


def main() -> None:
    raw = load_raw(ALL_SEASONS)
    features = build_features(raw)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_PATH, index=False)
    races = features.groupby(RACE).ngroups
    print(f"wrote {FEATURES_PATH.relative_to(FEATURES_PATH.parents[2])}: "
          f"{len(features)} rows, {races} races, {len(FEATURES)} features")
    coverage = features[FEATURES].notna().mean().mul(100).round(1)
    print("non-missing % per feature:")
    print(coverage.to_string())


if __name__ == "__main__":
    main()
