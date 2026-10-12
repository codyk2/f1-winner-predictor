"""Score one race from everything known on Saturday night.

    python -m f1pred.predict 2026 18                 # after qualifying, before the race
    python -m f1pred.predict 2026 18 --grid grid.csv # with the official starting grid
    python -m f1pred.predict 2026 17 --no-download   # re-score from data already on disk

The model is the frozen setup in ``config/model.json``, retrained on every race
before the target (including earlier rounds of the same season, so a new set
of regulations is learned as the season goes on). Predictions are written to
``predictions/{season}_R{round}.csv``; commit that file before lights out and
the commit timestamp proves the prediction came first.

Until the race has been run there is no official grid in the data, so the
grid defaults to qualifying order. Pass ``--grid`` (columns ``Abbreviation``,
``GridPosition``; 0 for a pit-lane start) once penalties are applied.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from f1pred.config import RACE, RAW, ROOT, TEST_SEASON, load_frozen
from f1pred.features import build_features, load_raw
from f1pred.ingest import RESULT_COLS, enable_cache, ingest
from f1pred.train import fit, predict

PREDICTIONS = ROOT / "predictions"
OUTCOME_COLS = ["Position", "ClassifiedPosition", "Status", "Points", "Laps"]


def entry_list(raw: dict[str, pd.DataFrame], season: int, rnd: int,
               grid: pd.DataFrame | None = None) -> pd.DataFrame:
    """Race-table rows for a race that has not been run: the qualifying entry
    list with the grid taken from qualifying order (or from ``grid``)."""
    quali = raw["quali"]
    q = quali[(quali["season"] == season) & (quali["round"] == rnd)].dropna(subset=["DriverId"])
    if q.empty:
        raise SystemExit(f"No qualifying results for {season} round {rnd} yet.")
    rows = q.reindex(columns=RESULT_COLS + ["season", "round", "session"]).copy()
    rows["session"] = "Race"
    rows["GridPosition"] = pd.to_numeric(q["Position"], errors="coerce").to_numpy()
    rows[["Q1", "Q2", "Q3"]] = pd.NaT
    if grid is not None:
        official = grid.set_index("Abbreviation")["GridPosition"]
        missing = set(rows["Abbreviation"]) - set(official.index)
        if missing:
            raise SystemExit(f"--grid is missing drivers: {sorted(missing)}")
        rows["GridPosition"] = rows["Abbreviation"].map(official).astype(float).to_numpy()
    rows[OUTCOME_COLS] = np.nan
    for col in ("ClassifiedPosition", "Status"):
        rows[col] = rows[col].astype("string")
    return rows


def prepare(raw: dict[str, pd.DataFrame], season: int, rnd: int,
            grid: pd.DataFrame | None = None) -> tuple[dict[str, pd.DataFrame], bool]:
    """Cut the tables off at the target race and blank its outcome.

    Returns the tables and whether the race has already been run. Either way
    the target race's result is removed before features are built, so a
    re-score of a past race sees exactly what a live run would have seen.
    """
    def upto(t: pd.DataFrame) -> pd.DataFrame:
        return t[(t["season"] < season) | ((t["season"] == season) & (t["round"] <= rnd))]

    raw = {name: upto(t).reset_index(drop=True) for name, t in raw.items()}
    race = raw["race"]
    target = (race["season"] == season) & (race["round"] == rnd)
    already_run = bool(target.any())
    if already_run and grid is None:
        race = race.copy()
        race.loc[target, OUTCOME_COLS] = np.nan
    else:
        race = pd.concat([race[~target], entry_list(raw, season, rnd, grid)], ignore_index=True)
    return {**raw, "race": race}, already_run


def score(season: int, rnd: int, grid: pd.DataFrame | None = None) -> pd.DataFrame:
    """Win probabilities for one race from the frozen setup, trained on the past only."""
    config = load_frozen()
    full = load_raw()
    raw, already_run = prepare(full, season, rnd, grid)
    features = build_features(raw)
    is_target = (features["season"] == season) & (features["round"] == rnd)
    train, target = features[~is_target], features[is_target]
    models = fit(train, config["features"], config["params"], season,
                 config["recency_decay"], tuple(config["seeds"]), allow_current_season=True)
    pred = predict(models, target, config["features"]).sort_values("rank")
    out = pred[RACE + ["event", "rank", "Abbreviation", "FullName", "TeamId", "grid_pos",
                       "quali_pos", "p"]].rename(columns={"p": "win_probability"})
    if already_run:  # report the outcome next to the prediction, never use it
        actual = full["race"][(full["race"]["season"] == season) & (full["race"]["round"] == rnd)]
        out = out.merge(actual[["Abbreviation", "Position"]].rename(
            columns={"Position": "actual_finish"}), on="Abbreviation", how="left")
    out.attrs.update(trained_races=train.groupby(RACE).ngroups, already_run=already_run)
    return out


def rounds_on_disk(year: int) -> int:
    """The latest round with race results in data/raw for this season (0 if none)."""
    path = RAW / f"race_{year}.parquet"
    return int(pd.read_parquet(path, columns=["round"])["round"].max()) if path.exists() else 0


def refresh(season: int, rnd: int) -> None:
    """Download what the target race needs from seasons after the locked test season.

    2019-2025 come from `make data` and are never re-downloaded here. Later
    seasons are fetched up to the target round, never fewer rounds than are
    already on disk, and earlier post-test seasons are completed first.
    """
    import fastf1

    for year in range(TEST_SEASON + 1, season + 1):
        if year == season:
            ingest([year], through=max(rnd, rounds_on_disk(year)))
        else:
            enable_cache(year)
            total = int(fastf1.get_event_schedule(year, include_testing=False)["RoundNumber"].max())
            if rounds_on_disk(year) < total:
                ingest([year])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("season", type=int)
    parser.add_argument("round", type=int)
    parser.add_argument("--grid", help="CSV with Abbreviation,GridPosition (0 = pit lane)")
    parser.add_argument("--no-download", action="store_true",
                        help="use the raw tables already in data/raw")
    args = parser.parse_args()

    if not args.no_download:
        refresh(args.season, args.round)

    grid = pd.read_csv(args.grid) if args.grid else None
    out = score(args.season, args.round, grid)
    event = out["event"].iloc[0]
    PREDICTIONS.mkdir(exist_ok=True)
    path = PREDICTIONS / f"{args.season}_R{args.round:02d}.csv"
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    kind = "re-score of a race already run" if out.attrs["already_run"] else "pre-race"
    with path.open("w") as fh:
        fh.write(f"# {args.season} round {args.round}: {event}. Generated {stamp} ({kind}); "
                 f"model trained on {out.attrs['trained_races']} earlier races.\n")
        out.to_csv(fh, index=False, float_format="%.4f")

    print(f"{args.season} round {args.round}: {event}  ({kind})")
    show = out.head(10).assign(win_probability=lambda d: d["win_probability"].map("{:.1%}".format))
    print(show.drop(columns=RACE + ["event", "FullName"]).to_string(index=False))
    print(f"\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
