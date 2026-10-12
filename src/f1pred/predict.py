"""Score one race from everything known before lights out.

    python -m f1pred.predict 2026 18                 # after qualifying, before the race
    python -m f1pred.predict 2026 18 --grid grid.csv # with the official starting grid
    python -m f1pred.predict 2026 17 --no-download   # re-score from data already on disk
    python -m f1pred.predict 2026 17 --season        # re-score every round so far with v1.0

Both frozen models are scored and written side by side: v2.0 (the current
model, ``config/model_v2.json``) and v1.0 (``config/model.json``). Each is
retrained on every race before the target, including earlier rounds of the
same season, so a new set of regulations is learned as the season goes on.
Predictions go to ``predictions/{season}_R{round}.csv``. Commit that file and
push it before lights out: a local commit time proves nothing, the push time
on GitHub does.

Only seasons after the locked 2025 test can be scored here. 2019-2025 belong
to validation and to the one locked run in ``evaluate.py``.

Until the race has been run there is no official grid in the data, so the
grid defaults to qualifying order. Pass ``--grid`` (columns ``Abbreviation``,
``GridPosition``; 0 for a pit-lane start) once penalties are applied.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from f1pred import baselines
from f1pred.config import FIRST_SEASON, RACE, RAW, ROOT, TEST_SEASON, load_frozen
from f1pred.experiments import V2_CONFIG, build_scorer
from f1pred.features import build_features
from f1pred.features_v2 import build_features_v2
from f1pred.ingest import RESULT_COLS, enable_cache, ingest, load_tables
from f1pred.train import fit, predict

PREDICTIONS = ROOT / "predictions"
OUTCOME_COLS = ["Position", "ClassifiedPosition", "Status", "Points", "Laps"]
V1, V2 = "v1.0", "v2.0"


def load_v2() -> dict:
    """The frozen version 2 setup."""
    if not V2_CONFIG.exists():
        raise FileNotFoundError(f"{V2_CONFIG} is missing: run `python -m f1pred.v2 freeze`")
    return json.loads(V2_CONFIG.read_text())


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


def _split(features: pd.DataFrame, season: int, rnd: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    is_target = (features["season"] == season) & (features["round"] == rnd)
    target = features[is_target].copy()
    assert target["won"].sum() == 0, "the target race's result must be hidden"
    return features[~is_target], target


def frames_v1(raw: dict[str, pd.DataFrame], season: int, rnd: int
              ) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """(training rows, target rows, fitted models) for v1.0 from prepared tables."""
    config = load_frozen()
    train, target = _split(build_features(raw), season, rnd)
    fitted = fit(train, config["features"], config["params"], season, config["recency_decay"],
                 tuple(config["seeds"]), allow_current_season=True)
    return train, target, fitted


def score_v1(raw: dict[str, pd.DataFrame], season: int, rnd: int) -> tuple[pd.DataFrame, int]:
    """v1.0's prediction frame for the target race, and how many races it trained on."""
    train, target, fitted = frames_v1(raw, season, rnd)
    pred = predict(fitted, target, load_frozen()["features"], name=V1)
    return pred, train.groupby(RACE).ngroups


def score_v2(raw: dict[str, pd.DataFrame], season: int, rnd: int) -> tuple[pd.DataFrame, int]:
    """v2.0's prediction frame for the target race, and how many races it trained on."""
    config = load_v2()
    train, target = _split(build_features_v2(raw), season, rnd)
    train = train[train["season"] >= config["train_from"]]
    scorer = build_scorer(config["model"], tuple(config["seeds"]), live=True)
    pred = scorer(train, target, config["features"], season).assign(model=V2)
    return pred, train.groupby(RACE).ngroups


def score(season: int, rnd: int, grid: pd.DataFrame | None = None) -> pd.DataFrame:
    """Win probabilities for one race from both frozen models, trained on the past only."""
    full = load_tables(range(FIRST_SEASON, season + 1))
    raw, already_run = prepare(full, season, rnd, grid)
    v1, n_races = score_v1(raw, season, rnd)
    keys = ["Abbreviation"]
    out = v1[RACE + ["event", "Abbreviation", "FullName", "TeamId", "grid_pos", "quali_pos"]]
    out = out.assign(p_v1=v1["p"], rank_v1=v1["rank"])
    order = "rank_v1"
    if V2_CONFIG.exists():
        v2, _ = score_v2(raw, season, rnd)
        out = out.merge(v2[keys + ["p", "rank"]].rename(columns={"p": "p_v2", "rank": "rank_v2"}),
                        on=keys, validate="one_to_one")
        order = "rank_v2"
    if already_run:  # report the outcome next to the prediction, never use it
        actual = full["race"][(full["race"]["season"] == season) & (full["race"]["round"] == rnd)]
        out = out.merge(actual[["Abbreviation", "Position"]].rename(
            columns={"Position": "actual_finish"}), on="Abbreviation", how="left")
    out = out.sort_values(order).reset_index(drop=True)
    out.attrs.update(trained_races=n_races, already_run=already_run)
    return out


def rescore_season(season: int, through: int, versions: tuple[str, ...] = (V1,)) -> pd.DataFrame:
    """Walk forward through a season that has been run.

    Each race is scored by the frozen setups trained on every earlier race, with
    that race's result blanked before features are built and restored only for
    scoring. Returns prediction frames for the chosen versions and the three
    baselines. This is an after-the-fact re-score, not a live prediction.
    """
    full = load_tables(range(FIRST_SEASON, season + 1))
    frames = []
    for rnd in range(1, through + 1):
        raw, already_run = prepare(full, season, rnd)
        if not already_run:
            raise SystemExit(f"{season} round {rnd} has no race result to score against.")
        actual = full["race"][(full["race"]["season"] == season) & (full["race"]["round"] == rnd)]
        finish = pd.to_numeric(actual.set_index("DriverId")["Position"], errors="coerce")

        def restore(pred: pd.DataFrame, finish: pd.Series = finish) -> pd.DataFrame:
            pred = pred.assign(finish_pos=pred["DriverId"].map(finish).to_numpy())
            return pred.assign(won=(pred["finish_pos"] == 1).astype(int))

        train, target, fitted = frames_v1(raw, season, rnd)
        if V1 in versions:
            name = baselines_name(V1, versions)
            frames.append(restore(predict(fitted, target, load_frozen()["features"], name=name)))
        if V2 in versions:
            frames.append(restore(score_v2(raw, season, rnd)[0]))
        scored = restore(target)
        frames += [baselines.pole_sitter(scored), baselines.form_leader(scored),
                   baselines.grid_logreg(train, scored)]
    return pd.concat(frames, ignore_index=True)


def baselines_name(version: str, versions: tuple[str, ...]) -> str:
    """v1.0 keeps its report name ("XGBoost") when it is the only model in the table."""
    from f1pred.train import MODEL_NAME

    return MODEL_NAME if versions == (V1,) else version


def season_report(season: int, through: int) -> None:
    """Re-score a season so far with v1.0, save it and fill the README's block for it."""
    from f1pred import evaluate as ev

    pred = rescore_season(season, through)
    PREDICTIONS.mkdir(exist_ok=True)
    pred.to_csv(PREDICTIONS / f"{season}_rescored.csv", index=False, float_format="%.6f")
    preds = ev.split_models(pred)
    model, pole = preds[ev.MODEL_NAME], preds[ev.POLE]

    def slot(grid: float, field: int) -> str:
        return ev.grid_slot(pd.Series([grid]), field)[0]

    rows = []
    for rnd, race in model.groupby("round"):
        pick, winner = race[race["rank"] == 1].iloc[0], race[race["won"] == 1].iloc[0]
        first = pole[(pole["round"] == rnd) & (pole["rank"] == 1)].iloc[0]
        rows.append({
            "Rd": rnd, "Race": pick["event"].replace(" Grand Prix", ""),
            "Winner (grid)": winner["Abbreviation"] + slot(winner["grid_pos"], len(race)),
            "Model pick (grid)": pick["Abbreviation"] + slot(pick["grid_pos"], len(race)),
            "P(pick)": f"{pick['p']:.0%}", "Pole sitter": first["Abbreviation"],
            "Pick right?": "yes" if pick["won"] else "**no**",
            "Pole right?": "yes" if first["won"] else "no",
            "P(winner)": f"{winner['p']:.0%}", "Winner's rank": int(winner["rank"]),
        })
    v = ev.log_loss_vs_logreg(preds)
    stamp = datetime.now(timezone.utc).date().isoformat()
    note = (f"Rounds 1-{through} of {season}, re-scored on {stamp} after the races were run. "
            f"Each race was scored by the frozen v1.0 setup retrained on every earlier race, "
            f"with that race's result hidden. Log loss against the grid-only logistic "
            f"regression: {v['diff']:+.4f} ({v['relative']:+.1%}; 95% CI {v['lo']:+.4f} to "
            f"{v['hi']:+.4f}). {ev.loss_gap_sensitivity(preds)}")
    sections = {f"season_{season}": ev.md_table(ev.results_table(preds)),
                f"season_{season}_notes": note,
                f"season_{season}_races": ev.md_table(pd.DataFrame(rows))}
    (PREDICTIONS / f"{season}_rescored.md").write_text(
        "\n\n".join(sections.values()) + "\n")
    ev.update_readme(sections)
    print("\n\n".join(sections.values()))


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

    reports = []
    for year in range(TEST_SEASON + 1, season + 1):
        if year == season:
            reports += ingest([year], through=max(rnd, rounds_on_disk(year)))
        else:
            enable_cache(year)
            total = int(fastf1.get_event_schedule(year, include_testing=False)["RoundNumber"].max())
            if rounds_on_disk(year) < total:
                reports += ingest([year])
    for line in (s for rep in reports for s in rep.skips):
        print("  " + line)


def check_ready(season: int, rnd: int) -> None:
    """Stop unless every earlier race of the season and the target's qualifying are on disk.

    A missing earlier race would shift every form window by one and the
    prediction would still be written, silently wrong.
    """
    def rounds(table: str) -> set[int]:
        path = RAW / f"{table}_{season}.parquet"
        return set(pd.read_parquet(path, columns=["round"])["round"]) if path.exists() else set()

    missing_races = sorted(set(range(1, rnd)) - rounds("race"))
    missing_quali = sorted(set(range(1, rnd + 1)) - rounds("quali"))
    if missing_races or missing_quali:
        raise SystemExit(
            f"Cannot score {season} round {rnd}: race results missing for rounds "
            f"{missing_races or 'none'}, qualifying missing for rounds {missing_quali or 'none'}. "
            "Rerun without --no-download once the results API has them.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("season", type=int)
    parser.add_argument("round", type=int)
    parser.add_argument("--grid", help="CSV with Abbreviation,GridPosition (0 = pit lane)")
    parser.add_argument("--no-download", action="store_true",
                        help="use the raw tables already in data/raw")
    parser.add_argument("--season", dest="whole_season", action="store_true",
                        help="re-score every round up to this one with v1.0 (races already run)")
    args = parser.parse_args()

    if args.season <= TEST_SEASON:
        raise SystemExit(
            f"predict.py scores seasons after {TEST_SEASON} only. {TEST_SEASON} is the locked "
            "test season (one run, in evaluate.py) and earlier seasons are validation data.")
    if not args.no_download:
        refresh(args.season, args.round)
    check_ready(args.season, args.round)
    if args.whole_season:
        season_report(args.season, args.round)
        return

    grid = pd.read_csv(args.grid) if args.grid else None
    out = score(args.season, args.round, grid)
    event = out["event"].iloc[0]
    PREDICTIONS.mkdir(exist_ok=True)
    path = PREDICTIONS / f"{args.season}_R{args.round:02d}.csv"
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    kind = "re-score of a race already run" if out.attrs["already_run"] else "pre-race"
    with path.open("w") as fh:
        fh.write(f"# {args.season} round {args.round}: {event}. Generated {stamp} ({kind}); "
                 f"models trained on {out.attrs['trained_races']} earlier races.\n")
        out.to_csv(fh, index=False, float_format="%.4f")

    print(f"{args.season} round {args.round}: {event}  ({kind})")
    show = out.head(10).drop(columns=RACE + ["event", "FullName"])
    for col in ("p_v1", "p_v2"):
        if col in show:
            show[col] = show[col].map("{:.1%}".format)
    print(show.to_string(index=False))
    print(f"\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
