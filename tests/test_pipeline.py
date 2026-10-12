"""End-to-end checks on the synthetic championship: features, baselines,
walk-forward training and metrics all run and obey their contracts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from f1pred import baselines
from f1pred.config import FROZEN_CONFIG, RACE, TEST_SEASON
from f1pred.features import FEATURE_GROUPS, FEATURES, build_features, circuit_id, load_features
from f1pred.metrics import paired_bootstrap, per_race, summarise
from f1pred.train import PARAMS, fit, recency_weights, walk_forward
from synthetic import make_raw

FAST = {**PARAMS, "n_estimators": 40}


@pytest.fixture(scope="module")
def raw() -> dict[str, pd.DataFrame]:
    return make_raw(seed=1)


@pytest.fixture(scope="module")
def features(raw) -> pd.DataFrame:
    return build_features(raw)


def test_one_row_per_driver_per_race_and_one_winner(raw, features):
    assert len(features) == len(raw["race"])
    assert not features.duplicated(RACE + ["DriverId"]).any()
    assert (features.groupby(RACE)["won"].sum() == 1).all()
    assert set(FEATURES) <= set(features.columns)
    assert len(FEATURES) == 19 == sum(len(v) for v in FEATURE_GROUPS.values())


def test_pit_lane_start_goes_behind_the_grid(raw, features):
    pit = raw["race"][raw["race"]["GridPosition"] == 0]
    assert len(pit) > 0
    merged = pit.merge(features, on=RACE + ["DriverId"])
    assert (merged["grid_pos"] == 21).all()
    assert features["grid_pos"].min() == 1


def test_renamed_team_keeps_its_form(features):
    first_after_rename = features[(features["season"] == 2022) & (features["round"] == 1)
                                  & (features["team_lineage"] == "racing_bulls")]
    assert len(first_after_rename) == 2
    assert first_after_rename["team_points_5"].notna().all()
    assert features["team_lineage"].nunique() == 5  # three names, one lineage


def test_circuit_ids_survive_renamed_venues():
    assert circuit_id("Monte Carlo") == circuit_id("Monaco") == "monaco"
    assert circuit_id("Miami Gardens") == circuit_id("Miami") == "miami"
    assert circuit_id("Marina Bay") == circuit_id("Singapore") == "singapore"
    assert circuit_id("Yas Island") == circuit_id("Yas Marina")
    assert circuit_id("São Paulo") == "sao_paulo"
    assert circuit_id("Sakhir", "Sakhir Grand Prix") != circuit_id("Sakhir", "Bahrain Grand Prix")


def test_sprint_and_missing_practice_are_empty_not_errors(features):
    sprint_weekend = features["round"] == 3
    assert features.loc[sprint_weekend, "sprint_pos"].notna().all()
    assert features.loc[~sprint_weekend, "sprint_pos"].isna().all()
    no_practice = features[features["round"] == 6]
    assert no_practice["longrun_gap_pct"].isna().all()
    assert (no_practice["has_longrun"] == 0).all()


def test_first_race_has_no_history(features):
    first = features[(features["season"] == 2021) & (features["round"] == 1)]
    history = FEATURE_GROUPS["form"] + ["drv_track_avg_finish", "track_pole_win_rate"]
    assert first[history].isna().all().all()


def test_walk_forward_gives_one_winner_per_race(features):
    pred = walk_forward(features, FEATURES, FAST, folds=(2022, 2023), seeds=(0, 1))
    assert set(pred["season"]) == {2022, 2023}
    assert np.allclose(pred.groupby(RACE)["p"].sum(), 1.0)
    assert pred["p"].between(0, 1).all()
    assert (pred.groupby(RACE)["rank"].min() == 1).all()
    assert not pred.duplicated(RACE + ["rank"]).any()
    stats = summarise(pred)
    assert stats["races"] == 12 and 0 <= stats["winner_hit"] <= stats["top3_hit"] <= 1
    assert stats["log_loss"] > 0


def test_training_never_sees_the_target_season(features):
    with pytest.raises(ValueError):
        fit(features, FEATURES, FAST, target_season=2023, seeds=(0,))


def test_walk_forward_refuses_the_locked_season(features):
    locked = features.assign(season=features["season"] + (TEST_SEASON - 2023))
    with pytest.raises(ValueError):
        walk_forward(locked, FEATURES, FAST, folds=(TEST_SEASON,), seeds=(0,))


def test_recency_weights():
    seasons = pd.Series([2019, 2020, 2021])
    assert np.allclose(recency_weights(seasons, 2022, 0.7), [0.49, 0.7, 1.0])


def test_baselines(features):
    val = features[features["season"] == 2023]
    pole = baselines.pole_sitter(val)
    picks = pole[pole["rank"] == 1]
    assert (picks["grid_pos"] == val.groupby(RACE)["grid_pos"].transform("min")[picks.index]).all()
    stats = summarise(pole)
    assert stats["hit_when_pole_lost"] == 0.0 or stats["pole_lost_races"] == 0
    assert np.isnan(summarise(pole)["log_loss"])

    form = baselines.form_leader(val)
    assert (form.groupby(RACE)["rank"].min() == 1).all()

    logreg = baselines.walk_forward_logreg(features, folds=(2022, 2023))
    assert np.allclose(logreg.groupby(RACE)["p"].sum(), 1.0)
    assert summarise(logreg)["log_loss"] > 0
    assert baselines.pick_hit_rate(val, "grid_pos") == summarise(pole)["winner_hit"]


def test_metrics_on_a_hand_made_example():
    rows = []
    # race 1: pick wins from pole; race 2: pick (P1) loses, winner ranked 3rd by the model
    for rnd, winner, probs in [(1, 0, [0.7, 0.2, 0.1, 0.0]), (2, 2, [0.5, 0.3, 0.15, 0.05])]:
        for i, p in enumerate(probs):
            rows.append({"season": 2030, "round": rnd, "DriverId": f"d{i}", "grid_pos": i + 1,
                         "quali_pos": i + 1, "won": int(i == winner), "p": p, "rank": i + 1})
    pred = pd.DataFrame(rows)
    stats = summarise(pred)
    assert stats["winner_hit"] == 0.5 and stats["top3_hit"] == 1.0
    assert stats["pole_lost_races"] == 1 and stats["hit_when_pole_lost"] == 0.0
    expected_brier = np.mean((pred["p"] - pred["won"]) ** 2)
    assert stats["brier"] == pytest.approx(expected_brier)
    p = pred["p"].clip(1e-6, 1 - 1e-6)
    expected_ll = -np.mean(pred["won"] * np.log(p) + (1 - pred["won"]) * np.log(1 - p))
    assert stats["log_loss"] == pytest.approx(expected_ll)

    race = per_race(pred)
    mean, lo, hi = paired_bootstrap(race["hit"], race["pole_won"], n=2000)
    assert mean == 0.0 and lo <= mean <= hi


def test_dev_loader_hides_the_test_season(tmp_path, features):
    path = tmp_path / "features.parquet"
    pd.concat([features, features.assign(season=TEST_SEASON)]).to_parquet(path)
    assert TEST_SEASON not in set(load_features(path=path)["season"])
    assert TEST_SEASON in set(load_features(include_test=True, path=path)["season"])


def test_frozen_config_uses_known_features():
    if not FROZEN_CONFIG.exists():
        pytest.skip("model not frozen yet")
    config = json.loads(FROZEN_CONFIG.read_text())
    assert set(config["features"]) <= set(FEATURES)
    assert "scale_pos_weight" not in config["params"]


def _target(df: pd.DataFrame, season: int, rnd: int) -> pd.DataFrame:
    rows = df[(df["season"] == season) & (df["round"] == rnd)]
    return rows.sort_values("DriverId").reset_index(drop=True)


def test_rescoring_a_past_race_hides_its_result(raw, features):
    from f1pred.predict import prepare

    cut, already_run = prepare(raw, 2023, 4)
    assert already_run
    race = cut["race"]
    assert not ((race["season"] == 2023) & (race["round"] > 4)).any()
    assert _target(race, 2023, 4)["Position"].isna().all()
    rebuilt = build_features(cut)
    pd.testing.assert_frame_equal(_target(rebuilt, 2023, 4)[FEATURES],
                                  _target(features, 2023, 4)[FEATURES])
    assert _target(rebuilt, 2023, 4)["won"].sum() == 0


def test_upcoming_race_is_built_from_the_qualifying_entry_list(raw, features):
    from f1pred.predict import prepare

    race = raw["race"]
    not_run = {**raw, "race": race[~((race["season"] == 2023) & (race["round"] >= 5))]}
    cut, already_run = prepare(not_run, 2023, 5)
    assert not already_run
    rebuilt = build_features(cut)
    # round 5 has no grid penalties in the synthetic data, so grid = qualifying order
    pd.testing.assert_frame_equal(_target(rebuilt, 2023, 5)[FEATURES],
                                  _target(features, 2023, 5)[FEATURES])

    grid = _target(cut["race"], 2023, 5)[["Abbreviation"]].assign(GridPosition=0)
    pit_lane, _ = prepare(not_run, 2023, 5, grid=grid)
    assert (_target(build_features(pit_lane), 2023, 5)["grid_pos"] == 21).all()


def test_model_code_unchanged_since_the_freeze():
    """After the 2025 run, the files that define the model must still match v1.0."""
    import subprocess

    from f1pred.config import REPORTS, ROOT

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)

    if not (REPORTS / "test_2025" / "metrics.json").exists():
        pytest.skip("2025 test not run yet")
    if git("rev-parse", "-q", "--verify", "refs/tags/v1.0").returncode != 0:
        pytest.skip("v1.0 tag not available in this checkout")
    locked = ["config", "src/f1pred/config.py", "src/f1pred/features.py",
              "src/f1pred/train.py", "src/f1pred/metrics.py"]
    diff = git("diff", "--stat", "v1.0", "--", *locked)
    assert diff.stdout.strip() == "", f"model files changed after the freeze:\n{diff.stdout}"

    # baselines.py: only the command-line printout (main) may differ from the tag.
    def before_main(text: str) -> str:
        return text.split("\ndef main()")[0]

    tagged = git("show", "v1.0:src/f1pred/baselines.py").stdout
    current = (ROOT / "src" / "f1pred" / "baselines.py").read_text()
    assert before_main(tagged) == before_main(current), \
        "baseline functions changed after the freeze"


def test_intervals_are_exact():
    from f1pred.intervals import paired_difference, wilson

    lo, hi = wilson(15, 24)
    assert lo == pytest.approx(0.4271, abs=1e-4) and hi == pytest.approx(0.7884, abs=1e-4)
    assert wilson(24, 24)[1] == 1.0 and wilson(0, 24)[0] == 0.0

    races = pd.RangeIndex(24)
    model = pd.Series([1] * 15 + [0] * 9, index=races)
    pole = pd.Series([1] * 13 + [0, 0] + [1] * 3 + [0] * 6, index=races)  # 2 gained, 3 lost
    mean, lo, hi = paired_difference(model, pole)
    assert mean == pytest.approx(-1 / 24)
    # brute force: the summed difference is a sum of 24 draws from {+1 x2, -1 x3, 0 x19}
    rng = np.random.default_rng(0)
    diff = (model - pole).to_numpy()
    sims = diff[rng.integers(0, 24, size=(400_000, 24))].mean(axis=1)
    assert lo == pytest.approx(np.percentile(sims, 2.5, method="inverted_cdf"))
    assert hi == pytest.approx(np.percentile(sims, 97.5, method="inverted_cdf"))


def test_results_table_handles_a_perfect_pole_record(features):
    from f1pred import evaluate as ev

    val = features[features["season"] == 2023]
    pole = baselines.pole_sitter(val)
    winners_from_pole = pole.assign(won=(pole["rank"] == 1).astype(int),
                                    finish_pos=pole["rank"].astype(float))
    table = ev.results_table({ev.POLE: winners_from_pole,
                              ev.FORM: baselines.form_leader(val).assign(
                                  won=winners_from_pole["won"])})
    assert table.loc[0, "Hits when pole lost"] == "0 of 0"
