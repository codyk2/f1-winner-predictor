"""Version 2 features: the same three leakage checks as v1, plus unit checks of
the rebuilt qualifying and long-run pace features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pred.features import FEATURES
from f1pred.features_v2 import (
    V2_FEATURES,
    V2_SEASONS,
    build_features_v2,
    longrun_v2,
    quali_v2,
)
from f1pred.ingest import load_tables
from synthetic import make_raw
from test_leakage import (
    REAL_TARGETS,
    SYNTHETIC_TARGETS,
    check_lights_out,
    check_no_future_data,
    check_no_same_race_data,
)

ALL = FEATURES + V2_FEATURES


@pytest.fixture(scope="module")
def synthetic_raw() -> dict[str, pd.DataFrame]:
    return make_raw(seed=0)


@pytest.fixture(scope="module")
def real_raw() -> dict[str, pd.DataFrame]:
    try:
        raw = load_tables(V2_SEASONS)
    except FileNotFoundError:
        pytest.skip("data/raw not built")
    if not set(V2_SEASONS) <= set(raw["race"]["season"]):
        pytest.skip("2014-2018 history not downloaded; run `make history`")
    return raw


def test_v2_no_future_data_synthetic(synthetic_raw):
    check_no_future_data(synthetic_raw, SYNTHETIC_TARGETS, build_features_v2, ALL)


def test_v2_no_same_race_data_synthetic(synthetic_raw):
    check_no_same_race_data(synthetic_raw, SYNTHETIC_TARGETS, build_features_v2, ALL)


def test_v2_lights_out_synthetic(synthetic_raw):
    check_lights_out(synthetic_raw, SYNTHETIC_TARGETS, build_features_v2, ALL)


def test_v2_lights_out_real(real_raw):
    targets = REAL_TARGETS + [(2014, 1), (2018, 21), (2019, 1), (2020, 11), (2025, 24)]
    check_lights_out(real_raw, targets, build_features_v2, ALL)


def test_v2_no_same_race_data_real(real_raw):
    check_no_same_race_data(real_raw, REAL_TARGETS, build_features_v2, ALL)


def test_v2_features_are_populated(synthetic_raw):
    table = build_features_v2(synthetic_raw)
    later = table[table["season"] > table["season"].min()]
    filled = later[V2_FEATURES].notna().mean()
    assert (filled > 0.3).all(), f"empty v2 features in the fixture: {filled[filled <= 0.3]}"
    assert not table.duplicated(["season", "round", "DriverId"]).any()


def test_v2_refuses_race_laps(synthetic_raw):
    laps = synthetic_raw["laps"]
    bad = {**synthetic_raw, "laps": pd.concat([laps, laps.head(3).assign(session="Race")])}
    with pytest.raises(ValueError):
        build_features_v2(bad)


def _quali(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["DriverId", "Q1", "Q2", "Q3"]).assign(season=2030, round=1)
    for seg in ("Q1", "Q2", "Q3"):
        df[seg] = pd.to_timedelta(df[seg], unit="s")
    return df


def test_quali_gap_is_measured_within_the_segment():
    # The track slows before Q3: the pole lap (90.5) is slower than the best Q2 lap (90.0).
    q = quali_v2(_quali([
        ("pole", 91.0, 90.2, 90.5),
        ("second", 91.2, 90.0, 90.95),
        ("out_q2", 91.1, 90.9, np.nan),
        ("out_q1", 92.0, np.nan, np.nan),
        ("no_time", np.nan, np.nan, np.nan),
    ])).set_index("DriverId")
    assert q.loc["pole", "quali_seg_gap_pct"] == 0
    assert q.loc["second", "quali_seg_gap_pct"] == pytest.approx((90.95 / 90.5 - 1) * 100)
    assert q.loc["pole", "quali_margin_pct"] == pytest.approx(-(90.95 / 90.5 - 1) * 100)
    assert q.loc["second", "quali_margin_pct"] == q.loc["second", "quali_seg_gap_pct"]
    # eliminated drivers are compared with the fastest time anyone set in their segment
    assert q.loc["out_q2", "quali_seg_gap_pct"] == pytest.approx((90.9 / 90.0 - 1) * 100)
    assert q.loc["out_q1", "quali_seg_gap_pct"] == pytest.approx((92.0 / 91.0 - 1) * 100)
    assert q.loc["no_time", ["quali_seg_gap_pct", "quali_margin_pct"]].isna().all()


def _laps(runs: dict[str, list[float]], session: str = "Practice 2") -> pd.DataFrame:
    rows = []
    for driver, times in runs.items():
        for k, sec in enumerate(times):
            rows.append({"season": 2030, "round": 1, "session": session, "DriverId": driver,
                         "Stint": 1.0, "Compound": "MEDIUM",
                         "LapTime": pd.to_timedelta(sec, unit="s"),
                         "IsAccurate": True, "TrackStatus": "1",
                         "PitInTime": pd.NaT, "PitOutTime": pd.NaT, "LapNumber": float(k + 1)})
    df = pd.DataFrame(rows)
    for col in ("PitInTime", "PitOutTime"):
        df[col] = pd.to_timedelta(df[col].astype(object))
    return df


def test_long_run_benchmark_is_the_field_median_and_ignores_short_runs():
    runs = {
        "a": [100.0] * 8, "b": [101.0] * 8, "c": [102.0] * 8, "d": [103.0] * 8,
        "e": [104.0] * 7 + [120.0],      # one traffic lap is dropped, seven remain
        "push": [95.0] * 5,              # five quick laps: not a long run
    }
    out = longrun_v2(_laps(runs)).set_index("DriverId")["longrun_gap_v2"]
    assert "push" not in out.index
    assert out["c"] == 0                                   # the median driver
    assert out["a"] == pytest.approx((100 / 102 - 1) * 100)
    assert out["e"] == pytest.approx((104 / 102 - 1) * 100)


def test_long_run_session_is_the_one_most_drivers_used():
    fp1 = _laps({d: [100.0 + i] * 8 for i, d in enumerate("abcde")}, "Practice 1")
    fp2 = _laps({d: [90.0 + i] * 8 for i, d in enumerate("abcd")}, "Practice 2")
    fp2 = fp2[fp2["DriverId"] != "a"]            # FP2 now has too few drivers for a benchmark
    out = longrun_v2(pd.concat([fp1, fp2])).set_index("DriverId")["longrun_gap_v2"]
    assert out["a"] == pytest.approx((100 / 102 - 1) * 100)   # FP1 values are used
    assert set(out.index) == set("abcde")


# ----------------------------------------------------------------- models

@pytest.fixture(scope="module")
def table(synthetic_raw) -> pd.DataFrame:
    return build_features_v2(synthetic_raw)


FAST = {"n_estimators": 30, "max_depth": 2, "learning_rate": 0.1, "subsample": 0.8,
        "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1.0,
        "eval_metric": "logloss"}


def _scorers() -> dict:
    from f1pred import models

    return {
        "binary": models.xgb_binary(FAST, (0, 1)),
        "binary_monotone": models.xgb_binary(FAST, (0, 1), monotone=True),
        "softmax": models.xgb_softmax(FAST, (0, 1)),
        "softmax_monotone": models.xgb_softmax(FAST, (0,), monotone=True),
        "cond_logit": models.cond_logit(1.0),
        "blend": models.blend(models.xgb_softmax(FAST, (0,)), models.cond_logit(1.0)),
        "tempered": models.tempered(models.cond_logit(1.0), inner_seasons=1),
    }


@pytest.mark.parametrize("name", ["binary", "binary_monotone", "softmax", "softmax_monotone",
                                  "cond_logit", "blend", "tempered"])
def test_every_scorer_gives_one_winner_per_race(table, name):
    from f1pred.experiments import walk_forward

    pred = walk_forward(table, ALL, _scorers()[name], train_from=2021, folds=(2022, 2023))
    assert set(pred["season"]) == {2022, 2023}
    assert np.allclose(pred.groupby(["season", "round"])["p"].sum(), 1.0)
    assert pred["p"].between(0, 1).all()
    assert not pred.duplicated(["season", "round", "rank"]).any()
    top = pred[pred["rank"] == 1]
    assert (top["p"].to_numpy() >= pred.groupby(["season", "round"])["p"].max().to_numpy()
            - 1e-12).all()


def test_softmax_and_logit_gradient():
    from scipy import optimize

    from f1pred.models import EPS, _softmax

    rng = np.random.default_rng(0)
    group = np.repeat(np.arange(6), 5)
    x = rng.normal(size=(30, 3))
    won = np.zeros(30)
    won[np.arange(6) * 5 + rng.integers(0, 5, 6)] = 1
    p = _softmax(rng.normal(size=30), group)
    assert np.allclose(np.bincount(group, weights=p), 1.0)

    def loss(w):
        return -np.log(np.clip(_softmax(x @ w, group)[won == 1], EPS, 1)).sum()

    def grad(w):
        return x.T @ (_softmax(x @ w, group) - won)

    assert optimize.check_grad(loss, grad, rng.normal(size=3)) < 1e-4


def test_conditional_logit_learns_an_obvious_signal():
    from f1pred.models import cond_logit

    rng = np.random.default_rng(1)
    rows = []
    for season in (2030, 2031):
        for rnd in range(1, 21):
            pace = rng.normal(size=8)
            winner = int(np.argmin(pace + rng.normal(0, 0.3, 8)))
            for i in range(8):
                rows.append({"season": season, "round": rnd, "event": "x", "DriverId": f"d{i}",
                             "Abbreviation": f"D{i}", "FullName": f"D{i}", "TeamId": "t",
                             "grid_pos": float(i + 1), "quali_pos": float(i + 1),
                             "pace": pace[i], "noise": rng.normal(), "won": int(i == winner),
                             "finish_pos": 1.0 if i == winner else 2.0})
    df = pd.DataFrame(rows)
    train, val = df[df["season"] == 2030], df[df["season"] == 2031]
    pred = cond_logit(1.0)(train, val, ["pace", "noise"], 2031)
    hit = pred[pred["rank"] == 1]["won"].mean()
    assert hit > 0.6                               # the fastest car usually wins here


def test_v2_walk_forward_refuses_2026(table):
    from f1pred import models
    from f1pred.experiments import walk_forward

    later = table.assign(season=table["season"] + 3)     # 2024-2026
    with pytest.raises(ValueError):
        walk_forward(later, ALL, models.cond_logit(1.0), train_from=2024, folds=(2025,))


def test_keep_rule_needs_a_gain_above_the_noise_and_three_folds():
    from f1pred.experiments import FOLDS, MIN_GAIN, better

    old = {"log_loss": 0.1000, **{f"log_loss_{s}": 0.1 for s in FOLDS}}
    small = {"log_loss": 0.1000 - MIN_GAIN / 2, **{f"log_loss_{s}": 0.09 for s in FOLDS}}
    two_folds = {"log_loss": 0.0900, **{f"log_loss_{s}": v for s, v in
                                        zip(FOLDS, (0.05, 0.05, 0.12, 0.12), strict=True)}}
    good = {"log_loss": 0.0950, **{f"log_loss_{s}": v for s, v in
                                   zip(FOLDS, (0.09, 0.09, 0.09, 0.11), strict=True)}}
    assert better(small, old) == (False, 4)
    assert better(two_folds, old) == (False, 2)
    assert better(good, old) == (True, 3)


# ------------------------------------------------------------ live scoring

def _fake_v2() -> dict:
    return {"features": ALL, "train_from": 2021, "seeds": [0, 1],
            "model": {"kind": "binary", "params": FAST, "monotone": True}}


def test_v2_scores_a_race_with_its_result_hidden(synthetic_raw, monkeypatch):
    from f1pred import predict

    monkeypatch.setattr(predict, "load_v2", _fake_v2)
    raw, already_run = predict.prepare(synthetic_raw, 2023, 4)
    assert already_run
    pred, trained_on = predict.score_v2(raw, 2023, 4)
    assert trained_on == 15                       # six + six + three earlier races
    assert (pred["model"] == predict.V2).all() and pred["won"].sum() == 0
    assert np.isclose(pred["p"].sum(), 1.0) and sorted(pred["rank"]) == list(range(1, 11))

    # the same race with a shuffled result must give the same probabilities
    race = synthetic_raw["race"].copy()
    idx = race.index[(race["season"] == 2023) & (race["round"] == 4)]
    race.loc[idx, ["Position", "ClassifiedPosition", "Points"]] = \
        race.loc[idx[::-1], ["Position", "ClassifiedPosition", "Points"]].to_numpy()
    again, _ = predict.score_v2(predict.prepare({**synthetic_raw, "race": race}, 2023, 4)[0],
                                2023, 4)
    assert np.allclose(pred.sort_values("DriverId")["p"], again.sort_values("DriverId")["p"])


def test_monotone_constraints_follow_the_feature_order():
    from f1pred.models import MONOTONE, _constraints

    feats = ["drv_wins_5", "has_longrun", "grid_pos", "longrun_gap_v2"]
    assert _constraints(feats) == "(1,0,-1,-1)"
    assert set(MONOTONE) <= set(ALL)


def test_v2_files_unchanged_since_its_tag():
    """Once v2 has been confirmed on 2026, its model files must still match v2.0."""
    import subprocess

    from f1pred.config import ROOT
    from f1pred.experiments import V2_DIR

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)

    if not list(V2_DIR.glob("confirmation_*.json")):
        pytest.skip("v2 not confirmed yet")
    if git("rev-parse", "-q", "--verify", "refs/tags/v2.0").returncode != 0:
        pytest.skip("v2.0 tag not available in this checkout")
    locked = ["config/model_v2.json", "src/f1pred/features_v2.py", "src/f1pred/models.py"]
    diff = git("diff", "--stat", "v2.0", "--", *locked)
    assert diff.stdout.strip() == "", f"v2 model files changed after the tag:\n{diff.stdout}"
