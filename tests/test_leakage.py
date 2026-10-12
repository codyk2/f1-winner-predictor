"""Leakage tests: every feature must be computable before lights out.

Three checks, for several target races:

1. No future data. Features built from all races equal features built from only
   the races up to the target.
2. No same-race data. Shuffling the target race's finishing order leaves its
   features unchanged.
3. Lights out. The strictest form: delete every later row from every table AND
   blank the target race's result (position, classification, status, points,
   laps). The target's features must not move. A permutation alone would miss
   race-level facts such as how many cars retired.

They run on a synthetic championship everywhere and, when ``data/raw`` has been
built, on the real 2019-2025 tables as well. A fourth test checks that the
practice and weather tables hold no race session at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pred.config import ALL_SEASONS
from f1pred.features import FEATURES, build_features
from f1pred.ingest import load_tables
from synthetic import make_raw

OUTCOME_COLS = ["Position", "ClassifiedPosition", "Status", "Points", "Laps"]
REAL_TARGETS = [(2023, 10), (2021, 1), (2022, 4), (2024, 24), (2025, 13)]
SYNTHETIC_TARGETS = [(2021, 1), (2022, 2), (2022, 3), (2023, 4), (2023, 6)]


def _real_raw() -> dict[str, pd.DataFrame] | None:
    try:
        return load_tables(ALL_SEASONS)
    except FileNotFoundError:
        return None


@pytest.fixture(scope="module")
def synthetic_raw() -> dict[str, pd.DataFrame]:
    return make_raw(seed=0)


@pytest.fixture(scope="module")
def real_raw() -> dict[str, pd.DataFrame]:
    raw = _real_raw()
    if raw is None:
        pytest.skip("data/raw not built; run `make data` to test on real sessions")
    return raw


def _race(df: pd.DataFrame, season: int, rnd: int) -> pd.DataFrame:
    rows = df[(df["season"] == season) & (df["round"] == rnd)]
    return rows.sort_values("DriverId").reset_index(drop=True)


def _up_to(raw: dict[str, pd.DataFrame], season: int, rnd: int) -> dict[str, pd.DataFrame]:
    """Every table cut off after the target race, as if later races never happened."""
    return {name: t[(t["season"] < season) | ((t["season"] == season) & (t["round"] <= rnd))]
            .reset_index(drop=True) for name, t in raw.items()}


def _shuffle_result(raw: dict[str, pd.DataFrame], season: int, rnd: int) -> dict[str, pd.DataFrame]:
    """Hand the target race's finishing order to different drivers."""
    race = raw["race"].copy()
    idx = race.index[(race["season"] == season) & (race["round"] == rnd)]
    perm = np.random.default_rng(season * 100 + rnd).permutation(len(idx))
    assert (perm != np.arange(len(idx))).any()
    race.loc[idx, OUTCOME_COLS] = race.loc[idx[perm], OUTCOME_COLS].to_numpy()
    return {**raw, "race": race}


def _assert_same_features(full: pd.DataFrame, other: pd.DataFrame, season: int, rnd: int) -> None:
    a, b = _race(full, season, rnd), _race(other, season, rnd)
    assert len(a) > 0, f"no rows for {season} round {rnd}"
    assert a["DriverId"].tolist() == b["DriverId"].tolist()
    pd.testing.assert_frame_equal(a[FEATURES], b[FEATURES], check_exact=False, rtol=1e-12)


def check_no_future_data(raw: dict[str, pd.DataFrame], targets: list[tuple[int, int]]) -> None:
    full = build_features(raw)
    for season, rnd in targets:
        truncated = build_features(_up_to(raw, season, rnd))
        _assert_same_features(full, truncated, season, rnd)


def check_no_same_race_data(raw: dict[str, pd.DataFrame], targets: list[tuple[int, int]]) -> None:
    full = build_features(raw)
    for season, rnd in targets:
        shuffled = build_features(_shuffle_result(raw, season, rnd))
        _assert_same_features(full, shuffled, season, rnd)
        # The shuffle must be real: the target changes, and so do later races.
        assert not _race(full, season, rnd)["finish_pos"].equals(
            _race(shuffled, season, rnd)["finish_pos"])
        later_full = full[(full["season"] > season)
                          | ((full["season"] == season) & (full["round"] > rnd))]
        if len(later_full):
            later_shuffled = shuffled.loc[later_full.index]
            assert not later_full[FEATURES].equals(later_shuffled[FEATURES]), \
                "history features ignore past results, so this test proves nothing"


def check_lights_out(raw: dict[str, pd.DataFrame], targets: list[tuple[int, int]],
                     build=build_features, features: list[str] = FEATURES) -> None:
    full = build(raw)
    for season, rnd in targets:
        cut = _up_to(raw, season, rnd)
        race = cut["race"].copy()
        target = (race["season"] == season) & (race["round"] == rnd)
        race.loc[target, ["Position", "Points", "Laps"]] = np.nan
        race.loc[target, ["ClassifiedPosition", "Status"]] = pd.NA
        blind = build({**cut, "race": race})
        a, b = _race(full, season, rnd), _race(blind, season, rnd)
        assert len(a) > 0 and a["DriverId"].tolist() == b["DriverId"].tolist()
        pd.testing.assert_frame_equal(a[features], b[features], check_exact=False, rtol=1e-12)
        assert b["won"].sum() == 0, "the blanked race must have no winner"


def test_lights_out_synthetic(synthetic_raw):
    check_lights_out(synthetic_raw, SYNTHETIC_TARGETS)


def test_lights_out_real(real_raw):
    check_lights_out(real_raw, REAL_TARGETS + [(2019, 1), (2020, 11), (2023, 4), (2025, 24)])


def test_only_pre_race_sessions_are_stored(real_raw):
    """Features read every lap and weather row they are given, so no race
    session may ever be in those tables."""
    assert set(real_raw["laps"]["session"]) <= {"Practice 1", "Practice 2", "Practice 3"}
    assert set(real_raw["weather"]["session"]) <= {"Practice 1", "Practice 2", "Practice 3",
                                                   "Qualifying"}


def test_no_future_data_synthetic(synthetic_raw):
    check_no_future_data(synthetic_raw, SYNTHETIC_TARGETS)


def test_no_same_race_data_synthetic(synthetic_raw):
    check_no_same_race_data(synthetic_raw, SYNTHETIC_TARGETS)


def test_no_future_data_real(real_raw):
    check_no_future_data(real_raw, REAL_TARGETS)


def test_no_same_race_data_real(real_raw):
    check_no_same_race_data(real_raw, REAL_TARGETS)


def test_leaky_feature_is_caught(synthetic_raw, monkeypatch):
    """The tests have teeth: a rolling mean without the shift reads the race's
    own result, so the same-race check must fail."""
    import f1pred.features as features

    def leaky(group, window, how):
        return group.transform(lambda s: getattr(s.rolling(window, min_periods=1), how)())

    monkeypatch.setattr(features, "_shifted", leaky)
    with pytest.raises(AssertionError):
        check_no_same_race_data(synthetic_raw, [(2022, 3)])


def test_future_leak_is_caught(synthetic_raw, monkeypatch):
    """A feature that looks ahead (whole-career mean) must fail the future test."""
    import f1pred.features as features

    def lookahead(group, window, how):
        return group.transform("mean")

    monkeypatch.setattr(features, "_shifted", lookahead)
    with pytest.raises(AssertionError):
        check_no_future_data(synthetic_raw, [(2022, 3)])
