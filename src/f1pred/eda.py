"""Phase 2 - the six EDA tables behind ``notebooks/01_eda.ipynb``.

One question: where is the signal beyond the starting grid? Every function
takes the 2019-2024 feature table (``load_features()`` never returns 2025
unless asked), so nothing here can look at the test season.
"""

from __future__ import annotations

import pandas as pd

from f1pred.config import RACE, TEST_SEASON


def _dev(df: pd.DataFrame) -> pd.DataFrame:
    if df["season"].max() >= TEST_SEASON:
        raise ValueError("EDA runs on 2019-2024 only")
    return df


def race_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per race: who started first, who won, and from where."""
    df = _dev(df).sort_values(RACE + ["grid_pos", "quali_pos", "DriverId"])
    winners = df[df["won"] == 1].set_index(RACE)
    g = df.groupby(RACE)
    out = pd.DataFrame({
        "event": g["event"].first(), "circuit": g["circuit"].first(),
        "street_circuit": g["street_circuit"].first(),
        "pole_won": g["won"].first(),
        "winner_grid": winners["grid_pos"],
        "winner_longrun_gap": winners["longrun_gap_pct"],
        "best_longrun_gap": g["longrun_gap_pct"].min(),
        "drivers_with_longrun": g["longrun_gap_pct"].count(),
        "winner_team_points": winners["team_points_5"],
        "best_team_points": g["team_points_5"].max(),
    })
    out["winner_fastest_longrun"] = (out["winner_longrun_gap"] == out["best_longrun_gap"]) \
        & out["winner_longrun_gap"].notna()
    out["winner_from_form_team"] = (out["winner_team_points"] == out["best_team_points"]) \
        & out["best_team_points"].notna()
    return out.reset_index()


def pole_by_season(df: pd.DataFrame) -> pd.DataFrame:
    """1. How strong is the pole-sitter baseline each year?"""
    races = race_level(df)
    out = races.groupby("season")["pole_won"].agg(races="size", pole_wins="sum").reset_index()
    out["rate"] = out["pole_wins"] / out["races"]
    return out


def wins_by_grid_slot(df: pd.DataFrame) -> pd.DataFrame:
    """2. Share of all wins taken from P1, P2, P3 and P4 or lower."""
    grid = race_level(df)["winner_grid"]
    slot = grid.map(lambda g: f"P{int(g)}" if g <= 3 else "P4+")
    out = slot.value_counts().reindex(["P1", "P2", "P3", "P4+"], fill_value=0).rename("wins")
    out = out.rename_axis("start").reset_index()
    out["share"] = out["wins"] / out["wins"].sum()
    return out


def pole_by_circuit(df: pd.DataFrame, min_races: int = 4) -> pd.DataFrame:
    """3. Pole conversion at each circuit with at least ``min_races`` races."""
    races = race_level(df)
    out = (races.groupby("circuit")
           .agg(races=("pole_won", "size"), pole_wins=("pole_won", "sum"),
                street=("street_circuit", "first")).reset_index())
    out["rate"] = out["pole_wins"] / out["races"]
    out = out[out["races"] >= min_races]
    return out.sort_values(["rate", "races"], ascending=[False, False]).reset_index(drop=True)


def practice_vs_grid(df: pd.DataFrame) -> pd.DataFrame:
    """4. When pole lost, how often was the winner the fastest driver on long runs?"""
    races = race_level(df)
    races = races[races["drivers_with_longrun"] >= 3]
    rows = []
    for label, part in [("all races", races), ("pole won", races[races["pole_won"] == 1]),
                        ("pole lost", races[races["pole_won"] == 0])]:
        had_run = part["winner_longrun_gap"].notna()
        rows.append({
            "races": label, "n": len(part),
            "winner had a long run": int(had_run.sum()),
            "winner fastest on long runs": int(part["winner_fastest_longrun"].sum()),
            "share (of winners with a run)":
                part["winner_fastest_longrun"].sum() / max(int(had_run.sum()), 1),
            "median winner gap %": part["winner_longrun_gap"].median(),
        })
    return pd.DataFrame(rows)


def form_vs_grid(df: pd.DataFrame) -> pd.DataFrame:
    """5. How often did the winner drive for the team with the best last-five-race points?"""
    races = race_level(df)
    races = races[races["best_team_points"].notna()]
    rows = []
    for label, part in [("all races", races), ("pole won", races[races["pole_won"] == 1]),
                        ("pole lost", races[races["pole_won"] == 0])]:
        rows.append({"races": label, "n": len(part),
                     "winner from best-form team": int(part["winner_from_form_team"].sum()),
                     "share": part["winner_from_form_team"].mean()})
    return pd.DataFrame(rows)


def missing_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """6. Per season: how complete is practice data? Plus the weekends with none."""
    df = _dev(df)
    per_race = df.groupby(RACE + ["event", "format"]).agg(
        drivers=("DriverId", "size"), with_short_run=("shortrun_gap_pct", "count"),
        with_long_run=("has_longrun", "sum"), with_long_gap=("longrun_gap_pct", "count"),
    ).reset_index()
    per_race["sprint_weekend"] = per_race["format"] != "conventional"
    season = per_race.groupby("season").agg(
        races=("round", "size"), sprint_weekends=("sprint_weekend", "sum"),
        no_practice_laps=("with_short_run", lambda s: int((s == 0).sum())),
        no_long_runs=("with_long_run", lambda s: int((s == 0).sum())),
        drivers=("drivers", "sum"), with_long_run=("with_long_run", "sum"),
        with_long_gap=("with_long_gap", "sum"),
    ).reset_index()
    season["drivers with a long run"] = season["with_long_run"] / season["drivers"]
    season["drivers with a long-run gap"] = season["with_long_gap"] / season["drivers"]
    season = season.drop(columns=["drivers", "with_long_run", "with_long_gap"])
    gaps = per_race[per_race["with_long_run"] == 0][
        ["season", "round", "event", "format", "with_short_run"]]
    return season, gaps.reset_index(drop=True)
