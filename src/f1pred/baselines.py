"""Phase 4 - the three baselines every model result is shown against.

    python -m f1pred.baselines               # baselines on 2019-2024
    python -m f1pred.baselines --check-2025  # pole-sitter count on 2025 (data check)

* Pole sitter: the driver starting first.
* Form leader: the driver with the best average finish over the last five races.
* Grid-only logistic regression: P(win) from grid position and qualifying gap.

Every predictor in this project returns the same frame: one row per driver per
race with ``p`` (win probability, empty for the two rule baselines) and
``rank`` (1 is the pick).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from f1pred.config import DEV_SEASONS, RACE, TEST_SEASON, VALIDATION_FOLDS
from f1pred.features import load_features

PRED_COLS = RACE + ["event", "DriverId", "Abbreviation", "grid_pos", "quali_pos", "won"]
LOGREG_FEATURES = ["grid_pos", "quali_gap_pct"]
GAP_CAP = 5.0  # % behind pole; wet-dry sessions produce gaps far beyond this


def pick_hit_rate(df: pd.DataFrame, score: str, lower_is_better: bool = True) -> float:
    """Share of races where the driver with the best score won."""
    grouped = df.groupby(RACE)[score]
    idx = grouped.idxmin() if lower_is_better else grouped.idxmax()
    return float(df.loc[idx, "won"].mean())


def with_rank(df: pd.DataFrame, by: list[str], ascending: list[bool]) -> pd.DataFrame:
    """Add ``rank`` within each race (1 = pick), ordering by ``by`` with ties broken in turn."""
    out = df.sort_values(RACE + by, ascending=[True, True] + ascending, na_position="last")
    out["rank"] = out.groupby(RACE).cumcount() + 1
    return out.sort_index()


def _frame(df: pd.DataFrame, model: str) -> pd.DataFrame:
    return df[PRED_COLS].assign(model=model, p=np.nan)


def pole_sitter(df: pd.DataFrame, by: str = "grid_pos") -> pd.DataFrame:
    """Rank drivers by starting grid (or by qualifying position with ``by='quali_pos'``)."""
    other = "quali_pos" if by == "grid_pos" else "grid_pos"
    out = _frame(df, "Pole sitter" if by == "grid_pos" else "Fastest qualifier")
    return with_rank(out, [by, other, "DriverId"], [True, True, True])


def form_leader(df: pd.DataFrame) -> pd.DataFrame:
    """Rank drivers by average finish over their previous five races."""
    out = _frame(df, "Form leader").assign(form=df["drv_avg_finish_5"])
    out = with_rank(out, ["form", "grid_pos", "DriverId"], [True, True, True])
    return out.drop(columns="form")


def _logreg_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Grid position and qualifying gap with the gaps filled and capped.

    A driver with no qualifying time gets the slowest gap in that session, and
    an unknown grid slot counts as the back of the grid. Both are pre-race facts.
    """
    x = df[LOGREG_FEATURES].copy()
    x["quali_gap_pct"] = x["quali_gap_pct"].fillna(
        df.groupby(RACE)["quali_gap_pct"].transform("max")).fillna(GAP_CAP).clip(upper=GAP_CAP)
    x["grid_pos"] = x["grid_pos"].fillna(df.groupby(RACE)["grid_pos"].transform("max"))
    return x


def normalise(df: pd.DataFrame, p: np.ndarray) -> pd.Series:
    """Rescale raw probabilities so each race sums to one winner."""
    raw = pd.Series(p, index=df.index)
    return raw / raw.groupby([df["season"], df["round"]]).transform("sum")


def grid_logreg(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Fit the grid-only logistic regression on ``train`` and score ``test``."""
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    model.fit(_logreg_inputs(train), train["won"])
    out = _frame(test, "Grid-only logistic regression")
    out["p"] = normalise(test, model.predict_proba(_logreg_inputs(test))[:, 1])
    return with_rank(out, ["p", "grid_pos", "DriverId"], [False, True, True])


def walk_forward_logreg(df: pd.DataFrame,
                        folds: tuple[int, ...] = VALIDATION_FOLDS) -> pd.DataFrame:
    """Each fold season predicted by a regression fitted on all earlier seasons."""
    return pd.concat([grid_logreg(df[df["season"] < s], df[df["season"] == s]) for s in folds])


def main() -> None:
    parser = argparse.ArgumentParser(description="Baselines for the F1 winner predictor")
    parser.add_argument("--check-2025", action="store_true",
                        help="count pole-sitter wins in 2025 (no model involved)")
    args = parser.parse_args()

    if args.check_2025:
        # A parameter-free data check from the build plan: the pole sitter by
        # starting grid should have won 16 of the 24 races in 2025.
        df = load_features(include_test=True)
        df = df[df["season"] == TEST_SEASON]
        for by in ("grid_pos", "quali_pos"):
            wins = df.loc[df.groupby(RACE)[by].idxmin(), "won"]
            print(f"2025 pole sitter by {by}: {int(wins.sum())} of {len(wins)} races")
        return

    df = load_features()
    rows = []
    for season in DEV_SEASONS:
        s = df[df["season"] == season]
        rows.append({"season": season, "races": s.groupby(RACE).ngroups,
                     "pole (grid)": pick_hit_rate(s, "grid_pos"),
                     "pole (quali)": pick_hit_rate(s, "quali_pos"),
                     "form leader": pick_hit_rate(s, "drv_avg_finish_5")})
    table = pd.DataFrame(rows).set_index("season")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
