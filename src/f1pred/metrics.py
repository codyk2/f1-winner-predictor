"""Scoring shared by validation and the final test.

All functions take a prediction frame: one row per driver per race with
``won``, ``grid_pos``, ``rank`` (1 = pick) and, for probabilistic models, ``p``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1pred.config import RACE

EPS = 1e-6
METRICS = ["winner_hit", "top3_hit", "hit_when_pole_lost", "log_loss", "brier"]


def per_race(pred: pd.DataFrame) -> pd.DataFrame:
    """One row per race: did the pick win, was the winner in the top three,
    did the pole sitter win, and the race's summed log loss and squared error."""
    pred = pred.sort_values(RACE + ["grid_pos", "quali_pos", "DriverId"])
    p = pred["p"].clip(EPS, 1 - EPS)
    pred = pred.assign(
        nll=-(pred["won"] * np.log(p) + (1 - pred["won"]) * np.log(1 - p)),
        sq=(pred["p"] - pred["won"]) ** 2,
        winner_rank=pred["rank"].where(pred["won"] == 1),
        pick_won=pred["won"].where(pred["rank"] == 1),
    )
    g = pred.groupby(RACE)
    out = pd.DataFrame({
        "hit": g["pick_won"].max(),
        "top3": (g["winner_rank"].min() <= 3).astype(float),
        "pole_won": g["won"].first(),   # first row is the lowest grid slot
        "nll": g["nll"].sum(min_count=1),
        "sq": g["sq"].sum(min_count=1),
        "rows": g.size(),
    })
    return out


def summarise(pred: pd.DataFrame) -> dict[str, float]:
    """Headline metrics for one model on one set of races."""
    r = per_race(pred)
    lost = r[r["pole_won"] == 0]
    has_p = r["nll"].notna().all()
    return {
        "races": len(r),
        "winner_hit": float(r["hit"].mean()),
        "top3_hit": float(r["top3"].mean()),
        "pole_lost_races": len(lost),
        "hit_when_pole_lost": float(lost["hit"].mean()) if len(lost) else float("nan"),
        "log_loss": float(r["nll"].sum() / r["rows"].sum()) if has_p else float("nan"),
        "brier": float(r["sq"].sum() / r["rows"].sum()) if has_p else float("nan"),
    }


def summary_table(preds: dict[str, pd.DataFrame], by_season: bool = False) -> pd.DataFrame:
    """Metrics for several models on the same races, optionally split by season."""
    rows = []
    for name, pred in preds.items():
        if by_season:
            for season, part in pred.groupby("season"):
                rows.append({"model": name, "season": str(season), **summarise(part)})
        rows.append({"model": name, "season": "pooled", **summarise(pred)})
    return pd.DataFrame(rows)


def paired_bootstrap(a: pd.Series, b: pd.Series, n: int = 10_000,
                     seed: int = 0) -> tuple[float, float, float]:
    """Mean of ``a - b`` over races with a 95% percentile interval.

    ``a`` and ``b`` are per-race values indexed by race. Races are resampled
    with replacement, keeping each race's pair together.
    """
    diff = (a - b.reindex(a.index)).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def bootstrap_rate(x: pd.Series, n: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """A per-race rate with a 95% percentile interval."""
    return paired_bootstrap(x, pd.Series(0.0, index=x.index), n, seed)


def bootstrap_log_loss_diff(a: pd.DataFrame, b: pd.DataFrame, n: int = 10_000,
                            seed: int = 0) -> tuple[float, float, float]:
    """Row-level log loss of ``a`` minus ``b`` with a 95% interval over races."""
    ra, rb = per_race(a), per_race(b)
    rb = rb.reindex(ra.index)
    nll = (ra["nll"] - rb["nll"]).to_numpy(dtype=float)
    rows = ra["rows"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(nll), size=(n, len(nll)))
    boot = nll[idx].sum(axis=1) / rows[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(nll.sum() / rows.sum()), float(lo), float(hi)


def calibration_table(pred: pd.DataFrame, edges: tuple[float, ...] =
                      (0, 0.01, 0.03, 0.06, 0.12, 0.25, 0.45, 0.65, 1.0)) -> pd.DataFrame:
    """Mean predicted probability against the observed win rate, per bin."""
    bins = pd.cut(pred["p"], list(edges), include_lowest=True)
    g = pred.groupby(bins, observed=True)
    out = pd.DataFrame({"predicted": g["p"].mean(), "observed": g["won"].mean(),
                        "wins": g["won"].sum(), "rows": g.size()})
    return out.reset_index(names="bin")


def expected_calibration_error(pred: pd.DataFrame) -> float:
    """Row-weighted mean gap between predicted and observed win rate."""
    t = calibration_table(pred)
    return float(((t["predicted"] - t["observed"]).abs() * t["rows"]).sum() / t["rows"].sum())
