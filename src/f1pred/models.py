"""Model classes tried for version 2. Each is a "scorer": a function that
takes training rows, the rows of one season to predict, a feature list and the
season, and returns the standard prediction frame (``p`` sums to one per race).

* ``xgb_binary``: v1's approach. A classifier per row ("won or not"), then the
  probabilities are rescaled per race. Optionally with monotone constraints.
* ``xgb_softmax``: boosted trees trained on the race itself. The loss is the
  softmax over the cars in a race, so "one winner per race" is built into
  training instead of patched on afterwards.
* ``cond_logit``: the classic model for racing odds. A linear score per car and
  a softmax over the field (a conditional logit), with an L2 penalty.
* ``blend``: the geometric mean of other scorers' probabilities.
* ``tempered``: one extra number, a temperature fitted on earlier seasons, that
  flattens or sharpens a scorer's probabilities.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import optimize

from f1pred import baselines
from f1pred.config import RACE
from f1pred.metrics import EPS
from f1pred.train import MODEL_NAME, fit, predict

Scorer = Callable[[pd.DataFrame, pd.DataFrame, list[str], int], pd.DataFrame]

# Direction in which a feature can only help (+1) or only hurt (-1) the chance
# of winning. Anything not listed is left free.
MONOTONE = {
    "grid_pos": -1, "quali_pos": -1, "quali_gap_pct": -1, "teammate_quali_delta": -1,
    "longrun_gap_pct": -1, "shortrun_gap_pct": -1, "sprint_pos": -1,
    "drv_avg_finish_5": -1, "drv_wins_5": 1, "drv_podiums_5": 1, "team_points_5": 1,
    "champ_rank": -1, "quali_seg_gap_pct": -1, "quali_margin_pct": -1,
    "drv_quali_gap_avg_5": -1, "team_quali_gap": -1, "team_quali_gap_avg_5": -1,
    "longrun_gap_v2": -1,
}


def _frame(val: pd.DataFrame, p: np.ndarray, name: str = MODEL_NAME) -> pd.DataFrame:
    """Rescale to one winner per race and rank (ties broken by grid, then name)."""
    out = val[baselines.PRED_COLS].assign(model=name)
    out["p"] = baselines.normalise(val, p)
    return baselines.with_rank(out, ["p", "grid_pos", "DriverId"], [False, True, True])


def _groups(df: pd.DataFrame) -> np.ndarray:
    """An integer id per race, in row order."""
    return df.groupby(RACE, sort=False).ngroup().to_numpy()


def _softmax(score: np.ndarray, group: np.ndarray) -> np.ndarray:
    top = np.full(group.max() + 1, -np.inf)
    np.maximum.at(top, group, score)
    e = np.exp(score - top[group])
    return e / np.bincount(group, weights=e)[group]


def _constraints(features: list[str]) -> str:
    return "(" + ",".join(str(MONOTONE.get(f, 0)) for f in features) + ")"


# ------------------------------------------------------------------ boosted

def xgb_binary(params: dict, seeds: tuple[int, ...], monotone: bool = False,
               live: bool = False) -> Scorer:
    """v1's model: a binary classifier per seed, averaged, rescaled per race.

    ``live`` allows training rows from the target season's earlier rounds, as
    ``predict.py`` needs; walk-forward validation never sets it.
    """
    def run(train: pd.DataFrame, val: pd.DataFrame, features: list[str], season: int):
        p = dict(params, monotone_constraints=_constraints(features)) if monotone else params
        fitted = fit(train, features, p, season, None, seeds, allow_current_season=live)
        return predict(fitted, val, features)
    return run


def xgb_softmax(params: dict, seeds: tuple[int, ...], monotone: bool = False) -> Scorer:
    """Boosted trees with a per-race softmax loss (gradient p - y within each race)."""
    def run(train: pd.DataFrame, val: pd.DataFrame, features: list[str], season: int):
        group = _groups(train)
        dtrain = xgb.DMatrix(train[features], label=train["won"])
        dval = xgb.DMatrix(val[features])

        def objective(margin: np.ndarray, data: xgb.DMatrix):
            p = _softmax(margin, group)
            return p - data.get_label(), np.maximum(p * (1 - p), 1e-6)

        booster = {
            "max_depth": params["max_depth"], "eta": params["learning_rate"],
            "subsample": params["subsample"], "colsample_bytree": params["colsample_bytree"],
            "min_child_weight": params["min_child_weight"], "lambda": params["reg_lambda"],
            "base_score": 0.0, "tree_method": "hist", "nthread": 1,
            "disable_default_eval_metric": 1,
        }
        if monotone:
            booster["monotone_constraints"] = _constraints(features)
        val_group = _groups(val)
        probs = []
        for seed in seeds:
            model = xgb.train({**booster, "seed": seed}, dtrain,
                              num_boost_round=params["n_estimators"], obj=objective)
            probs.append(_softmax(model.predict(dval, output_margin=True), val_group))
        return _frame(val, np.mean(probs, axis=0))
    return run


# ------------------------------------------------------------------- linear

class _Design:
    """Turn features into a matrix a linear model can use, learned on training rows
    only: clip to the 1st-99th percentile, fill gaps with the median, add a
    "was missing" column where gaps are common, then standardise."""

    def __init__(self, train: pd.DataFrame, features: list[str]) -> None:
        x = train[features]
        self.features = features
        self.lo, self.hi = x.quantile(0.01), x.quantile(0.99)
        self.median = x.clip(self.lo, self.hi, axis=1).median().fillna(0.0)
        self.flags = [f for f in features if x[f].isna().mean() > 0.01]
        m = self._raw(train)
        self.mean, self.std = m.mean(), m.std().replace(0, 1.0)

    def _raw(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.features].clip(self.lo, self.hi, axis=1).fillna(self.median)
        flags = df[self.flags].isna().astype(float).add_prefix("missing_")
        return pd.concat([x, flags], axis=1)

    def matrix(self, df: pd.DataFrame) -> np.ndarray:
        return ((self._raw(df) - self.mean) / self.std).to_numpy()


def _fit_cond_logit(x: np.ndarray, won: np.ndarray, group: np.ndarray, l2: float) -> np.ndarray:
    """Weights minimising mean race log loss plus an L2 penalty."""
    n_races = group.max() + 1

    def loss(w: np.ndarray) -> tuple[float, np.ndarray]:
        p = _softmax(x @ w, group)
        nll = -np.log(np.clip(p[won == 1], EPS, 1)).sum() / n_races
        grad = x.T @ (p - won) / n_races
        return nll + 0.5 * l2 * (w @ w) / n_races, grad + l2 * w / n_races

    result = optimize.minimize(loss, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    return result.x


def cond_logit(l2: float = 1.0) -> Scorer:
    """Conditional logit: linear score per car, softmax over the race."""
    def run(train: pd.DataFrame, val: pd.DataFrame, features: list[str], season: int):
        design = _Design(train, features)
        w = _fit_cond_logit(design.matrix(train), train["won"].to_numpy(dtype=float),
                            _groups(train), l2)
        return _frame(val, _softmax(design.matrix(val) @ w, _groups(val)))
    return run


# ------------------------------------------------------------- combinations

def blend(*scorers: Scorer) -> Scorer:
    """Geometric mean of the scorers' probabilities, rescaled per race."""
    def run(train: pd.DataFrame, val: pd.DataFrame, features: list[str], season: int):
        logs = [np.log(s(train, val, features, season)["p"].reindex(val.index).clip(EPS, 1))
                for s in scorers]
        return _frame(val, np.exp(np.mean(logs, axis=0)))
    return run


def _temper(p: pd.Series, df: pd.DataFrame, t: float) -> pd.Series:
    return baselines.normalise(df, np.power(p.clip(EPS, 1).to_numpy(), 1.0 / t))


def tempered(scorer: Scorer, inner_seasons: int = 3) -> Scorer:
    """Fit one temperature on the scorer's own out-of-sample record.

    The last ``inner_seasons`` training seasons are each predicted from the
    seasons before them; the temperature that minimises race log loss on those
    predictions is then applied to the target season. T > 1 flattens.
    """
    def run(train: pd.DataFrame, val: pd.DataFrame, features: list[str], season: int):
        seasons = sorted(train["season"].unique())
        inner = [s for s in seasons[-inner_seasons:] if s > seasons[0]]
        held = [scorer(train[train["season"] < s], train[train["season"] == s], features, s)
                for s in inner]
        out = scorer(train, val, features, season)
        if not held:
            return out
        past = pd.concat(held)
        rows = train.loc[past.index]

        def nll(t: float) -> float:
            p = _temper(past["p"], rows, t)
            return float(-np.log(p[past["won"] == 1].clip(EPS, 1)).mean())

        t = optimize.minimize_scalar(nll, bounds=(0.5, 3.0), method="bounded").x
        return _frame(val, _temper(out["p"].reindex(val.index), val, t).to_numpy())
    return run
