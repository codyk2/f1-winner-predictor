"""Stretch goal - a learning-to-rank model on the same walk-forward folds.

    python -m f1pred.ranker

``XGBRanker`` with ``rank:pairwise`` and one group per race learns the whole
finishing order instead of "won or not", so it should be better at podiums.
This compares it with the frozen classifier on the 2022-2024 validation folds:
same features, same tree parameters, same seeds.

An experiment run after v1.0, on validation data only. The frozen model and
the 2025 test are untouched, and nothing here is used by ``predict.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRanker

from f1pred import baselines
from f1pred.config import RACE, REPORTS, TEST_SEASON, VALIDATION_FOLDS, load_frozen
from f1pred.features import load_features
from f1pred.metrics import summarise
from f1pred.train import walk_forward

RANKER_NAME = "XGBRanker (pairwise)"


def relevance(df: pd.DataFrame) -> pd.Series:
    """Higher is better: field size minus finishing position; 0 when there is no position."""
    size = df.groupby(RACE)["DriverId"].transform("size")
    return (size - df["finish_pos"]).clip(lower=0).fillna(0).astype(int)


def walk_forward_ranker(df: pd.DataFrame, features: list[str], params: dict,
                        seeds: tuple[int, ...],
                        folds: tuple[int, ...] = VALIDATION_FOLDS) -> pd.DataFrame:
    """Each fold season ranked by a pairwise ranker trained on all earlier seasons."""
    if df["season"].max() >= TEST_SEASON:
        raise ValueError("the ranker experiment runs on 2019-2024 only")
    tree = {k: v for k, v in params.items() if k != "eval_metric"}
    out = []
    for season in folds:
        train = df[df["season"] < season].sort_values(RACE)  # groups must be contiguous
        val = df[df["season"] == season]
        qid = train.groupby(RACE, sort=False).ngroup()
        scores = []
        for seed in seeds:
            model = XGBRanker(objective="rank:pairwise", **tree, random_state=seed, n_jobs=1,
                              tree_method="hist")
            model.fit(train[features], relevance(train), qid=qid)
            scores.append(model.predict(val[features]))
        frame = val[baselines.PRED_COLS].assign(model=RANKER_NAME, p=np.nan,
                                                score=np.mean(scores, axis=0))
        frame = baselines.with_rank(frame, ["score", "grid_pos", "DriverId"],
                                    [False, True, True])
        out.append(frame.drop(columns="score"))
    return pd.concat(out)


def order_quality(pred: pd.DataFrame) -> dict[str, float]:
    """How well the predicted order matches the finishing order beyond the winner."""
    podium = pred[pred["finish_pos"] <= 3]
    spearman = pred.dropna(subset=["finish_pos"]).groupby(RACE)[["rank", "finish_pos"]] \
        .apply(lambda g: g["rank"].corr(g["finish_pos"], method="spearman"))
    return {
        "podium_in_top3": float((podium["rank"] <= 3).mean()),   # of 3 podium finishers
        "podium_exact": float(podium.groupby(RACE)["rank"].apply(lambda r: (r <= 3).all()).mean()),
        "spearman": float(spearman.mean()),
    }


def main() -> None:
    config = load_frozen()
    df = load_features()  # 2019-2024
    seeds = tuple(config["seeds"])
    preds = {
        "XGBoost classifier (frozen setup)":
            walk_forward(df, config["features"], config["params"],
                         recency_decay=config["recency_decay"], seeds=seeds),
        RANKER_NAME: walk_forward_ranker(df, config["features"], config["params"], seeds),
        "Starting grid": baselines.pole_sitter(df[df["season"].isin(VALIDATION_FOLDS)]),
    }
    rows = []
    for name, pred in preds.items():
        s, o = summarise(pred), order_quality(pred)
        rows.append({
            "Model": name, "Picked the winner": f"{round(s['winner_hit'] * s['races'])} of "
            f"{s['races']}", "Winner in top 3": f"{s['top3_hit']:.1%}",
            "Podium finishers in top 3": f"{o['podium_in_top3']:.1%}",
            "Whole podium right": f"{o['podium_exact']:.1%}",
            "Rank correlation with finish": f"{o['spearman']:.3f}",
        })
    table = pd.DataFrame(rows)
    table.to_csv(REPORTS / "cv" / "ranker.csv", index=False)

    from f1pred.evaluate import md_table, update_readme

    update_readme({"ranker": md_table(table)})
    print(md_table(table))


if __name__ == "__main__":
    main()
