"""Phase 5 - XGBoost win-probability model with walk-forward validation.

    python -m f1pred.train cv       # ablations, tuning, recency test on 2019-2024
    python -m f1pred.train freeze   # write the chosen setup to config/model.json

Each fold trains on the past and predicts the next season, exactly as the model
would be used. Nothing here can read 2025: ``walk_forward`` refuses any frame
that contains the test season. The final fit lives in ``fit_final`` and is only
called by ``evaluate.py`` and ``predict.py``.

Selection rules, fixed before any validation number was seen:

1. Feature set: add the feature groups one at a time and keep the step with the
   lowest pooled log loss.
2. Hyper-parameters: a small grid, chosen by pooled log loss (far more stable
   than a hit rate on 22-24 races).
3. Recency weighting: kept only if pooled log loss improves and at least two of
   the three folds improve.
4. Every model is the average of five seeds, so no result depends on one seed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from xgboost import XGBClassifier

from f1pred import baselines
from f1pred.config import (
    DEV_SEASONS,
    FROZEN_CONFIG,
    REPORTS,
    ROOT,
    TEST_SEASON,
    VALIDATION_FOLDS,
    load_frozen,
)
from f1pred.features import FEATURE_GROUPS, load_features
from f1pred.metrics import summarise, summary_table

CV_DIR = REPORTS / "cv"
MODEL_NAME = "XGBoost"

# Shallow, regularised trees: about 2,500 training rows and one winner per 20.
# No scale_pos_weight: it distorts the probabilities, and picking the top
# driver per race does not need it.
PARAMS: dict[str, float | int | str] = dict(
    n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, eval_metric="logloss",
)
GRID: dict[str, list[float | int]] = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.1],
    "n_estimators": [200, 400, 800],
    "min_child_weight": [1, 5, 10],
}
SEEDS = (0, 1, 2, 3, 4)
RECENCY_DECAY = 0.7


def recency_weights(seasons: pd.Series, target_season: int, decay: float) -> np.ndarray:
    """``decay ** (target_season - season - 1)``: the latest training season weighs 1."""
    return np.power(decay, (target_season - seasons - 1).to_numpy(dtype=float))


def fit(train: pd.DataFrame, features: list[str], params: dict, target_season: int,
        recency_decay: float | None = None, seeds: tuple[int, ...] = SEEDS) -> list[XGBClassifier]:
    """Fit one classifier per seed on rows from seasons before ``target_season``."""
    if train["season"].max() >= target_season:
        raise ValueError("training rows must come from seasons before the target season")
    weight = (recency_weights(train["season"], target_season, recency_decay)
              if recency_decay else None)
    models = []
    for seed in seeds:
        model = XGBClassifier(**params, random_state=seed, n_jobs=1, tree_method="hist")
        model.fit(train[features], train["won"], sample_weight=weight)
        models.append(model)
    return models


def predict(models: list[XGBClassifier], df: pd.DataFrame, features: list[str],
            name: str = MODEL_NAME) -> pd.DataFrame:
    """Seed-averaged win probabilities, rescaled so each race has one winner."""
    raw = np.mean([m.predict_proba(df[features])[:, 1] for m in models], axis=0)
    out = df[baselines.PRED_COLS].assign(model=name)
    out["p"] = baselines.normalise(df, raw)
    return baselines.with_rank(out, ["p", "grid_pos", "DriverId"], [False, True, True])


def walk_forward(df: pd.DataFrame, features: list[str], params: dict,
                 folds: tuple[int, ...] = VALIDATION_FOLDS,
                 recency_decay: float | None = None,
                 seeds: tuple[int, ...] = SEEDS) -> pd.DataFrame:
    """Predict each fold season from a model trained on every earlier season."""
    if df["season"].max() >= TEST_SEASON:
        raise ValueError("walk-forward validation must not see the locked test season")
    out = []
    for season in folds:
        train, val = df[df["season"] < season], df[df["season"] == season]
        models = fit(train, features, params, season, recency_decay, seeds)
        out.append(predict(models, val, features))
    return pd.concat(out)


def fit_final(df: pd.DataFrame, config: dict, target_season: int) -> list[XGBClassifier]:
    """The frozen model, trained on every season before ``target_season``."""
    train = df[df["season"] < target_season]
    return fit(train, config["features"], config["params"], target_season,
               config["recency_decay"], tuple(config["seeds"]))


# ------------------------------------------------------------------ selection

def _row(label: dict, pred: pd.DataFrame) -> dict:
    by_fold = {f"log_loss_{s}": summarise(part)["log_loss"] for s, part in pred.groupby("season")}
    pooled = summarise(pred)
    return {**label, **{k: pooled[k] for k in
                        ("winner_hit", "top3_hit", "hit_when_pole_lost", "log_loss", "brier")},
            **by_fold}


def ablation(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Add feature groups one at a time: grid, practice, sprint, form, circuit, weather."""
    rows, preds, features = [], {}, []
    for group, cols in FEATURE_GROUPS.items():
        features = features + cols
        label = "grid only" if group == "grid" else f"+ {group}"
        preds[label] = walk_forward(df, features, params)
        rows.append(_row({"step": label, "features": len(features)}, preds[label]))
    return pd.DataFrame(rows), preds


def grid_search(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Walk-forward results for every combination in ``GRID``."""
    combos = [dict(zip(GRID, values, strict=True)) for values in itertools.product(*GRID.values())]

    def run(combo: dict) -> dict:
        return _row(combo, walk_forward(df, features, {**PARAMS, **combo}))

    return pd.DataFrame(Parallel(n_jobs=-1)(delayed(run)(c) for c in combos))


def recency_test(df: pd.DataFrame, features: list[str], params: dict) -> tuple[pd.DataFrame, bool]:
    """Compare equal weights with ``RECENCY_DECAY`` weighting; keep it only if folds improve."""
    rows = [_row({"weighting": "equal"}, walk_forward(df, features, params)),
            _row({"weighting": f"{RECENCY_DECAY} ** seasons back"},
                 walk_forward(df, features, params, recency_decay=RECENCY_DECAY))]
    table = pd.DataFrame(rows)
    fold_cols = [f"log_loss_{s}" for s in VALIDATION_FOLDS]
    better_folds = int((table.loc[1, fold_cols] < table.loc[0, fold_cols]).sum())
    keep = bool(table.loc[1, "log_loss"] < table.loc[0, "log_loss"] and better_folds >= 2)
    return table, keep


def run_cv() -> None:
    df = load_features()  # 2019-2024 only
    assert set(df["season"]) == set(DEV_SEASONS)
    CV_DIR.mkdir(parents=True, exist_ok=True)

    print("1/4 ablation (default parameters)")
    abl, abl_preds = ablation(df, PARAMS)
    abl.to_csv(CV_DIR / "ablation.csv", index=False)
    print(abl.round(4).to_string(index=False))
    best_step = int(abl["log_loss"].idxmin())
    features = [f for g in list(FEATURE_GROUPS)[: best_step + 1] for f in FEATURE_GROUPS[g]]
    print(f"-> feature set: through '{abl.loc[best_step, 'step']}' ({len(features)} features)")

    print("\n2/4 grid search")
    grid = grid_search(df, features).sort_values("log_loss").reset_index(drop=True)
    grid.to_csv(CV_DIR / "grid.csv", index=False)
    print(grid.head(8).round(4).to_string(index=False))
    best = grid.iloc[0]
    params = {**PARAMS, **{k: (int(best[k]) if k != "learning_rate" else float(best[k]))
                           for k in GRID}}
    print(f"-> parameters: {({k: params[k] for k in GRID})}")

    print("\n3/4 recency weighting")
    rec, keep = recency_test(df, features, params)
    rec.to_csv(CV_DIR / "recency.csv", index=False)
    print(rec.round(4).to_string(index=False))
    recency = RECENCY_DECAY if keep else None
    print(f"-> recency weighting {'kept' if keep else 'not kept'}")

    print("\n4/4 validation results, chosen setup against the baselines")
    val = df[df["season"].isin(VALIDATION_FOLDS)]
    preds = {
        "Pole sitter": baselines.pole_sitter(val),
        "Form leader": baselines.form_leader(val),
        "Grid-only logistic regression": baselines.walk_forward_logreg(df),
        MODEL_NAME: walk_forward(df, features, params, recency_decay=recency),
    }
    pd.concat(preds.values()).to_csv(CV_DIR / "predictions.csv", index=False)
    results = summary_table(preds, by_season=True)
    results.to_csv(CV_DIR / "results.csv", index=False)
    print(results.round(4).to_string(index=False))

    choice = {"features": features, "params": params, "recency_decay": recency,
              "seeds": list(SEEDS), "validation": summarise(preds[MODEL_NAME])}
    (CV_DIR / "choice.json").write_text(json.dumps(choice, indent=2) + "\n")
    print(f"\nwrote {CV_DIR.relative_to(ROOT)}/. Run `make freeze` to lock this setup.")


def freeze() -> None:
    """Copy the validation choice to config/model.json: the gate before 2025."""
    choice = json.loads((CV_DIR / "choice.json").read_text())
    if FROZEN_CONFIG.exists():
        frozen = load_frozen()
        same = all(frozen[k] == choice[k] for k in ("features", "params", "recency_decay", "seeds"))
        print("config/model.json already frozen" + ("" if same else
              " and DIFFERS from the latest validation choice; delete it to re-freeze"))
        return
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    frozen = {
        "version": "1.0",
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen_from_commit": commit or None,
        "train_seasons": [DEV_SEASONS[0], DEV_SEASONS[-1]],
        "test_season": TEST_SEASON,
        **choice,
    }
    FROZEN_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    FROZEN_CONFIG.write_text(json.dumps(frozen, indent=2) + "\n")
    print(f"froze {len(choice['features'])} features and parameters in "
          f"{FROZEN_CONFIG.relative_to(ROOT)}. Commit it and tag v1.0 before `make test-2025`.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["cv", "freeze"])
    args = parser.parse_args()
    {"cv": run_cv, "freeze": freeze}[args.command]()


if __name__ == "__main__":
    main()
