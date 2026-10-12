"""Version 2 experiments: everything tried, on the same four walk-forward folds.

    python -m f1pred.experiments features   # pass 1: history and feature groups
    python -m f1pred.experiments models     # pass 2: other model classes and settings

Why a version 2 at all, and how it is kept honest
--------------------------------------------------
v1.0 was judged once on 2025. That result stands and its code is frozen. v2 is
built afterwards, with 2025's misses in plain view, so 2025 can no longer be a
test for it. The protocol:

* Development folds are 2022, 2023, 2024 and 2025: each is predicted by a model
  trained on earlier seasons only. 2025 is now an ordinary fold. Results are
  reported pooled over all four and for 2022-2024 alone, because the ideas
  came from 2025's errors and its fold could flatter them.
* Every candidate is scored six times, each with a different set of five seeds,
  and judged on the average. The spread between seed sets is the noise floor.
* A change is kept only if pooled log loss improves by more than ``MIN_GAIN``
  (twice the spread between seed sets) AND at least three of the four folds
  improve. Otherwise it is logged and dropped.
* Every candidate is written to ``reports/v2/experiments.csv``, kept or not.
* 2026 is not used to choose anything: ``walk_forward`` refuses it. After v2 is
  frozen and its tag pushed, rounds 1-17 are scored once as a confirmation, and
  the races still to come are the only clean test.

One correction to the record: the first run of pass 1 used a single seed set
and no minimum gain, and kept the circuit-stickiness group on a pooled gain of
0.0005. Six seed sets showed that gain to be the size of the noise (and the
picks consistently worse), so the rule above was tightened before pass 2.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

import numpy as np
import pandas as pd

from f1pred import baselines, models
from f1pred.config import RACE, REPORTS, ROOT, TEST_SEASON, load_frozen
from f1pred.features import FEATURES_PATH
from f1pred.features_v2 import V1_FROZEN, V2_FEATURES, V2_GROUPS, load_features_v2
from f1pred.metrics import EPS, per_race, summarise
from f1pred.models import Scorer

V2_DIR = REPORTS / "v2"
LOG = V2_DIR / "experiments.csv"
V2_CONFIG = ROOT / "config" / "model_v2.json"
FOLDS = (2022, 2023, 2024, 2025)
LAST_DEV_SEASON = TEST_SEASON          # 2025: the newest season v2 may be developed on
SEED_SETS = tuple(tuple(range(5 * k, 5 * k + 5)) for k in range(6))
MIN_GAIN = 0.0006                      # twice the spread of pooled log loss between seed sets
MIN_FOLDS_BETTER = 3
FINAL_SEEDS = tuple(range(15))         # the frozen v2 model averages fifteen seeds

Factory = Callable[[tuple[int, ...]], Scorer]   # seeds -> scorer


# ------------------------------------------------------------------- scoring

def race_nll(pred: pd.DataFrame) -> float:
    """Mean over races of -log P(the driver who won): the proper score for one winner."""
    winners = pred[pred["won"] == 1]
    return float(-np.log(winners["p"].clip(EPS, 1)).mean())


def score(pred: pd.DataFrame) -> dict[str, float]:
    """Pooled and per-fold metrics for one set of walk-forward predictions."""
    s = summarise(pred)
    out = {"hits": s["winner_hit"] * s["races"], "top3": s["top3_hit"] * s["races"],
           "log_loss": s["log_loss"], "race_nll": race_nll(pred), "brier": s["brier"]}
    early = summarise(pred[pred["season"] < LAST_DEV_SEASON])
    out.update(hits_22_24=early["winner_hit"] * early["races"], log_loss_22_24=early["log_loss"])
    for season, part in pred.groupby("season"):
        ps = summarise(part)
        out[f"hits_{season}"] = ps["winner_hit"] * ps["races"]
        out[f"log_loss_{season}"] = ps["log_loss"]
    return out


def better(new: dict, old: dict) -> tuple[bool, int]:
    """The keep rule: pooled log loss down by more than the noise, three of four folds better."""
    folds = sum(new[f"log_loss_{s}"] < old[f"log_loss_{s}"] for s in FOLDS)
    gain = old["log_loss"] - new["log_loss"]
    return bool(gain > MIN_GAIN and folds >= MIN_FOLDS_BETTER), folds


# -------------------------------------------------------------- walk-forward

def walk_forward(df: pd.DataFrame, features: list[str], scorer: Scorer,
                 train_from: int = 2019, folds: tuple[int, ...] = FOLDS) -> pd.DataFrame:
    """Predict each fold season from rows of earlier seasons (from ``train_from`` on)."""
    if df["season"].max() > LAST_DEV_SEASON or max(folds) > LAST_DEV_SEASON:
        raise ValueError("v2 is developed on seasons up to 2025; 2026 is for confirmation only")
    out = []
    for season in folds:
        train = df[(df["season"] >= train_from) & (df["season"] < season)]
        out.append(scorer(train, df[df["season"] == season], features, season))
    return pd.concat(out)


def evaluate(df: pd.DataFrame, features: list[str], factory: Factory, train_from: int = 2019,
             seeded: bool = True) -> tuple[dict, pd.DataFrame]:
    """Average score over the seed sets (one run for a model with no randomness)."""
    sets = SEED_SETS if seeded else SEED_SETS[:1]
    preds = [walk_forward(df, features, factory(seeds), train_from) for seeds in sets]
    runs = pd.DataFrame([score(p) for p in preds])
    mean = runs.mean().to_dict()
    mean.update(log_loss_sd=float(runs["log_loss"].std()) if seeded else 0.0,
                hits_min=float(runs["hits"].min()), hits_max=float(runs["hits"].max()))
    return mean, preds[0]


class Log:
    """The experiment log: one row per candidate, kept or not."""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        V2_DIR.mkdir(parents=True, exist_ok=True)
        old = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
        self.rows = [] if old.empty else old[old["stage"] != stage].to_dict("records")

    def add(self, name: str, what: str, s: dict, features: list[str],
            baseline: dict | None = None, **extra) -> bool:
        row = {"stage": self.stage, "candidate": name, "what": what,
               "features": len(features), **extra, **s, "gain": "", "folds_better": "", "kept": ""}
        keep = False
        if baseline is not None:
            keep, folds = better(s, baseline)
            row.update(gain=baseline["log_loss"] - s["log_loss"], folds_better=folds,
                       kept="yes" if keep else "no")
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(LOG, index=False)
        verdict = (f"gain {row['gain']:+.4f}, folds better {row['folds_better']}, "
                   f"kept {row['kept']}" if baseline else "reference")
        print(f"{name:<36} hits {s['hits']:5.1f}/92  log loss {s['log_loss']:.4f} "
              f"(sd {s['log_loss_sd']:.4f})  race NLL {s['race_nll']:.3f}  ({verdict})")
        return keep


def dev_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(v1's table plus the new columns, 2019-2025) and (the v2 table, 2014-2025).

    The first keeps v1's history windows, which start in 2019; the second feeds
    every window with 2014-2018 results and has those seasons as rows.
    """
    v1, v2 = pd.read_parquet(FEATURES_PATH), load_features_v2()
    plain = v1.merge(v2[RACE + ["DriverId", *V2_FEATURES]], on=RACE + ["DriverId"], how="left",
                     validate="one_to_one")
    return plain, v2


# ------------------------------------------------------------------- pass 1

def features_pass() -> None:
    """History, then feature groups one at a time, each judged by the keep rule."""
    params = load_frozen()["params"]

    def factory(seeds: tuple[int, ...]) -> Scorer:
        return models.xgb_binary(params, seeds)

    log = Log("1 features")
    plain, with_history = dev_tables()
    best, _ = evaluate(plain, V1_FROZEN, factory)
    log.add("v1.0 as frozen", "14 features, trained on 2019 onwards", best, V1_FROZEN,
            train_from=2019)
    name, features, train_from, table = "v1.0", list(V1_FROZEN), 2019, plain

    for step, what, start in (
            ("history in windows", "form and circuit windows see 2014-2018 results", 2019),
            ("train from 2014", "2014-2018 races added as training rows", 2014)):
        s, _ = evaluate(with_history, features, factory, start)
        if log.add(step, what, s, features, best, train_from=start):
            best, name, train_from, table = s, step, start, with_history

    # The rebuilt long-run gap replaces v1's; the segment gap is added next to v1's
    # session gap, which still orders the whole field better.
    swaps = {"practice": ["longrun_gap_pct", "has_longrun"]}
    for group, cols in V2_GROUPS.items():
        trial = [f for f in features if f not in swaps.get(group, [])] + cols
        what = (f"replaces {', '.join(swaps[group])}" if group in swaps else "added") \
            + f": {', '.join(cols)}"
        s, _ = evaluate(table, trial, factory, train_from)
        if log.add(f"+ {group}", what, s, trial, best, train_from=train_from):
            best, name, features = s, f"+ {group}", trial

    choice = {"stage": "1 features", "chosen": name, "features": features,
              "train_from": train_from, "history_in_windows": table is with_history,
              "params": params, "score": best}
    (V2_DIR / "pass1_choice.json").write_text(json.dumps(choice, indent=2) + "\n")
    print(f"\npass 1 keeps: {name} ({len(features)} features, trained from {train_from})")


# ------------------------------------------------------------------- pass 2

def models_pass() -> None:
    """Other model classes and settings on pass 1's features, same keep rule."""
    start = json.loads((V2_DIR / "pass1_choice.json").read_text())
    features, train_from, params = start["features"], start["train_from"], start["params"]
    plain, with_history = dev_tables()
    table = with_history if start["history_in_windows"] else plain
    log = Log("2 models")

    def binary(p: dict = params, monotone: bool = False) -> Factory:
        return lambda seeds: models.xgb_binary(p, seeds, monotone)

    def softmax(p: dict = params, monotone: bool = False) -> Factory:
        return lambda seeds: models.xgb_softmax(p, seeds, monotone)

    best, _ = evaluate(table, features, binary(), train_from)
    log.add("pass 1 model", "binary XGBoost, v1 parameters", best, features, kind="binary")
    chosen = {"kind": "binary", "params": params, "monotone": False}

    candidates: list[tuple[str, str, Factory, dict, bool]] = [
        ("monotone constraints", "each pace and form feature may only push one way",
         binary(monotone=True), {"kind": "binary", "params": params, "monotone": True}, True),
        ("softmax per race", "trees trained on the race: one winner per race in the loss",
         softmax(), {"kind": "softmax", "params": params, "monotone": False}, True),
        ("softmax + monotone", "both of the above",
         softmax(monotone=True), {"kind": "softmax", "params": params, "monotone": True}, True),
        ("conditional logit, L2 1", "linear score per car, softmax over the field",
         lambda seeds: models.cond_logit(1.0), {"kind": "cond_logit", "l2": 1.0}, False),
        ("conditional logit, L2 10", "the same with a stronger penalty",
         lambda seeds: models.cond_logit(10.0), {"kind": "cond_logit", "l2": 10.0}, False),
    ]
    for name, what, factory, spec, seeded in candidates:
        s, _ = evaluate(table, features, factory, train_from, seeded)
        if log.add(name, what, s, features, best, kind=spec["kind"]):
            best, chosen = s, spec

    # A linear model may gain more from extra rows than trees did.
    s, _ = evaluate(with_history, features, lambda seeds: models.cond_logit(1.0), 2014, False)
    log.add("conditional logit from 2014", "trained on 2014 onwards, history in windows", s,
            features, best, kind="cond_logit")

    def tree(seeds: tuple[int, ...]) -> Scorer:
        spec = chosen if chosen["kind"] in ("binary", "softmax") else \
            {"kind": "binary", "params": params, "monotone": False}
        return build_scorer(spec, seeds)

    s, _ = evaluate(table, features,
                    lambda seeds: models.blend(tree(seeds), models.cond_logit(1.0)), train_from)
    if log.add("blend: trees x conditional logit", "geometric mean of the two models", s,
               features, best, kind="blend"):
        tree_spec = chosen if chosen["kind"] in ("binary", "softmax") else \
            {"kind": "binary", "params": params, "monotone": False}
        best, chosen = s, {"kind": "blend", "tree": tree_spec, "l2": 1.0}

    if chosen["kind"] in ("binary", "softmax"):
        # A small check that v1's tree settings still suit the new features.
        for depth, trees in ((2, 200), (2, 800), (3, 200), (3, 400)):
            p = {**params, "max_depth": depth, "n_estimators": trees}
            make = softmax if chosen["kind"] == "softmax" else binary
            s, _ = evaluate(table, features, make(p, chosen["monotone"]), train_from)
            if log.add(f"depth {depth}, {trees} trees", "tree settings", s, features, best,
                       kind=chosen["kind"]):
                best, chosen = s, {**chosen, "params": p}

        current = dict(chosen)
        s, _ = evaluate(table, features,
                        lambda seeds: models.tempered(build_scorer(current, seeds)), train_from)
        if log.add("temperature", "one number fitted on earlier seasons flattens or sharpens",
                   s, features, best, kind=chosen["kind"]):
            best, chosen = s, {**chosen, "tempered": True}

    choice = {**start, "stage": "2 models", "model": chosen, "score": best}
    (V2_DIR / "pass2_choice.json").write_text(json.dumps(choice, indent=2) + "\n")
    print(f"\npass 2 keeps: {json.dumps({k: v for k, v in chosen.items() if k != 'params'})}")


# ------------------------------------------------------------ shared helpers

def build_scorer(model: dict, seeds: tuple[int, ...]) -> Scorer:
    """The scorer described by a ``model`` entry of a choice or config file."""
    kind = model["kind"]
    if kind == "blend":
        return models.blend(build_scorer(model["tree"], seeds), models.cond_logit(model["l2"]))
    if kind == "cond_logit":
        return models.cond_logit(model["l2"])
    make = models.xgb_softmax if kind == "softmax" else models.xgb_binary
    scorer = make(model["params"], seeds, model.get("monotone", False))
    return models.tempered(scorer) if model.get("tempered") else scorer


def reference_rows() -> pd.DataFrame:
    """Pole sitter and grid-only logistic regression on the same four folds."""
    v1 = pd.read_parquet(FEATURES_PATH)
    val = v1[v1["season"].isin(FOLDS)]
    rows = []
    for name, pred in (("Pole sitter", baselines.pole_sitter(val)),
                       ("Grid-only logistic regression",
                        baselines.walk_forward_logreg(v1, folds=FOLDS))):
        race = per_race(pred)
        s = summarise(pred)
        rows.append({"candidate": name, "hits": int(race["hit"].sum()), "races": len(race),
                     "top3": int(race["top3"].sum()), "log_loss": s["log_loss"],
                     "race_nll": race_nll(pred) if pred["p"].notna().all() else np.nan,
                     **{f"hits_{y}": int(race.loc[y, "hit"].sum()) for y in FOLDS}})
    out = pd.DataFrame(rows)
    V2_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(V2_DIR / "references.csv", index=False)
    return out


def report() -> None:
    """Write the experiment log as a table and fill the README's block with it."""
    from f1pred.evaluate import md_table, update_readme

    log, refs = pd.read_csv(LOG), reference_rows()
    rows = [{"Pass": "", "Candidate": r["candidate"], "What changed": "reference",
             "Winners (of 92)": str(r["hits"]), "Log loss": "n/a" if pd.isna(r["log_loss"])
             else f"{r['log_loss']:.4f}", "Gain": "", "Folds better": "", "Kept": "",
             "2022-24 (of 68)": str(r["hits"] - r["hits_2025"]),
             "2025 (of 24)": str(r["hits_2025"])} for r in refs.to_dict("records")]
    for r in log.to_dict("records"):
        judged = isinstance(r["kept"], str) and r["kept"] in ("yes", "no")
        rows.append({
            "Pass": r["stage"].split()[0], "Candidate": r["candidate"], "What changed": r["what"],
            "Winners (of 92)": f"{r['hits']:.1f} ({r['hits_min']:.0f}-{r['hits_max']:.0f})",
            "Log loss": f"{r['log_loss']:.4f}",
            "Gain": f"{float(r['gain']):+.4f}" if judged else "",
            "Folds better": f"{int(float(r['folds_better']))} of 4" if judged else "",
            "Kept": ("**yes**" if r["kept"] == "yes" else "no") if judged else "start",
            "2022-24 (of 68)": f"{r['hits_22_24']:.1f}", "2025 (of 24)": f"{r['hits_2025']:.1f}",
        })
    table = md_table(pd.DataFrame(rows))
    (V2_DIR / "experiments.md").write_text(table + "\n")
    update_readme({"v2_log": table})
    print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["features", "models", "references", "report"])
    args = parser.parse_args()
    if args.command == "references":
        print(reference_rows().round(4).to_string(index=False))
    else:
        {"features": features_pass, "models": models_pass, "report": report}[args.command]()
    print(f"log: {LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
