"""Version 2, frozen and checked.

    python -m f1pred.v2 freeze            # write config/model_v2.json from the experiment log
    python -m f1pred.v2 confirm 2026 17   # after the v2.0 tag is pushed: score 2026 so far, once

``freeze`` records the setup that the experiments kept, and scores that exact
setup (fifteen seeds) on the four development folds.

``confirm`` is v2's first look at 2026. It is not a clean test: v1.0 had
already been scored on those races when v2 was built, and the results were
seen. What it does guarantee is the order of events. It refuses to run unless
the ``v2.0`` tag exists on GitHub and matches the local files, so the push
time there is on record before any v2 number for 2026 exists, and it refuses
to run twice. The clean test is ``predict.py`` on races not yet run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone

import pandas as pd

from f1pred import baselines
from f1pred.config import ROOT, TEST_SEASON
from f1pred.experiments import (
    FINAL_SEEDS,
    FOLDS,
    V2_CONFIG,
    V2_DIR,
    build_scorer,
    dev_tables,
    race_nll,
    score,
    walk_forward,
)
from f1pred.intervals import paired_difference
from f1pred.metrics import bootstrap_log_loss_diff, per_race, summarise
from f1pred.predict import V1, V2, load_v2, rescore_season

TAG = "v2.0"
LOCKED = ["config/model_v2.json", "src/f1pred/features_v2.py", "src/f1pred/models.py",
          "src/f1pred/features.py", "src/f1pred/train.py", "src/f1pred/metrics.py"]
POLE, LOGREG = "Pole sitter", "Grid-only logistic regression"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def _json_safe(value):
    """NaN is not valid JSON; write it as null."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return None if isinstance(value, float) and pd.isna(value) else value


def freeze() -> None:
    """Write config/model_v2.json: the gate before v2 sees any 2026 race."""
    if V2_CONFIG.exists():
        raise SystemExit(f"{V2_CONFIG.relative_to(ROOT)} already exists; v2 is frozen.")
    choice = json.loads((V2_DIR / "pass2_choice.json").read_text())
    if choice["history_in_windows"]:
        raise SystemExit("the chosen setup needs 2014-2018 history; predict.py does not load it")
    plain, _ = dev_tables()
    scorer = build_scorer(choice["model"], FINAL_SEEDS)
    dev = walk_forward(plain, choice["features"], scorer, choice["train_from"])
    dev.assign(model=V2).to_csv(V2_DIR / "dev_predictions.csv", index=False)
    commit = _git("rev-parse", "--short", "HEAD").stdout.strip()
    config = {
        "version": "2.0",
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen_from_commit": commit or None,
        "developed_on": "walk-forward folds 2022-2025; 2025 was v1.0's test season and is "
                        "a development fold here",
        "features": choice["features"],
        "train_from": choice["train_from"],
        "model": choice["model"],
        "seeds": list(FINAL_SEEDS),
        "development_score": score(dev),
    }
    V2_CONFIG.write_text(json.dumps(config, indent=2) + "\n")
    s = config["development_score"]
    print(f"froze v2: {len(config['features'])} features, {config['model']['kind']} model, "
          f"{len(FINAL_SEEDS)} seeds. Development folds: {s['hits']:.0f} of 92 winners, "
          f"log loss {s['log_loss']:.4f}.")
    print(f"Commit {V2_CONFIG.relative_to(ROOT)}, tag {TAG} and push the tag before `confirm`.")


def check_pushed() -> str:
    """Refuse to continue unless the v2.0 tag is on GitHub and the files match it."""
    local = _git("rev-parse", "-q", "--verify", f"refs/tags/{TAG}")
    if local.returncode != 0:
        raise SystemExit(f"git tag {TAG} not found: commit the frozen config and tag it first.")
    remote = _git("ls-remote", "--tags", "origin", f"refs/tags/{TAG}").stdout.split()
    if not remote or remote[0] != local.stdout.strip():
        raise SystemExit(f"tag {TAG} is not on the remote (or differs). Push it first: the push "
                         "time on GitHub is what shows the model was fixed before this run.")
    changed = _git("diff", "--quiet", TAG, "--", *LOCKED).returncode != 0
    if changed or _git("status", "--porcelain", "--", *LOCKED).stdout.strip():
        raise SystemExit(f"model files differ from the {TAG} tag.")
    return _git("rev-parse", f"{TAG}^{{commit}}").stdout.strip()


def confirm(season: int, through: int) -> None:
    """Score a season's completed races with v2.0 and v1.0, once, after the tag is pushed."""
    if season <= TEST_SEASON:
        raise SystemExit("confirmation is for seasons after 2025")
    out = V2_DIR / f"confirmation_{season}.json"
    if out.exists():
        raise SystemExit(f"{out.relative_to(ROOT)} exists: the confirmation has been run.")
    tag_commit = check_pushed()
    pred = rescore_season(season, through, versions=(V2, V1))
    pred.to_csv(V2_DIR / f"confirmation_{season}.csv", index=False, float_format="%.6f")
    preds = {m: pred[pred["model"] == m] for m in (V2, V1, POLE, LOGREG)}
    races = {m: per_race(p) for m, p in preds.items()}
    diff, lo, hi = paired_difference(races[V2]["hit"], races[V1]["hit"])
    ll, ll_lo, ll_hi = bootstrap_log_loss_diff(preds[V2], preds[V1])
    record = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tag": TAG, "tag_commit": tag_commit, "season": season, "rounds": through,
        "models": {m: {**summarise(p), "race_nll": race_nll(p) if m in (V2, V1, LOGREG) else None}
                   for m, p in preds.items()},
        "v2_minus_v1": {"hits": diff, "hits_lo": lo, "hits_hi": hi,
                        "log_loss": ll, "log_loss_lo": ll_lo, "log_loss_hi": ll_hi},
    }
    out.write_text(json.dumps(_json_safe(record), indent=2, allow_nan=False) + "\n")
    print(f"{season} rounds 1-{through}, {TAG} @ {tag_commit[:7]}")
    for m, p in preds.items():
        s = summarise(p)
        print(f"  {m:<30} {s['winner_hit'] * s['races']:.0f} of {s['races']} winners, "
              f"top 3 {s['top3_hit'] * s['races']:.0f}, log loss {s['log_loss']:.4f}")
    report()


def results_table(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """v2.0, v1.0 and the references on the same races."""
    from f1pred.evaluate import interval, num
    from f1pred.intervals import wilson

    pole = per_race(preds[POLE])["hit"]
    rows = []
    for name, pred in preds.items():
        race, s = per_race(pred), summarise(pred)
        n, won, top3 = len(race), int(race["hit"].sum()), int(race["top3"].sum())
        diff, lo, hi = paired_difference(race["hit"], pole)
        rows.append({
            "Model": name, "Picked the winner": f"{won} of {n} ({won / n:.1%})",
            "95% CI": interval(*wilson(won, n)),
            "vs pole sitter (95% CI)": "baseline" if name == POLE
            else f"{diff:+.1%} ({interval(lo, hi, signed=True)})",
            "Winner in top 3": f"{top3} of {n}",
            "Log loss": num(s["log_loss"]),
            "Race log loss": num(race_nll(pred), 3) if pred["p"].notna().all() else "n/a",
        })
    return pd.DataFrame(rows)


def dev_frames() -> dict[str, pd.DataFrame]:
    """v2.0 (as frozen), v1.0 and the references on the four development folds."""
    from f1pred import models
    from f1pred.config import load_frozen
    from f1pred.features import FEATURES_PATH

    v1_config = load_frozen()
    table = pd.read_parquet(FEATURES_PATH)
    v1 = walk_forward(table, v1_config["features"],
                      models.xgb_binary(v1_config["params"], tuple(v1_config["seeds"])))
    val = table[table["season"].isin(FOLDS)]
    return {V2: pd.read_csv(V2_DIR / "dev_predictions.csv"), V1: v1,
            POLE: baselines.pole_sitter(val),
            LOGREG: baselines.walk_forward_logreg(table, folds=FOLDS)}


def report() -> None:
    """Fill the README's version 2 blocks from the frozen config and saved predictions."""
    from f1pred.evaluate import md_table, update_readme

    config = load_v2()
    dev = dev_frames()
    by_fold = pd.DataFrame([
        {"Fold": str(season), "Races": len(per_race(dev[V2][dev[V2]["season"] == season])),
         **{name: int(per_race(p[p["season"] == season])["hit"].sum())
            for name, p in dev.items()}}
        for season in FOLDS])
    total = {"Fold": "all four", "Races": int(by_fold["Races"].sum()),
             **{name: int(by_fold[name].sum()) for name in dev}}
    by_fold = pd.concat([by_fold, pd.DataFrame([total])], ignore_index=True)
    d, lo, hi = paired_difference(per_race(dev[V2])["hit"], per_race(dev[V1])["hit"])
    ll, ll_lo, ll_hi = bootstrap_log_loss_diff(dev[V2], dev[V1])
    p = config["model"]["params"]
    blocks = {
        "v2_dev": md_table(results_table(dev)),
        "v2_dev_folds": md_table(by_fold.astype(str)),
        "v2_dev_notes": (
            f"v2.0 as frozen: {len(config['features'])} features, binary XGBoost "
            f"(`max_depth` {p['max_depth']}, {p['n_estimators']} trees) with monotone "
            f"constraints, {len(config['seeds'])} seeds averaged. Against v1.0 on these 92 "
            f"races: winners {d:+.1%} (95% CI {lo:+.1%} to {hi:+.1%}), log loss {ll:+.4f} "
            f"(95% CI {ll_lo:+.4f} to {ll_hi:+.4f}). These are development numbers."),
    }
    done = sorted(V2_DIR.glob("confirmation_*.json"))
    if done:
        rec = json.loads(done[-1].read_text())
        pred = pd.read_csv(V2_DIR / f"confirmation_{rec['season']}.csv")
        preds = {m: pred[pred["model"] == m] for m in (V2, V1, POLE, LOGREG)}
        v = rec["v2_minus_v1"]
        blocks["v2_confirmation"] = md_table(results_table(preds))
        blocks["v2_confirmation_notes"] = (
            f"{rec['season']} rounds 1-{rec['rounds']}, scored once on {rec['run_at'][:10]} at "
            f"tag `{rec['tag']}` ({rec['tag_commit'][:7]}), which was pushed first. v2.0 against "
            f"v1.0: winners {v['hits']:+.1%} (95% CI {v['hits_lo']:+.1%} to {v['hits_hi']:+.1%}), "
            f"log loss {v['log_loss']:+.4f} (95% CI {v['log_loss_lo']:+.4f} to "
            f"{v['log_loss_hi']:+.4f}).")
    (V2_DIR / "report.md").write_text(
        "\n\n".join(f"## {k}\n\n{t}" for k, t in blocks.items()) + "\n")
    update_readme(blocks)
    print("\n\n".join(blocks.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["freeze", "confirm", "report"])
    parser.add_argument("season", type=int, nargs="?")
    parser.add_argument("round", type=int, nargs="?")
    args = parser.parse_args()
    if args.command == "confirm":
        if args.season is None or args.round is None:
            parser.error("confirm needs a season and the last completed round")
        confirm(args.season, args.round)
    else:
        {"freeze": freeze, "report": report}[args.command]()


if __name__ == "__main__":
    main()
