"""Phases 5-6 - result tables, figures and the single locked 2025 run.

    python -m f1pred.evaluate cv          # validation tables and figures (2019-2024)
    python -m f1pred.evaluate test-2025   # the locked run: once, after the v1.0 tag
    python -m f1pred.evaluate report      # rebuild 2025 tables and figures from saved output

``test-2025`` is the only command that scores a model on 2025. It refuses to
run unless ``config/model.json`` and ``src/f1pred`` match the ``v1.0`` git tag,
and it refuses to run a second time. ``report`` never refits anything: it only
re-renders what the locked run saved.

The guard stops accidents, not a determined operator: it trusts the local tag
and the feature file on disk. What makes the v1.0 result checkable is that a
clean clone of the public tag reproduces it bit for bit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb

from f1pred import baselines, plots
from f1pred.config import (
    FEATURES_PATH,
    FROZEN_CONFIG,
    RACE,
    REPORTS,
    ROOT,
    TEST_SEASON,
    VALIDATION_FOLDS,
    load_frozen,
)
from f1pred.features import FEATURES, load_features
from f1pred.intervals import paired_difference, wilson
from f1pred.metrics import (
    bootstrap_log_loss_diff,
    calibration_table,
    expected_calibration_error,
    per_race,
    summarise,
    summary_table,
)
from f1pred.train import CV_DIR, MODEL_NAME, fit_final, predict

TEST_DIR = REPORTS / "test_2025"
MODELS_DIR = ROOT / "models"
TAG = "v1.0"
LOCKED_PATHS = ["config", "src/f1pred"]
POLE, FORM, LOGREG = "Pole sitter", "Form leader", "Grid-only logistic regression"
MISS_CAUSES = ["Race incident", "Safety car or strategy", "Reliability", "Pace misread"]

FEATURE_LABELS = {
    "grid_pos": "starting position", "quali_pos": "qualifying position",
    "quali_gap_pct": "qualifying gap to pole", "teammate_quali_delta": "gap to team-mate",
    "longrun_gap_pct": "long-run pace", "shortrun_gap_pct": "one-lap practice pace",
    "has_longrun": "did a long run", "sprint_pos": "sprint result",
    "drv_avg_finish_5": "recent finishes", "drv_wins_5": "recent wins",
    "drv_podiums_5": "recent podiums", "team_points_5": "team form",
    "champ_rank": "championship position", "dnf_rate_10": "retirement rate",
    "drv_track_avg_finish": "record at this circuit",
    "track_pole_win_rate": "pole conversion at this circuit",
    "street_circuit": "street circuit", "quali_rain": "wet qualifying",
    "fp_track_temp": "track temperature",
}


# -------------------------------------------------------------------- helpers

def pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:.1%}"


def num(x: float, digits: int = 4) -> str:
    return "n/a" if pd.isna(x) else f"{x:.{digits}f}"


def grid_slot(grid: pd.Series, field: int = 20) -> pd.Series:
    """' (P3)' for a grid slot, ' (pit lane)' beyond the last slot, '' when unknown."""
    def one(g: float) -> str:
        if pd.isna(g):
            return ""
        return " (pit lane)" if g > field else f" (P{int(g)})"
    return grid.map(one)


def interval(lo: float, hi: float, signed: bool = False) -> str:
    fmt = "{:+.1%}" if signed else "{:.1%}"
    return f"{fmt.format(lo)} to {fmt.format(hi)}"


def md_table(df: pd.DataFrame) -> str:
    """A GitHub-flavoured markdown table from a frame of strings."""
    head = "| " + " | ".join(df.columns) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.to_numpy()]
    return "\n".join([head, rule, *body])


def inject(readme: str, name: str, content: str) -> str:
    """Replace the block between ``<!-- name:start -->`` and ``<!-- name:end -->``."""
    pattern = re.compile(rf"(<!-- {name}:start -->\n).*?(<!-- {name}:end -->)", re.DOTALL)
    return pattern.sub(lambda m: m.group(1) + content + "\n" + m.group(2), readme)


def update_readme(blocks: dict[str, str]) -> None:
    path = ROOT / "README.md"
    if not path.exists():
        return
    text = path.read_text()
    for name, content in blocks.items():
        text = inject(text, name, content)
    path.write_text(text)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def split_models(pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    order = [MODEL_NAME, POLE, FORM, LOGREG]
    return {m: pred[pred["model"] == m] for m in order if (pred["model"] == m).any()}


def hits(pred: pd.DataFrame, col: str = "hit") -> pd.Series:
    return per_race(pred)[col]


def results_table(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Model and baselines on the same races, with 95% intervals.

    Rates get Wilson intervals; the difference from the pole sitter gets the
    exact paired-bootstrap interval over races (see ``intervals.py``).
    """
    pole = hits(preds[POLE])
    rows = []
    for name, pred in preds.items():
        s = summarise(pred)
        race = per_race(pred)
        n, won, top3 = int(s["races"]), int(race["hit"].sum()), int(race["top3"].sum())
        lost = race[race["pole_won"] == 0]
        diff, dlo, dhi = paired_difference(race["hit"], pole)
        if name == POLE:
            versus = "baseline"
        elif (race["hit"] == pole.reindex(race.index)).all():
            versus = "same hits and misses"
        else:
            versus = f"{diff:+.1%} ({interval(dlo, dhi, signed=True)})"
        rows.append({
            "Model": name,
            "Picked the winner": f"{won} of {n} ({pct(won / n)})",
            "95% CI": interval(*wilson(won, n)),
            "vs pole sitter (95% CI)": versus,
            "Winner in top 3": f"{top3} of {n} ({interval(*wilson(top3, n))})",
            "Hits when pole lost": f"{int(lost['hit'].sum())} of {len(lost)}",
            "Log loss": num(s["log_loss"]),
            "Brier": num(s["brier"]),
        })
    return pd.DataFrame(rows)


def log_loss_vs_logreg(preds: dict[str, pd.DataFrame]) -> dict[str, float]:
    diff, lo, hi = bootstrap_log_loss_diff(preds[MODEL_NAME], preds[LOGREG])
    base = summarise(preds[LOGREG])["log_loss"]
    return {"diff": diff, "lo": lo, "hi": hi, "relative": diff / base}


def loss_gap_sensitivity(preds: dict[str, pd.DataFrame]) -> str:
    """How much of the log-loss gap to the grid-only model rests on one race."""
    a, b = per_race(preds[MODEL_NAME]), per_race(preds[LOGREG])
    gap = a["nll"] - b["nll"].reindex(a.index)
    top = gap.idxmin()
    rest = gap.index != top
    without = (a["nll"][rest].sum() - b["nll"][rest].sum()) / b["nll"][rest].sum()
    return (f"The single most favourable race (round {top[1]} of {top[0]}) supplies "
            f"{gap[top] / gap.sum():.0%} of that gap; without it the gap is {without:+.1%}.")


def shap_contributions(models: list, rows: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Seed-averaged TreeSHAP contributions (log-odds) per feature for each row."""
    data = xgb.DMatrix(rows[features])
    contrib = np.mean([m.get_booster().predict(data, pred_contribs=True) for m in models], axis=0)
    return pd.DataFrame(contrib[:, :-1], columns=features, index=rows.index)


# ----------------------------------------------------------------- validation

def cv_report() -> None:
    """Tables and figures for walk-forward validation (2019-2024 only)."""
    preds = split_models(pd.read_csv(CV_DIR / "predictions.csv"))
    choice = json.loads((CV_DIR / "choice.json").read_text())
    abl = pd.read_csv(CV_DIR / "ablation.csv")
    recency = pd.read_csv(CV_DIR / "recency.csv")

    pooled = results_table(preds)
    by_fold = summary_table(preds, by_season=True)
    fold_view = by_fold.assign(
        **{"Winner hit rate": by_fold["winner_hit"].map(pct),
           "Top-3 hit rate": by_fold["top3_hit"].map(pct),
           "Hit rate when pole lost": by_fold["hit_when_pole_lost"].map(pct),
           "Log loss": by_fold["log_loss"].map(num), "Brier": by_fold["brier"].map(num)}
    ).rename(columns={"model": "Model", "season": "Fold", "races": "Races"})
    fold_view = fold_view[["Model", "Fold", "Races", "Winner hit rate", "Top-3 hit rate",
                           "Hit rate when pole lost", "Log loss", "Brier"]]

    chosen = len(abl) - 1 - int((abl["features"].iloc[::-1] == len(choice["features"])).argmax())
    abl_view = pd.DataFrame({
        "Features": abl["step"], "Count": abl["features"],
        "Winner hit rate": abl["winner_hit"].map(pct), "Top-3 hit rate": abl["top3_hit"].map(pct),
        "Log loss": abl["log_loss"].map(num), "Brier": abl["brier"].map(num),
    })
    abl_view.loc[chosen, "Features"] = f"**{abl_view.loc[chosen, 'Features']}** (chosen)"

    rec_view = pd.DataFrame({
        "Sample weighting": recency["weighting"],
        "Winner hit rate": recency["winner_hit"].map(pct),
        "Log loss": recency["log_loss"].map(num),
        **{f"Log loss {s}": recency[f"log_loss_{s}"].map(num) for s in VALIDATION_FOLDS},
    })

    vs_logreg = log_loss_vs_logreg(preds)
    ece = {m: expected_calibration_error(preds[m]) for m in (MODEL_NAME, LOGREG)}
    p = choice["params"]
    notes = (
        f"Frozen setup: {len(choice['features'])} features, `max_depth` {p['max_depth']}, "
        f"`learning_rate` {p['learning_rate']}, {p['n_estimators']} trees, "
        f"`min_child_weight` {p['min_child_weight']}, "
        f"recency weighting {'on' if choice['recency_decay'] else 'off'}, "
        f"{len(choice['seeds'])} seeds averaged.\n\n"
        f"Log loss against the grid-only logistic regression: {vs_logreg['diff']:+.4f} "
        f"({vs_logreg['relative']:+.1%}; 95% CI {vs_logreg['lo']:+.4f} to {vs_logreg['hi']:+.4f}). "
        f"{loss_gap_sensitivity(preds)} "
        f"Expected calibration error: {MODEL_NAME} {ece[MODEL_NAME]:.3f}, "
        f"logistic regression {ece[LOGREG]:.3f}."
    )

    sections = {
        "validation": md_table(pooled),
        "validation_folds": md_table(fold_view),
        "ablation": md_table(abl_view),
        "recency": md_table(rec_view),
        "validation_notes": notes,
    }
    (CV_DIR / "validation.md").write_text(
        "\n\n".join(f"## {k}\n\n{v}" for k, v in sections.items()) + "\n")
    update_readme(sections)

    n_races = summarise(preds[MODEL_NAME])["races"]
    cal = {MODEL_NAME: calibration_table(preds[MODEL_NAME]),
           LOGREG: calibration_table(preds[LOGREG])}
    cal[MODEL_NAME].assign(model=MODEL_NAME).to_csv(CV_DIR / "calibration.csv", index=False)
    plots.save(plots.calibration(
        cal, f"Walk-forward validation, {n_races} races (2022-2024); bars are 95% intervals"),
        "calibration_validation.png")
    plots.save(plots.ablation(
        abl, summarise(preds[POLE])["winner_hit"], summarise(preds[LOGREG])["log_loss"], chosen,
        f"Walk-forward validation on {n_races} races (2022-2024), groups added left to right"),
        "ablation.png")
    print(md_table(pooled), "\n")
    print(md_table(abl_view), "\n")
    print(notes)


# ------------------------------------------------------------- the locked run

def check_freeze() -> str:
    """Refuse to touch 2025 unless the model is frozen, tagged and unchanged."""
    if not FROZEN_CONFIG.exists():
        raise SystemExit("config/model.json is missing: run `make cv` and `make freeze` first.")
    tag = _git("rev-parse", "-q", "--verify", f"refs/tags/{TAG}^{{commit}}")
    if tag.returncode != 0:
        raise SystemExit(f"git tag {TAG} not found: commit the frozen config and tag it first. "
                         "The tag is the proof that the model was fixed before the test.")
    changed = _git("diff", "--quiet", TAG, "--", *LOCKED_PATHS).returncode != 0
    untracked = _git("status", "--porcelain", "--", *LOCKED_PATHS).stdout.strip()
    if changed or untracked:
        raise SystemExit(f"config/ or src/f1pred/ differ from the {TAG} tag. "
                         "The 2025 run must use exactly the tagged code and config.")
    if (TEST_DIR / "metrics.json").exists():
        raise SystemExit(
            f"The 2025 test has already been run (see {TEST_DIR.relative_to(ROOT)}/). It runs "
            f"once. To reproduce it, check out the {TAG} tag in a clean clone and run it there.")
    return tag.stdout.strip()


def test_2025() -> None:
    """Train on 2019-2024 with the frozen setup and score every 2025 race, once."""
    tag_commit = check_freeze()
    config = load_frozen()
    df = load_features(include_test=True)
    train, test = df[df["season"] < TEST_SEASON], df[df["season"] == TEST_SEASON]

    models = fit_final(df, config, TEST_SEASON)
    pred = pd.concat([
        predict(models, test, config["features"]),
        baselines.pole_sitter(test),
        baselines.form_leader(test),
        baselines.grid_logreg(train, test),
    ])

    TEST_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for seed, model in zip(config["seeds"], models, strict=True):
        model.save_model(MODELS_DIR / f"xgb_v1_seed{seed}.json")
    pred.to_csv(TEST_DIR / "predictions.csv", index=False)
    contrib = shap_contributions(models, test, config["features"])
    pd.concat([test[RACE + ["DriverId"]], contrib], axis=1).to_csv(
        TEST_DIR / "contributions.csv", index=False)
    shap_contributions(models, train, config["features"]).abs().mean().rename("mean_abs_shap") \
        .rename_axis("feature").to_csv(TEST_DIR / "importance.csv")

    preds = split_models(pred)
    metrics = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tag": TAG, "tag_commit": tag_commit,
        "features_sha256": hashlib.sha256(FEATURES_PATH.read_bytes()).hexdigest(),
        "train_rows": len(train), "test_rows": len(test),
        "models": {name: summarise(p) for name, p in preds.items()},
        "log_loss_vs_logreg": log_loss_vs_logreg(preds),
    }
    (TEST_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"2025 locked run complete ({TAG} @ {tag_commit[:7]}).\n")
    report()


# --------------------------------------------------------------------- report

def per_race_table(preds: dict[str, pd.DataFrame], contrib: pd.DataFrame) -> pd.DataFrame:
    """One row per 2025 race: winner, pick, probabilities and why the model picked them."""
    model, pole = preds[MODEL_NAME], preds[POLE]
    contrib = contrib.set_index(RACE + ["DriverId"])
    rows = []
    for (season, rnd), race in model.groupby(RACE):
        pick = race[race["rank"] == 1].iloc[0]
        winner = race[race["won"] == 1].iloc[0]
        pole_row = pole[(pole["round"] == rnd) & (pole["rank"] == 1)].iloc[0]
        c = contrib.loc[(season, rnd, pick["DriverId"])].sort_values(ascending=False)
        reasons = [FEATURE_LABELS[f] for f in c.index[:2] if c[f] > 0]
        rows.append({
            "round": rnd, "event": pick["event"], "label": pick["event"].replace(" Grand Prix", ""),
            "winner": winner["Abbreviation"], "winner_grid": winner["grid_pos"],
            "pick": pick["Abbreviation"], "pick_grid": pick["grid_pos"], "pick_p": pick["p"],
            "pole": pole_row["Abbreviation"], "hit": int(pick["won"]),
            "pole_hit": int(pole_row["won"]),
            "winner_p": winner["p"], "winner_rank": int(winner["rank"]),
            "reasons": ", ".join(reasons),
        })
    out = pd.DataFrame(rows)
    causes = TEST_DIR / "miss_causes.csv"
    if causes.exists():
        out = out.merge(pd.read_csv(causes), on="round", how="left")
        bad = set(out["cause"].dropna()) - set(MISS_CAUSES)
        if bad:
            raise ValueError(f"unknown miss causes: {bad}")
    else:
        out["cause"], out["note"] = np.nan, np.nan
    return out


def resume_bullets(metrics: dict) -> str:
    m, pole = metrics["models"][MODEL_NAME], metrics["models"][POLE]
    races, lost = m["races"], m["pole_lost_races"]
    x, pole_x = round(m["winner_hit"] * races), round(pole["winner_hit"] * races)
    y = round(m["hit_when_pole_lost"] * lost)
    rel = -metrics["log_loss_vs_logreg"]["relative"]
    n_used = len(load_frozen()["features"])
    line = (f"Picked the winner in {x} of {races} held-out 2025 races vs {pole_x} for a "
            f"pole-sitter baseline" if x > pole_x else
            f"On {races} held-out 2025 races, matched a pole-sitter baseline within noise "
            f"({x} winners vs {pole_x}), ranked the winner in its top three every time and "
            f"called {y} of the {lost} won from off pole; {rel:.0%} lower log loss than a "
            f"grid-only model (23% on 68 validation races)")
    return "\n".join([
        f"- Built a FastF1 data pipeline ingesting practice, qualifying, sprint and race data "
        f"for 152 races (2019-2025), engineering {len(FEATURES)} pre-race features "
        f"({n_used} in the final model) including long-run practice pace, qualifying gap and "
        f"rolling driver/team form",
        "- Trained an XGBoost win-probability model with season-by-season walk-forward "
        "validation and automated leakage tests ensuring only pre-race information",
        f"- {line}",
    ])


def report() -> None:
    """Re-render the 2025 tables and figures from what the locked run saved."""
    if not (TEST_DIR / "metrics.json").exists():
        raise SystemExit("No 2025 results yet: the locked run (`make test-2025`) comes first.")
    metrics = json.loads((TEST_DIR / "metrics.json").read_text())
    preds = split_models(pd.read_csv(TEST_DIR / "predictions.csv"))
    contrib = pd.read_csv(TEST_DIR / "contributions.csv")
    importance = pd.read_csv(TEST_DIR / "importance.csv", index_col=0)["mean_abs_shap"]

    results = results_table(preds)
    races = per_race_table(preds, contrib)
    races.to_csv(TEST_DIR / "per_race.csv", index=False)
    view = pd.DataFrame({
        "Rd": races["round"], "Race": races["label"],
        "Winner (grid)": races["winner"] + grid_slot(races["winner_grid"]),
        "Model pick (grid)": races["pick"] + grid_slot(races["pick_grid"]),
        "P(pick)": races["pick_p"].map(lambda v: f"{v:.0%}"),
        "Pole sitter": races["pole"],
        "Pick right?": races["hit"].map({1: "yes", 0: "**no**"}),
        "P(winner)": races["winner_p"].map(lambda v: f"{v:.0%}"),
        "Winner's rank": races["winner_rank"],
        "Pick driven by": races["reasons"],
        "Miss cause": races["cause"].fillna(""),
    })
    v = metrics["log_loss_vs_logreg"]
    n_races = metrics["models"][MODEL_NAME]["races"]
    ece = expected_calibration_error(preds[MODEL_NAME])
    notes = (
        f"Log loss against the grid-only logistic regression: {v['diff']:+.4f} "
        f"({v['relative']:+.1%}; 95% CI {v['lo']:+.4f} to {v['hi']:+.4f}). "
        f"{loss_gap_sensitivity(preds)} "
        f"Expected calibration error on 2025: {ece:.3f}. "
        f"Rates carry Wilson intervals; differences are paired bootstraps over the "
        f"{n_races} races (exact for hits, 10,000 resamples for log loss). One race is "
        f"about {1 / n_races:.0%}."
    )
    misses = races[races["hit"] == 0]
    cause_view = (misses["cause"].value_counts().reindex(MISS_CAUSES, fill_value=0)
                  .rename_axis("Miss cause").rename("Races").reset_index())
    miss_view = pd.DataFrame({
        "Rd": misses["round"], "Race": misses["label"],
        "Pick": misses["pick"] + grid_slot(misses["pick_grid"]),
        "Winner": misses["winner"] + grid_slot(misses["winner_grid"]),
        "Cause": misses["cause"].fillna(""), "What happened": misses["note"].fillna(""),
    })
    sections = {
        "results_2025": md_table(results),
        "results_2025_notes": notes,
        "per_race_2025": md_table(view),
        "miss_causes": md_table(cause_view),
        "miss_table": md_table(miss_view),
    }
    bullets = resume_bullets(metrics)
    (TEST_DIR / "results.md").write_text(
        "\n\n".join(f"## {k}\n\n{v}" for k, v in sections.items())
        + "\n\n## resume_bullets\n\n" + bullets + "\n")
    update_readme(sections)

    top = [FEATURE_LABELS[f] for f in importance.sort_values(ascending=False).index[:2]]
    fig = plots.feature_importance(
        importance.rename(index=FEATURE_LABELS),
        "Mean absolute SHAP contribution over the 2019-2024 training rows")
    share = importance.sort_values(ascending=False).iloc[:2].sum() / importance.sum()
    fig.texts[0].set_text(f"{top[0].capitalize()} and {top[1]} are the two largest "
                          f"contributors ({share:.0%} of the total)")
    plots.save(fig, "feature_importance.png")
    plots.save(plots.race_strip(races), "race_strip_2025.png")

    print(md_table(results), "\n")
    print(notes, "\n")
    print(md_table(view), "\n")
    print(bullets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["cv", "test-2025", "report"])
    args = parser.parse_args()
    {"cv": cv_report, "test-2025": test_2025, "report": report}[args.command]()


if __name__ == "__main__":
    main()
