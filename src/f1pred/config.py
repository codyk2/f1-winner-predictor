"""Paths, season roles and the frozen-model config shared by every stage."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("F1PRED_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
CACHE = DATA / "cache"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
FEATURES_PATH = PROCESSED / "features.parquet"
FROZEN_CONFIG = ROOT / "config" / "model.json"

# Season roles. 2025 is the locked test season: nothing in train.py may read
# its outcomes, and evaluate.py scores it exactly once after the freeze.
FIRST_SEASON = 2019
TEST_SEASON = 2025
DEV_SEASONS = tuple(range(FIRST_SEASON, TEST_SEASON))  # 2019-2024
ALL_SEASONS = (*DEV_SEASONS, TEST_SEASON)
VALIDATION_FOLDS = (2022, 2023, 2024)  # each is predicted from all earlier seasons

RACE = ["season", "round"]  # one race
SEED = 0


def load_frozen() -> dict[str, Any]:
    """Read the frozen feature list and hyper-parameters (the Phase 5 gate)."""
    if not FROZEN_CONFIG.exists():
        raise FileNotFoundError(
            f"{FROZEN_CONFIG} is missing. Run `make cv` and `make freeze` first."
        )
    return json.loads(FROZEN_CONFIG.read_text())
