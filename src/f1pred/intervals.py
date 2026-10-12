"""Confidence intervals for race-level results, computed without simulation.

A hit rate over 24 races can only take 25 values, so a simulated bootstrap
percentile can land one race off when the true cut point sits near a step of
the distribution (three of the first README's bounds did). These are exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

LEVEL = 0.95


def wilson(hits: int, n: int, level: float = LEVEL) -> tuple[float, float]:
    """Wilson score interval for a proportion. Well behaved at 0 of n and n of n."""
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(0.5 + level / 2)
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    lo = 0.0 if hits == 0 else max(0.0, centre - half)
    hi = 1.0 if hits == n else min(1.0, centre + half)
    return float(lo), float(hi)


def paired_difference(a: pd.Series, b: pd.Series,
                      level: float = LEVEL) -> tuple[float, float, float]:
    """Mean of ``a - b`` over races with the exact percentile-bootstrap interval.

    ``a`` and ``b`` are per-race 0/1 results indexed by race. Resampling races
    with replacement makes the summed difference a sum of n independent draws
    from the observed differences (each -1, 0 or +1), whose distribution is an
    n-fold convolution. No Monte Carlo error.
    """
    diff = (a - b.reindex(a.index)).to_numpy(dtype=float)
    n = len(diff)
    values = np.array([-1.0, 0.0, 1.0])
    if not np.isin(diff, values).all():
        raise ValueError("paired_difference expects 0/1 results")
    step = np.array([(diff == v).mean() for v in values])
    pmf = np.array([1.0])
    for _ in range(n):
        pmf = np.convolve(pmf, step)
    support = np.arange(-n, n + 1) / n          # possible mean differences
    cdf = np.cumsum(pmf)
    alpha = (1 - level) / 2
    lo = support[np.searchsorted(cdf, alpha - 1e-12)]
    hi = support[np.searchsorted(cdf, 1 - alpha - 1e-12)]
    return float(diff.mean()), float(lo), float(hi)
