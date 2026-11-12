"""Bootstrap confidence intervals over seeds (numpy only).

Used by ``app_logic.comparison_stats`` for the app's multi-seed comparison. The experiment
scripts keep their own copies; these are the shared versions for new code.

Both helpers return ``(mean, lo, hi)``: the sample mean and the percentile-bootstrap bounds of
the mean at ``level`` (95 % by default). With fewer than 2 values the bounds are NaN; a
constant input gives a zero-width interval. The result is deterministic for a given generator
state.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

_NAN = float("nan")


def _as_rng(rng: np.random.Generator | int | Sequence[int] | None) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def bootstrap_mean_ci(values, n_boot: int = 2000,
                      rng: np.random.Generator | int | Sequence[int] | None = None,
                      level: float = 0.95) -> tuple[float, float, float]:
    """Mean of ``values`` and its percentile-bootstrap ``level`` interval (resample with
    replacement ``n_boot`` times). ``rng`` is a Generator or a seed for ``default_rng``."""
    v = np.asarray(values, float).ravel()
    if v.size == 0:
        return _NAN, _NAN, _NAN
    mean = float(v.mean())
    if v.size < 2:
        return mean, _NAN, _NAN
    if np.all(v == v[0]):
        return mean, mean, mean
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    idx = _as_rng(rng).integers(0, v.size, size=(int(n_boot), v.size))
    boot = v[idx].mean(axis=1)
    a = 0.5 * (1.0 - level)
    lo, hi = np.quantile(boot, [a, 1.0 - a])
    return mean, float(lo), float(hi)


def paired_bootstrap_mean_ci(a, b, n_boot: int = 2000,
                             rng: np.random.Generator | int | Sequence[int] | None = None,
                             level: float = 0.95) -> tuple[float, float, float]:
    """Mean of the paired differences ``a - b`` and its bootstrap interval: the pairs (one per
    seed) are resampled together, so the seed-to-seed variation common to both cancels."""
    x, y = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"paired samples differ in length: {x.size} and {y.size}")
    return bootstrap_mean_ci(x - y, n_boot, rng, level)
