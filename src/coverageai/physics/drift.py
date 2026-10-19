"""Drift and pattern displacement (first-order).

A droplet falling for t seconds in a steady wind is displaced x = U * t, where
t = H / Vt (with shrinking diameter, see evaporation.py). Cross-track displacement
x_c = U_cross t decides drift: the volume fraction with |x_c| > threshold is drift.
The along-track component x_a = U_along t only enters the reported pattern shift.
To avoid a staircase in the 40-bin discretization, the in/out decision is a linear
ramp in ln(x) over each bin's own ln(x) span (exact for a locally log-linear x(d)).
"""
from __future__ import annotations

import numpy as np


def in_range_fraction(x_cross, threshold_m: float):
    """Fraction of each diameter bin whose |cross-track displacement| <= threshold.

    x_cross has shape (N, B), bins ordered by diameter (monotone displacement).
    """
    lx = np.log(np.maximum(np.abs(x_cross), 1e-12))
    g = np.abs(np.gradient(lx, axis=1)) if lx.shape[1] > 1 else np.full_like(lx, 1e-9)
    g = np.maximum(g, 1e-9)
    return np.clip(0.5 + (np.log(threshold_m) - lx) / g, 0.0, 1.0)


def mean_displacement_mm(weights, x_cross, x_along, threshold_m: float):
    """Volume-weighted mean displacement vector magnitude of deposited droplets, mm.

    Per-bin components are capped at +-threshold (beyond it a droplet is drift, not on
    the card; along-track has no drift criterion so the cap just bounds the output).
    With nothing deposited the displacement saturates at threshold * 1000 mm.
    """
    w = np.asarray(weights, float)
    ws = w.sum(axis=1)
    xc = np.clip(x_cross, -threshold_m, threshold_m)
    xa = np.clip(x_along, -threshold_m, threshold_m)
    safe = np.where(ws > 1e-12, ws, 1.0)
    mc = (w * xc).sum(axis=1) / safe
    ma = (w * xa).sum(axis=1) / safe
    mag = 1000.0 * np.hypot(mc, ma)
    return np.where(ws > 1e-12, mag, 1000.0 * threshold_m)
