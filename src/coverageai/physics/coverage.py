"""Card coverage by an analytic Boolean (Poisson) model, and droplet sampling.

Deposited droplets of final diameter d land at random positions (Poisson process,
n per cm^2 per bin) and leave circular stains of diameter SF * d (SF = spread factor,
one value per segment). Expected covered area fraction of a Boolean model of disks is
C = 1 - exp(-sum_i n_i pi (SF d_i)^2 / 4). Only droplets with d >= min detectable
diameter leave a registered stain. Overlap is accounted for by the exponential; edge
effects are exact because sampling places centers on a card extended by one stain
radius, so stains straddling the card edge are kept (renderers clip them).
"""
from __future__ import annotations

import numpy as np

UL_TO_UM3 = 1e9   # 1 uL = 1 mm^3 = 1e9 um^3


def counts_per_cm2(dep_ul_cm2, d_um):
    """Droplets per cm^2 of each bin given deposited volume (uL/cm^2) and diameter."""
    d = np.maximum(d_um, 1e-3)
    return dep_ul_cm2 * UL_TO_UM3 / (np.pi / 6.0 * d ** 3)


def ln_bin_width(d_um):
    """Local ln(d) width of each bin along axis 1 (diameters below 1 um treated as 1 um).

    Equals the spectrum's dln when droplets do not evaporate; evaporation compresses it.
    """
    ld = np.log(np.maximum(d_um, 1.0))
    return np.maximum(np.abs(np.gradient(ld, axis=-1)), 1e-6)


def detectable_fraction(d_um, width, min_detectable_um: float):
    """Fraction of each bin (uniform in ln d over +-width/2) at or above the cutoff.

    A soft version of ``d >= cutoff`` that avoids a staircase when evaporation moves
    whole bins across the detection limit.
    """
    ld = np.log(np.maximum(d_um, 1e-6))
    return np.clip(0.5 + (ld - np.log(min_detectable_um)) / width, 0.0, 1.0)


def boolean_coverage(n_cm2, d_um, spread_factor, min_detectable_um: float):
    """Returns (coverage fraction (N,), detectable stains per cm^2 (N,)). Shapes (N, B)."""
    n_det = n_cm2 * detectable_fraction(d_um, ln_bin_width(d_um), min_detectable_um)
    stain_cm2 = np.pi / 4.0 * (np.atleast_1d(spread_factor)[:, None] * d_um * 1e-4) ** 2
    cov = 1.0 - np.exp(-(n_det * stain_cm2).sum(axis=1))
    return cov, n_det.sum(axis=1)


def sample_card_droplets(n_cm2, d_um, spread_factor, shift_x_mm, shift_y_mm,
                         min_detectable_um, card_w_mm, card_h_mm, rng):
    """Sample actual droplets for one segment. Inputs are 1-D over bins.

    Count per bin ~ Poisson(n * detectable fraction * area of the card extended by the
    stain radius). Diameters are uniform in ln d over the part of the bin above the
    detection cutoff (consistent with ``boolean_coverage``). Positions are uniform on the
    extended card, then shifted by the bin's mean displacement with toroidal wrap (a
    homogeneous field is invariant to the shift; it is kept so a renderer sees the
    contract's shifted pattern). Returns (diameters_um, x_mm, y_mm).
    """
    width = ln_bin_width(d_um)
    ld = np.log(np.maximum(d_um, 1e-6))
    hi = ld + width / 2.0
    lo = np.maximum(ld - width / 2.0, np.log(min_detectable_um))
    frac = np.clip((hi - lo) / width, 0.0, 1.0)
    margin = spread_factor * np.exp(hi) * 1e-3 / 2.0                  # max stain radius in bin, mm
    area_cm2 = (card_w_mm + 2 * margin) * (card_h_mm + 2 * margin) / 100.0
    lam = np.where(frac > 0, n_cm2 * frac * area_cm2, 0.0)
    cnt = rng.poisson(lam)
    total = int(cnt.sum())
    if total == 0:
        e = np.empty(0)
        return e, e.copy(), e.copy()
    idx = np.repeat(np.arange(len(d_um)), cnt)
    diam = np.exp(lo[idx] + rng.random(total) * (hi[idx] - lo[idx]))
    m = margin[idx]
    wx, wy = card_w_mm + 2 * m, card_h_mm + 2 * m
    x = np.mod(rng.random(total) * wx + shift_x_mm[idx], wx) - m
    y = np.mod(rng.random(total) * wy + shift_y_mm[idx], wy) - m
    return diam, x, y
