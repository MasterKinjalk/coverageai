"""Droplet spectrum and terminal velocity (first-order).

Spectrum: log-normal in volume. The volume-weighted diameter distribution has median
VMD and geometric std ln(sigma_g) = s. The relative span (Dv0.9 - Dv0.1) / Dv0.5 of a
log-normal is exp(z s) - exp(-z s) = 2 sinh(z s) with z = 1.2816 (the 90th percentile
of the standard normal), so s = asinh(span / 2) / z. For the default span 1.2,
s = 0.444 (sigma_g = 1.56).

Terminal velocity: steady fall under gravity (with buoyancy) against quadratic drag,
Cd = 24/Re (1 + 0.15 Re^0.687) for Re < 1000 (Schiller-Naumann) and 0.44 above.
The force balance is solved by bisection once on a fine grid and then interpolated.
Droplet shape deformation and wind-induced turbulence are ignored.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import ndtr

from coverageai.config import spec

Z90 = 1.2815515655446004   # standard normal 90th percentile
D_GRID_MAX_UM = 6000.0
D_GRID_STEP_UM = 0.5


def sigma_from_span(span: float) -> float:
    """Log-normal shape s = ln(sigma_g) that reproduces a given relative span."""
    return float(np.arcsinh(span / 2.0) / Z90)


def _cd(re: np.ndarray) -> np.ndarray:
    re = np.maximum(re, 1e-12)
    return np.where(re < 1000.0, 24.0 / re * (1.0 + 0.15 * re ** 0.687), 0.44)


def solve_terminal_velocity(d_um: np.ndarray, rho_w: float, rho_a: float, mu: float,
                            g: float) -> np.ndarray:
    """Bisection on the drag/weight balance; returns V_t in m/s for diameters in um."""
    d = np.maximum(np.asarray(d_um, float), 1e-6) * 1e-6
    area = np.pi * d * d / 4.0
    weight = (rho_w - rho_a) * g * np.pi * d ** 3 / 6.0
    lo = np.full_like(d, 1e-9)
    hi = np.full_like(d, 100.0)
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        drag = 0.5 * rho_a * _cd(rho_a * mid * d / mu) * area * mid * mid
        up = drag < weight       # still accelerating -> terminal velocity is larger
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return np.sqrt(lo * hi)


@lru_cache(maxsize=4)
def _tables(rho_w: float, rho_a: float, mu: float, g: float):
    d = np.arange(0.0, D_GRID_MAX_UM + D_GRID_STEP_UM, D_GRID_STEP_UM)
    vt = solve_terminal_velocity(d, rho_w, rho_a, mu, g)
    vt[0] = 0.0
    # F(d) = integral_0^d 2 s Vt(s) ds  [m um^2 / s]; fall height = (F(d0) - F(d)) / lambda
    integrand = 2.0 * d * vt
    f = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(d))])
    return d, vt, f


def tables():
    p = spec()["physics"]
    return _tables(p["water_density_kg_m3"], p["air_density_kg_m3"],
                   p["air_viscosity_pa_s"], p["gravity_m_s2"])


def terminal_velocity(d_um):
    """Terminal fall speed (m/s) of a water droplet of diameter d_um in still air."""
    d, vt, _ = tables()
    return np.interp(d_um, d, vt)


def fall_integral(d_um):
    """F(d) used by the evaporation module (see evaporation.fall_with_evaporation)."""
    d, _, f = tables()
    return np.interp(d_um, d, f)


def invert_fall_integral(f_val):
    d, _, f = tables()
    return np.interp(f_val, f, d)


def lognormal_volume_bins(vmd_um, sigma: float, n_bins: int = 40, zmax: float = 3.5):
    """Discretize the volume-weighted spectrum into n_bins equal-ln(d) bins.

    Returns (d (N, B) bin-center diameters in um, w (B,) volume fractions summing to 1,
    dln (float) bin width in ln(d)). Count fraction of a bin is proportional to w / d^3.
    """
    edges = np.linspace(-zmax, zmax, n_bins + 1)
    zc = 0.5 * (edges[1:] + edges[:-1])
    w = np.diff(ndtr(edges))
    w = w / w.sum()
    vmd = np.atleast_1d(np.asarray(vmd_um, float))
    d = vmd[:, None] * np.exp(sigma * zc)[None, :]
    return d, w, float(sigma * (edges[1] - edges[0]))
