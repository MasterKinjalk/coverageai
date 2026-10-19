"""Droplet evaporation during fall (first-order, no ventilation term).

d^2 law: d(t)^2 = d0^2 - lambda t with lambda = coeff * dT_wb [um^2/s] where
dT_wb = T - T_wb (wet-bulb depression, Stull 2011 approximation) and coeff = 84.76
(Williamson-Threadgill). As d shrinks the fall speed drops; the fall is integrated
exactly in closed form by the change of variable dz = Vt(d) dt = Vt(d) 2 d dd / lambda,
i.e. z = (F(d0) - F(d)) / lambda with F(d) = integral_0^d 2 s Vt(s) ds (tabulated in
droplets.py). Droplets with F(d0) <= lambda H evaporate completely before reaching the
target and are lost.
"""
from __future__ import annotations

import numpy as np

from coverageai.physics.droplets import fall_integral, invert_fall_integral, terminal_velocity


def wet_bulb_c(temp_c, rh_pct):
    """Stull (2011) wet-bulb temperature, valid for RH 5-99 % and T -20..50 C."""
    t = np.asarray(temp_c, float)
    rh = np.clip(np.asarray(rh_pct, float), 5.0, 99.0)
    return (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(t + rh)
            - np.arctan(rh - 1.676331) + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
            - 4.686035)


def wet_bulb_depression(temp_c, rh_pct):
    """T - T_wb in C, floored at 0 (saturated air does not condense onto droplets here)."""
    return np.maximum(np.asarray(temp_c, float) - wet_bulb_c(temp_c, rh_pct), 0.0)


def evaporation_lambda(temp_c, rh_pct, coeff: float = 84.76):
    """d^2 shrink rate, um^2/s."""
    return coeff * wet_bulb_depression(temp_c, rh_pct)


def fall_with_evaporation(d0_um, height_m, lam):
    """Fall of droplets of initial diameter d0 through height H with evaporation.

    Arrays broadcast. Returns (d_final um, t_fall s, alive bool). For dead droplets
    d_final = 0 and t_fall is the no-evaporation time (unused by callers).
    """
    d0 = np.asarray(d0_um, float)
    h = np.asarray(height_m, float)
    lam = np.asarray(lam, float)
    t0 = h / terminal_velocity(d0)
    negligible = lam * t0 < 1e-3 * d0 * d0       # < 0.1 % of d^2 lost: skip the inversion
    f_target = fall_integral(d0) - lam * h
    alive = negligible | (f_target > 0.0)
    d_final = invert_fall_integral(np.maximum(f_target, 0.0))
    lam_safe = np.where(lam > 0, lam, 1.0)
    t_evap = (d0 * d0 - d_final * d_final) / lam_safe
    d_final = np.where(negligible, d0, np.where(alive, d_final, 0.0))
    t_fall = np.where(negligible | ~alive, t0, t_evap)
    return d_final, t_fall, alive
