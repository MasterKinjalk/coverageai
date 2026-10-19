"""Drift and pattern displacement (first-order, decision D10).

Droplets leave the nozzle at v0 = coeff * sqrt(2 P / rho_w) (coeff 0.6, ~14 m/s at 40 psi)
with zero lateral velocity, and relax toward terminal velocity Vt and wind speed U under
linear drag with time constant tau = Vt / g:
    z(t) = Vt t + (v0 - Vt) tau (1 - exp(-t/tau))        (fall, downward)
    x(t) = U (t - tau (1 - exp(-t/tau)))                  (lateral, per wind component)
The fall time t_f solves z(t_f) = H (Newton, monotone). Cross-track displacement
x_c = x(t_f) with U_cross decides drift: the volume fraction with |x_c| > threshold is
drift. The along-track component only enters the reported pattern shift.
Fine droplets (small tau, small v0 tau penetration) spend most of H settling at Vt and
drift; medium and coarse droplets cross H in tens of ms and barely move.
To avoid a staircase in the 40-bin discretization, the in/out decision is a linear
ramp in ln(x) over each bin's own ln(x) span (exact for a locally log-linear x(d)).
"""
from __future__ import annotations

import numpy as np

from coverageai.config import spec
from coverageai.physics.droplets import terminal_velocity
from coverageai.physics.evaporation import fall_with_evaporation

PHASE_TAUS = 3.0     # near-nozzle penetration phase lasts this many tau


def exit_velocity(pressure_psi, coeff: float, rho_w: float):
    """Nozzle exit velocity v0 = coeff * sqrt(2 P / rho_water), m/s."""
    return coeff * np.sqrt(2.0 * np.asarray(pressure_psi, float) * 6894.757 / rho_w)


def fall_with_exit_velocity(d0_um, height_m, v0, lam, u_cross, u_along):
    """Fall of droplets from the nozzle through H with exit velocity and evaporation.

    Arrays broadcast to (N, B); height, v0, lam, u_* are (N, 1). Returns
    (d_final um, t_fall s, alive, x_cross m, x_along m).

    Evaporation approximation: tau and the relaxation use the initial diameter d0. If the
    droplet reaches H inside the penetration phase (first 3 tau), the d^2 law is applied
    over the exact fall time. Otherwise the phase is flown at d0 (diameter shrunk by the
    d^2 law over 3 tau), and the remaining height H - z(3 tau) is settled with the exact
    closed-form evaporating-settling solution (evaporation.fall_with_evaporation); the
    lateral displacement is x(3 tau) plus U times the settling time (the residual 5 % of
    the exit-velocity and wind lag at 3 tau is ignored).
    """
    g = spec()["physics"]["gravity_m_s2"]
    d0 = np.asarray(d0_um, float)
    vt = terminal_velocity(d0)
    tau = vt / g
    h = np.broadcast_to(np.asarray(height_m, float), d0.shape)
    v0 = np.broadcast_to(np.asarray(v0, float), d0.shape)
    lam = np.broadcast_to(np.asarray(lam, float), d0.shape)
    dv = v0 - vt
    # exact fall time without evaporation: Newton on z(t) = H, started at H / v0
    t = h / v0
    for _ in range(14):
        e = np.exp(-t / tau)
        z = vt * t + dv * tau * (1.0 - e)
        t = np.maximum(t - (z - h) / (vt + dv * e), 1e-9)
    t_ex = t

    def lateral(u, tt):
        return u * (tt - tau * (1.0 - np.exp(-tt / tau)))

    t3 = PHASE_TAUS * tau
    e3 = np.exp(-PHASE_TAUS)
    z3 = vt * t3 + dv * tau * (1.0 - e3)
    in_phase = z3 >= h

    # A: reaches H within the penetration phase
    d_a2 = d0 * d0 - lam * t_ex
    alive_a = d_a2 > 1.0
    d_a = np.sqrt(np.maximum(d_a2, 0.0))
    # B: penetration phase, then evaporating settling over the remaining height
    d1_2 = d0 * d0 - lam * t3
    alive_1 = d1_2 > 1.0
    d1 = np.sqrt(np.maximum(d1_2, 1.0))
    h_rem = np.where(in_phase, 1.0, h - z3)
    d_b, t_s, alive_b = fall_with_evaporation(d1, h_rem, lam)
    t_b = t3 + t_s

    d_f = np.where(in_phase, np.where(alive_a, d_a, 0.0), np.where(alive_1 & alive_b, d_b, 0.0))
    alive = np.where(in_phase, alive_a, alive_1 & alive_b)
    t_f = np.where(in_phase, t_ex, t_b)
    uc, ua = np.asarray(u_cross, float), np.asarray(u_along, float)
    x_c = np.where(in_phase, lateral(uc, t_ex), lateral(uc, t3) + uc * t_s)
    x_a = np.where(in_phase, lateral(ua, t_ex), lateral(ua, t3) + ua * t_s)
    return d_f, t_f, alive, x_c, x_a


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
