"""Spray lab: one nozzle setting under one set of conditions, simulated for display.

Pure Python (no Streamlit). ``run_lab`` returns everything the Spray lab tab shows:

1. The simulator's analytic outcome (no hidden mismatch): card coverage, deposit, spray
   drift, pattern displacement, stains per cm^2, plus the rendered water-sensitive paper card.
2. The fate of the spray VOLUME per droplet-size bin, split three ways and computed with
   exactly the simulator's own pipeline (``SpraySimulator._core`` inputs and physics):
     landed   = w * in_range * alive * (d_f / d0)^3   (liquid that reaches the target)
     blown    = w * (1 - in_range) * alive * (d_f / d0)^3   (liquid landing > threshold away)
     evaporated = w * (1 - alive * (d_f / d0)^3)     (water lost to the air in flight)
   ``landed`` summed over bins is the simulator's deposited fraction, so
   ``blown + evaporated == Outcome.drift_frac`` (the simulator counts evaporation as drift).
3. A particle sample for the browser animation. Every particle carries the parameters its
   trajectory needs so the JavaScript can evaluate its position in closed form each frame
   (``particle_state`` is the Python mirror of that JavaScript, used by the tests).

Particle sampling (the "how many of each size" choice). By count, a spray is almost all
tiny droplets (count ~ volume / d^3), so a count-faithful sample shows no large droplets;
by volume, it shows almost only large ones. We sample the 40 simulator bins with
probability p_b ~ w_b * d_b^-1.5 (halfway, in log terms, between volume weighting d^0 and
count weighting d^-3), allocate whole particles per bin (largest remainder, at least one
particle for bins holding >= 0.2 % of the volume), place each bin's particles at evenly
spaced (stratified) diameters across the bin's ln(d) width, and give each particle the
volume weight w_b / n_b. So the particle cloud over-represents large droplets relative to
their count, but volume-weighted tallies of particle fates reproduce the analytic fate
fractions (to within a percentage point or so).

Trajectory (same physics as physics/drift.py, per particle):
  * penetration phase, t <= 3 tau (or the whole fall if the droplet reaches the ground
    inside it): fall  z(t) = vt t + (v0 - vt) tau (1 - e^{-t/tau}),
    crosswind push  x(t) = U (t - tau (1 - e^{-t/tau})), tau = vt / g;
  * settling phase: the simulator's closed-form evaporating fall. For display the fallen
    height over the settling time T is drawn as  z3 + zS q (1 - beta q) / (1 - beta),
    q = s / T, beta = lam T / (2 d1^2) -- exact in the Stokes regime (vt ~ d^2) and exact
    at both ends in general (it lands at the simulator's fall time); x = x3 + U s.
  * diameter d(t)^2 = d0^2 - lam t (d^2 law); a droplet whose diameter reaches zero before
    it lands has evaporated.
  * fan spread (display only): droplets leave along straight rays inside the nozzle's fan
    angle, adding (ray slope) * fallen height to the cross-track position; landing
    positions follow the cos^2 single-nozzle pattern of ``app_logic.pattern_overlap``
    over the fan width W = 2 H tan(angle / 2). The
    simulator itself has no fan geometry; drift is decided only by the wind displacement
    vs ``drift_threshold_m``, exactly as in the simulator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from coverageai.config import spec
from coverageai.contracts import Conditions, Derived, Outcome, Setting, coverage_class
from coverageai.physics import drift as drift_mod
from coverageai.physics.droplets import fall_integral, lognormal_volume_bins, terminal_velocity
from coverageai.physics.evaporation import evaporation_lambda, fall_with_evaporation

# ------------------------------------------------------------------ plain-English vocabulary

DROPLET_CLASS_NAMES = {"VF": "Very fine", "F": "Fine", "M": "Medium", "C": "Coarse",
                       "VC": "Very coarse", "XC": "Extremely coarse", "UC": "Ultra coarse"}
COVERAGE_NAMES = {"acceptable": "Good", "marginal": "Borderline", "critical": "Poor"}

FAMILY_INFO: dict[str, dict[str, str]] = {
    "XR": {"name": "XR extended-range flat fan",
           "desc": "Standard 110° flat-fan tip that works over a wide pressure range. "
                   "Makes fine-to-medium droplets: good coverage, but drifts easily in wind."},
    "TP65": {"name": "TP65 standard flat fan (65°)",
             "desc": "Plain flat-fan tip with a narrow 65° fan. Small sizes make fine droplets, "
                     "so it covers well but is drift-prone."},
    "3D": {"name": "3D angled flat fan",
           "desc": "100° fan tip angled forward/backward for better coverage of upright plants. "
                   "Medium-to-coarse droplets."},
    "ULD": {"name": "ULD air-induction (ultra-low drift)",
            "desc": "Wide 120° fan that mixes air into the liquid to make large, air-filled "
                    "droplets. Very low drift, but fewer droplets per card."},
    "DGX40": {"name": "DGX40 narrow-fan drift-reduction tip",
              "desc": "Narrow 40° fan producing very large (extremely to ultra coarse) "
                      "droplets. Lowest drift, lowest coverage."},
}

# Wind direction relative to the sprayer -> relative angle (deg) the air moves TOWARD,
# measured from the travel direction (heading 0). +90 = air moves to the sprayer's right,
# i.e. left -> right when looking from behind the sprayer along the travel direction.
WIND_DIRECTIONS: dict[str, float] = {
    "Crosswind, blowing left → right": 90.0,
    "Crosswind, blowing right → left": 270.0,
    "Diagonal, from front-left": 135.0,
    "Diagonal, from front-right": 225.0,
    "Headwind (blowing at the front of the sprayer)": 180.0,
    "Tailwind (blowing from behind the sprayer)": 0.0,
}

FATE_LABELS = {"landed": "Landed on target", "blown": "Blown off target",
               "evaporated": "Evaporated in the air"}

EMIT_WINDOW_S = 1.0          # simulated seconds of spraying shown in the animation
EXTRA_AIR_TIME_S = 3.0       # after the last release, show at most this much more air time
DEFAULT_SLOWMO = 0.25        # animation plays at this fraction of real time
SIZE_WEIGHT_EXPONENT = 1.5   # particle sampling p_b ~ w_b * d_b^-1.5 (see module docstring)
D_COLOR_RANGE = (30.0, 1000.0)   # um, fixed log colour scale so runs are comparable
TRAJ_DIAMETERS_UM = (50.0, 100.0, 150.0, 250.0, 400.0, 600.0)


def fan_pattern_samples(n: int, rng: np.random.Generator) -> np.ndarray:
    """Lateral fan position s in [-1/2, 1/2] (units of the fan width W = 2 H tan(angle/2))
    with density 2 cos^2(pi s): the illustrative single-nozzle pattern of
    ``app_logic.pattern_overlap``."""
    g = np.linspace(-0.5, 0.5, 2001)
    cdf = np.cumsum(np.cos(np.pi * g) ** 2)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    return np.interp(rng.random(n), cdf, g)


def droplet_class_name(code: str) -> str:
    return DROPLET_CLASS_NAMES.get(code, str(code))


def family_label(fam: str) -> str:
    return FAMILY_INFO.get(fam, {}).get("name", fam)


def coverage_rating(coverage_pct: float, target_pct: float) -> tuple[str, str]:
    """(class key, plain label) relative to the user's target coverage.

    Same rule as the project spec (acceptable >= 30 %, marginal >= 15 %) with the
    thresholds scaled to the target: Good >= target, Borderline >= target * 15/30.
    """
    cls = spec()["coverage_classes"]
    ratio = float(cls["marginal_min"]) / float(cls["acceptable_min"])
    key = coverage_class(coverage_pct, target_pct, target_pct * ratio)
    return key, COVERAGE_NAMES[key]


def make_conditions(wind_speed_ms: float, wind_rel_deg: float, temp_c: float, rh_pct: float,
                    travel_speed_kmh: float, boom_height_m: float) -> Conditions:
    """Conditions with the sprayer heading 0 deg and the wind given relative to it."""
    return Conditions(wind_speed_ms=float(wind_speed_ms), wind_dir_deg=float(wind_rel_deg) % 360.0,
                      heading_deg=0.0, temp_c=float(temp_c), rh_pct=float(rh_pct),
                      travel_speed_kmh=float(travel_speed_kmh), boom_height_m=float(boom_height_m))


# ------------------------------------------------------------------ result record

@dataclass
class LabResult:
    setting: Setting
    conditions: Conditions
    seed: int
    target_pct: float
    derived: Derived
    outcome: Outcome                       # analytic simulator outcome (no hidden mismatch)
    fate: dict[str, float]                 # landed / blown / evaporated volume fractions (sum 1)
    bins: pd.DataFrame                     # per-size-bin spectrum and fate
    particles: dict[str, Any]              # JSON-serializable animation payload
    particle_fate: dict[str, float]        # the same fractions tallied from the particle sample
    card_image: np.ndarray | None = None   # rendered water-sensitive paper card (RGB uint8)
    card_mask: np.ndarray | None = None
    rating_key: str = "acceptable"
    rating: str = "Good"
    figures: dict[str, Any] = field(default_factory=dict)   # companion figure data (build_figures)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def drift_pct(self) -> float:
        return 100.0 * self.outcome.drift_frac

    @property
    def droplet_class(self) -> str:
        return droplet_class_name(self.derived.droplet_class)


# ------------------------------------------------------------------ physics helpers

def _sim():
    from coverageai.simulator import SpraySimulator
    return SpraySimulator()


def _phys():
    return spec()["physics"]


def bin_fate(sim, x: Conditions, u: Setting, derived: Derived | None = None) -> pd.DataFrame:
    """Per-bin spectrum and volume fate, using the simulator's own inputs (no mismatch).

    Columns: d0_um, d_final_um, volume_frac, landed, blown, evaporated (fractions of the
    TOTAL emitted volume, so each column sums to the overall fraction), x_cross_m, alive.
    """
    from coverageai.simulator import NO_MISMATCH
    vmd, _gpa, uc, _ua, h, t, rh, _eta, _sf, p_psi = sim._inputs(x, u, NO_MISMATCH, 1.0, derived)
    p = sim.p
    d0, w, dln = lognormal_volume_bins(vmd, sim.sigma_ln, sim.n_bins)
    lam = evaporation_lambda(t, rh, p["evaporation_lambda_coeff"])[:, None]
    v0 = drift_mod.exit_velocity(p_psi, p["exit_velocity_coeff"], p["water_density_kg_m3"])[:, None]
    d_f, _t_f, alive, x_c, _x_a = drift_mod.fall_with_exit_velocity(
        d0, h[:, None], v0, lam, uc[:, None], np.zeros_like(uc)[:, None])
    inr = drift_mod.in_range_fraction(x_c, float(p["drift_threshold_m"]))
    r3 = np.where(alive, (d_f / d0) ** 3, 0.0)
    landed = w[None, :] * inr * r3
    blown = w[None, :] * (1.0 - inr) * r3
    evap = w[None, :] * (1.0 - r3)
    return pd.DataFrame({"d0_um": d0[0], "d_final_um": d_f[0], "volume_frac": w,
                         "landed": landed[0], "blown": blown[0], "evaporated": evap[0],
                         "x_cross_m": x_c[0], "alive": alive[0].astype(bool),
                         "dln": np.full(len(w), dln)})


def particle_kinematics(d0_um, height_m: float, v0: float, lam: float, u_cross: float
                        ) -> dict[str, np.ndarray]:
    """Per-particle trajectory parameters; mirrors physics.drift.fall_with_exit_velocity.

    Returns arrays: vt, tau, mode (0 = lands inside the penetration phase, 1 = two-phase),
    t3, z3, x3, tS (settling duration until landing or death), zS (height fallen while
    settling), beta (settling-curve shape), t_end (flight time until landing / full
    evaporation), alive, d_final_um, x_cross_m (wind displacement at landing).
    """
    g = float(_phys()["gravity_m_s2"])
    d0 = np.asarray(d0_um, float)
    vt = terminal_velocity(d0)
    tau = vt / g
    h = float(height_m)
    dv = v0 - vt
    t = np.full_like(d0, h / v0)
    for _ in range(14):                                   # same Newton as drift.py
        e = np.exp(-t / tau)
        z = vt * t + dv * tau * (1.0 - e)
        t = np.maximum(t - (z - h) / (vt + dv * e), 1e-9)
    t_ex = t
    t3 = drift_mod.PHASE_TAUS * tau
    z3 = vt * t3 + dv * tau * (1.0 - math.exp(-drift_mod.PHASE_TAUS))
    x3 = u_cross * (t3 - tau * (1.0 - math.exp(-drift_mod.PHASE_TAUS)))
    in_phase = z3 >= h
    lam_safe = lam if lam > 0 else 1.0

    # A: reaches the ground inside the penetration phase
    alive_a = d0 * d0 - lam * t_ex > 1.0
    t_death_a = d0 * d0 / lam_safe
    # B: penetration phase, then evaporating settling over the remaining height
    d1_2 = d0 * d0 - lam * t3
    alive_1 = d1_2 > 1.0
    d1 = np.sqrt(np.maximum(d1_2, 1.0))
    h_rem = np.where(in_phase, 1.0, h - z3)
    d_b, t_s, alive_b = fall_with_evaporation(d1, h_rem, lam)
    settle_dead = alive_1 & ~alive_b
    t_settle_dead = d1 * d1 / lam_safe
    z_settle_dead = np.minimum(fall_integral(d1) / lam_safe, h_rem)

    mode = np.where(in_phase, 0, 1)
    alive = np.where(in_phase, alive_a, alive_1 & alive_b)
    t_s_all = np.where(settle_dead, t_settle_dead, t_s)
    zS = np.where(settle_dead, z_settle_dead, h_rem)
    beta = np.where(settle_dead, 0.5,
                    np.clip(lam * t_s_all / (2.0 * d1 * d1), 0.0, 0.5))
    t_end = np.where(in_phase, np.where(alive_a, t_ex, np.minimum(t_death_a, t_ex)),
                     np.where(~alive_1, np.minimum(t_death_a, t3), t3 + t_s_all))
    # inside phase B the penetration may end the flight (dead before t3) -> tS irrelevant
    t_s_all = np.where(in_phase | ~alive_1, 0.0, t_s_all)
    zS = np.where(in_phase | ~alive_1, 0.0, zS)

    def lateral(tt):
        return u_cross * (tt - tau * (1.0 - np.exp(-tt / tau)))

    d_f = np.where(in_phase, np.where(alive_a, np.sqrt(np.maximum(d0 * d0 - lam * t_ex, 0.0)), 0.0),
                   np.where(alive_1 & alive_b, d_b, 0.0))
    x_c = np.where(in_phase, lateral(t_ex), x3 + u_cross * t_s)
    return dict(vt=vt, tau=tau, mode=mode, t3=t3, z3=z3, x3=x3, tS=t_s_all, zS=zS, beta=beta,
                t_end=t_end, alive=alive, d_final_um=d_f, x_cross_m=x_c)


def _allocate(n_total: int, p: np.ndarray, w: np.ndarray, min_w: float = 2e-3) -> np.ndarray:
    """Whole particle counts per bin by largest remainder; >= 1 where w >= min_w."""
    raw = n_total * p / p.sum()
    n = np.floor(raw).astype(int)
    rem = n_total - n.sum()
    if rem > 0:
        n[np.argsort(-(raw - n))[:rem]] += 1
    n = np.where((w >= min_w) & (n == 0), 1, n)
    return n


def sample_particles(bins: pd.DataFrame, x: Conditions, u: Setting, derived: Derived,
                     n_nozzles: int, n_particles: int, rng: np.random.Generator
                     ) -> tuple[dict[str, Any], dict[str, float]]:
    """Animation payload (column arrays) and the particle-tallied volume fate fractions."""
    p = _phys()
    thr = float(p["drift_threshold_m"])
    w = bins["volume_frac"].to_numpy()
    dc = bins["d0_um"].to_numpy()
    dln = float(bins["dln"].iloc[0])
    pb = w * dc ** (-SIZE_WEIGHT_EXPONENT)
    nb = _allocate(int(n_particles), pb, w)
    idx = np.repeat(np.arange(len(w)), nb)
    # stratified diameters across each bin's ln(d) width
    pos = np.concatenate([(np.arange(k) + 0.5) / k - 0.5 for k in nb if k > 0]) if nb.sum() else np.empty(0)
    d0 = dc[idx] * np.exp(dln * pos)
    weight = w[idx] / nb[idx]
    n = len(d0)

    lam = float(evaporation_lambda(x.temp_c, x.rh_pct, p["evaporation_lambda_coeff"]))
    v0 = float(drift_mod.exit_velocity(u.pressure_psi, p["exit_velocity_coeff"],
                                       p["water_density_kg_m3"]))
    U = float(derived.u_cross_ms)
    H = float(x.boom_height_m)
    k = particle_kinematics(d0, H, v0, lam, U)

    alive = k["alive"].astype(bool)
    r3 = np.where(alive, (k["d_final_um"] / d0) ** 3, 0.0)
    on_target = alive & (np.abs(k["x_cross_m"]) <= thr)
    fate = np.where(~alive, 2, np.where(on_target, 0, 1))      # 0 landed, 1 blown, 2 evaporated

    half = math.radians(float(derived.spray_angle_deg)) / 2.0
    # lateral fan offset at the ground: W * s, s ~ cos^2 pattern (same as pattern_overlap)
    fan = 2.0 * math.tan(half) * fan_pattern_samples(n, rng)    # slope of the droplet's ray
    noz = rng.integers(0, n_nozzles, n)
    t0 = (rng.permutation(n) + rng.random(n)) / max(n, 1) * EMIT_WINDOW_S

    spacing = float(spec()["boom"]["nozzle_spacing_in"]) * 0.0254
    noz_x = (np.arange(n_nozzles) - (n_nozzles - 1) / 2.0) * spacing
    x_land = noz_x[noz] + fan * H + k["x_cross_m"]

    tally = {"landed": float((weight * r3)[fate == 0].sum()),
             "blown": float((weight * r3)[fate == 1].sum()),
             "evaporated": float((weight * (1.0 - r3)).sum())}
    tot = sum(tally.values()) or 1.0
    tally = {kk: v / tot for kk, v in tally.items()}

    def col(a, nd=5):
        a = np.asarray(a, float)
        return [float(f"{v:.{nd}g}") for v in a]

    swath_half = n_nozzles * spacing / 2.0            # target swath = n_nozzles x spacing
    thr_line = swath_half + thr                         # drift-threshold lines beyond the swath
    windy = abs(U) > 0.2
    down, up = (1.4, 0.35) if windy else (0.35, 0.35)
    xmin, xmax = (-(thr_line + up), thr_line + down) if U >= 0 else \
        (-(thr_line + down), thr_line + up)
    t_last = float(np.max(t0 + k["t_end"])) if n else 0.0
    payload = {
        "n": n,
        "t0": col(t0), "noz": [int(v) for v in noz], "fan": col(fan, 4), "d0": col(d0, 4),
        "vt": col(k["vt"]), "tau": col(k["tau"]), "mode": [int(v) for v in k["mode"]],
        "tS": col(k["tS"]), "zS": col(k["zS"]), "beta": col(k["beta"], 4),
        "tEnd": col(k["t_end"]), "fate": [int(v) for v in fate], "r3": col(r3, 4),
        "w": col(weight, 5), "xLand": col(x_land, 4),
        "H": H, "v0": v0, "U": U, "lam": lam, "thr": thr,
        "nozX": col(noz_x, 5), "spacing": spacing, "swathHalf": swath_half,
        "targetHalf": swath_half, "thrLine": thr_line, "dLo": D_COLOR_RANGE[0],
        "dHi": D_COLOR_RANGE[1],
        "sprayAngle": float(derived.spray_angle_deg),
        "xmin": xmin, "xmax": xmax,
        "emit": EMIT_WINDOW_S, "tFinal": min(t_last, EMIT_WINDOW_S + EXTRA_AIR_TIME_S),
        "slowmo": DEFAULT_SLOWMO,
    }
    return payload, tally


def particle_state(payload: dict[str, Any], i: int, t: float) -> tuple[float, float, float]:
    """(cross-track x m, height above ground m, diameter um) of particle i at flight time t.

    Python mirror of the JavaScript ``state()`` in app/static_spray_lab.html.
    """
    P = payload
    H, v0, U, lam = P["H"], P["v0"], P["U"], P["lam"]
    vt, tau = P["vt"][i], P["tau"][i]
    t = min(max(t, 0.0), P["tEnd"][i])
    t3 = drift_mod.PHASE_TAUS * tau

    def pen(tt):
        e = 1.0 - math.exp(-tt / tau)
        return vt * tt + (v0 - vt) * tau * e, U * (tt - tau * e)

    if P["mode"][i] == 0 or t <= t3:
        z, xw = pen(t)
    else:
        z3, x3 = pen(t3)
        T, b = P["tS"][i], P["beta"][i]
        q = min(max((t - t3) / T, 0.0), 1.0) if T > 0 else 1.0
        z = z3 + P["zS"][i] * q * (1.0 - b * q) / (1.0 - b)
        xw = x3 + U * (t - t3)
    z = min(z, H)
    x = P["nozX"][P["noz"][i]] + P["fan"][i] * z + xw
    d = math.sqrt(max(P["d0"][i] ** 2 - lam * t, 0.0))
    return x, H - z, d


# ------------------------------------------------------------------ main entry

def run_lab(setting: Setting, conditions: Conditions, seed: int = 0, n_nozzles: int = 5,
            n_particles: int = 1000, target_pct: float | None = None,
            render_card: bool = True, with_figures: bool = True) -> LabResult:
    """Simulate one setting under one set of conditions (no hidden mismatch).

    The analytic outcome comes from ``SpraySimulator.simulate(..., mismatch="none",
    sample_droplets=True)`` with ``default_rng(seed)``; the card is rendered from its
    sampled droplets (clean scan, no photo degradation).
    """
    from coverageai.simulator import SpraySimulator
    sim = SpraySimulator()
    derived = sim.derive_fn(conditions, setting)
    target = float(spec()["c_req_pct"] if target_pct is None else target_pct)
    rng = np.random.default_rng(int(seed))
    outcome = sim.simulate(conditions, setting, rng, "none", sample_droplets=render_card)

    bins = bin_fate(sim, conditions, setting, derived)
    fate = {k: float(bins[k].sum()) for k in ("landed", "blown", "evaporated")}
    tot = sum(fate.values())
    fate = {k: v / tot for k, v in fate.items()}

    payload, tally = sample_particles(bins, conditions, setting, derived, n_nozzles,
                                      n_particles, np.random.default_rng([int(seed), 7]))
    key, label = coverage_rating(outcome.coverage_pct, target)
    payload["summary"] = {
        "coverage": outcome.coverage_pct, "target": target, "rating": label, "ratingKey": key,
        "drift": 100.0 * outcome.drift_frac, "landed": 100.0 * fate["landed"],
        "blown": 100.0 * fate["blown"], "evaporated": 100.0 * fate["evaporated"],
        "dropletClass": droplet_class_name(derived.droplet_class),
        "windSpeed": float(conditions.wind_speed_ms),
        "windAlong": float(derived.u_along_ms),
        "vmd": float(derived.vmd_um),
    }

    img = mask = None
    if render_card:
        from coverageai.vision.render import WSPRenderer
        img, mask = WSPRenderer().render(outcome, np.random.default_rng([int(seed), 11]),
                                         degrade=False)
    return LabResult(setting=setting, conditions=conditions, seed=int(seed), target_pct=target,
                     derived=derived, outcome=outcome, fate=fate, bins=bins, particles=payload,
                     particle_fate=tally, card_image=img, card_mask=mask, rating_key=key,
                     rating=label,
                     figures=build_figures(sim, conditions, setting, derived, bins, payload,
                                           n_nozzles) if with_figures else {})


def fate_chart_frame(bins: pd.DataFrame, n_groups: int = 12, min_frac: float = 1e-3) -> pd.DataFrame:
    """Long-format frame for the droplet-fate chart: bins with >= min_frac of the volume,
    merged into about ``n_groups`` consecutive size groups.

    Columns: group (label "120–150 µm"), order, d_mid_um, fate (plain label), pct (% of
    total spray volume).
    """
    b = bins[bins["volume_frac"] >= min_frac].reset_index(drop=True)
    if b.empty:
        return pd.DataFrame(columns=["group", "order", "d_mid_um", "fate", "pct"])
    per = max(1, int(math.ceil(len(b) / n_groups)))
    rows = []
    edge = math.exp(float(bins["dln"].iloc[0]) / 2.0)
    for gi, start in enumerate(range(0, len(b), per)):
        g = b.iloc[start:start + per]
        lo = float(g["d0_um"].iloc[0]) / edge
        hi = float(g["d0_um"].iloc[-1]) * edge
        label = f"{lo:.0f}–{hi:.0f} µm"
        mid = math.sqrt(lo * hi)
        for key, name in FATE_LABELS.items():
            rows.append({"group": label, "order": gi, "d_mid_um": mid, "fate": name,
                         "pct": 100.0 * float(g[key].sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ scientific figure data
# All frames below come from the same physics as the simulator (no hidden mismatch) and
# are deterministic. Column names are internal; the view supplies plain-English labels.

FATE_SHORT = {0: "Deposited on target", 1: "Drifts (displaced > threshold)",
              2: "Evaporates before landing"}


def _scalars(x: Conditions, u: Setting, derived: Derived) -> tuple[float, float, float, float]:
    """(lam um^2/s, v0 m/s, crosswind U m/s, boom height H m)."""
    p = _phys()
    lam = float(evaporation_lambda(x.temp_c, x.rh_pct, p["evaporation_lambda_coeff"]))
    v0 = float(drift_mod.exit_velocity(u.pressure_psi, p["exit_velocity_coeff"],
                                       p["water_density_kg_m3"]))
    return lam, v0, float(derived.u_cross_ms), float(x.boom_height_m)


def kinematic_positions(k: dict[str, np.ndarray], d0, v0: float, lam: float, U: float,
                        H: float, t) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized ``particle_state`` without fan/nozzle offset: (wind displacement x m,
    height above target m, diameter um) at flight times t (broadcast against k arrays)."""
    tau, vt = k["tau"], k["vt"]
    t = np.clip(t, 0.0, k["t_end"])
    e = 1.0 - np.exp(-t / tau)
    z_pen = vt * t + (v0 - vt) * tau * e
    x_pen = U * (t - tau * e)
    ts = np.where(k["tS"] > 0, k["tS"], 1.0)
    q = np.where(k["tS"] > 0, np.clip((t - k["t3"]) / ts, 0.0, 1.0), 1.0)
    z_set = k["z3"] + k["zS"] * q * (1.0 - k["beta"] * q) / (1.0 - k["beta"])
    x_set = k["x3"] + U * (t - k["t3"])
    pen = (k["mode"] == 0) | (t <= k["t3"])
    z = np.minimum(np.where(pen, z_pen, z_set), H)
    xw = np.where(pen, x_pen, x_set)
    d = np.sqrt(np.maximum(np.asarray(d0, float) ** 2 - lam * t, 0.0))
    return xw, H - z, d


def trajectory_frame(x: Conditions, u: Setting, derived: Derived,
                     diameters=TRAJ_DIAMETERS_UM, n_t: int = 160
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Height vs wind displacement for representative initial diameters, one nozzle at
    x = 0 releasing straight down (fan centre). Returns (lines, end points).

    lines: d_um, t_s, x_m, h_m, step.  ends: d_um, x_m, h_m, t_s, outcome ("lands" /
    "evaporates").
    """
    lam, v0, U, H = _scalars(x, u, derived)
    d0 = np.asarray(diameters, float)
    k = particle_kinematics(d0, H, v0, lam, U)
    frac = np.linspace(0.0, 1.0, n_t)[:, None] ** 3      # dense early: the exit phase is short
    t = frac * k["t_end"][None, :]
    xw, h, _ = kinematic_positions(k, d0, v0, lam, U, H, t)
    lines = pd.DataFrame({"d_um": np.repeat(d0[None, :], n_t, 0).ravel(), "t_s": t.ravel(),
                          "x_m": xw.ravel(), "h_m": h.ravel(),
                          "step": np.repeat(np.arange(n_t), len(d0))})
    xe, he, _ = kinematic_positions(k, d0, v0, lam, U, H, k["t_end"])
    ends = pd.DataFrame({"d_um": d0, "x_m": xe, "h_m": he, "t_s": k["t_end"],
                         "outcome": np.where(k["alive"], "lands", "evaporates")})
    return lines, ends


def _dv(bins: pd.DataFrame, q: float) -> float:
    """Diameter below which a fraction q of the emitted volume lies (log interpolation)."""
    edge = math.exp(float(bins["dln"].iloc[0]) / 2.0)
    d_edges = np.concatenate([[bins["d0_um"].iloc[0] / edge], bins["d0_um"].to_numpy() * edge])
    cum = np.concatenate([[0.0], np.cumsum(bins["volume_frac"].to_numpy())])
    return float(np.exp(np.interp(q, cum, np.log(d_edges))))


def drift_potential_frame(x: Conditions, u: Setting, derived: Derived, bins: pd.DataFrame,
                          n: int = 140, d_range=(20.0, 1000.0)) -> tuple[pd.DataFrame, dict]:
    """Wind displacement at landing and flight time vs initial diameter (the classic
    drift-potential curve). Returns (frame, meta).

    frame: d_um, x_land_m (|displacement|, NaN if it evaporates), x_plot_m (floored at
    1 mm for log axes), t_s (fall time, or time until fully evaporated), fate.
    meta: d_crit_um (smaller droplets drift or evaporate; None if none do), vmd_um,
    dv10_um, dv90_um, thr_m, wind_ms (crosswind).
    """
    lam, v0, U, H = _scalars(x, u, derived)
    thr = float(_phys()["drift_threshold_m"])
    d = np.geomspace(*d_range, n)
    k = particle_kinematics(d, H, v0, lam, U)
    alive = k["alive"].astype(bool)
    ax = np.abs(k["x_cross_m"])
    fate = np.where(~alive, 2, np.where(ax <= thr, 0, 1))
    lost = np.nonzero(fate != 0)[0]
    d_crit = None
    if len(lost):
        i = int(lost[-1])
        if i + 1 >= n:
            d_crit = float(d[-1])
        elif alive[i] and alive[i + 1] and ax[i] > 0 and ax[i + 1] > 0:
            a, b = np.log(ax[i]), np.log(ax[i + 1])          # log-linear crossing of |x| = thr
            f = (a - math.log(thr)) / (a - b) if a != b else 0.5
            d_crit = float(np.exp(np.log(d[i]) + f * (np.log(d[i + 1]) - np.log(d[i]))))
        else:
            d_crit = float(np.sqrt(d[i] * d[i + 1]))
    frame = pd.DataFrame({"d_um": d, "x_land_m": np.where(alive, ax, np.nan),
                          "x_plot_m": np.where(alive, np.maximum(ax, 1e-3), np.nan),
                          "t_s": k["t_end"], "fate": [FATE_SHORT[int(f)] for f in fate]})
    meta = {"d_crit_um": d_crit, "vmd_um": _dv(bins, 0.5), "dv10_um": _dv(bins, 0.1),
            "dv90_um": _dv(bins, 0.9), "thr_m": thr, "wind_ms": abs(U)}
    return frame, meta


SPECTRUM_SERIES = {"volume_frac": "Emitted", "landed": "Deposited on target",
                   "blown": "Drifted (displaced > threshold)", "evaporated": "Evaporated"}


def spectrum_frames(bins: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Volume per size class (% of emitted volume, by initial diameter) for emitted /
    deposited / drifted / evaporated, the cumulative volume curve, and DV10/50/90.

    Bins have equal width in ln(d), so % per bin is proportional to the volume density
    per ln(d). Returns (per_bin long frame: d_um, series, pct; cumulative: d_um, cum_pct;
    meta: dv10_um, dv50_um, dv90_um, dln).
    """
    per_bin = pd.concat([pd.DataFrame({"d_um": bins["d0_um"], "series": name,
                                       "pct": 100.0 * bins[key].to_numpy()})
                         for key, name in SPECTRUM_SERIES.items()], ignore_index=True)
    edge = math.exp(float(bins["dln"].iloc[0]) / 2.0)
    d_edges = np.concatenate([[bins["d0_um"].iloc[0] / edge], bins["d0_um"].to_numpy() * edge])
    cum = pd.DataFrame({"d_um": d_edges, "cum_pct": 100.0 * np.concatenate(
        [[0.0], np.cumsum(bins["volume_frac"].to_numpy())])})
    meta = {"dv10_um": _dv(bins, 0.1), "dv50_um": _dv(bins, 0.5), "dv90_um": _dv(bins, 0.9),
            "dln": float(bins["dln"].iloc[0])}
    return per_bin, cum, meta


def deposition_profile(bins: pd.DataFrame, x: Conditions, derived: Derived, n_nozzles: int,
                       xlim: tuple[float, float], n_x: int = 320
                       ) -> tuple[pd.DataFrame, dict]:
    """Deposit (uL/cm^2) across the boom: sum over nozzles of each nozzle's cos^2 fan
    pattern (width W = 2 H tan(angle/2), as app_logic.pattern_overlap) shifted by each
    size bin's wind displacement, weighted by the bin's deposited and drifted volume.

    Normalization: a nozzle delivers ``applied * spacing`` (uL/cm^2 x m) per unit travel,
    so the integral over x of the on-target deposit equals
    applied * n_nozzles * spacing * fate["landed"] (exact when every deposit is in range).
    Returns (frame: x_m, component, ul_cm2; meta: applied_ul_cm2, cv_pct, cv_range, dx_m,
    swath_half_m, thr_line_m, spacing_m, fan_width_m).
    """
    p = _phys()
    H = float(x.boom_height_m)
    spacing = float(spec()["boom"]["nozzle_spacing_in"]) * 0.0254
    noz_x = (np.arange(n_nozzles) - (n_nozzles - 1) / 2.0) * spacing
    W = 2.0 * H * math.tan(math.radians(float(derived.spray_angle_deg)) / 2.0)
    xc = bins["x_cross_m"].to_numpy()
    edges = np.linspace(xlim[0], xlim[1], n_x + 1)
    dx = float(edges[1] - edges[0])
    centers = 0.5 * (edges[1:] + edges[:-1])
    applied = float(derived.app_rate_gpa) * float(p["gpa_to_ul_cm2"])

    def cdf(s):                       # CDF of the density 2 cos^2(pi s) on [-1/2, 1/2]
        s = np.clip(s, -0.5, 0.5)
        return s + 0.5 + np.sin(2.0 * np.pi * s) / (2.0 * np.pi)

    # exact share of each (nozzle, size bin) footprint falling in each x cell: (N, B, n_x)
    c = noz_x[:, None, None] + xc[None, :, None]
    share = cdf((edges[None, None, 1:] - c) / W) - cdf((edges[None, None, :-1] - c) / W)
    out, comps = [], {}
    for key, name in (("landed", "Deposited on target"),
                      ("blown", "Drifted (displaced > threshold)")):
        dens = (share * bins[key].to_numpy()[None, :, None]).sum(axis=(0, 1))
        comps[key] = applied * spacing * dens / dx          # uL/cm^2
        out.append(pd.DataFrame({"x_m": centers, "component": name, "ul_cm2": comps[key]}))
    total = comps["landed"] + comps["blown"]
    cv, rng_cv = None, None
    if n_nozzles >= 3:
        lo, hi = float(noz_x[1]), float(noz_x[-2])
        m = (centers >= lo) & (centers <= hi)
        if m.sum() > 3 and total[m].mean() > 0:
            cv = float(100.0 * total[m].std() / total[m].mean())
            rng_cv = (lo, hi)
    meta = {"applied_ul_cm2": applied, "cv_pct": cv, "cv_range": rng_cv, "dx_m": dx,
            "swath_half_m": n_nozzles * spacing / 2.0,
            "thr_line_m": n_nozzles * spacing / 2.0 + float(p["drift_threshold_m"]),
            "spacing_m": spacing, "fan_width_m": W}
    return pd.concat(out, ignore_index=True), meta


def setting_label(u: Setting) -> str:
    return f"{u.family} {u.size} at {u.pressure_psi:g} psi"


def alternative_nozzle(u: Setting, x: Conditions, derive_fn=None) -> Setting | None:
    """A contrasting nozzle with about the same flow: air-induction ULD (or XR flat fan if
    the current nozzle is already a ULD), at the same pressure clipped to its range."""
    from coverageai import catalog
    derive_fn = derive_fn or catalog.derive
    fam = "ULD" if u.family != "ULD" else "XR"
    flow = derive_fn(x, u).flow_gpm
    rngs = spec()["pressure_range_psi"]
    best, best_err = None, float("inf")
    for sz in catalog.sizes(fam):
        clo, chi = catalog.pressure_range(fam, sz)
        lo, hi = max(clo, float(rngs[fam][0])), min(chi, float(rngs[fam][1]))
        if lo > hi:
            continue
        pr = float(min(max(u.pressure_psi, lo), hi))
        f, _ = catalog.flow_and_class(fam, sz, pr)
        err = abs(math.log(f / flow))
        if err < best_err:
            best, best_err = Setting(fam, sz, pr), err
    return best


def drift_vs_wind_frame(sim, x: Conditions, u: Setting, n: int = 29,
                        alt: Setting | None = None) -> tuple[pd.DataFrame, dict]:
    """Spray drift (%) vs wind speed from 0 to the top of the operating range, wind
    direction kept as chosen, for this nozzle and (optionally) an alternative.
    Returns (frame: wind_ms, drift_pct, nozzle; meta: current wind/drift, wind limit)."""
    import dataclasses
    ws = np.linspace(0.0, float(spec()["ranges"]["wind_speed_ms"][1]), n)
    xs = [dataclasses.replace(x, wind_speed_ms=float(w)) for w in ws]
    rows = []
    for setting in [u] + ([alt] if alt is not None else []):
        a = sim.simulate_arrays(xs, [setting] * n, np.random.default_rng(0), "none")
        rows.append(pd.DataFrame({"wind_ms": ws, "drift_pct": 100.0 * a["drift_frac"],
                                  "nozzle": setting_label(setting)}))
    cur = sim.simulate_arrays([x], [u], np.random.default_rng(0), "none")
    meta = {"wind_ms": float(x.wind_speed_ms), "drift_pct": float(100.0 * cur["drift_frac"][0]),
            "nozzle": setting_label(u), "alt": setting_label(alt) if alt is not None else None,
            "wind_limit_ms": float(spec()["wind_limit_ms"])}
    return pd.concat(rows, ignore_index=True), meta


def build_figures(sim, x: Conditions, u: Setting, derived: Derived, bins: pd.DataFrame,
                  payload: dict[str, Any], n_nozzles: int) -> dict[str, Any]:
    """All data for the companion figures (see the individual builders)."""
    traj, ends = trajectory_frame(x, u, derived)
    pot, pot_meta = drift_potential_frame(x, u, derived, bins)
    spec_bins, spec_cum, spec_meta = spectrum_frames(bins)
    prof, prof_meta = deposition_profile(bins, x, derived, n_nozzles,
                                         (payload["xmin"], payload["xmax"]))
    try:
        alt = alternative_nozzle(u, x, sim.derive_fn)
    except ValueError:         # catalog edge case: show this nozzle only
        alt = None
    sens, sens_meta = drift_vs_wind_frame(sim, x, u, alt=alt)
    return {"trajectories": traj, "trajectory_ends": ends, "potential": pot,
            "potential_meta": pot_meta, "spectrum": spec_bins, "cumulative": spec_cum,
            "spectrum_meta": spec_meta, "profile": prof, "profile_meta": prof_meta,
            "sensitivity": sens, "sensitivity_meta": sens_meta}
