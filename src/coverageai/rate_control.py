"""Fixed application rate (opt-in, operator console).

An operator sets the nozzle (family, size) and a target application rate in gal/acre (GPA). The
rate is a label rate: it must be held within a tolerance (spec ``rate_control.tolerance_frac``,
5 %). With the nozzle and the rate fixed, only pressure and boom height are free. Flow follows
Q = k sqrt(P) (catalog), and GPA is proportional to Q / speed, so every pressure has exactly one
travel speed that gives the target rate. This module holds that arithmetic: the speed for a
pressure, the pressure for a speed, the "iso-rate" candidate settings the selector may choose
from, and a ranking of other nozzles that could hold the same rate with better predicted
coverage. Everything here uses the catalog and the model's predictions only; it does not touch
the simulator. Nothing in this module runs unless a rate is set, so default results are unchanged.
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np

from coverageai import catalog, units
from coverageai.config import spec
from coverageai.contracts import Conditions, Setting

_EPS = 1e-9
EXACT_TOL = 1e-6       # relative rate deviation counted as "exactly on target"


def tolerance_frac() -> float:
    """Allowed relative deviation of the applied rate (spec, 0.05 if the block is missing)."""
    return float(spec().get("rate_control", {}).get("tolerance_frac", 0.05))


def pressure_bounds(family: str, size: str) -> tuple[float, float]:
    """Catalog pressure range of the nozzle intersected with the spec family range."""
    lo, hi = catalog.pressure_range(family, size)
    r = spec()["pressure_range_psi"][family]
    return max(lo, float(r[0])), min(hi, float(r[1]))


def _spacing_in() -> float:
    return float(spec()["boom"]["nozzle_spacing_in"])


def speed_for_rate(family: str, size: str, pressure_psi: float, gpa: float) -> float:
    """Travel speed (km/h) that gives exactly ``gpa`` at ``pressure_psi`` (the inverse of
    ``catalog.derive(...).app_rate_gpa`` in speed)."""
    flow, _ = catalog.flow_and_class(family, size, pressure_psi)
    mph = 5940.0 * flow / (float(gpa) * _spacing_in())
    return float(units.mph_to_kmh(mph))


def pressure_for_rate(family: str, size: str, gpa: float, speed_kmh: float) -> float | None:
    """Pressure (psi) that gives ``gpa`` at ``speed_kmh``, or None if no pressure inside
    ``pressure_bounds`` does. Flow rises with pressure, so a bisection finds it."""
    lo, hi = pressure_bounds(family, size)
    need = float(gpa) * units.kmh_to_mph(speed_kmh) * _spacing_in() / 5940.0     # gal/min

    def q(p: float) -> float:
        return catalog.flow_and_class(family, size, p)[0]

    if need < q(lo) * (1 - 1e-12) or need > q(hi) * (1 + 1e-12):
        return None
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if q(mid) < need:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return float(0.5 * (lo + hi))


def pressure_grid(family: str, size: str, step_psi: float = 5.0) -> list[float]:
    """Pressures from the lower bound in whole steps, the upper bound always included."""
    lo, hi = pressure_bounds(family, size)
    if hi < lo - _EPS:
        return []
    n = int(np.floor((hi - lo) / step_psi + _EPS))
    ps = [lo + i * step_psi for i in range(n + 1)]
    if hi - ps[-1] > _EPS:
        ps.append(hi)
    return [float(round(p, 6)) for p in ps]


def iso_rate_candidates(family: str, size: str, gpa: float, tol_frac: float | None = None,
                        pressure_step_psi: float = 5.0,
                        levels: Sequence[float] | None = None) -> list[Setting]:
    """Settings of one nozzle whose speed is set so the rate is ``gpa * level``. ``levels`` are
    multipliers of the target, default (1 - tol, 1, 1 + tol). Settings whose speed falls outside
    the spec travel-speed range are dropped. Order: exact-rate settings first (level 1), then the
    other levels in the order given, each by rising pressure."""
    tol = tolerance_frac() if tol_frac is None else float(tol_frac)
    levels = (1.0 - tol, 1.0, 1.0 + tol) if levels is None else tuple(float(v) for v in levels)
    lo_v, hi_v = (float(v) for v in spec()["ranges"]["travel_speed_kmh"])
    order = sorted(range(len(levels)), key=lambda i: (abs(levels[i] - 1.0) > EXACT_TOL, i))
    out: list[Setting] = []
    ps = pressure_grid(family, size, pressure_step_psi)
    for i in order:
        for p in ps:
            v = speed_for_rate(family, size, p, gpa * levels[i])
            if lo_v - _EPS <= v <= hi_v + _EPS:
                out.append(Setting(family, size, p, travel_speed_kmh=v))
    return out


def rate_deviation(x: Conditions, u: Setting, gpa: float) -> float:
    """Relative deviation (applied - target) / target of the rate at (x, u)."""
    return float(catalog.derive(x, u).app_rate_gpa / float(gpa) - 1.0)


def alternative_nozzles(conditions: Sequence[Conditions], model, c_req: float, gpa: float,
                        bias: float = 0.0, exclude: tuple[str, str] | None = None,
                        tol_frac: float | None = None, top: int = 5, selector=None) -> list[dict]:
    """Other nozzles that could hold the same rate with a better predicted result.

    For every nozzle in the catalog (except ``exclude``) and every section, the pressure is chosen
    as the rate-locked selector would: lowest predicted drift among settings that meet the
    target with the selector's margin (corrected coverage - margin >= c_req), else the highest
    coverage. Only exact-rate settings are used and the boom height stays as in the conditions.
    Each nozzle costs one ``predict_batch`` call. Returns up to ``top`` dicts (family, size,
    share_meeting, mean_cov, mean_drift, pressure_range_psi, speed_range_kmh), ranked by
    share_meeting (high first), then mean_drift (low first). ``selector`` supplies the margin
    rule (a default ConstrainedSelector if omitted)."""
    from coverageai.selector import ConstrainedSelector
    sel = selector if selector is not None else ConstrainedSelector()
    xs = list(conditions)
    if not xs:
        raise ValueError("alternative_nozzles needs at least one section")
    n_s = len(xs)
    rows = []
    for fam in catalog.families():
        for size in catalog.sizes(fam):
            if exclude is not None and (fam, size) == (str(exclude[0]), str(exclude[1])):
                continue
            cands = iso_rate_candidates(fam, size, gpa, tol_frac, levels=(1.0,))
            n_c = len(cands)
            if n_c == 0:
                continue
            preds = model.predict_batch([x for x in xs for _ in range(n_c)], cands * n_s)
            cov = np.array([p.coverage_pct for p in preds], float).reshape(n_s, n_c)
            lo = np.array([p.coverage_lo for p in preds], float).reshape(n_s, n_c)
            drift = np.array([p.drift_frac for p in preds], float).reshape(n_s, n_c)
            corrected = cov + bias
            feas = corrected - sel.margin(cov, lo) >= c_req
            ok = feas.any(axis=1)
            pick = np.where(ok, np.where(feas, drift, np.inf).argmin(axis=1),
                            corrected.argmax(axis=1))
            seg = np.arange(n_s)
            ps = np.array([cands[j].pressure_psi for j in pick])
            vs = np.array([cands[j].travel_speed_kmh for j in pick])
            rows.append({"family": fam, "size": size, "share_meeting": float(ok.mean()),
                         "mean_cov": float(corrected[seg, pick].mean()),
                         "mean_drift": float(drift[seg, pick].mean()),
                         "pressure_range_psi": (float(ps.min()), float(ps.max())),
                         "speed_range_kmh": (float(vs.min()), float(vs.max()))})
    rows.sort(key=lambda r: (-r["share_meeting"], r["mean_drift"], r["family"], r["size"]))
    return rows[:max(int(top), 0)]


def with_rate_speed(u: Setting, gpa: float, pressure_psi: float | None = None) -> Setting:
    """``u`` at ``pressure_psi`` (default its own) with the speed that holds ``gpa``."""
    p = u.pressure_psi if pressure_psi is None else float(pressure_psi)
    return dataclasses.replace(u, pressure_psi=p,
                               travel_speed_kmh=speed_for_rate(u.family, u.size, p, gpa))
