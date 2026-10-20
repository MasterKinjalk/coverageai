"""Nozzle catalog: manufacturer chart lookups and the derived quantities (WP-01).

Data come from ``data/catalog/nozzles.csv`` (transcribed manufacturer charts).
Flow follows Q = k * sqrt(P) between chart rows: the coefficient k = Q / sqrt(P)
is interpolated linearly in pressure between the two bracketing chart rows, so
chart rows are reproduced exactly and flow is continuous in P. Droplet class is
taken from the nearest chart pressure row (classes are a step function of P).
"""
from __future__ import annotations

import dataclasses
import math
from functools import lru_cache

import numpy as np
import pandas as pd

from coverageai import units
from coverageai.config import ROOT, spec
from coverageai.contracts import Conditions, Derived, Setting, effective_speed_kmh

CATALOG_PATH = ROOT / "data" / "catalog" / "nozzles.csv"
_EPS = 1e-9
_P_REF_PSI = 40.0     # reference pressure for the within-class VMD trend


@lru_cache(maxsize=1)
def load() -> pd.DataFrame:
    """The full catalog (one row per chart pressure row). Do not mutate."""
    df = pd.read_csv(CATALOG_PATH, dtype={"family": str, "size": str, "droplet_class": str})
    df["notes"] = df["notes"].fillna("")
    return df.sort_values(["family", "size", "pressure_psi"], kind="stable").reset_index(drop=True)


def _size_key(size: str) -> float:
    # Size codes read as flow in 0.1 gpm units at 40 psi: "015" < "02" < "025" < ... < "10" < "15".
    return float("0." + size)


@lru_cache(maxsize=1)
def _index() -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, tuple[str, ...]]]:
    idx = {}
    for (fam, size), g in load().groupby(["family", "size"], sort=False):
        g = g.sort_values("pressure_psi")
        idx[(fam, size)] = (g["pressure_psi"].to_numpy(float), g["flow_gpm"].to_numpy(float),
                            tuple(g["droplet_class"]))
    return idx


def _rows(family: str, size: str):
    try:
        return _index()[(family, size)]
    except KeyError:
        raise ValueError(f"unknown nozzle {family}{size}") from None


def families() -> list[str]:
    order = spec()["nozzle_families"]
    present = set(load()["family"])
    return [f for f in order if f in present]


def sizes(family: str) -> list[str]:
    s = {sz for (f, sz) in _index() if f == family}
    if not s:
        raise ValueError(f"unknown family {family}")
    return sorted(s, key=_size_key)


def pressure_range(family: str, size: str) -> tuple[float, float]:
    p, _, _ = _rows(family, size)
    return float(p[0]), float(p[-1])


def flow_and_class(family: str, size: str, pressure_psi: float) -> tuple[float, str]:
    """(flow_gpm, droplet_class) for one nozzle at ``pressure_psi``.

    Raises ValueError if the pressure is outside the chart range for that size.
    """
    p, q, cls = _rows(family, size)
    if not (p[0] - _EPS <= pressure_psi <= p[-1] + _EPS):
        raise ValueError(f"{family}{size}: {pressure_psi} psi outside chart range "
                         f"[{p[0]:g}, {p[-1]:g}] psi")
    k = q / np.sqrt(p)
    flow = float(np.interp(pressure_psi, p, k)) * math.sqrt(pressure_psi)
    i = int(np.argmin(np.abs(p - pressure_psi)))   # ties -> lower pressure (coarser class)
    return flow, cls[i]


def vmd_um(droplet_class: str, pressure_psi: float, family: str | None = None) -> float:
    """Volume median diameter: class center * (P/40)^exponent, clipped to class bounds.

    ``family`` is accepted for interface stability; v1 uses one class-based law for all families.
    """
    d = spec()["droplets"]
    center = d["vmd_center_um"][droplet_class]
    lo, hi = d["vmd_bounds_um"][droplet_class]
    v = center * (pressure_psi / _P_REF_PSI) ** d["pressure_exponent"]
    return float(min(max(v, lo), hi))


def derive(x: Conditions, u: Setting) -> Derived:
    """Catalog lookups and wind geometry for one (conditions, setting) pair."""
    s = spec()
    flow, dc = flow_and_class(u.family, u.size, u.pressure_psi)
    mph = units.kmh_to_mph(effective_speed_kmh(x, u))
    gpa = units.gpa(flow, mph, s["boom"]["nozzle_spacing_in"])
    rel = units.wrap_deg(x.wind_dir_deg - x.heading_deg)
    r = math.radians(rel)
    return Derived(
        flow_gpm=flow, app_rate_gpa=gpa, app_rate_lha=units.gpa_to_lha(gpa),
        droplet_class=dc, vmd_um=vmd_um(dc, u.pressure_psi, u.family),
        spray_angle_deg=float(s["spray_angle_deg"][u.family]),
        rel_wind_deg=rel, u_cross_ms=x.wind_speed_ms * math.sin(r),
        u_along_ms=x.wind_speed_ms * math.cos(r),
    )


def speed_grid(current_kmh: float, options_kmh=None, band_frac: float | None = None) -> list[float]:
    """Admissible controllable travel speeds for a segment currently at ``current_kmh``.

    The option grid (spec selector.speed_options_kmh) is clipped to the spec travel-speed
    range and to +-band_frac around the current speed (an operator can adjust speed only
    within limits). The current speed itself is not listed: it is the implicit
    ``travel_speed_kmh=None`` candidate. Empty options -> [] (speed control off).
    """
    s = spec()
    opts = s["selector"].get("speed_options_kmh", []) if options_kmh is None else options_kmh
    if not opts:
        return []
    band = float(s["selector"].get("speed_band_frac", 0.25) if band_frac is None else band_frac)
    lo, hi = s["ranges"]["travel_speed_kmh"]
    lo, hi = max(float(lo), current_kmh * (1 - band)), min(float(hi), current_kmh * (1 + band))
    return [float(v) for v in sorted(opts)
            if lo - _EPS <= v <= hi + _EPS and abs(v - current_kmh) > _EPS]


def with_speeds(base: list[Setting], current_kmh: float, options_kmh=None,
                band_frac: float | None = None) -> list[Setting]:
    """``base`` (speed = current) plus every base setting at each admissible speed."""
    sp = speed_grid(current_kmh, options_kmh, band_frac)
    return list(base) + [dataclasses.replace(u, travel_speed_kmh=v) for v in sp for u in base]


def min_boom_height_m(family: str, overlap_factor: float | None = None) -> float:
    """Minimum release height for adequate fan overlap between adjacent nozzles.

    Rule used (standard flat-fan boom setup guidance, e.g. Spraying Systems TeeJet catalog /
    ASABE-extension practice): a fan of included angle theta at height H spans 2*H*tan(theta/2)
    on the target; with nozzle spacing S, single coverage (no overlap) needs
    H0 = S / (2 tan(theta/2)) and a ~30 % overlap (overlap_factor = 1.3, the usual minimum for
    uniform distribution; 100 % overlap would be 2.0) needs H = factor * H0. Lower than this gives
    striping / poor uniformity, which the (uniform-field) simulator does not model, so the selector
    enforces it as a constraint. The spec range ``ranges.boom_height_m`` lower bound still applies.
    """
    s = spec()
    f = float(s["selector"].get("boom_overlap_factor", 1.3) if overlap_factor is None else overlap_factor)
    spacing_m = s["boom"]["nozzle_spacing_in"] * 0.0254
    theta = math.radians(float(s["spray_angle_deg"][family]))
    return f * spacing_m / (2.0 * math.tan(theta / 2.0))


def boom_height_grid(current_m: float, family: str, options_m=None, band_frac: float | None = None,
                     overlap_factor: float | None = None) -> list[float]:
    """Admissible controllable release heights for a segment currently at ``current_m``.

    Options (spec selector.boom_height_options_m) are clipped to the spec boom range, to
    +-band_frac around the current height (automatic boom control can change height within
    seconds but only in a band around the canopy-following set point) and to the family's
    minimum overlap height. The current height is the implicit ``boom_height_m=None``
    candidate and is not listed. Empty options -> [] (height control off).
    """
    s = spec()
    opts = s["selector"].get("boom_height_options_m", []) if options_m is None else options_m
    if not opts:
        return []
    band = float(s["selector"].get("boom_height_band_frac", 0.25) if band_frac is None else band_frac)
    lo, hi = s["ranges"]["boom_height_m"]
    lo = max(lo, min_boom_height_m(family, overlap_factor))
    return [float(v) for v in opts
            if lo - 1e-9 <= v <= hi + 1e-9 and abs(v - current_m) <= band * current_m + 1e-9
            and abs(v - current_m) > 1e-9]


def with_boom_heights(base: list[Setting], current_m: float, options_m=None,
                      band_frac: float | None = None, overlap_factor: float | None = None) -> list[Setting]:
    """``base`` (height = current) plus every base setting at each admissible height for its family."""
    out = list(base)
    grids: dict[str, list[float]] = {}
    for u in base:
        g = grids.get(u.family)
        if g is None:
            g = grids[u.family] = boom_height_grid(current_m, u.family, options_m, band_frac, overlap_factor)
        out += [dataclasses.replace(u, boom_height_m=h) for h in g]
    return out


def candidates(pressure_step_psi: float = 5.0) -> list[Setting]:
    """Every family/size on a pressure grid spanning the chart range.

    Grid: chart minimum, then +step, ... up to the chart maximum (always included),
    intersected with the family operating range in spec.yaml.
    """
    if pressure_step_psi <= 0:
        raise ValueError("pressure_step_psi must be positive")
    rng = spec()["pressure_range_psi"]
    out: list[Setting] = []
    for fam in families():
        for size in sizes(fam):
            cmin, cmax = pressure_range(fam, size)
            lo, hi = max(cmin, float(rng[fam][0])), min(cmax, float(rng[fam][1]))
            if lo > hi + _EPS:
                continue
            n = int(math.floor((hi - lo) / pressure_step_psi + _EPS))
            ps = [lo + i * pressure_step_psi for i in range(n + 1)]
            if hi - ps[-1] > _EPS:
                ps.append(hi)
            out.extend(Setting(fam, size, float(round(p, 6))) for p in ps)
    return out
