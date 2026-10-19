"""Nozzle catalog: manufacturer chart lookups and the derived quantities (WP-01).

Data come from ``data/catalog/nozzles.csv`` (transcribed manufacturer charts).
Flow follows Q = k * sqrt(P) between chart rows: the coefficient k = Q / sqrt(P)
is interpolated linearly in pressure between the two bracketing chart rows, so
chart rows are reproduced exactly and flow is continuous in P. Droplet class is
taken from the nearest chart pressure row (classes are a step function of P).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd

from coverageai import units
from coverageai.config import ROOT, spec
from coverageai.contracts import Conditions, Derived, Setting

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
    mph = units.kmh_to_mph(x.travel_speed_kmh)
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
