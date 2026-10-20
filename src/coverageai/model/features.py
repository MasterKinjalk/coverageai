"""Feature builder for the predictive model (WP-10).

Inputs are plain arrays so a batch of ~550 candidates for one set of conditions is
built in well under 10 ms:

  * ``condition_array(xs)``  -> (n, 7) float array of Conditions fields
  * ``setting_array(us)``    -> (n, 6) float array of static per-setting properties
    [flow_gpm, droplet-class ordinal, vmd_um, spray_angle_deg, family index, pressure_psi],
    looked up through ``catalog`` once per distinct Setting and cached
  * ``build_features(X, S)`` -> (n, n_features) matrix (vectorized)

Wind enters as cross/along components, |cross| and sin/cos of the relative angle, never
as raw degrees (a circular variable). Physics features: wet-bulb depression, fall time
H / V_t(VMD), the first-order drift index |U_cross| H / V_t, an evaporation index
lambda H / (V_t d^2) and the coverage proxy GPA / VMD (coverage ~ volume / diameter).
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np

from coverageai import catalog, units
from coverageai.config import spec
from coverageai.contracts import Conditions, Setting
from coverageai.physics.droplets import terminal_velocity
from coverageai.physics.evaporation import wet_bulb_depression

FEATURE_VERSION = 1
X_FIELDS = tuple(f.name for f in dataclasses.fields(Conditions))
FAMILIES = tuple(spec()["nozzle_families"])
CLASS_ORDER = tuple(spec()["droplets"]["class_order"])
_CLASS_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
_FAM_IDX = {f: i for i, f in enumerate(FAMILIES)}
_SPACING_IN = float(spec()["boom"]["nozzle_spacing_in"])
_LAMBDA = float(spec()["physics"]["evaporation_lambda_coeff"])

FEATURE_NAMES = (
    ["wind_speed", "u_cross", "u_along", "abs_u_cross", "sin_rel", "cos_rel",
     "boom_height", "temp_c", "rh_pct", "wb_depression", "speed_kmh",
     "flow_gpm", "gpa", "class_ord", "vmd_um", "spray_angle", "pressure_psi"]
    + [f"fam_{f}" for f in FAMILIES]
    + ["vt_vmd", "fall_time", "drift_index", "wind_drift_index", "evap_index",
       "gpa_over_vmd", "log_gpa"]
)
N_FEATURES = len(FEATURE_NAMES)
_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def feature_index(name: str) -> int:
    return _IDX[name]


# ------------------------------------------------------------------ setting cache

_SETTING_CACHE: dict[Setting, tuple[float, ...]] = {}


def _lookup_setting(u: Setting) -> tuple[float, ...]:
    row = _SETTING_CACHE.get(u)
    if row is None:
        flow, cls = catalog.flow_and_class(u.family, u.size, u.pressure_psi)
        row = (flow, float(_CLASS_IDX[cls]), catalog.vmd_um(cls, u.pressure_psi, u.family),
               float(spec()["spray_angle_deg"][u.family]), float(_FAM_IDX[u.family]),
               float(u.pressure_psi))
        _SETTING_CACHE[u] = row
    return row


def clear_cache() -> None:
    """Drop cached catalog lookups (call after the catalog or spec changes)."""
    _SETTING_CACHE.clear()


def setting_array(us: Sequence[Setting]) -> np.ndarray:
    get = _SETTING_CACHE.get
    rows = []
    for u in us:
        r = get(u)
        rows.append(r if r is not None else _lookup_setting(u))
    return np.array(rows, dtype=float).reshape(len(rows), 6)


def condition_array(xs: Sequence[Conditions]) -> np.ndarray:
    """(n, 7) array; a list repeating one object (selector use) is built from a single row."""
    n = len(xs)
    if n == 0:
        return np.empty((0, 7))
    x0 = xs[0]
    if n > 1 and all(x is x0 for x in xs):
        return np.tile(np.array(dataclasses.astuple(x0), dtype=float), (n, 1))
    return np.array([dataclasses.astuple(x) for x in xs], dtype=float)


# ------------------------------------------------------------------ builder

def build_features(X: np.ndarray, S: np.ndarray) -> np.ndarray:
    """X: (n, 7) conditions in Conditions field order; S: (n, 6) from ``setting_array``."""
    n = X.shape[0]
    u, wdir, head, temp, rh, speed, h = (X[:, i] for i in range(7))
    flow, cls, vmd, ang, fam, press = (S[:, i] for i in range(6))
    rel = np.radians(wdir - head)
    s_rel, c_rel = np.sin(rel), np.cos(rel)
    u_cross, u_along = u * s_rel, u * c_rel
    dtwb = wet_bulb_depression(temp, rh)
    gpa = units.gpa(flow, units.kmh_to_mph(speed), _SPACING_IN)
    vt = terminal_velocity(vmd)
    fall = h / vt
    F = np.empty((n, N_FEATURES))
    cols = [u, u_cross, u_along, np.abs(u_cross), s_rel, c_rel, h, temp, rh, dtwb, speed,
            flow, gpa, cls, vmd, ang, press]
    for i, c in enumerate(cols):
        F[:, i] = c
    k = len(cols)
    for j in range(len(FAMILIES)):
        F[:, k + j] = (fam == j)
    k += len(FAMILIES)
    F[:, k] = vt
    F[:, k + 1] = fall
    F[:, k + 2] = np.abs(u_cross) * fall
    F[:, k + 3] = u * fall
    F[:, k + 4] = _LAMBDA * dtwb * fall / (vmd ** 2)
    F[:, k + 5] = gpa / vmd
    F[:, k + 6] = np.log(gpa)
    return F


def featurize(xs: Sequence[Conditions], us: Sequence[Setting]) -> np.ndarray:
    if len(xs) != len(us):
        raise ValueError("xs and us must have equal length")
    X = condition_array(xs)
    sp = [u.travel_speed_kmh for u in us]
    if any(s is not None for s in sp):   # setting-controlled speed overrides the condition's
        X = X.copy()
        X[:, 5] = [x_s if s is None else s for x_s, s in zip(X[:, 5], sp)]
    hs = [u.boom_height_m for u in us]
    if any(h is not None for h in hs):   # setting-controlled boom height overrides the condition's
        X = X.copy()
        X[:, 6] = [x_h if h is None else h for x_h, h in zip(X[:, 6], hs)]
    return build_features(X, setting_array(us))


def featurize_frame(df) -> np.ndarray:
    """Features from a dataset frame (columns of Conditions and Setting)."""
    X = df[list(X_FIELDS)].to_numpy(float)
    keys = df[["family", "size", "pressure_psi"]].drop_duplicates()
    table = {(r.family, r.size, r.pressure_psi): _lookup_setting(Setting(r.family, r.size, float(r.pressure_psi)))
             for r in keys.itertuples(index=False)}
    S = np.array([table[k] for k in zip(df["family"], df["size"], df["pressure_psi"])], dtype=float)
    return build_features(X, S)
