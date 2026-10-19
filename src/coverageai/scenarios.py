"""Field-pass scenario library (WP-12).

A scenario is a sequence of K segment conditions ``x`` (``contracts.Conditions``)
that mimics one pass of a ground boom sprayer: AR(1) wind with occasional gust
bursts and a slowly drifting direction, linear temperature and humidity trends,
back-and-forth headings by lap, and small machine noise. Everything is generated
from a named, documented ``ScenarioSpec`` and an integer seed, so passes are
reproducible. All values are clipped to the operating ranges in
``config/spec.yaml``; wind may exceed the label limit only when the spec sets
``allow_exceed_limit`` (the marginal preset).

Wind direction follows decision D1: degrees the airflow moves toward.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from .config import spec as _spec
from .contracts import Conditions
from .units import wrap_deg

# Per-segment random-walk noise on wind direction, deg (turbulent meander).
_DIR_NOISE_DEG = 3.0
# Temperature noise (AR(1)), deg C; RH responds with this slope plus own noise.
_T_NOISE_C = 0.3
_T_NOISE_PHI = 0.8
_RH_PER_C = -2.5          # % RH per deg C of short-term temperature excursion
_RH_NOISE_PCT = 1.0
_NOISE_PHI = 0.5          # persistence of speed and boom-height noise


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    description: str
    rationale: str
    segments: int
    base_wind_ms: float
    wind_ar1_phi: float
    wind_sigma: float                 # stationary std of wind speed around base, m/s
    gust_prob: float                  # per-segment probability a gust burst starts
    gust_mult: float                  # wind multiplier during a burst
    wind_dir_deg: float               # initial direction the wind moves toward
    wind_dir_drift_deg_per_seg: float
    temp_start_c: float
    temp_end_c: float
    rh_start_pct: float
    rh_end_pct: float
    speed_kmh: float
    speed_sigma: float
    boom_height_m: float
    boom_sigma: float
    headings: tuple[float, ...] = (0.0, 180.0)
    segments_per_lap: int = 10
    allow_exceed_limit: bool = False  # True lets wind use the full range up to 7 m/s


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    """Stationary AR(1) series with marginal std sigma."""
    x = np.empty(n)
    x[0] = rng.normal(0.0, sigma)
    innov = sigma * np.sqrt(max(1.0 - phi * phi, 0.0))
    for k in range(1, n):
        x[k] = phi * x[k - 1] + rng.normal(0.0, innov)
    return x


def _gust_multiplier(rng: np.random.Generator, n: int, prob: float, mult: float) -> np.ndarray:
    """Per-segment wind multiplier; bursts last 1 to 3 segments."""
    m = np.ones(n)
    k = 0
    while k < n:
        if rng.random() < prob:
            length = int(rng.integers(1, 4))
            m[k:k + length] = mult * rng.uniform(0.9, 1.1)
            k += length
        else:
            k += 1
    return m


def generate(spec: ScenarioSpec, seed: int) -> list[Conditions]:
    """Return ``spec.segments`` conditions for one pass; deterministic given seed."""
    r = _spec()["ranges"]
    n = spec.segments
    rng = np.random.default_rng(seed)

    # wind speed: AR(1) around the base, then gust bursts, then clip
    wind = spec.base_wind_ms + _ar1(rng, n, spec.wind_ar1_phi, spec.wind_sigma)
    wind = wind * _gust_multiplier(rng, n, spec.gust_prob, spec.gust_mult)
    w_hi = r["wind_speed_ms"][1] if spec.allow_exceed_limit else min(
        r["wind_speed_ms"][1], _spec()["wind_limit_ms"])
    wind = np.clip(wind, r["wind_speed_ms"][0], w_hi)

    # wind direction: linear drift plus random-walk meander
    meander = np.cumsum(rng.normal(0.0, _DIR_NOISE_DEG, n)) * 0.5
    wdir = spec.wind_dir_deg + spec.wind_dir_drift_deg_per_seg * np.arange(n) + meander

    # temperature and humidity: linear trends, RH falls when T is above trend
    frac = np.arange(n) / max(n - 1, 1)
    t_noise = _ar1(rng, n, _T_NOISE_PHI, _T_NOISE_C)
    temp = spec.temp_start_c + (spec.temp_end_c - spec.temp_start_c) * frac + t_noise
    rh = (spec.rh_start_pct + (spec.rh_end_pct - spec.rh_start_pct) * frac
          + _RH_PER_C * t_noise + rng.normal(0.0, _RH_NOISE_PCT, n))
    temp = np.clip(temp, *r["temp_c"])
    rh = np.clip(rh, *r["rh_pct"])

    # machine state
    speed = np.clip(spec.speed_kmh + _ar1(rng, n, _NOISE_PHI, spec.speed_sigma),
                    *r["travel_speed_kmh"])
    boom = np.clip(spec.boom_height_m + _ar1(rng, n, _NOISE_PHI, spec.boom_sigma),
                   *r["boom_height_m"])
    lap = np.arange(n) // spec.segments_per_lap
    heading = np.array([spec.headings[i % len(spec.headings)] for i in lap], dtype=float)

    return [
        Conditions(
            wind_speed_ms=float(wind[k]),
            wind_dir_deg=float(wdir[k] % 360.0),
            heading_deg=float(heading[k] % 360.0),
            temp_c=float(temp[k]),
            rh_pct=float(rh[k]),
            travel_speed_kmh=float(speed[k]),
            boom_height_m=float(boom[k]),
        )
        for k in range(n)
    ]


def crosswind_ms(x: Conditions) -> float:
    """Signed crosswind component U*sin(wind_dir - heading), m/s."""
    return x.wind_speed_ms * float(np.sin(np.radians(wrap_deg(x.wind_dir_deg - x.heading_deg))))


# --------------------------------------------------------------------- presets

_SEGMENTS = int(_spec()["pass"]["segments"])


def _preset(**kw) -> ScenarioSpec:
    return ScenarioSpec(segments=_SEGMENTS, **kw)


PRESETS: dict[str, ScenarioSpec] = {s.name: s for s in [
    _preset(
        name="calm_morning",
        description="Light, steady wind; cool and humid; slow warming.",
        rationale=(
            "Early-morning passes usually have the lowest and steadiest wind of the day, so "
            "horizontal displacement is small and most droplets reach the target. Cool, humid air "
            "slows evaporation, which lets even fine droplets keep their size during the fall "
            "(UNL G1773). This is the benign reference: coverage is limited by application "
            "volume and droplet spectrum rather than by weather, so it checks that the selector "
            "does not over-react when conditions are easy and can choose low-drift settings "
            "at modest volume."),
        base_wind_ms=1.2, wind_ar1_phi=0.8, wind_sigma=0.4, gust_prob=0.02, gust_mult=1.5,
        wind_dir_deg=45.0, wind_dir_drift_deg_per_seg=0.3,
        temp_start_c=14.0, temp_end_c=19.0, rh_start_pct=85.0, rh_end_pct=72.0,
        speed_kmh=14.0, speed_sigma=0.8, boom_height_m=0.50, boom_sigma=0.02),
    _preset(
        name="gusty_afternoon",
        description="Moderate mean wind with frequent gust bursts; warm, moderately dry.",
        rationale=(
            "Afternoon heating produces turbulent, intermittent wind. The mean can sit well "
            "below the label limit while individual gusts approach it, so a setting chosen from "
            "the mean wind under-predicts displacement during bursts. Because drift distance "
            "scales with crosswind and fall time, fine droplets from a high boom are the most "
            "exposed (EPA mitigation menu; Nuyttens et al. on boom height). The scenario stresses "
            "the feedback loop: card residuals jump when a gust hits and the correction must "
            "react without chasing noise."),
        base_wind_ms=4.0, wind_ar1_phi=0.6, wind_sigma=1.4, gust_prob=0.12, gust_mult=1.6,
        wind_dir_deg=120.0, wind_dir_drift_deg_per_seg=0.5,
        temp_start_c=27.0, temp_end_c=31.0, rh_start_pct=45.0, rh_end_pct=38.0,
        speed_kmh=16.0, speed_sigma=1.0, boom_height_m=0.55, boom_sigma=0.03),
    _preset(
        name="hot_dry",
        description="Moderate wind, high temperature, low and falling humidity.",
        rationale=(
            "High temperature and low relative humidity raise the wet-bulb depression, so water "
            "droplets shrink faster on the way down. UNL G1773 notes that hot, dry air speeds "
            "evaporation, increases drift potential and can reduce deposition on the target. "
            "Droplets below roughly 200 um are the most sensitive, and the smallest can "
            "evaporate completely. Coverage therefore falls even when wind is moderate, which "
            "favors coarser spectra or higher volume. This preset isolates the "
            "temperature-humidity effect on the model and selector."),
        base_wind_ms=2.5, wind_ar1_phi=0.7, wind_sigma=0.7, gust_prob=0.04, gust_mult=1.4,
        wind_dir_deg=90.0, wind_dir_drift_deg_per_seg=0.4,
        temp_start_c=33.0, temp_end_c=38.0, rh_start_pct=28.0, rh_end_pct=16.0,
        speed_kmh=16.0, speed_sigma=0.8, boom_height_m=0.60, boom_sigma=0.03),
    _preset(
        name="crosswind_shift",
        description="Moderate wind whose direction rotates about 140 degrees over the pass.",
        rationale=(
            "With a back-and-forth pattern the crosswind component U*sin(wind direction minus "
            "heading) changes sign between laps, so a wind that pushes spray to one side on "
            "one lap pushes it to the other on the next. A slowly rotating wind also moves "
            "between crosswind and along-track (head or tail wind) geometry, which changes "
            "displacement without any change in wind speed. EPA guidance stresses wind "
            "direction relative to the application path. The scenario checks that models and "
            "selectors use the relative angle rather than wind speed alone."),
        base_wind_ms=3.0, wind_ar1_phi=0.7, wind_sigma=0.6, gust_prob=0.04, gust_mult=1.4,
        wind_dir_deg=80.0, wind_dir_drift_deg_per_seg=3.5,
        temp_start_c=20.0, temp_end_c=24.0, rh_start_pct=60.0, rh_end_pct=52.0,
        speed_kmh=15.0, speed_sigma=0.8, boom_height_m=0.50, boom_sigma=0.02),
    _preset(
        name="marginal_limits",
        description="High mean wind near the label limit with gusts that exceed it.",
        rationale=(
            "EPA guidance and product labels commonly restrict application at wind speeds in "
            "the 10-15 mph range (4.5-6.7 m/s), and the proposed modeling range ends at 7 m/s. "
            "This preset sits at that edge, with a mean of about 6 m/s and gust bursts that "
            "cross 6.7 m/s, to test behavior where drift risk is highest and a real operator "
            "would normally stop spraying. Feasible settings may not exist, so it exercises "
            "the selector's infeasible path and the retreat flag. It is the only preset "
            "allowed to exceed the limit."),
        base_wind_ms=6.0, wind_ar1_phi=0.6, wind_sigma=0.8, gust_prob=0.12, gust_mult=1.25,
        wind_dir_deg=90.0, wind_dir_drift_deg_per_seg=0.5,
        temp_start_c=24.0, temp_end_c=28.0, rh_start_pct=50.0, rh_end_pct=44.0,
        speed_kmh=16.0, speed_sigma=1.0, boom_height_m=0.65, boom_sigma=0.03,
        allow_exceed_limit=True),
    _preset(
        name="inversion_evening",
        description="Near-calm evening air, cooling and humidifying.",
        rationale=(
            "Near sunset the ground cools faster than the air above it and wind often drops "
            "toward zero, which favors a surface temperature inversion. Under an inversion, "
            "fine droplets can stay suspended in a stable layer and move off target in light, "
            "variable air even though wind speed is low; product labels commonly warn against "
            "spraying under inversions (EPA label statements). Cooling and rising humidity "
            "slow evaporation, so coverage is usually good. The challenge is that low "
            "measured wind makes a wind-based drift predictor look safe, so this is a "
            "limitation test rather than a regime the simulator models directly."),
        base_wind_ms=0.6, wind_ar1_phi=0.8, wind_sigma=0.3, gust_prob=0.01, gust_mult=1.5,
        wind_dir_deg=200.0, wind_dir_drift_deg_per_seg=0.2,
        temp_start_c=22.0, temp_end_c=15.0, rh_start_pct=55.0, rh_end_pct=80.0,
        speed_kmh=14.0, speed_sigma=0.8, boom_height_m=0.50, boom_sigma=0.02),
]}


def list_presets() -> list[str]:
    return list(PRESETS)


def get(name: str) -> ScenarioSpec:
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown scenario {name!r}; available: {list_presets()}") from None


# ------------------------------------------------------------ random conditions

_RANGE_KEYS = ("wind_speed_ms", "wind_dir_deg", "heading_deg", "temp_c", "rh_pct",
               "travel_speed_kmh", "boom_height_m")


def random_conditions(n: int, seed: int) -> list[Conditions]:
    """Independent Latin-hypercube samples over the full operating ranges."""
    r = _spec()["ranges"]
    u = qmc.LatinHypercube(d=len(_RANGE_KEYS), seed=seed).random(n)
    lo = np.array([r[k][0] for k in _RANGE_KEYS])
    hi = np.array([r[k][1] for k in _RANGE_KEYS])
    v = lo + u * (hi - lo)
    return [Conditions(*(float(a) for a in row)) for row in v]
