"""Unit conversions used across the project."""
from __future__ import annotations

MPH_PER_KMH = 0.621371
L_PER_GAL = 3.785411784
M_PER_IN = 0.0254
GPA_TO_LHA = 9.353958        # 1 US gal/acre in L/ha
LHA_TO_UL_CM2 = 0.01         # 1 L/ha = 1e6 uL / 1e8 cm^2


def kmh_to_mph(v: float) -> float:
    return v * MPH_PER_KMH


def mph_to_kmh(v: float) -> float:
    return v / MPH_PER_KMH


def kmh_to_ms(v: float) -> float:
    return v / 3.6


def gpm_to_lmin(q: float) -> float:
    return q * L_PER_GAL


def lmin_to_gpm(q: float) -> float:
    return q / L_PER_GAL


def gpa_to_lha(r: float) -> float:
    return r * GPA_TO_LHA


def lha_to_gpa(r: float) -> float:
    return r / GPA_TO_LHA


def gpa_to_ul_cm2(r: float) -> float:
    return gpa_to_lha(r) * LHA_TO_UL_CM2


def gpa(flow_gpm: float, speed_mph: float, spacing_in: float = 20.0) -> float:
    """Boom-sprayer calibration: GPA = 5940 * GPM / (MPH * W)."""
    return 5940.0 * flow_gpm / (speed_mph * spacing_in)


def lha(flow_lmin: float, speed_kmh: float, width_m: float) -> float:
    """ASABE form: L/ha = 600 * q / (w * s)."""
    return 600.0 * flow_lmin / (width_m * speed_kmh)


def wrap_deg(a: float) -> float:
    """Wrap an angle to (-180, 180]."""
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a
