"""Mock modules that satisfy the contracts, so every module can be built and
tested before its dependencies land. Behavior is smooth and roughly physical,
not calibrated; never use mocks for reported results.
"""
from __future__ import annotations

import math
import time

import numpy as np

from coverageai import units
from coverageai.config import spec
from coverageai.contracts import (CardMeasurement, Conditions, Derived, Outcome,
                                  Prediction, Setting, coverage_class, effective_boom_height_m)

# family -> size -> (flow_gpm at 40 psi, droplet class at 40 psi)
STUB_CATALOG: dict[str, dict[str, tuple[float, str]]] = {
    "TP65": {"02": (0.20, "F"), "04": (0.40, "M"), "08": (0.80, "M")},
    "XR": {"02": (0.20, "F"), "04": (0.40, "M"), "08": (0.80, "C")},
    "DGX40": {"03": (0.30, "UC"), "04": (0.40, "UC"), "06": (0.60, "UC")},
    "ULD": {"02": (0.20, "VC"), "04": (0.40, "XC"), "08": (0.80, "XC")},
    "3D": {"02": (0.20, "M"), "04": (0.40, "C"), "08": (0.80, "C")},
}


def mock_derive(x: Conditions, u: Setting) -> Derived:
    s = spec()
    q40, dc = STUB_CATALOG[u.family][u.size]
    flow = q40 * math.sqrt(u.pressure_psi / 40.0)
    mph = units.kmh_to_mph(u.travel_speed_kmh if u.travel_speed_kmh is not None else x.travel_speed_kmh)
    gpa = units.gpa(flow, mph, s["boom"]["nozzle_spacing_in"])
    vmd = s["droplets"]["vmd_center_um"][dc] * (u.pressure_psi / 40.0) ** -0.3
    rel = units.wrap_deg(x.wind_dir_deg - x.heading_deg)
    return Derived(
        flow_gpm=flow, app_rate_gpa=gpa, app_rate_lha=units.gpa_to_lha(gpa),
        droplet_class=dc, vmd_um=vmd, spray_angle_deg=s["spray_angle_deg"][u.family],
        rel_wind_deg=rel, u_cross_ms=x.wind_speed_ms * math.sin(math.radians(rel)),
        u_along_ms=x.wind_speed_ms * math.cos(math.radians(rel)),
    )


def mock_candidates() -> list[Setting]:
    out = []
    for fam, sizes in STUB_CATALOG.items():
        lo, hi = spec()["pressure_range_psi"][fam]
        for size in sizes:
            for p in np.linspace(lo, min(hi, 70), 4):
                out.append(Setting(fam, size, float(round(p))))
    return out


def _closed_form(x: Conditions, u: Setting) -> tuple[float, float, float, float]:
    d = mock_derive(x, u)
    h = effective_boom_height_m(x, u)
    drift = 1 - math.exp(-abs(d.u_cross_ms) * h * 60.0 / d.vmd_um)
    drift = min(max(drift, 0.0), 0.95)
    dep = units.gpa_to_ul_cm2(d.app_rate_gpa) * (1 - drift)
    stain_um = 1.9 * d.vmd_um
    n_per_cm2 = dep * 1e9 / (math.pi / 6 * d.vmd_um ** 3)   # uL -> um^3
    cov = 100 * (1 - math.exp(-n_per_cm2 * math.pi / 4 * (stain_um * 1e-4) ** 2))
    disp = d.u_cross_ms * h / max(d.vmd_um * 4e-3, 0.05) * 1000 * 0.01
    return cov, dep, drift, disp


class MockSimulator:
    def simulate(self, x, u, rng, mismatch="none", sample_droplets=False) -> Outcome:
        cov, dep, drift, disp = _closed_form(x, u)
        factor = {"none": 1.0, "low": 0.92, "med": 0.82, "high": 0.7}[mismatch]
        cov = float(np.clip(cov * factor + rng.normal(0, 1.0 if mismatch != "none" else 0.0), 0, 100))
        return Outcome(coverage_pct=cov, deposition_ul_cm2=dep * factor, drift_frac=drift,
                       displacement_mm=disp, stains_per_cm2=cov * 1.5, droplets=None)


class MockModel:
    def predict(self, x, u) -> Prediction:
        cov, dep, drift, disp = _closed_form(x, u)
        return Prediction(cov, cov - 4, cov + 4, dep, drift, disp)

    def predict_batch(self, xs, us):
        return [self.predict(x, u) for x, u in zip(xs, us)]

    def fit(self, records) -> None:
        pass


class MockSelector:
    def select(self, x, model, bias, c_req):
        cands = mock_candidates()
        preds = model.predict_batch([x] * len(cands), cands)
        ok = [(p.drift_frac, mock_derive(x, u).app_rate_gpa, u, p)
              for u, p in zip(cands, preds) if p.coverage_pct + bias - 3.0 >= c_req]
        if ok:
            ok.sort(key=lambda t: (t[0], t[1]))
            return ok[0][2], {"infeasible": False, "candidates": len(cands)}
        best = max(zip(cands, preds), key=lambda t: t[1].coverage_pct)
        return best[0], {"infeasible": True, "candidates": len(cands)}


class MockRenderer:
    """Draws random blue disks on a yellow card until the target coverage is reached."""

    def render(self, outcome, rng, degrade=True):
        s = spec()
        ppm = s["card"]["dpi"] / 25.4
        h, w = int(s["card"]["height_mm"] * ppm), int(s["card"]["width_mm"] * ppm)
        mask = np.zeros((h, w), bool)
        yy, xx = np.ogrid[:h, :w]
        target = outcome.coverage_pct / 100.0
        r = 6
        while mask.mean() < target:
            cy, cx = rng.integers(0, h), rng.integers(0, w)
            y0, y1, x0, x1 = max(cy - r, 0), min(cy + r + 1, h), max(cx - r, 0), min(cx + r + 1, w)
            mask[y0:y1, x0:x1] |= (yy[y0:y1] - cy) ** 2 + (xx[:, x0:x1] - cx) ** 2 <= r * r
        img = np.empty((h, w, 3), np.uint8)
        img[:] = (235, 210, 40)
        img[mask] = (30, 40, 120)
        return img, mask


class MockVerifier:
    def measure(self, image, px_per_mm) -> CardMeasurement:
        t0 = time.perf_counter()
        stained = image[..., 2].astype(int) > image[..., 0].astype(int)
        cov = 100.0 * float(stained.mean())
        return CardMeasurement(cov, None, None, None, None, coverage_class(cov),
                               cov > 60, time.perf_counter() - t0)
