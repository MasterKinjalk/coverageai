"""Physics-informed spray simulator (WP-02): (x, u, hidden mismatch) -> Outcome.

Pipeline per segment (first-order, fit to published trends; NOT a validated drift model):
  1. Derived (flow, application rate, VMD, wind components) from ``derive_fn``.
  2. Hidden mismatch applied: gust factor on wind, canopy interception, nozzle wear.
  3. Log-normal volume spectrum around VMD, 40 equal-ln(d) bins (physics/droplets.py).
  4. Fall through boom height H with d^2-law evaporation (physics/evaporation.py);
     displacement x = U t_fall; drift = out-of-range volume + evaporated volume
     (physics/drift.py).
  5. Deposition D = GPA * wear * (1 - drift_frac) * canopy_eta, split by bin.
  6. Boolean-model coverage with spread factor SF ~ U(spread_factor) (physics/coverage.py).

Mismatch. ``mismatch`` in {none, low, med, high} selects a row of spec.mismatch.levels.
Hidden factors: gust multiplier on wind g = max(0, 1 + sigma e), canopy interception
eta ~ U(canopy_eta), nozzle wear flow * (1 + U(wear_flow)) and VMD * (1 + U(wear_vmd)).

  * Per-segment draws (RNG contract). ``simulate`` consumes ``rng`` in a fixed order that
    does not depend on x or u: gust normal (only if mismatch != none), spread factor SF,
    then (only without ``begin_pass``) canopy and wear. So an oracle that calls
    ``simulate(x, u', default_rng([seed, k]), mismatch)`` for many candidates u' sees the
    same hidden realization for every u'. Droplet sampling (``sample_droplets=True``) draws
    after the analytic outcome is computed, so the analytic outcome does not depend on it.
  * Pass-level state. ``begin_pass(seed, mismatch)`` draws canopy, flow wear and VMD wear
    once per pass from default_rng([seed, 999]) and makes the gust an AR(1) process
    e_k = phi e_{k-1} + sqrt(1 - phi^2) eps_k (eps_k = the segment's first rng normal;
    stationary start). The AR state advances once per distinct eps, so repeated calls with
    the same per-segment rng (oracle candidates, then the real outcome) see the same gust.
  * Without ``begin_pass`` (or for another mismatch level) ``simulate`` is pure: iid gust,
    canopy and wear from ``rng``. ``simulate_batch`` is always iid per row.

Card frame of sampled droplets: x_mm is cross-track (76 mm side), y_mm along-track (26 mm).
Sampled stain centers may lie up to one stain radius outside [0, W] x [0, H] so that
stains straddling the edge are not lost; renderers must clip.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from coverageai.config import spec
from coverageai.contracts import (Conditions, Derived, DropletSample, MismatchLevel, Outcome,
                                  Setting)
from coverageai.physics import coverage as cov_mod
from coverageai.physics import drift as drift_mod
from coverageai.physics.droplets import lognormal_volume_bins, sigma_from_span
from coverageai.physics.evaporation import evaporation_lambda, fall_with_evaporation


def _default_derive() -> Callable[[Conditions, Setting], Derived]:
    try:
        from coverageai.catalog import derive  # WP-01
        return derive
    except ImportError:
        from coverageai.mocks import mock_derive
        return mock_derive


@dataclass(frozen=True)
class HiddenState:
    """Realization of the hidden mismatch factors for one segment."""
    gust_mult: float = 1.0       # multiplies wind speed
    canopy_eta: float = 1.0      # fraction of non-drifted volume reaching the card
    flow_factor: float = 1.0     # 1 + wear_flow
    vmd_factor: float = 1.0      # 1 + wear_vmd


NO_MISMATCH = HiddenState()


class SpraySimulator:
    """Implements ``contracts.Simulator`` plus ``simulate_batch``."""

    def __init__(self, derive_fn: Callable[[Conditions, Setting], Derived] | None = None,
                 n_bins: int = 40, physics_overrides: dict | None = None):
        self.derive_fn = derive_fn or _default_derive()
        s = spec()
        self.p = dict(s["physics"])
        if physics_overrides:
            self.p.update(physics_overrides)
        self.levels = s["mismatch"]["levels"]
        self.phi = float(s["mismatch"]["gust_ar1_phi"])
        self.sigma_ln = sigma_from_span(s["droplets"]["relative_span"])
        self.n_bins = n_bins
        self.card_w = float(s["card"]["width_mm"])
        self.card_h = float(s["card"]["height_mm"])
        self._pass_level: str | None = None
        self._pass_factors = (1.0, 1.0, 1.0)          # canopy_eta, flow_factor, vmd_factor
        self._e_cur: float | None = None              # AR(1) gust state after the last segment
        self._e_prev: float | None = None
        self._last_eps: float | None = None

    # ------------------------------------------------------------ mismatch
    def begin_pass(self, seed: int, mismatch: MismatchLevel = "none") -> None:
        """Start a field pass: draw canopy/wear once from default_rng([seed, 999]) and
        enable AR(1) gusts. Replaces any earlier pass state."""
        self._e_cur = self._e_prev = self._last_eps = None
        if mismatch == "none":
            self._pass_level, self._pass_factors = "none", (1.0, 1.0, 1.0)
            return
        lv = self.levels[mismatch]
        r = np.random.default_rng([int(seed), 999])
        self._pass_level = mismatch
        self._pass_factors = (float(r.uniform(*lv["canopy_eta"])),
                              1.0 + float(r.uniform(*lv["wear_flow"])),
                              1.0 + float(r.uniform(*lv["wear_vmd"])))

    def end_pass(self) -> None:
        """Back to pure per-call behavior."""
        self._pass_level = None
        self._e_cur = self._e_prev = self._last_eps = None

    def _gust_e(self, eps: float) -> float:
        """AR(1) gust state for the segment whose first normal draw is eps (idempotent)."""
        if self._last_eps is not None and eps == self._last_eps:
            return self._e_cur
        self._e_prev = self._e_cur
        self._e_cur = eps if self._e_prev is None else (
            self.phi * self._e_prev + np.sqrt(1.0 - self.phi ** 2) * eps)
        self._last_eps = eps
        return self._e_cur

    def _draw_segment(self, rng: np.random.Generator, mismatch: str):
        """Returns (HiddenState, spread factor); fixed rng order, independent of x and u."""
        if mismatch == "none":
            return NO_MISMATCH, float(rng.uniform(*self.p["spread_factor"]))
        lv = self.levels[mismatch]
        eps = float(rng.standard_normal())
        sf = float(rng.uniform(*self.p["spread_factor"]))
        in_pass = self._pass_level == mismatch
        e = self._gust_e(eps) if in_pass else eps
        gust = max(0.0, 1.0 + lv["gust_sigma"] * e)
        if in_pass:
            eta, ff, vf = self._pass_factors
        else:
            eta = float(rng.uniform(*lv["canopy_eta"]))
            ff = 1.0 + float(rng.uniform(*lv["wear_flow"]))
            vf = 1.0 + float(rng.uniform(*lv["wear_vmd"]))
        return HiddenState(gust, eta, ff, vf), sf

    def _draw_hidden_batch(self, n: int, rng: np.random.Generator, mismatch: str):
        if mismatch == "none":
            one = np.ones(n)
            return one, one.copy(), one.copy(), one.copy()
        lv = self.levels[mismatch]
        g = np.maximum(0.0, 1.0 + lv["gust_sigma"] * rng.standard_normal(n))
        eta = rng.uniform(*lv["canopy_eta"], n)
        ff = 1.0 + rng.uniform(*lv["wear_flow"], n)
        vf = 1.0 + rng.uniform(*lv["wear_vmd"], n)
        return g, eta, ff, vf

    # ------------------------------------------------------------ core
    def _core(self, vmd, gpa, u_cross, u_along, height, temp, rh, eta, sf):
        """Vectorized over N segments. Returns dict of arrays plus per-bin arrays."""
        p = self.p
        thr = float(p["drift_threshold_m"])
        d0, w, dln = lognormal_volume_bins(vmd, self.sigma_ln, self.n_bins)
        lam = evaporation_lambda(temp, rh, p["evaporation_lambda_coeff"])[:, None]
        d_f, t_fall, alive = fall_with_evaporation(d0, height[:, None], lam)
        x_c = u_cross[:, None] * t_fall
        x_a = u_along[:, None] * t_fall
        inr = drift_mod.in_range_fraction(x_c, thr)
        frac_dep = w[None, :] * inr * alive * (d_f / d0) ** 3     # emitted-volume fraction
        drift_frac = np.clip(1.0 - frac_dep.sum(axis=1), 0.0, 1.0)
        total_ul = np.asarray(gpa) * p["gpa_to_ul_cm2"]
        dep_bins = total_ul[:, None] * eta[:, None] * frac_dep    # uL/cm^2 per bin
        n_cm2 = cov_mod.counts_per_cm2(dep_bins, d_f)
        cov, stains = cov_mod.boolean_coverage(n_cm2, d_f, sf, p["min_detectable_droplet_um"])
        disp = drift_mod.mean_displacement_mm(frac_dep, x_c, x_a, thr)
        return dict(coverage=cov * 100.0, deposition=dep_bins.sum(axis=1), drift=drift_frac,
                    displacement=disp, stains=stains, d_f=d_f, n_cm2=n_cm2, x_c=x_c, x_a=x_a,
                    dln=dln, frac_dep=frac_dep)

    def _inputs(self, x: Conditions, u: Setting, hid: HiddenState, sf: float,
                derived: Derived | None = None):
        d = derived or self.derive_fn(x, u)
        return (np.array([d.vmd_um * hid.vmd_factor]),
                np.array([d.app_rate_gpa * hid.flow_factor]),
                np.array([d.u_cross_ms * hid.gust_mult]),
                np.array([d.u_along_ms * hid.gust_mult]),
                np.array([x.boom_height_m]), np.array([x.temp_c]), np.array([x.rh_pct]),
                np.array([hid.canopy_eta]), np.array([sf]))

    # ------------------------------------------------------------ public API
    def simulate(self, x: Conditions, u: Setting, rng: np.random.Generator,
                 mismatch: MismatchLevel = "none", sample_droplets: bool = False) -> Outcome:
        """One segment; see the module docstring for the rng draw order."""
        hid, sf = self._draw_segment(rng, mismatch)
        r = self._core(*self._inputs(x, u, hid, sf))
        sample = self._sample(r, sf, rng) if sample_droplets else None
        return Outcome(coverage_pct=float(r["coverage"][0]),
                       deposition_ul_cm2=float(r["deposition"][0]),
                       drift_frac=float(r["drift"][0]),
                       displacement_mm=float(r["displacement"][0]),
                       stains_per_cm2=float(r["stains"][0]), droplets=sample)

    def _sample(self, r, sf: float, rng: np.random.Generator) -> DropletSample:
        shift_x = 1000.0 * np.clip(r["x_c"][0], -self.p["drift_threshold_m"], self.p["drift_threshold_m"])
        shift_y = 1000.0 * np.clip(r["x_a"][0], -self.p["drift_threshold_m"], self.p["drift_threshold_m"])
        diam, xs, ys = cov_mod.sample_card_droplets(
            r["n_cm2"][0], r["d_f"][0], sf, shift_x, shift_y,
            self.p["min_detectable_droplet_um"], self.card_w, self.card_h, rng)
        return DropletSample(diameters_um=diam, x_mm=xs, y_mm=ys, spread_factor=sf)

    def simulate_arrays(self, xs: Sequence[Conditions], us: Sequence[Setting],
                        rng: np.random.Generator, mismatch: MismatchLevel = "none",
                        derived: Sequence[Derived] | None = None,
                        chunk: int = 4096) -> dict[str, np.ndarray]:
        """Vectorized simulation of many independent segments (iid hidden draws).

        Returns arrays coverage_pct, deposition_ul_cm2, drift_frac, displacement_mm,
        stains_per_cm2. Chunked to bound the (N, 40) temporaries.
        """
        n = len(xs)
        ders = derived if derived is not None else [self.derive_fn(x, u) for x, u in zip(xs, us)]
        vmd = np.array([d.vmd_um for d in ders])
        gpa = np.array([d.app_rate_gpa for d in ders])
        uc = np.array([d.u_cross_ms for d in ders])
        ua = np.array([d.u_along_ms for d in ders])
        h = np.array([x.boom_height_m for x in xs])
        t = np.array([x.temp_c for x in xs])
        rh = np.array([x.rh_pct for x in xs])
        g, eta, ff, vf = self._draw_hidden_batch(n, rng, mismatch)
        sf = rng.uniform(*self.p["spread_factor"], n)
        keys = ("coverage", "deposition", "drift", "displacement", "stains")
        out = {k: np.empty(n) for k in keys}
        for a in range(0, n, chunk):
            s = slice(a, min(a + chunk, n))
            r = self._core(vmd[s] * vf[s], gpa[s] * ff[s], uc[s] * g[s], ua[s] * g[s],
                           h[s], t[s], rh[s], eta[s], sf[s])
            for k in keys:
                out[k][s] = r[k]
        return {"coverage_pct": out["coverage"], "deposition_ul_cm2": out["deposition"],
                "drift_frac": out["drift"], "displacement_mm": out["displacement"],
                "stains_per_cm2": out["stains"]}

    def simulate_batch(self, xs: Sequence[Conditions], us: Sequence[Setting],
                       rng: np.random.Generator, mismatch: MismatchLevel = "none"
                       ) -> list[Outcome]:
        a = self.simulate_arrays(xs, us, rng, mismatch)
        return [Outcome(float(a["coverage_pct"][i]), float(a["deposition_ul_cm2"][i]),
                        float(a["drift_frac"][i]), float(a["displacement_mm"][i]),
                        float(a["stains_per_cm2"][i]), None) for i in range(len(xs))]
