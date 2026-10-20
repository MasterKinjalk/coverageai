"""WP-11 parameter selector (proposal eq. 2).

    u* = argmin_{u in U}  lam * D_n(x,u) + (1 - lam) * V_n(u)
         s.t.  C_hat(x,u) + bias - m(x,u) >= C_req

D_n, V_n are min-max normalized over the *feasible* set (zero range -> 0).
V(u) is the application volume per area, derive(x,u).app_rate_gpa (uses the effective
travel speed: u.travel_speed_kmh if set, else x's).

Optional speed control (``speed_options_kmh``, off by default): each base candidate is also
offered at every grid speed within +-speed_band_frac of x's speed and the spec speed range.
Slower travel raises GPA and deposition (and lowers drift per unit liquid), faster does the
reverse, so the selector can trade volume for coverage without changing the nozzle.

Optional boom-height control (``boom_height_options_m``, off by default, combinable with speed):
candidates are also offered at grid heights within +-boom_height_band_frac of x's height, the
spec height range and above the family's minimum overlap height (catalog.min_boom_height_m).
Lower release means shorter fall, less drift; too low causes fan-overlap striping.

Margin m (pp): with ``use_interval_margin`` it is the lower half-width of the
model's prediction interval, floored so a degenerate (zero-width) interval cannot
remove all safety:  m = max(0.5 * margin_pp, C_hat - C_lo).
Without an interval, m = margin_pp.

If no candidate is feasible the setting with the highest corrected coverage is
returned and ``info["infeasible"]`` is True.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Callable, Sequence

import numpy as np

from coverageai.config import spec
from coverageai.contracts import Conditions, Derived, Setting

CandidatesFn = Callable[[], Sequence[Setting]]
DeriveFn = Callable[[Conditions, Setting], Derived]


def _default_fns() -> tuple[CandidatesFn, DeriveFn]:
    try:
        from coverageai import catalog
        return catalog.candidates, catalog.derive
    except Exception:  # catalog (WP-01) not merged yet, or data missing
        from coverageai.mocks import mock_candidates, mock_derive
        return mock_candidates, mock_derive


def _minmax(v: np.ndarray) -> np.ndarray:
    rng = float(v.max() - v.min()) if v.size else 0.0
    if rng <= 1e-12:
        return np.zeros_like(v, dtype=float)
    return (v - v.min()) / rng


class ConstrainedSelector:
    def __init__(self, lam: float | None = None, margin_pp: float | None = None,
                 use_interval_margin: bool | None = None,
                 candidates_fn: CandidatesFn | None = None, derive_fn: DeriveFn | None = None,
                 speed_options_kmh: Sequence[float] | None = None,
                 speed_band_frac: float | None = None,
                 boom_height_options_m: Sequence[float] | None = None,
                 boom_height_band_frac: float | None = None,
                 boom_overlap_factor: float | None = None):
        s = spec()["selector"]
        # Opt-in travel-speed control: None -> spec default (empty = off).
        self.speed_options_kmh = list(s.get("speed_options_kmh", []) if speed_options_kmh is None
                                      else speed_options_kmh)
        self.speed_band_frac = float(s.get("speed_band_frac", 0.25) if speed_band_frac is None
                                     else speed_band_frac)
        # Opt-in boom-height control (same pattern; empty = off).
        self.boom_height_options_m = list(s.get("boom_height_options_m", [])
                                          if boom_height_options_m is None else boom_height_options_m)
        self.boom_height_band_frac = float(s.get("boom_height_band_frac", 0.25)
                                           if boom_height_band_frac is None else boom_height_band_frac)
        self.boom_overlap_factor = float(s.get("boom_overlap_factor", 1.3)
                                         if boom_overlap_factor is None else boom_overlap_factor)
        self.lam = float(s["lambda"] if lam is None else lam)
        self.margin_pp = float(s["margin_pp"] if margin_pp is None else margin_pp)
        self.use_interval_margin = bool(s["use_interval_margin"] if use_interval_margin is None
                                        else use_interval_margin)
        dc, dd = _default_fns()
        self._candidates_fn = candidates_fn or dc
        self._derive_fn = derive_fn or dd
        self._cands: list[Setting] | None = None
        self.wind_limit_ms = float(spec()["wind_limit_ms"])
        self.last_model_time_s = 0.0
        self.last_total_time_s = 0.0

    @property
    def candidates(self) -> list[Setting]:
        if self._cands is None:
            self._cands = list(self._candidates_fn())
        return self._cands

    def candidates_for(self, x: Conditions) -> list[Setting]:
        """Base candidates, plus speed variants within the band around x's speed if enabled."""
        out = self.candidates
        if self.speed_options_kmh:
            from coverageai.catalog import with_speeds
            out = with_speeds(out, x.travel_speed_kmh, self.speed_options_kmh, self.speed_band_frac)
        if self.boom_height_options_m:   # combinable with speed: full product of the two grids
            from coverageai.catalog import with_boom_heights
            out = with_boom_heights(out, x.boom_height_m, self.boom_height_options_m,
                                    self.boom_height_band_frac, self.boom_overlap_factor)
        return out

    def margin(self, cov: np.ndarray, lo: np.ndarray) -> np.ndarray:
        if self.use_interval_margin:
            return np.maximum(0.5 * self.margin_pp, cov - lo)
        return np.full_like(cov, self.margin_pp)

    def select(self, x: Conditions, model, bias: float, c_req: float) -> tuple[Setting, dict]:
        t0 = time.perf_counter()
        cands = self.candidates_for(x)
        n = len(cands)
        if n == 0:
            raise ValueError("no candidate settings")
        preds = model.predict_batch([x] * n, cands)
        t1 = time.perf_counter()
        cov = np.array([p.coverage_pct for p in preds], float)
        lo = np.array([p.coverage_lo for p in preds], float)
        drift = np.array([p.drift_frac for p in preds], float)
        corrected = cov + bias
        m = self.margin(cov, lo)
        feasible = corrected - m >= c_req

        cache: dict[Setting, Derived] = {}   # derive once per candidate for this x

        def gpa(u: Setting) -> float:
            if u not in cache:
                cache[u] = self._derive_fn(x, u)
            return cache[u].app_rate_gpa

        info: dict = {"n_candidates": n, "n_feasible": int(feasible.sum()),
                      "infeasible": not bool(feasible.any()),
                      "wind_limit_exceeded": bool(x.wind_speed_ms > self.wind_limit_ms),
                      "lam": self.lam, "c_req": c_req, "bias": bias}
        score = np.full(n, np.nan)
        if info["infeasible"]:
            best = int(np.argmax(corrected))
            order = np.argsort(-corrected, kind="stable")[:5]
        else:
            idx = np.flatnonzero(feasible)
            vol = np.array([gpa(cands[i]) for i in idx], float)
            sc = self.lam * _minmax(drift[idx]) + (1 - self.lam) * _minmax(vol)
            score[idx] = sc
            # tie-break: score, then drift, then volume
            order = idx[np.lexsort((vol, drift[idx], sc))][:5]
            best = int(order[0])
        info["top"] = [{
            **dataclasses.asdict(cands[i]), "pred_cov": float(cov[i]),
            "corrected_cov": float(corrected[i]), "margin": float(m[i]),
            "drift": float(drift[i]), "gpa": float(gpa(cands[i])),
            "score": None if np.isnan(score[i]) else float(score[i]),
            "feasible": bool(feasible[i]),
        } for i in order]
        t2 = time.perf_counter()
        self.last_model_time_s = t1 - t0
        self.last_total_time_s = t2 - t0
        info["model_time_s"] = t1 - t0
        info["selector_time_s"] = t2 - t1
        return cands[best], info


def _stats_for_candidates(scenarios: Sequence[Conditions], truth, candidates: Sequence[Setting],
                          derive_fn: DeriveFn, seed: int = 0):
    """Per-candidate coverage and drift matrices [n_cand, n_scen] and mean volume [n_cand]."""
    n_s = len(scenarios)
    cov = np.zeros((len(candidates), n_s))
    drift = np.zeros_like(cov)
    if hasattr(truth, "predict_batch"):
        for j, u in enumerate(candidates):
            preds = truth.predict_batch(list(scenarios), [u] * n_s)
            cov[j] = [p.coverage_pct for p in preds]
            drift[j] = [p.drift_frac for p in preds]
    else:  # Simulator protocol, no mismatch
        rng = np.random.default_rng(seed)
        for j, u in enumerate(candidates):
            for i, x in enumerate(scenarios):
                o = truth.simulate(x, u, rng, "none")
                cov[j, i], drift[j, i] = o.coverage_pct, o.drift_frac
    vol = np.array([[derive_fn(x, u).app_rate_gpa for x in scenarios] for u in candidates]).mean(axis=1)
    return cov, drift, vol


def best_fixed_setting(scenarios: Sequence[Conditions], simulator_or_model, c_req: float,
                       candidates: Sequence[Setting] | None = None,
                       derive_fn: DeriveFn | None = None, return_stats: bool = False):
    """Single setting with the highest success rate (coverage >= c_req) across scenarios.
    Ties: lowest mean drift, then lowest mean volume. Accepts a Model (predict_batch)
    or a Simulator (simulate, run with mismatch 'none')."""
    dc, dd = _default_fns()
    candidates = list(candidates if candidates is not None else dc())
    derive_fn = derive_fn or dd
    cov, drift, vol = _stats_for_candidates(scenarios, simulator_or_model, candidates, derive_fn)
    success = (cov >= c_req).mean(axis=1)
    mean_drift = drift.mean(axis=1)
    j = int(np.lexsort((vol, mean_drift, -success))[0])
    if return_stats:
        return candidates[j], {"success": success, "mean_drift": mean_drift, "mean_gpa": vol,
                               "index": j, "candidates": candidates}
    return candidates[j]
