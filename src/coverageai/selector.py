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

Optional nozzle lock (``nozzle_lock``, off by default, decision D19): a sprayer carries one
nozzle on the boom for a whole field, so with ``nozzle_lock = (family, size)`` the candidate
set is that nozzle's pressure grid only (speed and boom-height variants still apply when those
options are on). ``best_nozzle`` picks the nozzle for a field from the model's predictions.
With the lock off the candidate list, and therefore every result, is unchanged.

Optional fixed application rate (``rate_lock`` = target gal/acre, off by default, needs
``nozzle_lock``): the rate is a label rate and must stay within +-``rate_tol_frac`` (spec
``rate_control.tolerance_frac``). Speed is then tied to pressure (see ``rate_control``), so the
candidates are the locked nozzle's pressure grid, each with the speed that holds the rate (and
the two band ends), times the boom-height variants if those are on; ``speed_options_kmh`` is
ignored. The choice has three tiers: the lowest-drift feasible setting at exactly the target
rate, else the lowest-drift feasible one inside the band, else the highest corrected coverage in
the band (``info["infeasible"]``, ``info["target_unreachable_at_rate"]``). Volume is fixed by
the rate, so ``lam`` and the volume term are not used. With the lock off nothing changes.
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
    from coverageai import catalog      # a broken catalog must fail loudly, not fall back to mocks
    return catalog.candidates, catalog.derive


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
                 boom_overlap_factor: float | None = None,
                 nozzle_lock: tuple[str, str] | None = None,
                 rate_lock: float | None = None, rate_tol_frac: float | None = None):
        s = spec()["selector"]
        # Opt-in nozzle lock: None (default) = any nozzle per segment, the v1.0 behaviour.
        self.nozzle_lock = None if nozzle_lock is None else (str(nozzle_lock[0]), str(nozzle_lock[1]))
        # Opt-in fixed application rate (gal/acre): None (default) = volume is a free trade-off.
        self.rate_lock = None if rate_lock is None else float(rate_lock)
        self.rate_tol_frac = float(spec().get("rate_control", {}).get("tolerance_frac", 0.05)
                                   if rate_tol_frac is None else rate_tol_frac)
        self._rate_cands: dict[tuple, list[Setting]] = {}
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

    @property
    def candidates(self) -> list[Setting]:
        if self._cands is None:
            self._cands = list(self._candidates_fn())
        return self._cands

    def candidates_for(self, x: Conditions) -> list[Setting]:
        """Base candidates (only the locked nozzle's if ``nozzle_lock`` is set), plus speed
        variants within the band around x's speed if enabled."""
        if self.rate_lock is not None:
            return self._rate_candidates_for(x)
        out = self.candidates
        if self.nozzle_lock is not None:
            out = [u for u in out if (u.family, u.size) == self.nozzle_lock]
        if self.speed_options_kmh:
            from coverageai.catalog import with_speeds
            out = with_speeds(out, x.travel_speed_kmh, self.speed_options_kmh, self.speed_band_frac)
        if self.boom_height_options_m:   # combinable with speed: full product of the two grids
            from coverageai.catalog import with_boom_heights
            out = with_boom_heights(out, x.boom_height_m, self.boom_height_options_m,
                                    self.boom_height_band_frac, self.boom_overlap_factor)
        return out

    def _rate_candidates_for(self, x: Conditions) -> list[Setting]:
        """Rate mode: iso-rate settings of the locked nozzle (cached), plus boom-height variants."""
        if self.nozzle_lock is None:
            raise ValueError("rate_lock needs nozzle_lock: the rate is held for one nozzle")
        from coverageai import rate_control
        key = (self.nozzle_lock, self.rate_lock, self.rate_tol_frac)
        out = self._rate_cands.get(key)
        if out is None:
            out = self._rate_cands[key] = rate_control.iso_rate_candidates(
                *self.nozzle_lock, self.rate_lock, self.rate_tol_frac)
        if self.boom_height_options_m:
            from coverageai.catalog import with_boom_heights
            out = with_boom_heights(out, x.boom_height_m, self.boom_height_options_m,
                                    self.boom_height_band_frac, self.boom_overlap_factor)
        return list(out)

    def margin(self, cov: np.ndarray, lo: np.ndarray) -> np.ndarray:
        if self.use_interval_margin:
            return np.maximum(0.5 * self.margin_pp, cov - lo)
        return np.full_like(cov, self.margin_pp)

    def select(self, x: Conditions, model, bias: float, c_req: float) -> tuple[Setting, dict]:
        if self.rate_lock is not None:
            return self._select_rate(x, model, bias, c_req)
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
        info["model_time_s"] = t1 - t0
        info["selector_time_s"] = t2 - t1
        return cands[best], info

    def _select_rate(self, x: Conditions, model, bias: float, c_req: float) -> tuple[Setting, dict]:
        """Rate-locked choice (see the module docstring): exact-rate feasible, else band
        feasible, else best coverage in the band. Order within a tier: drift, then pressure."""
        t0 = time.perf_counter()
        cands = self.candidates_for(x)
        n = len(cands)
        if n == 0:
            raise ValueError("no candidate settings at this rate (speed range)")
        preds = model.predict_batch([x] * n, cands)
        t1 = time.perf_counter()
        cov = np.array([p.coverage_pct for p in preds], float)
        lo = np.array([p.coverage_lo for p in preds], float)
        drift = np.array([p.drift_frac for p in preds], float)
        corrected = cov + bias
        m = self.margin(cov, lo)
        feasible = corrected - m >= c_req
        gpas = np.array([self._derive_fn(x, u).app_rate_gpa for u in cands], float)
        dev = np.abs(gpas / self.rate_lock - 1.0)
        exact = dev < 1e-6
        press = np.array([u.pressure_psi for u in cands], float)
        group = np.where(feasible, np.where(exact, 0, 1), 2)
        # feasible tiers: lowest drift; unreachable tier: highest corrected coverage
        primary = np.where(group == 2, -corrected, drift)
        order = np.lexsort((press, primary, group))
        best = int(order[0])
        tier = ("exact", "band", "unreachable")[int(group[best])]
        info: dict = {"n_candidates": n, "n_feasible": int(feasible.sum()),
                      "infeasible": not bool(feasible.any()),
                      "wind_limit_exceeded": bool(x.wind_speed_ms > self.wind_limit_ms),
                      "lam": self.lam, "c_req": c_req, "bias": bias,
                      "rate_mode": True, "rate_target_gpa": self.rate_lock, "rate_tier": tier,
                      "target_unreachable_at_rate": tier == "unreachable"}
        info["top"] = [{
            **dataclasses.asdict(cands[i]), "pred_cov": float(cov[i]),
            "corrected_cov": float(corrected[i]), "margin": float(m[i]),
            "drift": float(drift[i]), "gpa": float(gpas[i]), "score": None,
            "feasible": bool(feasible[i]),
        } for i in order[:5]]
        t2 = time.perf_counter()
        self.last_model_time_s = t1 - t0
        info["model_time_s"] = t1 - t0
        info["selector_time_s"] = t2 - t1
        return cands[best], info

    def best_nozzle(self, conditions: Sequence[Conditions], model, c_req: float,
                    bias: float = 0.0) -> tuple[tuple[str, str], dict]:
        """The one nozzle (family, size) to carry for a whole field (decision D19).

        For every nozzle in the base grid and every segment, the pressure is the lowest-drift
        one that meets the selector's own constraint (corrected coverage - margin >= c_req), or
        the highest-coverage one if none does. Nozzles are ranked by the share of segments that
        can meet the target, then by the selector's own trade-off (lam x mean drift +
        (1 - lam) x mean volume, min-max normalized over the nozzles tied on that share). Only
        pressure is varied; speed and boom-height options do not enter the choice. Uses the model's
        predictions only (what an operator knows before the pass), one ``predict_batch`` call.
        Returns ((family, size), stats) with one entry per nozzle in ``stats["nozzles"]``."""
        xs, cands = list(conditions), self.candidates
        n_s, n_c = len(xs), len(cands)
        if n_s == 0 or n_c == 0:
            raise ValueError("best_nozzle needs at least one segment and one candidate")
        preds = model.predict_batch([x for x in xs for _ in range(n_c)], cands * n_s)
        cov = np.array([p.coverage_pct for p in preds], float).reshape(n_s, n_c)
        lo = np.array([p.coverage_lo for p in preds], float).reshape(n_s, n_c)
        drift = np.array([p.drift_frac for p in preds], float).reshape(n_s, n_c)
        corrected = cov + bias
        feasible = corrected - self.margin(cov, lo) >= c_req
        groups: dict[tuple[str, str], list[int]] = {}
        for j, u in enumerate(cands):
            groups.setdefault((u.family, u.size), []).append(j)
        rows = []
        seg = np.arange(n_s)
        for key, cols in groups.items():
            cols_a = np.array(cols)
            f, d, c = feasible[:, cols_a], drift[:, cols_a], corrected[:, cols_a]
            ok = f.any(axis=1)
            pick = np.where(ok, np.where(f, d, np.inf).argmin(axis=1), c.argmax(axis=1))
            chosen = cols_a[pick]
            gpa = np.mean([self._derive_fn(xs[i], cands[chosen[i]]).app_rate_gpa for i in seg])
            rows.append({"family": key[0], "size": key[1], "success": float(ok.mean()),
                         "mean_drift": float(d[seg, pick].mean()), "mean_gpa": float(gpa)})
        # Same trade-off as select(): among the nozzles with the best success rate, the lowest
        # lam * drift + (1 - lam) * volume, both min-max normalized over that group.
        success = np.array([r["success"] for r in rows])
        md, mv = (np.array([r[k] for r in rows]) for k in ("mean_drift", "mean_gpa"))
        top = np.flatnonzero(success >= success.max() - 1e-12)
        score = np.full(len(rows), np.inf)
        score[top] = self.lam * _minmax(md[top]) + (1 - self.lam) * _minmax(mv[top])
        for r, s_ in zip(rows, score):
            r["score"] = None if np.isinf(s_) else float(s_)
        order = np.lexsort((mv, md, score, -success))
        best = rows[int(order[0])]
        return (best["family"], best["size"]), {"nozzles": [rows[int(i)] for i in order],
                                                 "c_req": c_req, "n_segments": n_s}


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
