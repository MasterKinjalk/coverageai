"""WP-11 parameter selector (proposal eq. 2).

    u* = argmin_{u in U}  lam * D_n(x,u) + (1 - lam) * V_n(u)
         s.t.  C_hat(x,u) + bias - m(x,u) >= C_req

D_n, V_n are min-max normalized over the *feasible* set (zero range -> 0).
V(u) is the application volume per area, derive(x,u).app_rate_gpa (uses the effective
travel speed: u.travel_speed_kmh if set, else x's).

Optional speed control (``speed_options_kmh``, off by default): each base candidate is also
offered at grid speeds within +-speed_band_frac of x's speed and the spec speed range. If a
GPA target cannot be reached in that band, the selector can expand to the full spec speed range.
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


def _near_target(candidates: Sequence[Setting], values: Sequence[float], target: float,
                 tolerance_frac: float) -> tuple[list[Setting], bool]:
    """Keep values within a relative target tolerance, or the nearest available values."""
    if not candidates:
        return [], False
    errors = np.abs(np.asarray(values, dtype=float) - float(target))
    tolerance = abs(float(target)) * float(tolerance_frac)
    within = errors <= tolerance + 1e-9
    if within.any():
        return [u for u, keep in zip(candidates, within) if keep], True
    nearest = errors.min()
    return [u for u, error in zip(candidates, errors) if error <= nearest + 1e-9], False


def _within_target(candidates: Sequence[Setting], values: Sequence[float], target: float,
                   tolerance_frac: float) -> list[Setting]:
    """Keep only values inside a relative target tolerance."""
    if not candidates:
        return []
    errors = np.abs(np.asarray(values, dtype=float) - float(target))
    tolerance = abs(float(target)) * float(tolerance_frac)
    return [u for u, error in zip(candidates, errors) if error <= tolerance + 1e-9]


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
                 target_gpa_gal_acre: float | None = None,
                 gpa_tolerance_frac: float = 0.05,
                 pressure_target_psi: float | None = None,
                 pressure_control: bool = False,
                 pressure_band_frac: float = 0.25):
        s = spec()["selector"]
        # Opt-in nozzle lock: None (default) = any nozzle per segment, the v1.0 behaviour.
        self.nozzle_lock = None if nozzle_lock is None else (str(nozzle_lock[0]), str(nozzle_lock[1]))
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
        self.target_gpa_gal_acre = target_gpa_gal_acre
        self.gpa_tolerance_frac = float(gpa_tolerance_frac)
        self.pressure_target_psi = pressure_target_psi
        self.pressure_control = bool(pressure_control)
        self.pressure_band_frac = float(pressure_band_frac)
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

    def _candidates_before_gpa(self, x: Conditions, *, relax_pressure: bool = False,
                                relax_speed: bool = False) -> list[Setting]:
        out = self.candidates
        if self.nozzle_lock is not None:
            out = [u for u in out if (u.family, u.size) == self.nozzle_lock]
        if self.pressure_target_psi is not None and not (
                relax_pressure and self.pressure_control):
            pressure_tolerance = self.pressure_band_frac if self.pressure_control else 0.0
            out, _ = _near_target(out, [u.pressure_psi for u in out],
                                  self.pressure_target_psi, pressure_tolerance)
        if self.speed_options_kmh:
            from coverageai.catalog import with_speeds
            speed_band = self.speed_band_frac
            speed_options = self.speed_options_kmh
            if relax_speed:
                speed_lo, speed_hi = spec()["ranges"]["travel_speed_kmh"]
                speed_options = sorted({
                    *map(float, self.speed_options_kmh), float(speed_lo), float(speed_hi)
                })
                current = float(x.travel_speed_kmh)
                if current > 0:
                    speed_band = max((current - float(speed_lo)) / current,
                                     (float(speed_hi) - current) / current, 0.0)
            out = with_speeds(out, x.travel_speed_kmh, speed_options, speed_band)
        if self.boom_height_options_m:   # combinable with speed: full product of the two grids
            from coverageai.catalog import with_boom_heights
            out = with_boom_heights(out, x.boom_height_m, self.boom_height_options_m,
                                    self.boom_height_band_frac, self.boom_overlap_factor)
        return out

    def _gpa_candidate_sets(self, x: Conditions) -> list[list[Setting]]:
        """Progressively relax setpoint bands, never the nozzle chart or legal speed range."""
        can_relax_pressure = self.pressure_control and self.pressure_target_psi is not None
        can_relax_speed = bool(self.speed_options_kmh)
        relaxations = [(False, False)]
        if can_relax_pressure:
            relaxations.append((True, False))
        if can_relax_speed:
            relaxations.append((False, True))
        if can_relax_pressure and can_relax_speed:
            relaxations.append((True, True))
        return [self._candidates_before_gpa(x, relax_pressure=pressure, relax_speed=speed)
                for pressure, speed in relaxations]

    def candidates_for(self, x: Conditions) -> list[Setting]:
        """Candidates within GPA, or the closest available rates after all relaxations."""
        if self.target_gpa_gal_acre is None:
            return self._candidates_before_gpa(x)
        closest_candidates: list[Setting] = []
        for candidates in self._gpa_candidate_sets(x):
            rates = [self._derive_fn(x, u).app_rate_gpa for u in candidates]
            nearest, within = _near_target(candidates, rates, self.target_gpa_gal_acre,
                                           self.gpa_tolerance_frac)
            if within:
                return nearest
            if nearest:
                closest_candidates = nearest
        return closest_candidates

    def gpa_target_status(self, x: Conditions) -> tuple[bool, float | None]:
        """Check all allowed pressure/speed ranges and return the closest rate if infeasible."""
        if self.target_gpa_gal_acre is None:
            return True, None
        closest: float | None = None
        for options in self._gpa_candidate_sets(x):
            options_rates = [self._derive_fn(x, u).app_rate_gpa for u in options]
            if _within_target(options, options_rates, self.target_gpa_gal_acre,
                              self.gpa_tolerance_frac):
                return True, None
            if options_rates:
                nearest = min(options_rates, key=lambda rate: abs(rate - self.target_gpa_gal_acre))
                if closest is None or abs(nearest - self.target_gpa_gal_acre) < abs(
                        closest - self.target_gpa_gal_acre):
                    closest = nearest
        return False, closest

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
        if self.target_gpa_gal_acre is not None:
            chosen_gpa = gpa(cands[best])
            info.update(
                target_gpa_gal_acre=float(self.target_gpa_gal_acre),
                selected_gpa_gal_acre=float(chosen_gpa),
                gpa_target_met=abs(chosen_gpa - self.target_gpa_gal_acre)
                <= self.target_gpa_gal_acre * self.gpa_tolerance_frac + 1e-9,
            )
        t2 = time.perf_counter()
        self.last_model_time_s = t1 - t0
        info["model_time_s"] = t1 - t0
        info["selector_time_s"] = t2 - t1
        return cands[best], info

    def best_nozzle(self, conditions: Sequence[Conditions], model, c_req: float,
                    bias: float = 0.0) -> tuple[tuple[str, str], dict]:
        """The one nozzle (family, size) to carry for a whole field (decision D19).

        With no GPA target, it retains the selector's original ranking: for every nozzle and
        segment, choose the lowest-drift pressure meeting the coverage constraint, or the
        highest-coverage pressure if none does. With a GPA target, first restrict pressure and
        rate to their configured bounds, then choose for coverage and drift. Nozzles are ranked
        by target-and-coverage success, then by drift and rate error. Only pressure is varied;
        speed and boom-height options do not enter this field-level choice. Uses one
        ``predict_batch`` call and returns ((family, size), stats) with one entry per nozzle."""
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
        rate = None
        if self.target_gpa_gal_acre is not None:
            rate = np.asarray([
                [self._derive_fn(x, u).app_rate_gpa for u in cands]
                for x in xs
            ], dtype=float)
        groups: dict[tuple[str, str], list[int]] = {}
        for j, u in enumerate(cands):
            groups.setdefault((u.family, u.size), []).append(j)
        rows = []
        seg = np.arange(n_s)
        for key, cols in groups.items():
            cols_a = np.array(cols)
            f, d, c = feasible[:, cols_a], drift[:, cols_a], corrected[:, cols_a]
            if rate is None:
                ok = f.any(axis=1)
                pick = np.where(ok, np.where(f, d, np.inf).argmin(axis=1), c.argmax(axis=1))
            else:
                group_rate = rate[:, cols_a]
                pressure_tol = self.pressure_band_frac if self.pressure_control else 0.0
                p_mask = np.ones(len(cols_a), dtype=bool)
                if self.pressure_target_psi is not None:
                    pressure_values = np.asarray([cands[j].pressure_psi for j in cols_a])
                    p_error = np.abs(pressure_values - self.pressure_target_psi)
                    p_mask = p_error <= abs(self.pressure_target_psi) * pressure_tol + 1e-9
                    if not p_mask.any():
                        p_mask = p_error <= p_error.min() + 1e-9
                pick = np.empty(n_s, dtype=int)
                ok = np.zeros(n_s, dtype=bool)
                for i in seg:
                    pressure_idx = np.flatnonzero(p_mask)
                    pressure_rates = group_rate[i, pressure_idx]
                    rate_mask = np.abs(pressure_rates - self.target_gpa_gal_acre) <= (
                        self.target_gpa_gal_acre * self.gpa_tolerance_frac + 1e-9
                    )
                    rate_met = bool(rate_mask.any())
                    if not rate_met:
                        rate_error = np.abs(pressure_rates - self.target_gpa_gal_acre)
                        rate_mask = rate_error <= rate_error.min() + 1e-9
                    eligible = pressure_idx[rate_mask]
                    good = eligible[f[i, eligible]]
                    if good.size:
                        local = np.lexsort((np.abs(group_rate[i, good] - self.target_gpa_gal_acre),
                                            d[i, good]))
                        pick[i] = int(good[local[0]])
                        ok[i] = rate_met
                    else:
                        local = np.lexsort((d[i, eligible],
                                            np.abs(group_rate[i, eligible] - self.target_gpa_gal_acre),
                                            -c[i, eligible]))
                        pick[i] = int(eligible[local[0]])
            chosen = cols_a[pick]
            gpa = np.mean([self._derive_fn(xs[i], cands[chosen[i]]).app_rate_gpa for i in seg])
            rows.append({"family": key[0], "size": key[1], "success": float(ok.mean()),
                         "mean_drift": float(d[seg, pick].mean()), "mean_gpa": float(gpa),
                         "mean_gpa_error": (None if self.target_gpa_gal_acre is None else
                                            float(abs(gpa - self.target_gpa_gal_acre)))})
        # Same trade-off as select(): among the nozzles with the best success rate, the lowest
        # lam * drift + (1 - lam) * volume, both min-max normalized over that group.
        success = np.array([r["success"] for r in rows])
        md = np.array([r["mean_drift"] for r in rows])
        mv = np.array([r["mean_gpa"] for r in rows])
        top = np.flatnonzero(success >= success.max() - 1e-12)
        score = np.full(len(rows), np.inf)
        if self.target_gpa_gal_acre is None:
            score[top] = self.lam * _minmax(md[top]) + (1 - self.lam) * _minmax(mv[top])
        else:
            rate_error = np.abs(mv - self.target_gpa_gal_acre)
            score[top] = self.lam * _minmax(md[top]) + (1 - self.lam) * _minmax(rate_error[top])
        for r, s_ in zip(rows, score):
            r["score"] = None if np.isinf(s_) else float(s_)
        order = (np.lexsort((mv, md, score, -success)) if self.target_gpa_gal_acre is None
                 else np.lexsort((rate_error, md, score, -success)))
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
