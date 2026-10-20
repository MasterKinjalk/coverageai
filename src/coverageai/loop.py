"""Closed-loop harness: runs a field pass segment by segment with injected modules.

Policies (argument `policy` of `run_pass`):
  open       selector with bias = 0 for the whole pass.
  fast       selector with the BiasFeedback correction (eq. 3), per-regime.
  fast_slow  same as `fast` inside a pass; `run_passes_slow` adds the between-pass
             slow loop (residual-correction refit on all verified records, see there).
  oracle     FAIR oracle (FairOracleModel): the selector queries the true simulator with
             the hidden mismatch realization of that segment (gust, canopy, wear) but NOT
             the per-card WSP spread factor, which no controller can know before the card
             is sprayed. It predicts coverage at the mean spread factor with the exact 10/90 %
             spread-factor quantiles as its interval, so the selector applies the same margin
             rule (lower interval bound, q = 0.1, as the learned model's 80 % CQR interval)
             as the deployable policies. Reference for the proposal's closed-loop criterion.
  oracle_card card-aware oracle (OracleModel, the "oracle" of the midterm): also sees the
             per-card spread factor, so its interval is degenerate and it needs no margin.
             Kept as an upper bound that no deployable controller can match.
  best_fixed one fixed Setting (kwarg `fixed_setting`) for every segment, no selector.

Randomness / the hidden mismatch (important contract for the simulator):
  Segment k of a pass with integer `seed` draws its simulator randomness from
  np.random.default_rng([seed, k]); the renderer uses default_rng([seed, k, 1]).
  OracleModel evaluates candidates by calling simulator.simulate(x, u, default_rng([seed, k]),
  mismatch, sample_droplets=False), i.e. with a fresh generator in the same state as the one
  later used for the real outcome. The oracle therefore sees exactly the hidden mismatch
  realization (gust, canopy, wear) of segment k provided simulate() (a) draws its hidden
  mismatch from the first draws of `rng`, independent of `u`, and (b) gives the same
  coverage_pct with sample_droplets True or False.
  Note: if the mismatch is meant to PERSIST across a pass (canopy, nozzle wear), the
  simulator must derive those factors from something shared across segments; with a
  per-segment rng alone they are redrawn at every segment.

Pass state: if the simulator has begin_pass / end_pass (SpraySimulator) they bracket each
pass, so canopy interception and nozzle wear persist over the pass and the gust is AR(1).
A verifier with a `measure_outcome(outcome, rng)` method (integration.NoisyVerifier) is
read directly from the true outcome (no image, no droplet sampling).

Wind forecaster (optional, `forecaster=` of run_pass / run_passes_slow; default None = the
behaviour above, bit for bit). Policies open / fast / fast_slow only (oracles know the gust,
best_fixed does not select). For segment k the forecaster (coverageai.forecast) turns the
nominal x_k into x_point (point-forecast wind speed, direction kept) and x_plan (the wind the
selector plans for: = x_point, or forecast + z sd when gust-aware). The selector sees x_plan;
the logged prediction, the fast-loop regime/bias update and the slow-loop records use
x_point; the plant always uses the true x_k. After the plant call the anemometer reading of
segment k (simulator.last_realized wind + N(0, obs_sd^2) from default_rng([seed, k, 3]))
is passed to the forecaster, so the forecast for k never uses k's gust. SegmentLog.x stays
the nominal x_k; forecast vs realized wind is in extra["fc_*"].

Measurement: if `render` is True and both renderer and verifier are available, the
card image is rendered and read by the verifier. Otherwise (verifier=None, renderer=None
or render=False) the "perfect measurement" is used: meas.coverage_pct = outcome.coverage_pct.

Logged fields: `bias` is the correction applied to this segment's prediction
(corrected_pred_cov = pred.coverage_pct + bias), `residual` = meas - corrected_pred
(what the controller sees); the raw eq. (3) error e_k is in extra["raw_residual"] and the
bias after the update in extra["bias_after"].
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from coverageai.config import px_per_mm, spec
from coverageai.contracts import (CardMeasurement, Conditions, Derived, Outcome, Prediction,
                                  SegmentLog, Setting, coverage_class, flatten_log,
                                  segment_log_to_json)
from coverageai.feedback import BiasFeedback, ResidualCorrectedModel, retreat_flag
from coverageai.forecast import WindForecaster, make_forecaster

POLICIES = ("open", "fast", "fast_slow", "oracle", "oracle_card", "best_fixed")
ORACLE_POLICIES = ("oracle", "oracle_card")
FAIR_ORACLE_INTERVAL_Q = 0.1       # same one-sided level as SprayModel's q_lo (80 % interval)
SLOW_MODES = ("residual", "refit")
FORECAST_POLICIES = ("open", "fast", "fast_slow")   # policies that use a wind forecaster


@dataclass
class Modules:
    simulator: Any
    model: Any
    selector: Any
    verifier: Any            # None => perfect measurement (outcome.coverage_pct)
    renderer: Any
    derive_fn: Callable[[Conditions, Setting], Derived]


class FastMockRenderer:
    """~10 ms stand-in for mocks.MockRenderer (which takes ~3 s/card): random 8x8-px
    blocks, each stained with probability coverage/100, in the same yellow/blue palette.
    Satisfies the Renderer protocol; mocks only, never for reported results."""

    def render(self, outcome, rng, degrade: bool = True):
        s = spec()["card"]
        ppm = s["dpi"] / 25.4
        h, w = int(s["height_mm"] * ppm), int(s["width_mm"] * ppm)
        b = 8
        small = rng.random((-(-h // b), -(-w // b))) < outcome.coverage_pct / 100.0
        mask = np.kron(small, np.ones((b, b), bool))[:h, :w]
        img = np.empty((h, w, 3), np.uint8)
        img[:] = (235, 210, 40)
        img[mask] = (30, 40, 120)
        return img, mask


def default_mock_modules() -> Modules:
    from coverageai.mocks import (MockModel, MockSelector, MockSimulator, MockVerifier,
                                  mock_derive)
    return Modules(MockSimulator(), MockModel(), MockSelector(), MockVerifier(),
                   FastMockRenderer(), mock_derive)


class OracleModel:
    """Model wrapper that answers with the simulator's true outcome for segment `k`.

    Shares the hidden mismatch by re-seeding: for segment k it calls the simulator with
    np.random.default_rng([seed, k]), the same stream the harness uses for the real
    outcome. `k` must be set by the caller before each selection (run_pass does).
    Interval is degenerate (lo = hi = truth). predict_batch is vectorized
    (simulator.simulate_candidates) when the simulator provides it; else one simulate() per candidate.
    """

    def __init__(self, simulator, mismatch: str, seed: int):
        self.simulator, self.mismatch, self.seed, self.k = simulator, mismatch, seed, 0

    def predict(self, x: Conditions, u: Setting) -> Prediction:
        o = self.simulator.simulate(x, u, segment_rng(self.seed, self.k), self.mismatch,
                                    sample_droplets=False)
        return Prediction(o.coverage_pct, o.coverage_pct, o.coverage_pct, o.deposition_ul_cm2,
                          o.drift_frac, o.displacement_mm)

    def predict_batch(self, xs, us):
        # Fast path: one conditions object repeated over many candidates (what the selector
        # sends) -> a single vectorized evaluation with ONE shared hidden draw.
        if hasattr(self.simulator, "simulate_candidates") and len(xs) and all(x is xs[0] for x in xs):
            a = self.simulator.simulate_candidates(xs[0], us, segment_rng(self.seed, self.k),
                                                   self.mismatch)
            c = a["coverage_pct"]
            return [Prediction(float(c[i]), float(c[i]), float(c[i]),
                               float(a["deposition_ul_cm2"][i]), float(a["drift_frac"][i]),
                               float(a["displacement_mm"][i])) for i in range(len(us))]
        return [self.predict(x, u) for x, u in zip(xs, us)]

    def fit(self, records) -> None:
        pass


class FairOracleModel:
    """Oracle that knows segment k's hidden MISMATCH realization but not the card noise.

    For segment k it draws the hidden state exactly as the plant will (simulator
    ``_draw_segment`` on ``default_rng([seed, k])``, so the AR(1) gust state and the pass
    canopy/wear are the ones the real outcome sees) and discards the spread factor SF. It
    then evaluates the physics with that hidden state at the mean SF (point prediction)
    and at the ``interval_q`` / ``1 - interval_q`` quantiles of SF's uniform range
    (coverage_lo / coverage_hi). Coverage is increasing in SF, so coverage_lo is the exact
    ``interval_q`` quantile of the true coverage given everything a controller could know
    before spraying; the selector's margin rule (C_hat - C_lo, floored) is then applied
    exactly as for the learned model, whose CQR interval targets the same quantile.

    Needs a SpraySimulator-like plant (``_draw_segment``, ``_arrays_from_hidden``,
    ``p["spread_factor"]``); for any other simulator (mocks) it falls back to the
    card-aware OracleModel and sets ``self.fallback = True``.
    """

    def __init__(self, simulator, mismatch: str, seed: int,
                 interval_q: float = FAIR_ORACLE_INTERVAL_Q):
        self.simulator, self.mismatch, self.seed, self.k = simulator, mismatch, seed, 0
        self.interval_q = float(interval_q)
        self.fallback = not (hasattr(simulator, "_draw_segment")
                             and hasattr(simulator, "_arrays_from_hidden")
                             and hasattr(simulator, "p"))
        self._card = OracleModel(simulator, mismatch, seed)

    def hidden_state(self):
        """Hidden mismatch realization of segment k (the spread factor is discarded)."""
        hid, _sf = self.simulator._draw_segment(segment_rng(self.seed, self.k), self.mismatch)
        return hid

    def predict_arrays(self, xs, us) -> dict[str, np.ndarray]:
        from coverageai.physics import coverage as cov_mod
        sim = self.simulator
        n = len(us)
        hid = self.hidden_state()
        ders = [sim.derive_fn(x, u) for x, u in zip(xs, us)]
        a, b = (float(v) for v in sim.p["spread_factor"])
        full = lambda v: np.full(n, float(v))  # noqa: E731
        r = sim._arrays_from_hidden(list(xs), list(us), ders, full(hid.gust_mult),
                                    full(hid.canopy_eta), full(hid.flow_factor),
                                    full(hid.vmd_factor), full(0.5 * (a + b)), want_bins=True)
        d_f, n_cm2 = r.pop("d_f"), r.pop("n_cm2")
        md = sim.p["min_detectable_droplet_um"]
        for key, q in (("coverage_lo", self.interval_q), ("coverage_hi", 1.0 - self.interval_q)):
            c, _ = cov_mod.boolean_coverage(n_cm2, d_f, full(a + q * (b - a)), md)
            r[key] = c * 100.0
        return r

    def predict_batch(self, xs, us) -> list[Prediction]:
        if len(xs) == 0:
            return []
        if self.fallback:
            self._card.k = self.k
            return self._card.predict_batch(xs, us)
        r = self.predict_arrays(xs, us)
        c, lo, hi = r["coverage_pct"], r["coverage_lo"], r["coverage_hi"]
        return [Prediction(float(c[i]), float(min(lo[i], c[i])), float(max(hi[i], c[i])),
                           float(r["deposition_ul_cm2"][i]), float(r["drift_frac"][i]),
                           float(r["displacement_mm"][i])) for i in range(len(us))]

    def predict(self, x: Conditions, u: Setting) -> Prediction:
        return self.predict_batch([x], [u])[0]

    def fit(self, records) -> None:
        pass


def make_oracle(policy: str, simulator, mismatch: str, seed: int,
                interval_q: float | None = None):
    """'oracle' -> FairOracleModel (interval_q defaults to FAIR_ORACLE_INTERVAL_Q),
    'oracle_card' -> OracleModel."""
    if policy == "oracle":
        return FairOracleModel(simulator, mismatch, seed,
                               FAIR_ORACLE_INTERVAL_Q if interval_q is None else interval_q)
    if policy == "oracle_card":
        return OracleModel(simulator, mismatch, seed)
    raise ValueError(policy)


def segment_rng(seed: int, k: int) -> np.random.Generator:
    return np.random.default_rng([seed, k])


def _perfect_measurement(outcome: Outcome) -> CardMeasurement:
    s = spec()["coverage_classes"]
    c = outcome.coverage_pct
    return CardMeasurement(c, outcome.stains_per_cm2, None, None, None,
                           coverage_class(c, s["acceptable_min"], s["marginal_min"]), False, 0.0)


def _scalar_info(info: dict) -> dict:
    cands = info.get("candidates")
    n = len(cands) if hasattr(cands, "__len__") else cands
    return {"sel_n_candidates": n}


def run_pass(conditions: list[Conditions], policy: str, modules: Modules, seed: int,
             mismatch: str, c_req: float, pass_id: str,
             feedback: BiasFeedback | None = None, render: bool = True,
             fixed_setting: Setting | None = None,
             oracle_interval_q: float | None = None,
             forecaster: WindForecaster | str | None = None) -> list[SegmentLog]:
    """Run one pass over `conditions` and return one SegmentLog per segment.

    `feedback` (fast / fast_slow only) is created from spec if None; a passed-in object
    keeps its state (call .reset() for a fresh pass). `oracle_interval_q` overrides the
    fair oracle's risk level (sensitivity analysis only). `forecaster` (open / fast /
    fast_slow only; ignored by the other policies): None (default, no forecast), a name
    ("nominal", "last", "ewma", "ar1", optionally "+gust") for a fresh forecaster, or a
    WindForecaster object (keeps its fitted history across calls; begin_pass is called).
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    if policy == "best_fixed" and fixed_setting is None:
        raise ValueError("policy 'best_fixed' needs fixed_setting")
    m = modules
    use_fb = policy in ("fast", "fast_slow")
    fb = (feedback or BiasFeedback()) if use_fb else None
    oracle = (make_oracle(policy, m.simulator, mismatch, seed, oracle_interval_q)
              if policy in ORACLE_POLICIES else None)
    direct = m.verifier is not None and hasattr(m.verifier, "measure_outcome")
    vision = render and not direct and m.verifier is not None and m.renderer is not None
    ppm = px_per_mm()
    logs: list[SegmentLog] = []
    fc = make_forecaster(forecaster) if policy in FORECAST_POLICIES else None
    if fc is not None:
        fc.begin_pass(seed=seed, mismatch=mismatch, simulator=m.simulator)
    if hasattr(m.simulator, "begin_pass"):   # persistent canopy/wear + AR(1) gust for the pass
        m.simulator.begin_pass(seed, mismatch)

    for k, x in enumerate(conditions):
        rng_k = segment_rng(seed, k)
        t0 = time.perf_counter()
        extra: dict[str, Any] = {}
        # controller's view of the wind: x_pt (point forecast), x_sel (planning wind)
        x_pt = x_sel = x
        fcast = None
        if fc is not None:
            fcast = fc.forecast(k, x)
            x_pt, x_sel = fc.conditions(x, fcast)
            extra.update(fcast.as_extra())
            extra["fc_phi"] = fc.phi()

        # ---- decision
        if policy == "best_fixed":
            u, info = fixed_setting, {"infeasible": False, "candidates": 1}
            derived, regime, bias = m.derive_fn(x, u), "fixed", 0.0
        elif policy in ORACLE_POLICIES:
            oracle.k = k
            u, info = m.selector.select(x, oracle, 0.0, c_req)
            derived, regime, bias = m.derive_fn(x, u), policy, 0.0
        elif policy == "open":
            u, info = m.selector.select(x_sel, m.model, 0.0, c_req)
            derived, regime, bias = m.derive_fn(x_pt, u), "open", 0.0
        else:
            # The regime depends on the chosen droplet class, which depends on the bias.
            # Resolve with one refinement: select with the pooled bias, look up the regime
            # bias of that choice, and re-select if the bias differs.
            # (Without a forecaster x_sel is x_pt is x, i.e. the original code path.)
            hint = fb.global_bias()
            u, info = m.selector.select(x_sel, m.model, hint, c_req)
            derived = m.derive_fn(x_pt, u)
            regime = fb.regime(x_pt, derived)
            bias = fb.bias(regime)
            if abs(bias - hint) > 1e-9:
                u, info = m.selector.select(x_sel, m.model, bias, c_req)
                derived = m.derive_fn(x_pt, u)
                regime = fb.regime(x_pt, derived)
                extra["selection_bias"] = bias
                bias = fb.bias(regime)
            else:
                extra["selection_bias"] = hint
        t_select = time.perf_counter() - t0
        d_ctl = derived                         # derived as the controller sees it (x_pt)
        if x_pt is not x:
            derived = m.derive_fn(x, u)         # logged record: nominal conditions

        pred = m.model.predict(x_pt, u)         # raw nominal prediction (also for oracle)
        corrected = pred.coverage_pct + bias
        if hasattr(m.model, "correction"):      # slow-loop wrapper: log the offset it applied
            extra["slow_corr"] = float(m.model.correction(x_pt, u, d_ctl))

        # ---- plant
        t1 = time.perf_counter()
        outcome = m.simulator.simulate(x, u, rng_k, mismatch, sample_droplets=vision)
        t_sim = time.perf_counter() - t1
        if fc is not None:                      # anemometer: segment k AFTER it was sprayed
            extra.update(_observe_wind(fc, k, x, fcast, m.simulator, seed))

        # ---- measurement
        t_render = 0.0
        if vision:
            t2 = time.perf_counter()
            image, _ = m.renderer.render(outcome, np.random.default_rng([seed, k, 1]))
            t_render = time.perf_counter() - t2
            meas = m.verifier.measure(image, ppm)
        elif direct:        # image-free noisy verifier: measures the true outcome plus error
            meas = m.verifier.measure_outcome(outcome, np.random.default_rng([seed, k, 2]))
        else:
            meas = _perfect_measurement(outcome)

        # ---- feedback
        raw_resid = meas.coverage_pct - pred.coverage_pct
        if use_fb:
            fb.update(regime, meas.coverage_pct, pred.coverage_pct)
            extra["bias_after"] = fb.bias(regime)
        extra.update(regime=regime, raw_residual=raw_resid, true_cov_err=outcome.coverage_pct - meas.coverage_pct,
                     t_select_s=t_select, t_sim_s=t_sim, t_render_s=t_render,
                     t_measure_s=meas.runtime_s, **_scalar_info(info))
        logs.append(SegmentLog(
            pass_id=pass_id, seed=seed, k=k, policy=policy, x=x, u=u, derived=derived,
            pred=pred, corrected_pred_cov=corrected, outcome=outcome, meas=meas,
            residual=meas.coverage_pct - corrected, bias=bias,
            flagged_retreat=retreat_flag(meas),
            selector_infeasible=bool(info.get("infeasible", False)), extra=extra))
    if hasattr(m.simulator, "end_pass"):
        m.simulator.end_pass()
    return logs


def _observe_wind(fc: WindForecaster, k: int, x: Conditions, fcast, simulator,
                  seed: int) -> dict:
    """Anemometer reading of segment k (realized wind + sensor noise), fed to `fc`.
    Uses its own stream default_rng([seed, k, 3]) so no other draw is affected."""
    real = getattr(simulator, "last_realized", None)
    if not real:
        return {"fc_obs_wind_ms": None, "fc_obs_used": False}
    w = float(real["wind_speed_ms"])
    obs = max(0.0, w + fc.obs_sd_ms * float(np.random.default_rng([seed, k, 3]).standard_normal()))
    out = fc.observe(k, x, obs)
    out.update(fc_realized_wind_ms=w, fc_abs_err_ms=abs(fcast.wind_ms - w),
               fc_nominal_abs_err_ms=abs(float(x.wind_speed_ms) - w))
    return out


def run_passes_slow(conditions_list, modules: Modules, seed: int, mismatch: str, c_req: float,
                    n_passes: int = 3, render: bool = True,
                    feedback: BiasFeedback | None = None,
                    reset_bias_between_passes: bool = True,
                    slow_mode: str = "residual",
                    correction: ResidualCorrectedModel | None = None,
                    forecaster: WindForecaster | str | None = None) -> list[SegmentLog]:
    """Policy "fast_slow": `n_passes` consecutive passes with fast feedback inside each and a
    slow-loop update between passes. Returns all logs (policy="fast_slow", pass_id
    "fast_slow_p{i}", extra["slow_pass"]=i).

    `conditions_list` is either a list of passes (list[list[Conditions]], pass i uses entry
    i mod len) or a single list[Conditions] reused for every pass. Pass i uses hidden-noise
    seed `seed + 1009 * i` (a new realization of gust, canopy and wear per pass).

    Verified records: for each segment (x, u, Outcome) where the Outcome carries the
    MEASURED coverage (and measured stains/cm2 if available) and all other fields
    (deposition, drift, displacement) from the simulator outcome, because the card only
    measures coverage.

    slow_mode "residual" (default, proposal section 4.4 as redesigned after the midterm):
      The offline model (`modules.model`) is frozen and wrapped in a
      `feedback.ResidualCorrectedModel` for the duration of the call (pass `correction` to
      supply a configured wrapper; its `base` is then used as is). After every pass the
      new verified records join the slow-loop training set and the correction (additive
      coverage offset, per fast-loop regime, empirical-Bayes shrunk toward a global offset
      and toward 0) is refit on ALL records so far, against the FROZEN base predictions,
      so successive refits never compound. The selector and the fast loop then see
      base + correction; the prediction interval is shifted, not narrowed.
      `modules.model` is restored on exit, so no refit leaks into later calls.
    slow_mode "refit" (legacy, the midterm slow loop): `modules.model.fit(new_records)` is
      called between passes (SprayModel appends them with `record_weight` and refits all
      LightGBM boosters and the CQR offset). Mutates `modules.model`.

    Fast-loop bias: reset at the start of every pass by default, because it is defined
    relative to the model that produced the predictions, which has just been refit. The
    cross-pass persistent part of the error lives in the slow correction and the bias only
    has to track this pass's deviation from it (no double counting: the fast-loop residual
    e_k is measured against base + correction, and the correction is fit against base).

    `forecaster` (default None): one WindForecaster (a name is turned into one) is used for
    all passes, so its gust model keeps learning across passes; the verified records then
    carry the point-forecast wind (the conditions the controller predicted with).
    """
    if slow_mode not in SLOW_MODES:
        raise ValueError(f"slow_mode must be one of {SLOW_MODES}")
    passes = conditions_list
    if len(passes) and isinstance(passes[0], Conditions):
        passes = [passes]
    fb = feedback or BiasFeedback()
    fc = make_forecaster(forecaster)
    saved_model = modules.model
    corr = None
    if slow_mode == "residual":
        corr = correction or ResidualCorrectedModel(saved_model, modules.derive_fn)
        modules.model = corr
    all_logs: list[SegmentLog] = []
    try:
        for i in range(n_passes):
            if reset_bias_between_passes or i == 0:
                fb.reset()
            logs = run_pass(list(passes[i % len(passes)]), "fast_slow", modules, seed + 1009 * i,
                            mismatch, c_req, f"fast_slow_p{i}", feedback=fb, render=render,
                            forecaster=fc)
            new = []
            for lg in logs:
                lg.extra["slow_pass"] = i
                if corr is not None:
                    lg.extra["slow_global_corr"] = corr.global_corr
                    lg.extra["slow_n_records"] = corr.n_records
                o = lg.outcome
                stains = lg.meas.stains_per_cm2 if lg.meas.stains_per_cm2 is not None else o.stains_per_cm2
                x_rec = (lg.x if "fc_wind_ms" not in lg.extra
                         else dataclasses.replace(lg.x, wind_speed_ms=lg.extra["fc_wind_ms"]))
                new.append((x_rec, lg.u, Outcome(lg.meas.coverage_pct, o.deposition_ul_cm2,
                                                o.drift_frac, o.displacement_mm, stains, None)))
            if i < n_passes - 1:
                modules.model.fit(new)       # residual: refit correction; refit: legacy
            all_logs.extend(logs)
    finally:
        modules.model = saved_model
    return all_logs


# ------------------------------------------------------------------ outputs

def logs_to_frame(logs: Sequence[SegmentLog]) -> pd.DataFrame:
    return pd.DataFrame([flatten_log(l) for l in logs])


def save_logs(logs: Sequence[SegmentLog], path: str | Path) -> tuple[Path, Path]:
    """Write `<path>.parquet` and `<path>.jsonl` (any suffix on `path` is replaced)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pq, jl = p.with_suffix(".parquet"), p.with_suffix(".jsonl")
    logs_to_frame(logs).to_parquet(pq, index=False)
    jl.write_text("\n".join(segment_log_to_json(l) for l in logs) + "\n", encoding="utf8")
    return pq, jl


def summarize(logs: Sequence[SegmentLog], c_req: float) -> dict:
    """Pass-level metrics using TRUE outcome coverage (not the measurement)."""
    if not logs:
        return {"n": 0}
    cov = np.array([l.outcome.coverage_pct for l in logs])
    gpa = np.array([l.derived.app_rate_gpa for l in logs])
    return {
        "n": len(logs),
        "frac_met": float(np.mean(cov >= c_req)),
        "mean_shortfall_pp": float(np.mean(np.maximum(c_req - cov, 0.0))),
        "total_volume_gpa": float(gpa.sum()),
        "mean_volume_gpa": float(gpa.mean()),
        "mean_drift_frac": float(np.mean([l.outcome.drift_frac for l in logs])),
        "n_retreat": int(sum(l.flagged_retreat for l in logs)),
        "n_infeasible": int(sum(l.selector_infeasible for l in logs)),
    }
