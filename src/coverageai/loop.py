"""Closed-loop harness: runs a field pass segment by segment with injected modules.

Policies (argument `policy` of `run_pass`):
  open       selector with bias = 0 for the whole pass.
  fast       selector with the BiasFeedback correction (eq. 3), per-regime.
  fast_slow  same as `fast` inside a pass; `run_passes_slow` adds the between-pass
             model refit (slow loop).
  oracle     selector queries the TRUE simulator (OracleModel) with the hidden mismatch
             realization of that segment. Upper bound, not deployable.
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

Measurement: if `render` is True and both renderer and verifier are available, the
card image is rendered and read by the verifier. Otherwise (verifier=None, renderer=None
or render=False) the "perfect measurement" is used: meas.coverage_pct = outcome.coverage_pct.

Logged fields: `bias` is the correction applied to this segment's prediction
(corrected_pred_cov = pred.coverage_pct + bias), `residual` = meas - corrected_pred
(what the controller sees); the raw eq. (3) error e_k is in extra["raw_residual"] and the
bias after the update in extra["bias_after"].
"""
from __future__ import annotations

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
from coverageai.feedback import BiasFeedback, retreat_flag

POLICIES = ("open", "fast", "fast_slow", "oracle", "best_fixed")


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
    Interval is degenerate (lo = hi = truth). Cost: one simulate() call per candidate.
    """

    def __init__(self, simulator, mismatch: str, seed: int):
        self.simulator, self.mismatch, self.seed, self.k = simulator, mismatch, seed, 0

    def predict(self, x: Conditions, u: Setting) -> Prediction:
        o = self.simulator.simulate(x, u, segment_rng(self.seed, self.k), self.mismatch,
                                    sample_droplets=False)
        return Prediction(o.coverage_pct, o.coverage_pct, o.coverage_pct, o.deposition_ul_cm2,
                          o.drift_frac, o.displacement_mm)

    def predict_batch(self, xs, us):
        return [self.predict(x, u) for x, u in zip(xs, us)]

    def fit(self, records) -> None:
        pass


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
             fixed_setting: Setting | None = None) -> list[SegmentLog]:
    """Run one pass over `conditions` and return one SegmentLog per segment.

    `feedback` (fast / fast_slow only) is created from spec if None; a passed-in object
    keeps its state (call .reset() for a fresh pass).
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    if policy == "best_fixed" and fixed_setting is None:
        raise ValueError("policy 'best_fixed' needs fixed_setting")
    m = modules
    use_fb = policy in ("fast", "fast_slow")
    fb = (feedback or BiasFeedback()) if use_fb else None
    oracle = OracleModel(m.simulator, mismatch, seed) if policy == "oracle" else None
    vision = render and m.verifier is not None and m.renderer is not None
    ppm = px_per_mm()
    logs: list[SegmentLog] = []

    for k, x in enumerate(conditions):
        rng_k = segment_rng(seed, k)
        t0 = time.perf_counter()
        extra: dict[str, Any] = {}

        # ---- decision
        if policy == "best_fixed":
            u, info = fixed_setting, {"infeasible": False, "candidates": 1}
            derived, regime, bias = m.derive_fn(x, u), "fixed", 0.0
        elif policy == "oracle":
            oracle.k = k
            u, info = m.selector.select(x, oracle, 0.0, c_req)
            derived, regime, bias = m.derive_fn(x, u), "oracle", 0.0
        elif policy == "open":
            u, info = m.selector.select(x, m.model, 0.0, c_req)
            derived, regime, bias = m.derive_fn(x, u), "open", 0.0
        else:
            # The regime depends on the chosen droplet class, which depends on the bias.
            # Resolve with one refinement: select with the pooled bias, look up the regime
            # bias of that choice, and re-select if the bias differs.
            hint = fb.global_bias()
            u, info = m.selector.select(x, m.model, hint, c_req)
            derived = m.derive_fn(x, u)
            regime = fb.regime(x, derived)
            bias = fb.bias(regime)
            if abs(bias - hint) > 1e-9:
                u, info = m.selector.select(x, m.model, bias, c_req)
                derived = m.derive_fn(x, u)
                regime = fb.regime(x, derived)
                extra["selection_bias"] = bias
                bias = fb.bias(regime)
            else:
                extra["selection_bias"] = hint
        t_select = time.perf_counter() - t0

        pred = m.model.predict(x, u)            # raw nominal prediction (also for oracle)
        corrected = pred.coverage_pct + bias

        # ---- plant
        t1 = time.perf_counter()
        outcome = m.simulator.simulate(x, u, rng_k, mismatch, sample_droplets=vision)
        t_sim = time.perf_counter() - t1

        # ---- measurement
        t_render = 0.0
        if vision:
            t2 = time.perf_counter()
            image, _ = m.renderer.render(outcome, np.random.default_rng([seed, k, 1]))
            t_render = time.perf_counter() - t2
            meas = m.verifier.measure(image, ppm)
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
    return logs


def run_passes_slow(conditions_list, modules: Modules, seed: int, mismatch: str, c_req: float,
                    n_passes: int = 3, render: bool = True,
                    feedback: BiasFeedback | None = None,
                    reset_bias_between_passes: bool = True) -> list[SegmentLog]:
    """Policy "fast_slow": `n_passes` consecutive passes with fast feedback inside each and a
    model refit between passes. Returns all logs (policy="fast_slow", pass_id
    "fast_slow_p{i}", extra["slow_pass"]=i).

    `conditions_list` is either a list of passes (list[list[Conditions]], pass i uses entry
    i mod len) or a single list[Conditions] reused for every pass.

    Slow-loop records: for each segment (x, u, Outcome) where the Outcome carries the
    MEASURED coverage (and measured stains/cm2 if available) and all other fields
    (deposition, drift, displacement) from the simulator outcome, because the card only
    measures coverage. Records are accumulated in a buffer.

    CONTRACT ASSUMPTION: `model.fit(records)` is incremental: it receives ONLY the new
    records of the pass just finished and must append them to whatever training data it
    already holds (the original offline dataset plus earlier slow-loop records) before
    refitting. The harness keeps the full buffer (`logs`' source) but never resends it.

    Each pass uses seed `seed + 1009 * i` (new hidden-noise realization per pass).
    The bias is reset between passes by default because it is defined relative to the
    model that produced the predictions, which has just changed.
    """
    passes = conditions_list
    if len(passes) and isinstance(passes[0], Conditions):
        passes = [passes]
    fb = feedback or BiasFeedback()
    buffer: list[tuple[Conditions, Setting, Outcome]] = []
    all_logs: list[SegmentLog] = []
    for i in range(n_passes):
        if reset_bias_between_passes or i == 0:
            fb.reset()
        logs = run_pass(list(passes[i % len(passes)]), "fast_slow", modules, seed + 1009 * i,
                        mismatch, c_req, f"fast_slow_p{i}", feedback=fb, render=render)
        new = []
        for lg in logs:
            lg.extra["slow_pass"] = i
            o = lg.outcome
            stains = lg.meas.stains_per_cm2 if lg.meas.stains_per_cm2 is not None else o.stains_per_cm2
            new.append((lg.x, lg.u, Outcome(lg.meas.coverage_pct, o.deposition_ul_cm2,
                                            o.drift_frac, o.displacement_mm, stains, None)))
        buffer.extend(new)
        if i < n_passes - 1:
            modules.model.fit(new)
        all_logs.extend(logs)
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
