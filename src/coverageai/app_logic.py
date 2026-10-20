"""Non-UI logic behind the Streamlit demo (WP-22): module construction, pass runner
wrapper, KPI computation, field-strip layout, chart frames, nozzle-explorer curves.

Everything here is plain Python/pandas so it can be unit-tested without Streamlit
(tests/test_app_logic.py). ``app/main.py`` only handles widgets and layout.

Module construction. ``build_modules_adapter`` first tries the integration module
(``coverageai.integration.build_modules``). If that is not importable it builds a
working stack from the real simulator, selector and renderer, with the nominal
simulator (no hidden mismatch, mean spread factor) standing in for the learned model.
Swap-in therefore needs no change in the UI.

Honesty note shown in the app: the plant is the simulator, so everything on screen is
simulation-based evidence, not field validation.
"""
from __future__ import annotations

import copy
import dataclasses
import importlib
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from coverageai import scenarios
from coverageai.config import px_per_mm, spec
from coverageai.contracts import (Conditions, Derived, Outcome, Prediction, Setting,
                                  coverage_class)
from coverageai.loop import (FORECAST_POLICIES, Modules, POLICIES, default_mock_modules, run_pass,
                             run_passes_slow, segment_rng, summarize)

MISMATCH_LEVELS = ("none", "low", "med", "high")
APP_POLICIES = ("open", "fast", "fast_slow", "oracle", "oracle_card", "best_fixed")
N_LEARN_PASSES = 3          # earlier "learning" passes before the displayed fast_slow pass
MODEL_CHOICES = ("oracle_nominal", "spray", "mock")
VERIFIER_CHOICES = ("perfect", "noisy", "b0", "b1")
# Opt-in controller options (all off by default; see config/spec.yaml selector.*_options and
# coverageai.forecast). The grids are the ones the evaluation scripts use
# (experiments/eval_speed_control.py, experiments/eval_boom_control.py); the selector further
# restricts them to +-band around the field conditions and to the legal ranges.
SPEED_OPTIONS_KMH = tuple(float(v) for v in range(8, 31, 2))
BOOM_HEIGHT_OPTIONS_M = (0.30, 0.40, 0.50, 0.60, 0.75)
FORECAST_CHOICES = ("off", "last", "ewma", "ar1", "ar1+gust")
CLASS_ORDER = ("acceptable", "marginal", "critical")
CLASS_COLORS = {"acceptable": "#3ddc97", "marginal": "#f5b942", "critical": "#ef5b5b",
                "pending": "#2b3340"}


# ----------------------------------------------------------------- modules

class NominalSimModel:
    """Model protocol backed by the simulator with NO hidden mismatch and the mean spread
    factor. It is what a perfect nominal-physics model would predict (decision: option
    ``oracle_nominal``). Interval: +-``interval_pp`` around the coverage."""

    def __init__(self, derive_fn: Callable | None = None, interval_pp: float = 3.0):
        from coverageai.simulator import SpraySimulator
        sf = spec()["physics"]["spread_factor"]
        mid = float(np.mean(sf))
        self.sim = SpraySimulator(derive_fn=derive_fn, physics_overrides={"spread_factor": [mid, mid]})
        self.interval_pp = float(interval_pp)

    def predict_batch(self, xs, us) -> list[Prediction]:
        a = self.sim.simulate_arrays(list(xs), list(us), np.random.default_rng(0), "none")
        c = a["coverage_pct"]
        return [Prediction(float(c[i]), float(max(c[i] - self.interval_pp, 0.0)),
                           float(c[i] + self.interval_pp), float(a["deposition_ul_cm2"][i]),
                           float(a["drift_frac"][i]), float(a["displacement_mm"][i]))
                for i in range(len(c))]

    def predict(self, x, u) -> Prediction:
        return self.predict_batch([x], [u])[0]

    def fit(self, records) -> None:  # nominal physics: nothing to learn
        pass


def _find_verifier_class(module_name: str):
    mod = importlib.import_module(module_name)
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ == mod.__name__ and hasattr(obj, "measure"):
            return obj
    raise ImportError(f"no verifier class with measure() in {module_name}")


def _fallback_modules(model: str, verifier: str) -> Modules:
    """Real simulator/selector/renderer with the nominal model; mocks only if asked."""
    if model == "mock":
        m = default_mock_modules()
        if verifier != "perfect":
            raise ValueError("mock model supports only the perfect verifier")
        return dataclasses.replace(m, verifier=None)
    from coverageai import catalog
    from coverageai.selector import ConstrainedSelector
    from coverageai.simulator import SpraySimulator
    from coverageai.vision.render import WSPRenderer
    ver = None
    if verifier == "b0":
        ver = _find_verifier_class("coverageai.vision.classical")()
    elif verifier == "b1":
        ver = _find_verifier_class("coverageai.vision.unet")()
    elif verifier == "noisy":
        raise ValueError("verifier 'noisy' needs integration.build_modules")
    elif verifier != "perfect":
        raise ValueError(f"unknown verifier {verifier!r}")
    if model == "spray":
        from coverageai.model.predictor import SprayModel
        mdl = SprayModel.load()      # models/spray_model_v1.joblib (FileNotFoundError if untrained)
    else:
        mdl = NominalSimModel(derive_fn=catalog.derive)
    return Modules(SpraySimulator(), mdl, ConstrainedSelector(), ver, WSPRenderer(), catalog.derive)


def build_modules_adapter(model: str = "oracle_nominal", verifier: str = "perfect") -> Modules:
    """integration.build_modules if present, else the fallback stack above."""
    try:
        from coverageai.integration import build_modules  # type: ignore
    except ImportError:
        return _fallback_modules(model, verifier)
    return build_modules(model=model, verifier=verifier)


class RecordingSelector:
    """Transparent proxy that remembers the info dict of every selection, keyed by the
    (conditions, chosen setting) pair, so the UI can show the top-5 table for a segment
    (SegmentLog itself only stores a scalar summary)."""

    def __init__(self, inner):
        self.inner = inner
        self.records: dict[tuple[Conditions, Setting], dict] = {}

    def select(self, x, model, bias, c_req):
        u, info = self.inner.select(x, model, bias, c_req)
        self.records[(x, u)] = info
        return u, info

    def __getattr__(self, name):
        return getattr(self.inner, name)


# ------------------------------------------------------------- pass runner

@dataclass
class PassResult:
    logs: list
    conds: list
    policy: str
    seed: int
    mismatch: str
    c_req: float
    lam: float | None
    modules: Modules
    records: dict
    fixed_setting: Setting | None = None
    used_vision: bool = False
    elapsed_s: float = 0.0
    label: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.conds)

    def info_for(self, k: int) -> dict | None:
        lg = self.logs[k]
        return self.records.get((lg.x, lg.u))


def make_conditions(preset: str, seed: int, n_segments: int | None = None) -> list[Conditions]:
    sp = scenarios.get(preset)
    if n_segments is not None:
        sp = dataclasses.replace(sp, segments=int(n_segments))
    return scenarios.generate(sp, int(seed))


def best_fixed_for(modules: Modules, conds: Sequence[Conditions], c_req: float) -> Setting:
    """Best fixed baseline: highest model-predicted success rate over the scenario, ties by
    lowest drift (selector.best_fixed_setting)."""
    from coverageai.selector import best_fixed_setting
    return best_fixed_setting(list(conds), modules.model, c_req, derive_fn=modules.derive_fn)


def run_pass_wrapper(modules: Modules, conds: Sequence[Conditions], policy: str, seed: int,
                     mismatch: str, c_req: float, lam: float | None = None,
                     fixed_setting: Setting | None = None, render: bool | None = None,
                     pass_id: str | None = None, n_learn: int = N_LEARN_PASSES,
                     speed_control: bool = False, boom_control: bool = False,
                     forecaster: str | None = None) -> PassResult:
    """Run one pass and keep what the UI needs (selector records, timing).

    ``fast_slow``: runs ``n_learn`` earlier passes (seeds ``seed + 1009 i``, no card
    rendering) through ``loop.run_passes_slow`` so the slow loop learns a residual
    correction, then the displayed pass (seed ``seed + 1009 n_learn``) with the fitted
    ``ResidualCorrectedModel`` as ``modules.model``. ``meta["passes"]`` holds a per-pass
    summary (the last row is the displayed pass).

    Starts a simulator pass (persistent canopy/wear + AR(1) gusts) when the simulator
    supports it, applies ``lam`` to the selector for the duration of the call, and
    computes the best-fixed setting if the policy needs one and none was given.

    Opt-in options, all off by default (then nothing below changes the run): ``speed_control`` /
    ``boom_control`` let the selector also choose travel speed / boom height from
    ``SPEED_OPTIONS_KMH`` / ``BOOM_HEIGHT_OPTIONS_M`` (set on the selector for the duration of the
    call, like ``lam``; best_fixed keeps its single nozzle setting). ``forecaster`` (one of
    ``FORECAST_CHOICES``; "off"/None = none) turns on the short-horizon wind forecaster for the
    policies that support it (open, fast, fast_slow; ignored by the others). For fast_slow one
    forecaster object is shared by the learning passes and the displayed pass.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    sel = modules.selector
    rec = sel if isinstance(sel, RecordingSelector) else RecordingSelector(sel)
    mods = dataclasses.replace(modules, selector=rec)
    if render is None:
        render = mods.verifier is not None
    # An image-free verifier (NoisyVerifier.measure_outcome) never reads rendered cards in the
    # loop (loop.run_pass: ``direct``), so it must not be reported as image analysis.
    vision = bool(render and mods.verifier is not None and mods.renderer is not None
                  and not hasattr(mods.verifier, "measure_outcome"))
    if policy == "best_fixed" and fixed_setting is None:
        fixed_setting = best_fixed_for(mods, conds, c_req)
    old_lam = getattr(sel, "lam", None)
    if lam is not None and old_lam is not None:
        sel.lam = float(lam)
    inner = sel.inner if isinstance(sel, RecordingSelector) else sel
    saved_opts: dict[str, Any] = {}
    for on, attr, grid, what in ((speed_control, "speed_options_kmh", SPEED_OPTIONS_KMH, "travel-speed"),
                                 (boom_control, "boom_height_options_m", BOOM_HEIGHT_OPTIONS_M, "boom-height")):
        if on:
            if not hasattr(inner, attr):
                raise ValueError(f"this selector does not support {what} control")
            saved_opts[attr] = getattr(inner, attr)
            setattr(inner, attr, list(grid))
    fc = _make_forecaster(forecaster) if policy in FORECAST_POLICIES else None
    t0 = time.perf_counter()
    meta: dict = {}
    run_seed = int(seed)
    try:
        if policy == "fast_slow":
            run_seed, passes = _learn_between_passes(mods, conds, int(seed), mismatch,
                                                     float(c_req), int(n_learn), forecaster=fc)
            meta["passes"] = passes
        if hasattr(mods.simulator, "begin_pass"):
            mods.simulator.begin_pass(run_seed, mismatch)
        logs = run_pass(list(conds), policy, mods, run_seed, mismatch, float(c_req),
                        pass_id or f"{policy}_s{run_seed}", render=vision,
                        fixed_setting=fixed_setting, forecaster=fc)
    finally:
        if lam is not None and old_lam is not None:
            sel.lam = old_lam
        for attr, old in saved_opts.items():
            setattr(inner, attr, old)
    if policy == "fast_slow":
        meta["passes"].append(_pass_summary(len(meta["passes"]) + 1, logs, float(c_req),
                                            mods.model, displayed=True))
    return PassResult(logs=logs, conds=list(conds), policy=policy, seed=run_seed,
                      mismatch=mismatch, c_req=float(c_req), lam=lam, modules=mods,
                      records=rec.records, fixed_setting=fixed_setting, used_vision=vision,
                      elapsed_s=time.perf_counter() - t0, meta=meta)


def _make_forecaster(name: str | None):
    """None/"off" -> None; otherwise a fresh WindForecaster (names: FORECAST_CHOICES)."""
    if name in (None, "off"):
        return None
    if name not in FORECAST_CHOICES:
        raise ValueError(f"unknown forecaster {name!r}; choose from {FORECAST_CHOICES}")
    from coverageai.forecast import make_forecaster
    return make_forecaster(name)


def _pass_summary(i: int, logs: Sequence, c_req: float, model, displayed: bool = False) -> dict:
    cov = np.array([l.outcome.coverage_pct for l in logs])
    err = np.array([abs(l.outcome.coverage_pct - l.corrected_pred_cov) for l in logs])
    return {"pass": i, "met_pct": 100.0 * float((cov >= c_req).mean()),
            "mean_error_pp": float(err.mean()),
            "slow_offset_pp": float(getattr(model, "global_corr", 0.0)),
            "displayed": displayed}


def _learn_between_passes(mods: Modules, conds: Sequence[Conditions], seed: int, mismatch: str,
                          c_req: float, n_learn: int, forecaster=None) -> tuple[int, list[dict]]:
    """Slow loop: ``n_learn`` passes (no rendering) through ``run_passes_slow``, then one
    more fit on the last pass so ``mods.model`` is the ResidualCorrectedModel as it stands
    at the displayed pass. Returns (seed of the displayed pass, summaries of the earlier
    passes). Mutates only ``mods`` (a private copy), never the cached module stack."""
    from coverageai.feedback import ResidualCorrectedModel
    corr = ResidualCorrectedModel(mods.model, mods.derive_fn)
    passes: list[dict] = []
    if n_learn > 0:
        logs = run_passes_slow(list(conds), mods, seed, mismatch, c_req, n_passes=n_learn,
                               render=False, correction=corr, forecaster=forecaster)
        by_pass: dict[int, list] = {}
        for lg in logs:
            by_pass.setdefault(int(lg.extra["slow_pass"]), []).append(lg)
        for i in sorted(by_pass):
            passes.append(_pass_summary(i + 1, by_pass[i], c_req, corr))
            passes[-1]["slow_offset_pp"] = float(by_pass[i][0].extra.get("slow_global_corr", 0.0))
        # run_passes_slow does not refit after its last pass: do it here
        last = by_pass[max(by_pass)]
        corr.fit([(lg.x, lg.u, Outcome(lg.meas.coverage_pct, lg.outcome.deposition_ul_cm2,
                                       lg.outcome.drift_frac, lg.outcome.displacement_mm,
                                       lg.meas.stains_per_cm2 if lg.meas.stains_per_cm2 is not None
                                       else lg.outcome.stains_per_cm2, None)) for lg in last])
    mods.model = corr
    return seed + 1009 * n_learn, passes


# --------------------------------------------------------------------- KPIs

def compute_kpis(logs: Sequence, c_req: float, upto: int | None = None) -> dict:
    """Header KPIs over the first ``upto + 1`` segments (all if None).

    Requirement-met uses TRUE outcome coverage (what the field would get); the measured
    mean is what the controller saw.
    """
    ls = list(logs if upto is None else logs[: upto + 1])
    if not ls:
        return {"n": 0, "n_met": 0, "frac_met": 0.0, "mean_cov": 0.0, "mean_meas": 0.0,
                "mean_gpa": 0.0, "mean_drift": 0.0, "n_retreat": 0, "n_infeasible": 0}
    s = summarize(ls, c_req)
    true_cov = np.array([l.outcome.coverage_pct for l in ls])
    return {
        "n": len(ls), "n_met": int((true_cov >= c_req).sum()), "frac_met": s["frac_met"],
        "mean_cov": float(true_cov.mean()),
        "mean_meas": float(np.mean([l.meas.coverage_pct for l in ls])),
        "mean_gpa": s["mean_volume_gpa"], "mean_drift": s["mean_drift_frac"],
        "n_retreat": s["n_retreat"], "n_infeasible": s["n_infeasible"],
    }


# ---------------------------------------------------------------- field strip

def lap_layout(conds: Sequence[Conditions], default_lap_len: int | None = None
               ) -> list[tuple[int, int]]:
    """(row, col) of each segment for a serpentine (back-and-forth) layout.

    A new lap starts when the heading changes; if the heading never changes, laps are
    chunks of ``default_lap_len`` (spec pass.segments_per_lap or 10). Odd laps run right
    to left, like the sprayer returning along the next swath.
    """
    n = len(conds)
    laps = np.zeros(n, int)
    for k in range(1, n):
        laps[k] = laps[k - 1] + (abs(((conds[k].heading_deg - conds[k - 1].heading_deg + 180) % 360) - 180) > 1e-6)
    if laps.max(initial=0) == 0 and n > 1:
        L = default_lap_len or 10
        laps = np.arange(n) // L
    pos = np.zeros(n, int)
    for lap in np.unique(laps):
        idx = np.flatnonzero(laps == lap)
        pos[idx] = np.arange(len(idx))
    out = []
    for k in range(n):
        idx = np.flatnonzero(laps == laps[k])
        c = int(pos[k]) if laps[k] % 2 == 0 else int(len(idx) - 1 - pos[k])
        out.append((int(laps[k]), c))
    return out


def field_strip_data(conds: Sequence[Conditions], logs: Sequence, cursor: int,
                     basis: str = "measured", c_req: float | None = None) -> pd.DataFrame:
    """One row per segment for the field-strip chart.

    ``status`` is the coverage class (acceptable/marginal/critical) for segments up to and
    including ``cursor`` and "pending" afterwards. ``basis``: "measured" uses the card
    class, "true" classifies the simulator outcome. ``retreat`` marks flagged segments.
    """
    cc = spec()["coverage_classes"]
    layout = lap_layout(conds)
    rows = []
    for k, (row, col) in enumerate(layout):
        status, cov, retreat, meas_cov, true_cov = "pending", np.nan, False, np.nan, np.nan
        if k < len(logs) and k <= cursor:
            lg = logs[k]
            meas_cov, true_cov = lg.meas.coverage_pct, lg.outcome.coverage_pct
            if basis == "true":
                cov = true_cov
                status = coverage_class(cov, cc["acceptable_min"], cc["marginal_min"])
            else:
                cov = meas_cov
                status = lg.meas.coverage_class
            retreat = bool(lg.flagged_retreat)
        rows.append({"k": k, "seg": k + 1, "lap": row, "row": row, "col": col, "status": status,
                     "coverage": cov, "measured": meas_cov, "true": true_cov, "retreat": retreat,
                     "current": k == cursor,
                     "heading": conds[k].heading_deg})
    return pd.DataFrame(rows)


# ----------------------------------------------------------- series / tables

def timeseries_frame(logs: Sequence, upto: int | None = None) -> pd.DataFrame:
    ls = list(logs if upto is None else logs[: upto + 1])
    return pd.DataFrame({
        "k": [l.k + 1 for l in ls],
        "predicted": [l.pred.coverage_pct for l in ls],
        "corrected": [l.corrected_pred_cov for l in ls],
        "measured": [l.meas.coverage_pct for l in ls],
        "true": [l.outcome.coverage_pct for l in ls],
        "bias": [l.bias for l in ls],
        "bias_after": [l.extra.get("bias_after", l.bias) for l in ls],
        "residual": [l.residual for l in ls],
    })


def top_candidates_frame(info: dict | None) -> pd.DataFrame:
    """Top-5 candidate table from the selector info (empty frame if unavailable)."""
    cols = ["Setting", "Pred %", "Corr. %", "Drift", "GPA", "Score", "Feasible"]
    if not info or not info.get("top"):
        return pd.DataFrame(columns=cols)
    rows = []
    for c in info["top"]:
        extra = "".join([f", {c['travel_speed_kmh']:g} km/h" if c.get("travel_speed_kmh") is not None else "",
                         f", boom {c['boom_height_m']:.2f} m" if c.get("boom_height_m") is not None else ""])
        rows.append({"Setting": f"{c['family']}-{c['size']} @ {c['pressure_psi']:g} psi{extra}",
                     "Pred %": round(c["pred_cov"], 1),
                     "Corr. %": round(c["corrected_cov"], 1),
                     "Drift": round(c["drift"], 4), "GPA": round(c["gpa"], 2),
                     "Score": None if c["score"] is None else round(c["score"], 3),
                     "Feasible": bool(c["feasible"])})
    return pd.DataFrame(rows, columns=cols)


def explain_choice(res: PassResult, k: int) -> list[str]:
    """Short plain-language reasons for the setting chosen at segment k."""
    lg, info = res.logs[k], res.info_for(k)
    out: list[str] = []
    if res.policy == "best_fixed":
        return ["Fixed setting for the whole pass: the single catalog setting with the highest "
                "predicted requirement-met rate over this scenario (ties: lowest drift)."]
    if res.policy == "oracle":
        out.append("Oracle: the selector queried the true simulator, including the hidden "
                   "mismatch realization of this segment (gusts, canopy, nozzle wear) but not the "
                   "random card-to-card spread. Fair upper bound, not deployable.")
    if res.policy == "oracle_card":
        out.append("Card-aware oracle: the selector knew the hidden mismatch AND this card's random "
                   "spread. Unreachable bound, not deployable.")
    if res.policy == "fast_slow":
        n_prev = max(len(res.meta.get("passes", [])) - 1, 0)
        out.append(f"Learning between passes: a correction learned from {n_prev} earlier pass(es) "
                   "was added to the model before this pass.")
    if not info:
        return out + ["No selector record available for this segment."]
    if "n_candidates" in info:
        out.append(f"{info['n_feasible']} of {info['n_candidates']} candidates meet "
                   f"corrected coverage - margin >= C_req = {res.c_req:g}%.")
    if info.get("infeasible"):
        out.append("No candidate is feasible, so the selector fell back to the highest "
                   "corrected coverage and flagged the segment.")
    elif info.get("top"):
        lam = info.get("lam", res.lam)
        lam_txt = f" (lambda = {lam:g})" if lam is not None else ""
        out.append("Among feasible candidates it picked the lowest score = lambda x drift "
                   f"+ (1 - lambda) x volume{lam_txt}.")
    if abs(lg.bias) > 1e-9:
        out.append(f"Feedback bias {lg.bias:+.1f} pp was added to the model's coverage "
                   "(fast loop, eq. 3).")
    if wind_limit_exceeded(lg.x, info):
        out.append("Wind is above the label limit; application would normally be prohibited.")
    return out


def wind_limit_exceeded(x: Conditions, info: dict | None = None) -> bool:
    if info and "wind_limit_exceeded" in info:
        return bool(info["wind_limit_exceeded"])
    return bool(x.wind_speed_ms > spec()["wind_limit_ms"])


def wind_arrow_deg(x: Conditions) -> float:
    """Angle of the wind arrow clockwise from the vehicle's forward direction (the
    direction the air moves TOWARD, decision D1), wrapped to (-180, 180]."""
    a = (x.wind_dir_deg - x.heading_deg + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else float(a)


def wind_arrow_svg(x: Conditions, size: int = 150) -> str:
    """Inline SVG: vehicle heading is up, the arrow shows where the wind is going."""
    ang = wind_arrow_deg(x)
    c = size / 2
    L = 18 + 8 * min(x.wind_speed_ms, 7.0)           # arrow length grows with speed
    L = min(L, size / 2 - 10)
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" '
        'xmlns="http://www.w3.org/2000/svg">'
        f'<circle cx="{c}" cy="{c}" r="{c - 4}" fill="#10151c" stroke="#2b3340"/>'
        f'<polygon points="{c},{c - 28} {c - 9},{c - 8} {c + 9},{c - 8}" fill="#8a94a6"/>'
        f'<rect x="{c - 7}" y="{c - 8}" width="14" height="28" rx="3" fill="#5b6575"/>'
        f'<g transform="rotate({ang:.1f} {c} {c})">'
        f'<line x1="{c}" y1="{c}" x2="{c}" y2="{c - L:.1f}" stroke="#4da3ff" stroke-width="4" '
        'stroke-linecap="round"/>'
        f'<polygon points="{c},{c - L - 9:.1f} {c - 8},{c - L + 4:.1f} {c + 8},{c - L + 4:.1f}" '
        'fill="#4da3ff"/></g>'
        f'<text x="{c}" y="14" fill="#8a94a6" font-size="10" text-anchor="middle">heading</text>'
        '</svg>')


# ------------------------------------------------------ replay / card render

@dataclass
class CardView:
    image: np.ndarray
    gt_mask: np.ndarray
    pred_mask: np.ndarray | None
    outcome: Outcome
    consistent: bool
    meas: Any | None


def replay_card(res: PassResult, k: int, degrade: bool = True) -> CardView:
    """Re-create segment k's WSP card: replay the simulator's hidden state on a copy of
    the simulator (persistent mismatch + AR(1) gusts depend only on the rng stream, not
    on the settings), sample droplets for segment k with the same per-segment rng the
    loop used, render with the loop's renderer rng, and (if a verifier exists) measure.
    Rendering costs ~0.5-0.7 s, so the UI calls this only for the selected segment."""
    m = res.modules
    sim = copy.deepcopy(m.simulator)
    if hasattr(sim, "begin_pass"):
        sim.begin_pass(res.seed, res.mismatch)
        if res.mismatch != "none":
            for j in range(k):          # advance the gust AR(1) state
                sim.simulate(res.conds[j], res.logs[j].u, segment_rng(res.seed, j), res.mismatch)
    lg = res.logs[k]
    outcome = sim.simulate(lg.x, lg.u, segment_rng(res.seed, k), res.mismatch,
                           sample_droplets=True)
    consistent = abs(outcome.coverage_pct - lg.outcome.coverage_pct) < 1e-6
    image, gt = m.renderer.render(outcome, np.random.default_rng([res.seed, k, 1]),
                                  degrade=degrade)
    meas, pred_mask = None, None
    if m.verifier is not None:
        ppm = px_per_mm()
        meas = m.verifier.measure(image, ppm)
        if hasattr(m.verifier, "segment"):
            try:
                try:
                    seg = m.verifier.segment(image, ppm)
                except TypeError:          # segment(image) signature (ClassicalVerifier)
                    seg = m.verifier.segment(image)
                seg = seg[0] if isinstance(seg, tuple) else seg
                pred_mask = np.asarray(seg).astype(bool)
            except Exception:  # overlay is optional
                pred_mask = None
    return CardView(image, np.asarray(gt, bool), pred_mask, outcome, consistent, meas)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(255, 64, 64), alpha: float = 0.55,
                 max_width: int = 1100) -> np.ndarray:
    """Blend a boolean mask over an RGB image and downscale for display."""
    import cv2
    img = image.copy()
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    col = np.array(color, float)
    img[mask] = ((1 - alpha) * img[mask] + alpha * col).astype(np.uint8)
    if img.shape[1] > max_width:
        s = max_width / img.shape[1]
        img = cv2.resize(img, (max_width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return img


# --------------------------------------------------------- policy comparison

def compare_policies(modules: Modules, conds: Sequence[Conditions], seed: int, mismatch: str,
                     c_req: float, lam: float | None = None,
                     policies: Sequence[str] = ("open", "fast", "fast_slow", "oracle"),
                     render: bool = False, fixed_setting: Setting | None = None,
                     n_learn: int = N_LEARN_PASSES, speed_control: bool = False,
                     boom_control: bool = False, forecaster: str | None = None
                     ) -> dict[str, PassResult]:
    """Run each policy on the same conditions, seed and hidden mismatch realization."""
    out = {}
    for p in policies:
        out[p] = run_pass_wrapper(modules, conds, p, seed, mismatch, c_req, lam,
                                  fixed_setting=fixed_setting if p == "best_fixed" else None,
                                  render=render, n_learn=n_learn, speed_control=speed_control,
                                  boom_control=boom_control, forecaster=forecaster)
    return out


def comparison_table(results: dict[str, PassResult]) -> pd.DataFrame:
    rows = []
    for p, r in results.items():
        s = summarize(r.logs, r.c_req)
        rows.append({"policy": p, "requirement met (%)": 100 * s["frac_met"],
                     "mean shortfall (pp)": s["mean_shortfall_pp"],
                     "mean volume (GPA)": s["mean_volume_gpa"],
                     "total volume (GPA)": s["total_volume_gpa"],
                     "mean drift (frac)": s["mean_drift_frac"],
                     "re-treatment flags": s["n_retreat"], "infeasible": s["n_infeasible"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------- nozzle explorer

_nominal: NominalSimModel | None = None


def _nominal_sim() -> NominalSimModel:
    global _nominal
    if _nominal is None:
        from coverageai import catalog
        _nominal = NominalSimModel(derive_fn=catalog.derive)
    return _nominal


def nozzle_pressure_bounds(family: str, size: str) -> tuple[float, float]:
    from coverageai import catalog
    lo, hi = catalog.pressure_range(family, size)
    r = spec()["pressure_range_psi"][family]
    return max(lo, float(r[0])), min(hi, float(r[1]))


def nozzle_point(setting: Setting, x: Conditions) -> dict:
    """Catalog-derived quantities and nominal simulator prediction at one point."""
    from coverageai import catalog
    d = catalog.derive(x, setting)
    p = _nominal_sim().predict(x, setting)
    return {"derived": d, "pred": p}


def _arrays(xs, us) -> dict[str, np.ndarray]:
    return _nominal_sim().sim.simulate_arrays(xs, us, np.random.default_rng(0), "none")


def coverage_vs_pressure(family: str, size: str, x: Conditions, n: int = 40) -> pd.DataFrame:
    lo, hi = nozzle_pressure_bounds(family, size)
    ps = np.linspace(lo, hi, n)
    us = [Setting(family, size, float(p)) for p in ps]
    from coverageai import catalog
    a = _arrays([x] * n, us)
    gpa = [catalog.derive(x, u).app_rate_gpa for u in us]
    cls = [catalog.derive(x, u).droplet_class for u in us]
    return pd.DataFrame({"pressure_psi": ps, "coverage_pct": a["coverage_pct"],
                         "drift_frac": a["drift_frac"], "gpa": gpa, "droplet_class": cls})


def drift_vs_wind(setting: Setting, x: Conditions, n: int = 36) -> pd.DataFrame:
    """Crosswind sweep: the heading is set so that the whole wind speed is crosswind."""
    ws = np.linspace(0.0, spec()["ranges"]["wind_speed_ms"][1], n)
    xs = [dataclasses.replace(x, wind_speed_ms=float(w), wind_dir_deg=90.0, heading_deg=0.0)
          for w in ws]
    a = _arrays(xs, [setting] * n)
    return pd.DataFrame({"wind_ms": ws, "drift_frac": a["drift_frac"],
                         "coverage_pct": a["coverage_pct"],
                         "displacement_mm": a["displacement_mm"]})


def coverage_grid(family: str, size: str, x: Conditions, n_p: int = 24, n_w: int = 24) -> pd.DataFrame:
    """Coverage and drift over a pressure x crosswind grid (Emery-style coverage map)."""
    lo, hi = nozzle_pressure_bounds(family, size)
    ps = np.linspace(lo, hi, n_p)
    ws = np.linspace(0.0, spec()["ranges"]["wind_speed_ms"][1], n_w)
    xs, us, pp, ww = [], [], [], []
    for p in ps:
        for w in ws:
            xs.append(dataclasses.replace(x, wind_speed_ms=float(w), wind_dir_deg=90.0,
                                          heading_deg=0.0))
            us.append(Setting(family, size, float(p)))
            pp.append(p)
            ww.append(w)
    a = _arrays(xs, us)
    return pd.DataFrame({"pressure_psi": pp, "wind_ms": ww, "coverage_pct": a["coverage_pct"],
                         "drift_frac": a["drift_frac"]})


def pattern_overlap(spray_angle_deg: float, height_m: float, spacing_m: float | None = None,
                    n_nozzles: int = 9) -> dict:
    """Illustrative geometric spray pattern (NOT simulator output): each nozzle puts a raised
    cosine volume profile over the fan width W = 2 H tan(angle / 2); neighboring nozzles
    at ``spacing_m`` overlap. Returns the single-nozzle cross-section, the summed boom
    profile, the uniformity coefficient 1 - CV over the central swath and the single-nozzle
    coverage width, mirroring the readouts of Emery's MATLAB pattern model."""
    spacing = spacing_m or spec()["boom"]["nozzle_spacing_in"] * 0.0254
    W = 2.0 * height_m * np.tan(np.radians(spray_angle_deg) / 2.0)
    xs = np.linspace(-0.5 * n_nozzles * spacing, 0.5 * n_nozzles * spacing, 1200)
    centers = (np.arange(n_nozzles) - (n_nozzles - 1) / 2) * spacing

    def prof(x, c):
        u = (x - c) / W
        return np.where(np.abs(u) < 0.5, np.cos(np.pi * u) ** 2, 0.0)

    single = prof(xs, 0.0)
    total = sum(prof(xs, c) for c in centers)
    mid = np.abs(xs) <= spacing / 2 + 1e-9     # one spacing period at the boom center
    seg = total[mid]
    cv = float(seg.std() / seg.mean()) if seg.mean() > 0 else float("nan")
    return {"x_m": xs, "single": single, "total": total, "fan_width_m": float(W),
            "uniformity": float(1.0 - cv) if np.isfinite(cv) else float("nan"),
            "cv": cv, "peak": float(total[mid].max())}
