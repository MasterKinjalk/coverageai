"""Non-UI logic behind the Streamlit demo (WP-22): module construction, pass runner
wrapper, KPI computation, field-strip layout, chart frames, nozzle-explorer curves.

Everything here is plain Python/pandas so it can be unit-tested without Streamlit
(tests/test_app_logic.py). ``app/main.py`` only handles widgets and layout.

Module construction is ``coverageai.integration.build_modules`` (``build_modules_adapter``
is the app's name for it).

One run, one record. ``run_pass_wrapper`` returns a ``PassResult`` that carries everything the
UI shows: the segment logs, the selector's candidate tables, card thumbnails captured while the
loop read them (``cards``), the nozzle held for the field (``meta["nozzle"]``), and, derived on
demand from the logs, the operator recommendations (``recommendations``), the per-section
decision table (``decision_frame``) and the feedback-loop numbers (``loop_numbers``) that both
the animation and the charts below it read.

Honesty note shown in the app: the plant is the simulator, so everything on screen is
simulation-based evidence, not field validation.

Shared stack: Streamlit caches one module stack for all browser sessions (``st.cache_resource``)
and every run writes its options onto that stack's selector and restarts its simulator's pass
state. ``_STACK_LOCK`` (a re-entrant lock) is held for the whole mutate-run-restore part of
``run_pass_wrapper`` and across ``compare_policies`` / ``compare_seeds``, so two sessions never
mix their state; a second session waits until the first run is done.
"""
from __future__ import annotations

import base64
import copy
import re
import dataclasses
import inspect
import os
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from coverageai import scenarios
from coverageai.config import px_per_mm, spec
from coverageai.feedback import FALLBACK_DAMPING
from coverageai.contracts import (Conditions, Outcome, Prediction, Setting,
                                  coverage_class)
from coverageai.loop import (FORECAST_POLICIES, Modules, POLICIES, logs_to_frame, run_pass,
                             run_passes_slow, segment_rng, summarize, verified_records)

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
CLASS_COLORS = {"acceptable": "#3ddc97", "marginal": "#f5b942", "critical": "#ef5b5b",
                "pending": "#2b3340"}
# Held while a run mutates the shared (cached) module stack; re-entrant because compare_policies
# and compare_seeds hold it across their loops of run_pass_wrapper calls.
_STACK_LOCK = threading.RLock()


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


def build_modules_adapter(model: str = "oracle_nominal", verifier: str = "perfect",
                          calibrate_reader: bool | str = False) -> Modules:
    """The module stack (simulator, model, selector, verifier, renderer). ``calibrate_reader``
    (opt-in, D30): remove the card reader's known mean error per coverage band (see
    ``integration.build_modules``)."""
    from coverageai.integration import build_modules
    return build_modules(model=model, verifier=verifier, calibrate_reader=calibrate_reader)


def reader_calibration_available(verifier: str) -> bool:
    """Whether a calibration table exists for this card reader (D30): always for the noisy reader,
    for an image reader only once it has been measured on closed-loop cards, never for perfect."""
    from coverageai.integration import reader_calibration_available as available
    return available(verifier)


class RecordingSelector:
    """Transparent proxy that remembers the info dict of every selection, so the UI can show the
    top-5 table for a segment (SegmentLog itself only stores a scalar summary).

    ``records`` keys each info by the (conditions, chosen setting) pair the selector was called
    with. ``calls`` keeps every call in order as (section tag, conditions, setting, info); the
    tag is the section count of ``tracker`` at call time (None without a tracker), which lets
    ``run_pass_wrapper`` find the selection of every section even when the selector was called
    with forecast conditions instead of the logged ones. Reads are forwarded to ``inner``;
    set attributes on ``inner``."""

    def __init__(self, inner):
        self.inner = inner
        self.records: dict[tuple[Conditions, Setting], dict] = {}
        self.calls: list[tuple[int | None, Conditions, Setting, dict]] = []
        self.tracker: _Sections | None = None

    def select(self, x, model, bias, c_req):
        u, info = self.inner.select(x, model, bias, c_req)
        self.records[(x, u)] = info
        t = self.tracker
        self.calls.append((None if t is None else t.started, x, u, info))
        return u, info

    def __getattr__(self, name):
        if name == "inner" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.inner, name)


class _Sections:
    """Counts the sections a run simulates and reports progress. ``loop.run_pass`` calls
    ``modules.model.predict`` exactly once per section, after the selection and before the plant
    call, for every policy; ``_CountingModel`` turns that call into ``begin_section``. So while
    section j (0-based, counted over all passes of the run) is being selected, ``started == j``.
    ``progress(done, total)`` gets the number of finished sections at the start of each section
    and ``(total, total)`` from ``finish``. Its exceptions propagate."""

    def __init__(self, total: int, progress: Callable[[int, int], None] | None = None):
        self.started, self.total, self.progress = 0, int(total), progress

    def begin_section(self) -> None:
        if self.progress is not None:
            self.progress(min(self.started, self.total), self.total)
        self.started += 1

    def finish(self) -> None:
        if self.progress is not None:
            self.progress(self.total, self.total)


class _CountingModel:
    """Model proxy that tells a ``_Sections`` counter about every single ``predict`` call (one per
    section in ``loop.run_pass``; the selector and the model wrappers use ``predict_batch``). The
    prediction itself is passed through unchanged; everything else is forwarded to ``inner``."""

    def __init__(self, inner, sections: _Sections):
        self.inner, self.sections = inner, sections

    def predict(self, x, u):
        p = self.inner.predict(x, u)
        self.sections.begin_section()
        return p

    def __getattr__(self, name):
        if name in ("inner", "sections") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.inner, name)


# ------------------------------------------------------------ card thumbnails

CARD_THUMB_W = 340                  # px wide JPEG per card in the replay payload
READER_COLOR = (255, 64, 64)        # stained area found by the card reader
TRUTH_COLOR = (0, 255, 200)         # true stained area


THUMB_ALPHA = 0.55                  # mask opacity, as overlay_mask


def card_thumb(image: np.ndarray, mask: np.ndarray | None = None, color=READER_COLOR,
               width: int = CARD_THUMB_W, quality: int = 60) -> str:
    """Base64 JPEG of a card photo downscaled to ``width`` (if wider), with ``mask`` blended over
    it. Same picture as blending at full resolution and then downscaling (``overlay_mask``), up
    to resampling and rounding, but about four times faster: blending is linear, so the
    thumbnail is area(image) + alpha (color area(mask) - area(image under the mask)), and only
    the three downscaled planes are blended."""
    import cv2
    img = np.ascontiguousarray(image)
    h, w = img.shape[:2]
    size = (w, h) if w <= width else (int(width), int(h * (width / w)))

    def area(a: np.ndarray) -> np.ndarray:
        return a if size == (w, h) else cv2.resize(a, size, interpolation=cv2.INTER_AREA)

    small = area(img)
    if mask is not None:
        m = np.asarray(mask, bool)
        if m.shape[:2] != (h, w):           # as overlay_mask: mask to the image frame first
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        m8 = m.astype(np.uint8)
        frac = area(m8 * np.uint8(255)).astype(np.float32)[..., None] / 255.0
        under = area(cv2.bitwise_and(img, img, mask=m8)).astype(np.float32)
        col = np.asarray(color, np.float32)
        out = small.astype(np.float32) + THUMB_ALPHA * (col * frac - under)
        small = np.clip(out, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


class _Proxy:
    """Attribute pass-through that is safe to copy (no recursion before ``inner`` exists)."""

    def __getattr__(self, name):
        if name == "inner" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.inner, name)


class RecordingRenderer(_Proxy):
    """Renderer proxy that keeps a thumbnail of every card the loop renders, in segment order,
    so the UI can replay the cards without rendering them a second time. ``truth`` also stores
    the card with its true stain mask (used when the reader produces no mask)."""

    def __init__(self, inner, cards: list[dict], truth: bool = False):
        self.inner, self.cards, self.truth = inner, cards, truth

    def render(self, outcome, rng, degrade: bool = True, **kw):
        image, mask = self.inner.render(outcome, rng, degrade=degrade, **kw)
        card = {"img": card_thumb(image)}
        if self.truth:
            card["truth"] = card_thumb(image, mask, TRUTH_COLOR)
        self.cards.append(card)
        return image, mask


class RecordingVerifier(_Proxy):
    """Verifier proxy that adds the reader's stain mask to the thumbnail of the card it has just
    measured (one detection: ``measure_full``). The measurement itself is unchanged. Anything
    else, including ``measure_outcome`` of the image-free reader, passes straight through."""

    def __init__(self, inner, cards: list[dict]):
        self.inner, self.cards = inner, cards

    def measure(self, image, px_per_mm):
        meas, mask, _roi = self.inner.measure_full(image, px_per_mm)
        if self.cards:
            self.cards[-1]["reader"] = card_thumb(image, mask, READER_COLOR)
        return meas


def _copy_simulator(sim):
    """A private copy of the (shared, cached) simulator for a replay.

    The copy is taken without the stack lock, so that looking at a card is not held up by a long
    run in another browser session. That run may be writing an attribute while we copy, which
    ``deepcopy`` reports as a RuntimeError ("dictionary changed size during iteration"); the copy
    is then simply taken again. The values copied do not matter: ``begin_pass`` resets them."""
    for _ in range(4):
        try:
            return copy.deepcopy(sim)
        except RuntimeError:
            time.sleep(0.01)
    return copy.deepcopy(sim)


def render_cards(res: "PassResult", max_workers: int | None = None) -> list[dict]:
    """Thumbnails for a run whose loop rendered no cards (perfect or simulated-sensor reader).
    Replays the hidden state once, like ``replay_card``, then renders the cards in threads.
    The cards are for viewing only: nothing in the run read them."""
    m = res.modules
    sim = _copy_simulator(m.simulator)
    if hasattr(sim, "begin_pass"):
        sim.begin_pass(res.seed, res.mismatch)
    outcomes = [sim.simulate(lg.x, lg.u, segment_rng(res.seed, k), res.mismatch,
                             sample_droplets=True) for k, lg in enumerate(res.logs)]

    def one(k: int) -> dict:
        image, gt = copy.copy(m.renderer).render(outcomes[k], np.random.default_rng([res.seed, k, 1]))
        return {"img": card_thumb(image), "truth": card_thumb(image, gt, TRUTH_COLOR)}

    with ThreadPoolExecutor(max_workers or min(8, os.cpu_count() or 2)) as ex:
        return list(ex.map(one, range(len(outcomes))))


def ensure_cards(res: "PassResult") -> list[dict] | None:
    """``res.cards``, rendering them now if the loop did not (None if this stack cannot).

    When it returns None it says why in ``res.meta["cards_note"]`` (plain English, for the
    page) and does not try again for this result; a rendering error is kept, as text, in
    ``res.meta["cards_error"]``."""
    if res.cards is None and "cards_note" not in res.meta:
        if res.modules.renderer is None:
            res.meta["cards_note"] = ("This setup has no card renderer, so there are no card "
                                      "pictures for this run.")
        else:
            try:
                res.cards = render_cards(res)
            except (ValueError, AttributeError, TypeError) as exc:   # e.g. a plant without droplets
                res.cards = None
                res.meta["cards_note"] = ("Card pictures could not be drawn for this run. The "
                                          "simulator in this setup does not produce the droplet "
                                          "detail a card picture needs.")
                res.meta["cards_error"] = f"{type(exc).__name__}: {exc}"
    return res.cards


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
    cards: list[dict] | None = None      # one thumbnail dict per section (see RecordingRenderer)
    # selector info of the selection applied in each section of the displayed pass (None for a
    # section without one, e.g. best_fixed); see run_pass_wrapper
    selections: list[dict | None] | None = None
    # what the controller would have done in each section of the displayed pass without card
    # feedback (run_pass_wrapper(explain_feedback=True), feedback strategies only); see
    # ``feedback_effect``
    no_feedback: list[dict] | None = None

    @property
    def n(self) -> int:
        return len(self.conds)

    def info_for(self, k: int) -> dict | None:
        """Selector info of the setting applied in section k (the final selection when the fast
        loop selected twice), or None. Uses ``selections`` when the run recorded them, which
        also works when a wind forecast made the selector plan with other conditions than the
        logged ones; otherwise the (conditions, setting) lookup in ``records``."""
        if self.selections is not None and 0 <= k < len(self.selections):
            info = self.selections[k]
            if info is not None:
                return info
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
                     forecaster: str | None = None,
                     fixed_nozzle: str | tuple[str, str] | None = None,
                     capture_cards: bool = False, same_field: bool = False,
                     progress: Callable[[int, int], None] | None = None,
                     feedback_model: Any = None, rate_lock: float | None = None,
                     explain_feedback: bool = False) -> PassResult:
    """Run one pass and keep what the UI needs (selector records, timing, card thumbnails).

    ``explain_feedback`` (D33; off by default, the app's Field run turns it on): for the card
    feedback strategies, also work out what the controller would have done in every section of
    the displayed pass without card feedback, and what the plant would then have given
    (``PassResult.no_feedback``, read through ``feedback_effect``). It changes no logged number.

    ``feedback_model`` (decision D29; None or "rule" = off, the eq. 3 running average): a trained
    ``feedback_learned.FeedbackModel``, or "xgboost" / "lightgbm" / a path to load one. The fast
    loop of ``fast`` then takes its correction from that model
    (``feedback_learned.LearnedFeedback``). Every other strategy ignores it, ``fast_slow``
    included (``TRAINED_FEEDBACK_POLICIES``). Reported in ``meta["feedback"]``.

    ``fast_slow``: runs ``n_learn`` learning passes (no card rendering) through
    ``loop.run_passes_slow`` so the slow loop learns a residual correction, then the displayed
    pass with the fitted ``ResidualCorrectedModel`` as ``modules.model``. ``meta["passes"]``
    holds a per-pass summary in the order they ran (the last row is the displayed pass).
    Seeds: with ``same_field`` the displayed pass runs on ``seed`` itself, so it sees the same
    gusts, canopy, wear and card noise as every other policy on ``seed``, and the learning
    passes run on ``seed + 1009 i`` for i = 1..n_learn. Without it (default, the behaviour the
    golden digests pin) the learning passes run on ``seed + 1009 i`` for i = 0..n_learn-1 and
    the displayed pass on ``seed + 1009 n_learn``. ``PassResult.seed`` is the displayed pass's
    seed. ``compare_policies`` uses ``same_field=True``.

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

    ``fixed_nozzle`` (decision D19; None or "free" = off, any nozzle per section): "auto" holds
    the one nozzle ``selector.best_nozzle`` picks for these conditions with the pass's model and
    ``lam``; a (family, size) pair or "FAMILY-SIZE" holds that nozzle. Only pressure (and speed /
    boom height if those options are on) then varies along the field. Set on the selector for the
    duration of the call; fast_slow learning passes use the same nozzle; best_fixed ignores it.
    The choice is reported in ``meta["nozzle"]``.

    ``rate_lock`` (operator console; None = off): a target application rate in gal/acre, held
    within the selector's ``rate_tol_frac`` (spec ``rate_control.tolerance_frac``). It needs a
    named nozzle (``fixed_nozzle`` a (family, size) pair or "FAMILY-SIZE", not None, "free" or
    "auto"), so best_fixed cannot use it. The selector then moves only pressure, with the speed
    that holds the rate, and boom height if ``boom_control`` is on (``selector`` module docstring).
    Checked before the selector is touched (ValueError: no named nozzle, a rate <= 0, a selector
    without rate mode, or no pressure of the nozzle that holds the rate inside the spec speed
    range); set on the selector for the duration of the call like ``nozzle_lock``. Reported in
    ``meta["rate"]`` as ``{"target_gpa", "tol_frac"}``.

    ``capture_cards`` keeps a thumbnail of every card the loop renders and reads
    (``PassResult.cards``; None if the loop read no images, see ``ensure_cards``). It wraps the
    renderer and verifier in recording proxies for this call only and does not change any number.

    ``progress(done, total)``, if given, is called at the start of every simulated section with
    the number of sections finished so far and once more with ``(total, total)`` at the end;
    ``total`` counts every simulated section (``len(conds) * (n_learn + 1)`` for fast_slow). It
    is driven by a transparent proxy around the model and changes no number. An exception it
    raises ends the run and propagates; the shared selector is restored first.

    The whole mutate-run-restore part runs under ``_STACK_LOCK``.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    with _STACK_LOCK:
        return _run_pass_locked(modules, conds, policy, seed, mismatch, c_req, lam, fixed_setting,
                                render, pass_id, n_learn, speed_control, boom_control, forecaster,
                                fixed_nozzle, capture_cards, same_field, progress, feedback_model,
                                rate_lock, explain_feedback)


FEEDBACK_CHOICES = ("rule", "xgboost", "lightgbm")     # app option; "rule" is the default
_FEEDBACK_MODELS: dict[str, Any] = {}


def feedback_model_available(name: str) -> bool:
    """True if a ``FEEDBACK_CHOICES`` entry can run here: its trained file is on disk and its
    package is installed (xgboost is an optional extra; "rule" needs neither)."""
    import importlib.util
    from coverageai.feedback_learned import DEFAULT_PATHS
    if name == "rule":
        return True
    return (name in DEFAULT_PATHS and DEFAULT_PATHS[name].exists()
            and importlib.util.find_spec(name) is not None)


def default_feedback_choice() -> str:
    """The card feedback the APP uses unless told otherwise (decision D31): the trained model,
    XGBoost if it can run here, else LightGBM (same accuracy, no extra package), else the rule.
    The library default (``run_pass_wrapper(feedback_model=None)``, ``loop.run_pass``) stays the
    eq. 3 rule, so the reported experiments and the golden digests do not move."""
    return next((c for c in ("xgboost", "lightgbm") if feedback_model_available(c)), "rule")


def _learned_feedback(feedback_model: Any):
    """None for the rule; otherwise a fresh ``LearnedFeedback`` (trained files are loaded once)."""
    if feedback_model in (None, "", "rule", "off"):
        return None
    from coverageai.feedback_learned import LearnedFeedback, load_feedback_model
    model = feedback_model
    if isinstance(feedback_model, (str, Path)):
        key = str(feedback_model)
        if key not in _FEEDBACK_MODELS:
            _FEEDBACK_MODELS[key] = load_feedback_model(key)
        model = _FEEDBACK_MODELS[key]
    return LearnedFeedback(model)


def _run_pass_locked(modules, conds, policy, seed, mismatch, c_req, lam, fixed_setting, render,
                     pass_id, n_learn, speed_control, boom_control, forecaster, fixed_nozzle,
                     capture_cards, same_field, progress, feedback_model=None,
                     rate_lock=None, explain_feedback=False) -> PassResult:
    """Body of ``run_pass_wrapper``; the caller holds ``_STACK_LOCK``."""
    learned = _learned_feedback(feedback_model) if policy in TRAINED_FEEDBACK_POLICIES else None
    sel = modules.selector
    rec = sel if isinstance(sel, RecordingSelector) else RecordingSelector(sel)
    mods = dataclasses.replace(modules, selector=rec)
    cards: list[dict] = []
    if capture_cards and mods.renderer is not None:
        masks = hasattr(mods.verifier, "measure_full")
        mods.renderer = RecordingRenderer(modules.renderer, cards, truth=not masks)
        if masks:
            mods.verifier = RecordingVerifier(modules.verifier, cards)
    if render is None:
        render = mods.verifier is not None
    # An image-free verifier (NoisyVerifier.measure_outcome) never reads rendered cards in the
    # loop (loop.run_pass: ``direct``), so it must not be reported as image analysis.
    vision = bool(render and mods.verifier is not None and mods.renderer is not None
                  and not hasattr(mods.verifier, "measure_outcome"))
    if policy == "best_fixed" and fixed_setting is None:
        fixed_setting = best_fixed_for(mods, conds, c_req)
    # Validate everything that can raise BEFORE touching the (st.cache_resource-shared) selector;
    # then mutate inside the try so the finally always restores, even on a partial failure.
    fc = _make_forecaster(forecaster) if policy in FORECAST_POLICIES else None
    inner = sel.inner if isinstance(sel, RecordingSelector) else sel
    old_lam = getattr(inner, "lam", None)   # set on inner: a RecordingSelector only forwards reads
    opt_grids: list[tuple[str, Any]] = []
    for on, attr, grid, what in ((speed_control, "speed_options_kmh", SPEED_OPTIONS_KMH, "travel-speed"),
                                 (boom_control, "boom_height_options_m", BOOM_HEIGHT_OPTIONS_M, "boom-height")):
        if on:
            if not hasattr(inner, attr):
                raise ValueError(f"this selector does not support {what} control")
            opt_grids.append((attr, grid))
    lock = _parse_nozzle(fixed_nozzle) if policy != "best_fixed" else None
    if lock is not None:
        if not hasattr(inner, "best_nozzle"):
            raise ValueError("this selector does not support holding one nozzle")
        if lock != "auto" and not any((u.family, u.size) == lock for u in inner.candidates):
            raise ValueError(f"nozzle {lock[0]}-{lock[1]} is not in the catalog")
    if rate_lock is not None:
        if lock is None or lock == "auto":
            raise ValueError("rate_lock needs a named nozzle (fixed_nozzle='FAMILY-SIZE')")
        if not hasattr(inner, "rate_lock"):
            raise ValueError("this selector does not support a fixed application rate")
        rate_lock = float(rate_lock)
        if not rate_lock > 0:
            raise ValueError(f"rate_lock must be a positive rate in gal/acre, got {rate_lock!r}")
        from coverageai.rate_control import iso_rate_candidates
        if not iso_rate_candidates(lock[0], lock[1], rate_lock, inner.rate_tol_frac):
            raise ValueError(f"no pressure of nozzle {lock[0]}-{lock[1]} holds {rate_lock:g} gal/acre "
                             "inside the travel-speed range")
    saved_opts: dict[str, Any] = {}
    lam_set = False
    t0 = time.perf_counter()
    meta: dict = {}
    run_seed = int(seed)
    sections = _Sections(len(conds) * (int(n_learn) + 1 if policy == "fast_slow" else 1), progress)
    old_tracker = rec.tracker
    try:
        rec.tracker = sections
        if lam is not None and old_lam is not None:
            inner.lam = float(lam)
            lam_set = True
        for attr, grid in opt_grids:
            saved_opts[attr] = getattr(inner, attr)
            setattr(inner, attr, list(grid))
        if lock is not None:
            meta["nozzle"] = {"mode": "auto" if lock == "auto" else "fixed"}
            if lock == "auto":      # after lam is set: the choice uses this pass's trade-off
                lock, stats = inner.best_nozzle(list(conds), mods.model, float(c_req))
                meta["nozzle"]["ranking"] = stats["nozzles"][:5]
            meta["nozzle"].update(family=lock[0], size=lock[1])
            saved_opts["nozzle_lock"] = inner.nozzle_lock
            inner.nozzle_lock = lock
        if rate_lock is not None:
            meta["rate"] = {"target_gpa": rate_lock, "tol_frac": float(inner.rate_tol_frac)}
            saved_opts["rate_lock"] = inner.rate_lock
            inner.rate_lock = rate_lock
        if policy == "fast_slow":
            run_seed, passes = _learn_between_passes(mods, conds, int(seed), mismatch,
                                                     float(c_req), int(n_learn), forecaster=fc,
                                                     same_field=same_field, sections=sections,
                                                     feedback=learned)
            meta["passes"] = passes
        if learned is not None:
            learned.reset()                 # the replayed pass starts with no card, like the rule
        meta["feedback"] = ({"kind": "learned", "backend": learned.model.backend,
                             "trees": learned.model.meta.get("trees"), "rows": learned.model.meta.get("n_rows")}
                            if learned is not None else {"kind": "rule" if policy in FEEDBACK_POLICIES else "none"})
        if hasattr(mods.simulator, "begin_pass"):
            mods.simulator.begin_pass(run_seed, mismatch)
        base, first_call = sections.started, len(rec.calls)
        logs = run_pass(list(conds), policy,
                        dataclasses.replace(mods, model=_CountingModel(mods.model, sections)),
                        run_seed, mismatch, float(c_req), pass_id or f"{policy}_s{run_seed}",
                        render=vision, fixed_setting=fixed_setting, forecaster=fc, feedback=learned)
        # the selection applied in section k is the last call tagged base + k (CLAUDE.md rule 7)
        selections: list[dict | None] = [None] * len(logs)
        plan_x: list[Conditions | None] = [None] * len(logs)
        for tag, x_call, _u, info in rec.calls[first_call:]:
            if tag is not None and 0 <= tag - base < len(logs):
                selections[tag - base] = info
                plan_x[tag - base] = x_call
        no_fb_sel = None
        if explain_feedback and policy in FEEDBACK_POLICIES:
            # while the run's selector options (lam, grids, nozzle and rate lock) are still set
            no_fb_sel = _no_feedback_selections(
                inner, mods.model.base if policy == "fast_slow" else mods.model, logs, plan_x,
                float(c_req), mods.derive_fn)
        sections.finish()
    finally:
        rec.tracker = old_tracker
        if lam_set:
            inner.lam = old_lam
        for attr, old in saved_opts.items():
            setattr(inner, attr, old)
    if policy == "fast_slow":
        meta["passes"].append(_pass_summary(len(meta["passes"]) + 1, logs, float(c_req),
                                            mods.model, displayed=True))
    meta.setdefault("nozzle", {"mode": "best_fixed" if policy == "best_fixed" else "free"})
    no_feedback = (_no_feedback_outcomes(mods.simulator, logs, no_fb_sel, run_seed, mismatch)
                   if no_fb_sel is not None else None)
    # the recording proxies are for this call only: the result keeps the real renderer/verifier
    mods = dataclasses.replace(mods, renderer=modules.renderer, verifier=modules.verifier)
    return PassResult(logs=logs, conds=list(conds), policy=policy, seed=run_seed,
                      mismatch=mismatch, c_req=float(c_req), lam=lam, modules=mods,
                      records=rec.records, fixed_setting=fixed_setting, used_vision=vision,
                      elapsed_s=time.perf_counter() - t0, meta=meta,
                      cards=cards if len(cards) == len(logs) and cards else None,
                      selections=selections, no_feedback=no_feedback)


def _no_feedback_selections(selector, model, logs: Sequence, plan_x: Sequence, c_req: float,
                            derive_fn) -> list[dict]:
    """What the selector picks in each section with no card feedback: zero correction and the
    model without the slow loop's learned offset, with the same options as the run (call this
    while they are set) and the same planning wind (``plan_x``, the conditions the run's
    selector was called with). That is the choice of the "Model only" strategy. Calls the inner
    selector, so nothing is recorded, and only the model: no simulator."""
    out = []
    for lg, xp in zip(logs, plan_x):
        u, info = selector.select(xp if xp is not None else lg.x, model, 0.0, c_req)
        out.append({"u": u, "infeasible": bool(info.get("infeasible", False))})
    # the model's raw prediction for that setting, with the conditions the run predicted with
    xs = [lg.x if "fc_wind_ms" not in lg.extra
          else dataclasses.replace(lg.x, wind_speed_ms=lg.extra["fc_wind_ms"]) for lg in logs]
    preds = model.predict_batch(xs, [o["u"] for o in out])
    for o, x, p in zip(out, xs, preds):
        o.update(pred_cov=float(p.coverage_pct), pred_drift=float(p.drift_frac),
                 derived=derive_fn(x, o["u"]))
    return out


def _no_feedback_outcomes(simulator, logs: Sequence, sel: list[dict], seed: int,
                          mismatch: str) -> list[dict]:
    """Adds the plant's answer to each no-feedback choice: the copied simulator replays the pass
    with the same per-section random streams. The hidden state (gust, canopy, wear) does not
    depend on the setting (``simulator`` docstring), so each section sees exactly the conditions
    it saw in the run, and a section whose setting is the same gets the same outcome."""
    sim = _copy_simulator(simulator)
    if hasattr(sim, "begin_pass"):
        sim.begin_pass(seed, mismatch)
    for k, (lg, o) in enumerate(zip(logs, sel)):
        out = sim.simulate(lg.x, o["u"], segment_rng(seed, k), mismatch)
        o.update(actual_cov=float(out.coverage_pct), actual_drift=float(out.drift_frac))
    if hasattr(sim, "end_pass"):
        sim.end_pass()
    return sel


def _parse_nozzle(v) -> tuple[str, str] | str | None:
    """None / "free" / False -> None, "auto" / True -> "auto", ("XR", "04") or "XR-04" -> pair."""
    if v in (None, False, "", "free", "off"):
        return None
    if v is True or v == "auto":
        return "auto"
    if isinstance(v, str):
        fam, sep, size = v.partition("-")
        if not sep or not fam or not size:
            raise ValueError(f"nozzle must be 'auto', 'free' or 'FAMILY-SIZE', got {v!r}")
        return fam, size
    fam, size = v
    return str(fam), str(size)


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
                          c_req: float, n_learn: int, forecaster=None, same_field: bool = False,
                          sections: _Sections | None = None, feedback=None) -> tuple[int, list[dict]]:
    """Slow loop: ``n_learn`` passes (no rendering) through ``run_passes_slow``, then one
    more fit on the last pass so ``mods.model`` is the ResidualCorrectedModel as it stands
    at the displayed pass. Returns (seed of the displayed pass, summaries of the learning
    passes). Seeds: ``same_field`` -> learning passes ``seed + 1009 i``, i = 1..n_learn, and
    the displayed pass on ``seed``; otherwise learning passes i = 0..n_learn-1 and the displayed
    pass on ``seed + 1009 n_learn``. ``sections`` counts the learning passes' sections (the
    correction is handed to the slow loop inside a counting proxy). Mutates only ``mods`` (a
    private copy), never the cached module stack."""
    from coverageai.feedback import ResidualCorrectedModel
    corr = ResidualCorrectedModel(mods.model, mods.derive_fn)
    passes: list[dict] = []
    first = seed + 1009 if same_field else seed
    if n_learn > 0:
        logs = run_passes_slow(list(conds), mods, first, mismatch, c_req, n_passes=n_learn,
                               render=False, forecaster=forecaster, feedback=feedback,
                               correction=corr if sections is None else _CountingModel(corr, sections))
        by_pass: dict[int, list] = {}
        for lg in logs:
            by_pass.setdefault(int(lg.extra["slow_pass"]), []).append(lg)
        for i in sorted(by_pass):
            passes.append(_pass_summary(i + 1, by_pass[i], c_req, corr))
            passes[-1]["slow_offset_pp"] = float(by_pass[i][0].extra.get("slow_global_corr", 0.0))
        # run_passes_slow does not refit after its last pass: do it here
        last = by_pass[max(by_pass)]
        corr.fit(verified_records(last))
    mods.model = corr
    return (seed if same_field else seed + 1009 * n_learn), passes


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

def loop_numbers(lg) -> dict[str, float]:
    """The feedback-loop numbers of one section. Single source for animation panels B and D
    (``field_anim``) and for the charts and table under the animation, so they cannot disagree.

    ``feedback_pp`` is the correction applied to this section's prediction, ``error_pp`` what was
    left after it against the actual coverage (panel D), ``card_error_pp`` the same against the
    card reading, ``raw_error_pp`` the card reading minus the model's own prediction (the e_k of
    eq. 3: this, not the corrected error, is what the fast loop learns from) and
    ``next_feedback_pp`` the correction after this card."""
    return {
        "predicted": float(lg.pred.coverage_pct),
        "feedback_pp": float(lg.bias),
        "after_feedback": float(lg.corrected_pred_cov),
        "card": float(lg.meas.coverage_pct),
        "actual": float(lg.outcome.coverage_pct),
        "error_pp": float(lg.outcome.coverage_pct - lg.corrected_pred_cov),
        "card_error_pp": float(lg.residual),
        "raw_error_pp": float(lg.extra.get("raw_residual", lg.meas.coverage_pct - lg.pred.coverage_pct)),
        "next_feedback_pp": float(lg.extra.get("bias_after", lg.bias)),
    }


FEEDBACK_POLICIES = ("fast", "fast_slow")       # strategies that correct the model from the cards
# The trained feedback model (D29) is used by "fast" only. With the learning strategy the slow
# loop's offset and the model's own starting correction add up and the correction overshoots
# (adoption check, D31), so "fast_slow" keeps the eq. 3 rule.
TRAINED_FEEDBACK_POLICIES = ("fast",)


def regime_text(regime: str | None) -> str:
    """Plain words for a fast-loop regime key such as ``"u2-4|coarse"`` ("" if it is not one)."""
    m = re.fullmatch(r"u([\d.]+)-([\d.]+)\|(\w+)", str(regime or ""))
    if not m:
        return ""
    lo, hi, group = float(m.group(1)), float(m.group(2)), m.group(3)
    top = float(spec()["feedback"]["regime_bins"]["u_cross_ms"][-1])
    wind = f"crosswind above {lo:g} m/s" if hi >= top else f"crosswind {lo:g} to {hi:g} m/s"
    return f"{wind}, {group} droplets"


def feedback_rule_text(res: "PassResult | None" = None) -> str:
    """How the fast loop gets its correction, in one paragraph: the eq. 3 rule with the numbers of
    ``spec.feedback``, or, for a run with a trained feedback model (D29), what that model is and
    what it looks at."""
    fb = spec()["feedback"]
    kind = (res.meta.get("feedback") or {}) if res is not None else {}
    if kind.get("kind") == "learned":
        name = {"xgboost": "XGBoost", "lightgbm": "LightGBM"}.get(kind.get("backend"), "gradient-boosted")
        size = (f" ({kind['trees']:,} trees, trained on {kind['rows']:,} simulated sections)"
                if kind.get("trees") and kind.get("rows") else "")
        return (f"The correction is the prediction of a trained {name} model{size}. It predicts the next "
                "card's error (card reading minus model prediction) from the section's wind, weather, "
                "speed, boom height and chosen setting, and from the cards read so far: how many, the "
                "last errors, their running averages, and what the running-average rule would apply. "
                f"It is limited to ±{float(fb['bias_clip_pp']):g} pp. It was trained on simulated passes "
                "with hidden surprises, on seeds kept apart from the ones evaluated.")
    a = float(fb["alpha"])
    edges = [float(v) for v in fb["regime_bins"]["u_cross_ms"]]
    bins = [f"{lo:g} to {hi:g}" for lo, hi in zip(edges[:-2], edges[1:-1])] + [f"above {edges[-2]:g}"]
    groups = " and ".join(fb["regime_bins"]["droplet_group"])
    return (f"Each card updates the correction of its group: new correction = {100 * (1 - a):.0f}% of the "
            f"old one + {100 * a:.0f}% of (card reading minus model prediction), limited to "
            f"±{float(fb['bias_clip_pp']):g} pp. A group is a crosswind range ({', '.join(bins)} m/s) "
            f"for {groups} droplets. A group's correction is used once it has "
            f"{int(fb['warmup_cards'])} cards. A group with no card yet takes "
            f"{100 * FALLBACK_DAMPING:.0f}% of the field-wide correction.")


def feedback_basis(res: "PassResult") -> list[dict]:
    """For each section, what the feedback correction applied to it rests on.

    ``uses_feedback`` is False for strategies without card feedback. Otherwise ``group`` names the
    section's regime in plain words, ``cards_before`` counts the earlier cards of this pass in the
    same regime, and ``source`` says where the applied correction came from: ``"group"`` (that
    regime's own running correction), ``"field"`` (no card in the regime yet, so a damped share of
    the pass-wide correction) or ``"none"`` (still warming up, nothing applied). With a trained
    feedback model (D29) the source is ``"learned"`` and ``cards_before`` counts every earlier card
    of the pass, because that model looks at all of them. ``text`` is one short line for the page."""
    uses = res.policy in FEEDBACK_POLICIES
    learned = (res.meta.get("feedback") or {}).get("kind") == "learned"
    warm = max(int(spec()["feedback"]["warmup_cards"]), 1)
    seen: dict[str, int] = {}
    errors: list[float] = []            # card - model of the earlier sections (what the model saw)
    out = []
    for k, lg in enumerate(res.logs):
        regime = str(lg.extra.get("regime", ""))
        group = regime_text(regime)
        n = seen.get(regime, 0)
        if not uses or not group:
            out.append({"uses_feedback": False, "group": "", "cards_before": 0, "source": "none",
                        "text": "This strategy does not use card feedback."})
            continue
        if learned:                     # D29: one trained model looks at every earlier card of the pass
            text = ("trained model, no card yet (its usual starting correction)" if not errors
                    else f"trained model, {len(errors)} earlier cards (last {errors[-1]:+.1f} pp, "
                         f"average {float(np.mean(errors)):+.1f} pp)")
            out.append({"uses_feedback": True, "group": group, "cards_before": len(errors),
                        "source": "learned", "text": text})
            errors.append(loop_numbers(lg)["raw_error_pp"])
            seen[regime] = n + 1
            continue
        if n >= warm:
            source, text = "group", f"{n} earlier cards with {group}"
        elif n == 0 and k >= warm:
            source = "field"
            text = (f"{100 * FALLBACK_DAMPING:.0f}% of the field-wide correction "
                    f"(no card yet with {group})")
        else:
            source = "none"
            text = f"none yet: needs {warm} cards with {group} (has {n})"
        out.append({"uses_feedback": True, "group": group, "cards_before": n, "source": source,
                    "text": text})
        seen[regime] = n + 1
    return out


def timeseries_frame(logs: Sequence, upto: int | None = None) -> pd.DataFrame:
    """One row per section (the first ``upto + 1`` if given) from ``loop_numbers``; ``error`` is
    panel D's prediction error, ``residual`` the card-based error the controller sees."""
    ls = list(logs if upto is None else logs[: upto + 1])
    nums = [loop_numbers(l) for l in ls]
    return pd.DataFrame({
        "k": [l.k + 1 for l in ls],
        "predicted": [n["predicted"] for n in nums],
        "corrected": [n["after_feedback"] for n in nums],
        "measured": [n["card"] for n in nums],
        "true": [n["actual"] for n in nums],
        "bias": [n["feedback_pp"] for n in nums],
        "bias_after": [n["next_feedback_pp"] for n in nums],
        "residual": [n["card_error_pp"] for n in nums],
        "raw_error": [n["raw_error_pp"] for n in nums],
        "error": [n["error_pp"] for n in nums],
    })


# ------------------------------------------------- model outputs and changes

def _fmt(v: float, nd: int) -> str:
    return f"{v:.{nd}f}".rstrip("0").rstrip(".") if nd else f"{v:.0f}"


def setting_changes(prev, cur) -> list[str]:
    """What the controller changed from section ``prev`` to ``cur`` (two SegmentLogs), in plain
    words: nozzle, pressure and, only where the controller sets them, travel speed and boom
    height. Changes in the field's own speed or height are conditions, not decisions."""
    return setting_diff(prev.x, prev.u, cur.x, cur.u)


def setting_diff(xa, a, xb, b) -> list[str]:
    """What differs from setting ``a`` (under conditions ``xa``) to setting ``b`` (under ``xb``),
    in the words of ``setting_changes``."""
    from coverageai.contracts import effective_boom_height_m, effective_speed_kmh
    out = []
    if (a.family, a.size) != (b.family, b.size):
        out.append(f"nozzle {a.family} {a.size} to {b.family} {b.size}")
    if abs(a.pressure_psi - b.pressure_psi) > 1e-9:
        out.append(f"pressure {a.pressure_psi:g} to {b.pressure_psi:g} psi "
                   f"({b.pressure_psi - a.pressure_psi:+g})")
    if a.travel_speed_kmh is not None or b.travel_speed_kmh is not None:
        va, vb = effective_speed_kmh(xa, a), effective_speed_kmh(xb, b)
        if abs(va - vb) > 1e-9:
            out.append(f"speed {_fmt(va, 1)} to {_fmt(vb, 1)} km/h ({vb - va:+.1f})")
    if a.boom_height_m is not None or b.boom_height_m is not None:
        ha, hb = effective_boom_height_m(xa, a), effective_boom_height_m(xb, b)
        if abs(ha - hb) > 1e-9:
            out.append(f"boom {ha:.2f} to {hb:.2f} m ({hb - ha:+.2f})")
    return out


def feedback_effect(res: "PassResult") -> list[dict] | None:
    """For each section, what card feedback changed in the controller's decision and what that
    did on the plant (D33). None when the run has no such record (a strategy without card
    feedback, or ``explain_feedback`` off).

    "Without feedback" is what the "Model only" strategy would have done in that section: the same
    selector options, the model's raw prediction with no correction (for the learning strategy
    also without the slow loop's offset), on the same field, gusts and hidden surprises. Its
    actual coverage is simulator truth for that setting. Keys: ``changed`` (True when the setting
    differs), ``changes`` (from the no-feedback setting to the one used, in the words of
    ``setting_changes``), ``setting`` / ``setting_without`` (text), ``actual`` / ``actual_without``
    and ``gain_pp`` (actual coverage, %, and the difference), ``predicted_without`` (the model's
    raw prediction for the no-feedback setting), ``drift_pct`` / ``drift_without_pct``, ``gpa`` /
    ``gpa_without``, and ``text``, one line for the page."""
    if not res.no_feedback or len(res.no_feedback) != len(res.logs):
        return None
    out = []
    for lg, o in zip(res.logs, res.no_feedback):
        changes = setting_diff(lg.x, o["u"], lg.x, lg.u)
        actual, without = float(lg.outcome.coverage_pct), float(o["actual_cov"])
        gain = actual - without
        if changes:
            text = (f"Without feedback: {setting_text(o['u'])}, actual {without:.1f}%. Feedback "
                    f"changed {'; '.join(changes)}: actual {actual:.1f}% ({gain:+.1f} pp).")
        else:
            text = ("Same setting as without feedback: the correction changed only the "
                    "expected coverage.")
        out.append({
            "changed": bool(changes), "changes": changes,
            "setting": setting_text(lg.u), "setting_without": setting_text(o["u"]),
            "actual": actual, "actual_without": without, "gain_pp": gain,
            "predicted_without": float(o["pred_cov"]),
            "drift_pct": 100.0 * float(lg.outcome.drift_frac),
            "drift_without_pct": 100.0 * float(o["actual_drift"]),
            "gpa": float(lg.derived.app_rate_gpa), "gpa_without": float(o["derived"].app_rate_gpa),
            "infeasible_without": bool(o["infeasible"]), "text": text,
        })
    return out


def feedback_effect_summary(res: "PassResult", upto: int | None = None) -> dict | None:
    """Run totals of ``feedback_effect`` over the first ``upto + 1`` sections (all if None): how
    many settings feedback changed, and actual coverage, sections on target, drift and spray
    volume with and without feedback. "On target" counts actual coverage at or above the target,
    as ``compute_kpis`` does, so ``n_met`` here is the KPI's ``n_met``."""
    eff = feedback_effect(res)
    if eff is None:
        return None
    eff = eff if upto is None else eff[: upto + 1]
    if not eff:
        return None
    c = float(res.c_req)
    changed = [e for e in eff if e["changed"]]

    def mean(key, rows=eff):
        return float(np.mean([e[key] for e in rows])) if rows else 0.0

    return {
        "n": len(eff), "n_changed": len(changed),
        "n_met": sum(e["actual"] >= c for e in eff),
        "n_met_without": sum(e["actual_without"] >= c for e in eff),
        "mean_cov": mean("actual"), "mean_cov_without": mean("actual_without"),
        "mean_gain_pp": mean("gain_pp"), "mean_gain_changed_pp": mean("gain_pp", changed),
        "mean_drift_pct": mean("drift_pct"), "mean_drift_without_pct": mean("drift_without_pct"),
        "mean_gpa": mean("gpa"), "mean_gpa_without": mean("gpa_without"),
    }


def feedback_effect_frame(res: "PassResult") -> pd.DataFrame:
    """One row per section: the setting without and with feedback and the actual coverage of
    each (column names: ``labels.FEEDBACK_EFFECT_COLS`` keys)."""
    eff = feedback_effect(res) or []
    return pd.DataFrame([{
        "k": k + 1, "setting_without": e["setting_without"], "setting": e["setting"],
        "changes": "; ".join(e["changes"]) or "same setting",
        "actual_without": e["actual_without"], "actual": e["actual"], "gain_pp": e["gain_pp"],
    } for k, e in enumerate(eff)], columns=["k", "setting_without", "setting", "changes",
                                            "actual_without", "actual", "gain_pp"])


def decision_frame(res: "PassResult") -> pd.DataFrame:
    """Everything the predictive model and the selector put out for each section of a run, one
    row per section: the chosen setting, the model's coverage prediction with its interval, the
    predicted drift, what the card and the simulator then showed, and what changed against the
    previous section. This is the table to export or diff."""
    from coverageai.contracts import effective_boom_height_m, effective_speed_kmh
    rows = []
    for k, lg in enumerate(res.logs):
        n = loop_numbers(lg)
        rows.append({
            "section": k + 1, "nozzle": f"{lg.u.family} {lg.u.size}",
            "pressure_psi": lg.u.pressure_psi,
            "speed_kmh": effective_speed_kmh(lg.x, lg.u),
            "boom_height_m": effective_boom_height_m(lg.x, lg.u),
            "volume_gpa": lg.derived.app_rate_gpa, "droplet_class": lg.derived.droplet_class,
            "wind_ms": lg.x.wind_speed_ms,
            "predicted_cov": n["predicted"], "predicted_lo": lg.pred.coverage_lo,
            "predicted_hi": lg.pred.coverage_hi, "feedback_pp": n["feedback_pp"],
            "after_feedback_cov": n["after_feedback"], "card_cov": n["card"],
            "actual_cov": n["actual"], "error_pp": n["error_pp"],
            "predicted_drift_pct": 100.0 * lg.pred.drift_frac,
            "actual_drift_pct": 100.0 * lg.outcome.drift_frac,
            "changed": "; ".join(setting_changes(res.logs[k - 1], lg)) if k else "",
        })
    return pd.DataFrame(rows)


def logs_frame(res: "PassResult") -> pd.DataFrame:
    """The complete flat run log (``loop.logs_to_frame``): every logged field of every section."""
    return logs_to_frame(res.logs)


def run_summary(res: "PassResult") -> dict[str, float]:
    kp = compute_kpis(res.logs, res.c_req)
    return {"Sections meeting target (%)": 100.0 * kp["frac_met"],
            "Average coverage (%)": kp["mean_cov"],
            "Average spray volume (gal/acre)": kp["mean_gpa"],
            "Average drift (%)": 100.0 * kp["mean_drift"],
            "Sections to re-spray": float(kp["n_retreat"])}


def run_changes(prev: "PassResult", cur: "PassResult") -> dict:
    """What differs between two runs: the inputs the user changed (``meta["inputs"]``, a dict of
    display name -> value set by the UI), the run-level results, and every section where the
    model chose a different setting (only when both runs cover the same number of sections)."""
    pin, cin = prev.meta.get("inputs", {}), cur.meta.get("inputs", {})
    inputs = [{"Input": k, "Previous run": pin.get(k, ""), "This run": cin.get(k, "")}
              for k in dict.fromkeys([*pin, *cin]) if pin.get(k) != cin.get(k)]
    a, b = run_summary(prev), run_summary(cur)
    results = [{"Result": k, "Previous run": a[k], "This run": b[k], "Change": b[k] - a[k]} for k in b]
    rows = []
    if prev.n == cur.n:
        for k, (p, c) in enumerate(zip(prev.logs, cur.logs)):
            diff = setting_changes(p, c)
            if diff:
                rows.append({"Section": k + 1, "Previous run": setting_text(p.u),
                             "This run": setting_text(c.u), "Changed": "; ".join(diff),
                             "Predicted coverage change (pp)": c.pred.coverage_pct - p.pred.coverage_pct,
                             "Actual coverage change (pp)": c.outcome.coverage_pct - p.outcome.coverage_pct})
    return {"inputs": pd.DataFrame(inputs, columns=["Input", "Previous run", "This run"]),
            "results": pd.DataFrame(results),
            "sections": pd.DataFrame(rows, columns=["Section", "Previous run", "This run", "Changed",
                                                    "Predicted coverage change (pp)",
                                                    "Actual coverage change (pp)"]),
            "comparable": prev.n == cur.n, "n_changed": len(rows), "n": cur.n}


def setting_text(u: Setting) -> str:
    """'XR 04 at 40 psi', plus the travel speed / boom height when the controller set them."""
    t = f"{u.family} {u.size} at {u.pressure_psi:g} psi"
    if u.travel_speed_kmh is not None:      # one decimal: a held rate gives speeds like 15.95 km/h
        t += f", {_fmt(u.travel_speed_kmh, 1)} km/h"
    if u.boom_height_m is not None:
        t += f", boom {u.boom_height_m:.2f} m"
    return t


def recommendations(res: "PassResult") -> list:
    """One ``recommend.Recommendation`` per section (cached on the result): for every section
    whose card reads below the target, which of pressure, speed and boom height to change and by
    how much, from the run's own predictive model."""
    if "recommendations" not in res.meta:
        from coverageai.recommend import recommend_for_logs
        res.meta["recommendations"] = recommend_for_logs(
            res.modules.model, res.logs, res.c_req, derive_fn=res.modules.derive_fn)
    return res.meta["recommendations"]


def recommendation_frame(res: "PassResult") -> pd.DataFrame:
    """Sections that need a change, with the advice in words."""
    rows = []
    for k, (lg, r) in enumerate(zip(res.logs, recommendations(res))):
        if r.status == "ok":
            continue
        rows.append({"Section": k + 1, "Setting used": setting_text(lg.u),
                     "Card reading (%)": r.observed_cov, "Short of target (pp)": r.shortfall_pp,
                     "Recommendation": r.headline(),
                     "Alternatives": "; ".join(s[3:] for s in r.lines()[1:])})
    return pd.DataFrame(rows, columns=["Section", "Setting used", "Card reading (%)",
                                       "Short of target (pp)", "Recommendation", "Alternatives"])


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
    sim = _copy_simulator(m.simulator)
    if hasattr(sim, "begin_pass"):
        sim.begin_pass(res.seed, res.mismatch)
        if res.mismatch != "none":
            for j in range(k):          # advance the gust AR(1) state
                sim.simulate(res.conds[j], res.logs[j].u, segment_rng(res.seed, j), res.mismatch)
    lg = res.logs[k]
    outcome = sim.simulate(lg.x, lg.u, segment_rng(res.seed, k), res.mismatch,
                           sample_droplets=True)
    consistent = abs(outcome.coverage_pct - lg.outcome.coverage_pct) < 1e-6
    # a shallow copy, as render_cards does, so a run in another session can share the renderer
    image, gt = copy.copy(m.renderer).render(outcome, np.random.default_rng([res.seed, k, 1]),
                                             degrade=degrade)
    meas, pred_mask = None, None
    v = m.verifier
    if v is not None:
        ppm = px_per_mm()
        if hasattr(v, "measure_full"):      # image readers B0 / B1: one detection
            meas, mask, _roi = v.measure_full(image, ppm)
        else:                               # e.g. the noisy sensor: reading, then its overlay
            meas = v.measure(image, ppm)
            mask = _segment_mask(v, image, ppm) if hasattr(v, "segment") else None
        if mask is not None:
            pred_mask = np.asarray(mask).astype(bool)
    return CardView(image, np.asarray(gt, bool), pred_mask, outcome, consistent, meas)


def _segment_mask(verifier, image: np.ndarray, ppm: float):
    """Stain mask from ``verifier.segment``, which is ``segment(image)`` on some readers and
    ``segment(image, px_per_mm)`` on others: the signature decides, so an error raised inside
    the reader is not mistaken for a signature mismatch."""
    seg = verifier.segment
    try:
        params = list(inspect.signature(seg).parameters.values())
    except (TypeError, ValueError):         # no introspectable signature: pass both arguments
        params = None
    positional = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    takes_ppm = (params is None
                 or sum(p.kind in positional for p in params) >= 2
                 or any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params))
    out = seg(image, ppm) if takes_ppm else seg(image)
    return out[0] if isinstance(out, tuple) else out


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
                     boom_control: bool = False, forecaster: str | None = None,
                     fixed_nozzle: str | tuple[str, str] | None = None,
                     progress: Callable[[int, int], None] | None = None,
                     feedback_model: Any = None) -> dict[str, PassResult]:
    """Run each policy on the same conditions, seed and hidden mismatch realization (fast_slow
    included: its displayed pass runs on ``seed``, its learning passes on later seeds, see
    ``run_pass_wrapper(same_field=True)``). With ``fixed_nozzle="auto"`` the nozzle is chosen
    once (first policy) and held for all of them. ``progress(done, total)`` is called after each
    policy run with ``total = len(policies)``. Holds ``_STACK_LOCK`` for the whole comparison."""
    out = {}
    with _STACK_LOCK:
        for i, p in enumerate(policies):
            out[p] = run_pass_wrapper(modules, conds, p, seed, mismatch, c_req, lam,
                                      fixed_setting=fixed_setting if p == "best_fixed" else None,
                                      render=render, n_learn=n_learn, speed_control=speed_control,
                                      boom_control=boom_control, forecaster=forecaster,
                                      fixed_nozzle=fixed_nozzle, same_field=True,
                                      feedback_model=feedback_model)
            nz = out[p].meta["nozzle"]
            if fixed_nozzle == "auto" and nz.get("family"):
                fixed_nozzle = (nz["family"], nz["size"])
            if progress is not None:
                progress(i + 1, len(policies))
    return out


SEED_COLUMNS = ["seed", "policy", "met_pct", "shortfall_pp", "volume_gpa", "drift_pct",
                "n_retreat", "n_infeasible"]


def compare_seeds(modules: Modules, preset: str, seeds: Sequence[int], mismatch: str,
                  c_req: float, lam: float | None = None,
                  policies: Sequence[str] = ("open", "fast", "fast_slow", "oracle"),
                  render: bool = False, n_learn: int = N_LEARN_PASSES,
                  speed_control: bool = False, boom_control: bool = False,
                  forecaster: str | None = None,
                  fixed_nozzle: str | tuple[str, str] | None = None,
                  progress: Callable[[int, int], None] | None = None,
                  feedback_model: Any = None) -> dict:
    """``compare_policies`` on several seeds, each an independent field: its own conditions
    (``make_conditions(preset, seed)``), its own best-fixed setting (if that policy is asked
    for) and, with ``fixed_nozzle="auto"``, its own held nozzle. Every policy runs on every
    seed, so the per-seed rows are paired by seed.

    Returns ``{"seeds", "policies", "per_seed", "first"}``: ``per_seed`` has one row per (seed,
    policy) with ``SEED_COLUMNS`` (met in percent of sections and drift in percent of sprayed
    volume, both from ``loop.summarize`` on the true outcome, as ``comparison_table``);
    ``first`` holds the full ``PassResult``s of the first seed only. ``progress(done, total)``
    is called after each (seed, policy) run, ``total = len(seeds) * len(policies)``. One seed
    gives the numbers of ``comparison_table(compare_policies(...))`` on that seed."""
    seeds, policies = [int(s) for s in seeds], list(policies)
    if not seeds or not policies:
        raise ValueError("compare_seeds needs at least one seed and one policy")
    total = len(seeds) * len(policies)
    rows: list[dict] = []
    first: dict[str, PassResult] = {}
    with _STACK_LOCK:
        for i, s in enumerate(seeds):
            conds = make_conditions(preset, s)
            fixed = best_fixed_for(modules, conds, float(c_req)) if "best_fixed" in policies else None
            hook = None if progress is None else (
                lambda d, _t, off=i * len(policies): progress(off + d, total))
            res = compare_policies(modules, conds, s, mismatch, c_req, lam, policies,
                                   render=render, fixed_setting=fixed, n_learn=n_learn,
                                   speed_control=speed_control, boom_control=boom_control,
                                   forecaster=forecaster, fixed_nozzle=fixed_nozzle,
                                   progress=hook, feedback_model=feedback_model)
            for p, r in res.items():
                sm = summarize(r.logs, r.c_req)
                rows.append({"seed": s, "policy": p, "met_pct": 100.0 * sm["frac_met"],
                             "shortfall_pp": sm["mean_shortfall_pp"],
                             "volume_gpa": sm["mean_volume_gpa"],
                             "drift_pct": 100.0 * sm["mean_drift_frac"],
                             "n_retreat": sm["n_retreat"], "n_infeasible": sm["n_infeasible"]})
            if i == 0:
                first = res
    return {"seeds": seeds, "policies": policies,
            "per_seed": pd.DataFrame(rows, columns=SEED_COLUMNS), "first": first}


_CI_METRICS = (("met", "met_pct"), ("volume", "volume_gpa"), ("drift", "drift_pct"))


def comparison_stats(per_seed: pd.DataFrame, reference: str | None = None,
                     n_boot: int = 2000, rng_seed: int = 0) -> pd.DataFrame:
    """One row per policy (order of first appearance in ``per_seed``, a ``compare_seeds``
    frame): means over seeds of met / volume / drift with 95 % percentile-bootstrap bounds over
    seeds (``_lo`` / ``_hi``, NaN with fewer than 2 seeds), and the means of shortfall, retreat
    flags and infeasible sections. With ``reference`` naming a present policy, also
    ``d_met_pp``: the mean over seeds of (met_pct - the reference's met_pct on the same seed),
    with a paired-bootstrap 95 % CI (``d_met_lo`` / ``d_met_hi``; seeds are resampled with
    their pairs kept together). The difference is NaN on the reference row; its bounds are NaN
    with fewer than 2 shared seeds. Each metric uses its own generator seeded by ``rng_seed``,
    so every policy is resampled with the same seed indices and the result is deterministic."""
    from coverageai.stats import bootstrap_mean_ci, paired_bootstrap_mean_ci
    pols = list(dict.fromkeys(per_seed["policy"]))
    paired = reference is not None and reference in pols
    ref = (per_seed[per_seed["policy"] == reference].set_index("seed")["met_pct"]
           if paired else None)
    nan = float("nan")
    rows = []
    for p in pols:
        d = per_seed[per_seed["policy"] == p]
        row: dict[str, Any] = {"policy": p, "n_seeds": int(d["seed"].nunique())}
        for j, (short, col) in enumerate(_CI_METRICS):
            m, lo, hi = bootstrap_mean_ci(d[col].to_numpy(float), n_boot,
                                          np.random.default_rng([int(rng_seed), j]))
            row[col], row[f"{short}_lo"], row[f"{short}_hi"] = m, lo, hi
        for col in ("shortfall_pp", "n_retreat", "n_infeasible"):
            row[col] = float(d[col].mean())
        if paired:
            if p == reference:
                row.update(d_met_pp=nan, d_met_lo=nan, d_met_hi=nan)
            else:
                own = d.set_index("seed")["met_pct"]
                common = [s for s in own.index if s in ref.index]
                m, lo, hi = paired_bootstrap_mean_ci(
                    own.loc[common].to_numpy(float), ref.loc[common].to_numpy(float), n_boot,
                    np.random.default_rng([int(rng_seed), len(_CI_METRICS)]))
                row.update(d_met_pp=m, d_met_lo=lo, d_met_hi=hi)
        rows.append(row)
    return pd.DataFrame(rows)


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
