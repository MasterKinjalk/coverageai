"""Operator recommendations: when a section's coverage is below target, which of pressure,
travel speed and boom height to change, and by how much (decision D20).

The nozzle stays the one on the boom. For each of the three in-cab controls the model is asked
what coverage it predicts along that control's legal range with the other two held, and the
smallest change whose predicted gain closes the shortfall is reported, and the controls are
listed smallest move first (as a share of each control's legal range). The model supplies the
*effect* of a change (predicted coverage at the new setting minus predicted coverage at the
current one); the *level* comes from what was observed (the card reading), so a model that is
biased here still gives a usable step size. If no single control is enough, the smallest
combined change is searched; if that fails too the best achievable setting is returned with
status "unreachable".

Works with any ``contracts.Model`` (``predict_batch``); nothing here touches the simulator, so
it is an output of the predictive model only. ``recommend_for_logs`` evaluates a whole pass with
one batched model call.
"""
from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from coverageai.config import spec
from coverageai.contracts import (Conditions, Derived, Setting, effective_boom_height_m,
                                  effective_speed_kmh)

KNOBS = ("pressure", "speed", "boom_height")
UNITS = {"pressure": "psi", "speed": "km/h", "boom_height": "m"}
STEPS = {"pressure": 5.0, "speed": 1.0, "boom_height": 0.05}      # size of one recommended step
MAX_JOINT_VALUES = 6       # values per control tried in a joint move (6^3 + 1 settings at most)
_DECIMALS = {"pressure": 0, "speed": 1, "boom_height": 2}
_VERBS = {"pressure": ("Raise pressure", "Lower pressure"), "speed": ("Speed up", "Slow down"),
          "boom_height": ("Raise the boom", "Lower the boom")}


@dataclass(frozen=True)
class Adjustment:
    """One control moved from ``current`` to ``proposed``; effects are model predictions."""
    knob: str
    current: float
    proposed: float
    gain_pp: float                 # predicted coverage change for this move alone
    expected_cov: float            # observed coverage + gain_pp
    drift_change_pp: float         # predicted drift change, percentage points of sprayed volume
    volume_change_gpa: float       # change in applied volume, gal/acre
    reaches_target: bool
    move_frac: float = 0.0         # size of the move as a share of the control's legal range

    @property
    def delta(self) -> float:
        return self.proposed - self.current

    @property
    def unit(self) -> str:
        return UNITS[self.knob]

    def describe(self, effect: bool = True) -> str:
        nd = _DECIMALS[self.knob]
        up, down = _VERBS[self.knob]
        s = (f"{up if self.delta > 0 else down} {self.current:.{nd}f} to {self.proposed:.{nd}f} "
             f"{self.unit} ({self.delta:+.{nd}f} {self.unit})")
        if effect:
            s += f": coverage about {self.expected_cov:.0f}% ({self.gain_pp:+.1f} pp)"
        return s


@dataclass(frozen=True)
class Recommendation:
    """Advice for one section. ``status``: "ok" (nothing to change), "unavailable" (below
    target but the nozzle is not in the catalog, so there is nothing to search), "adjust" (at least one
    single control reaches the target; ``options`` holds one entry per control, best first),
    "combine" (only the joint move in ``combined`` reaches it) or "unreachable" (``combined``
    is the best the three controls can do with this nozzle)."""
    status: str
    observed_cov: float
    target_cov: float
    options: tuple[Adjustment, ...] = ()
    combined: tuple[Adjustment, ...] = ()
    combined_cov: float | None = None
    setting: Setting | None = None           # the recommended setting, if any

    @property
    def shortfall_pp(self) -> float:
        return max(self.target_cov - self.observed_cov, 0.0)

    @property
    def best(self) -> Adjustment | None:
        return self.options[0] if self.options and self.options[0].reaches_target else None

    def headline(self) -> str:
        """One sentence for a status line."""
        if self.status == "ok":
            return "Coverage meets the target: no change needed."
        if self.status == "adjust":
            return self.best.describe()
        if self.status == "unavailable":
            return "Below target; no recommendation for a nozzle that is not in the catalog."
        joint = " and ".join(a.describe(effect=False) for a in self.combined)
        if self.status == "combine":
            return f"{joint}: coverage about {self.combined_cov:.0f}%"
        if not joint:
            return "No change of pressure, speed or boom height raises coverage with this nozzle."
        return (f"Target not reachable with this nozzle. Best: {joint} (coverage about "
                f"{self.combined_cov:.0f}%); re-spray or fit a larger nozzle.")

    def lines(self) -> list[str]:
        """Headline plus the other single-control options that also reach the target."""
        out = [self.headline()]
        if self.status == "adjust":
            out += ["or " + a.describe()[0].lower() + a.describe()[1:]
                    for a in self.options[1:] if a.reaches_target]
        return out


def _legal_range(knob: str, x: Conditions, u: Setting) -> tuple[float, float]:
    from coverageai import catalog
    s = spec()
    if knob == "pressure":
        lo, hi = catalog.pressure_range(u.family, u.size)
        r = s["pressure_range_psi"][u.family]
        return max(lo, float(r[0])), min(hi, float(r[1]))
    if knob == "speed":
        lo, hi = s["ranges"]["travel_speed_kmh"]
        return float(lo), float(hi)
    lo, hi = s["ranges"]["boom_height_m"]
    return max(float(lo), catalog.min_boom_height_m(u.family)), float(hi)


def current_value(knob: str, x: Conditions, u: Setting) -> float:
    if knob == "pressure":
        return float(u.pressure_psi)
    if knob == "speed":
        return float(effective_speed_kmh(x, u))
    return float(effective_boom_height_m(x, u))


def with_value(knob: str, u: Setting, value: float) -> Setting:
    field = {"pressure": "pressure_psi", "speed": "travel_speed_kmh", "boom_height": "boom_height_m"}[knob]
    return dataclasses.replace(u, **{field: float(value)})


def knob_values(knob: str, x: Conditions, u: Setting, step: float | None = None) -> list[float]:
    """Alternative values of one control: the current value plus or minus whole steps inside its
    legal range (catalog pressure range of this nozzle, spec speed range, spec boom range above
    the family's minimum overlap height), and the range ends. The current value is excluded."""
    lo, hi = _legal_range(knob, x, u)
    cur = current_value(knob, x, u)
    step = STEPS[knob] if step is None else float(step)
    vals = {round(lo, 4), round(hi, 4)} if hi > lo else set()
    k = 1
    while cur - k * step >= lo - 1e-9 or cur + k * step <= hi + 1e-9:
        for v in (cur - k * step, cur + k * step):
            if lo - 1e-9 <= v <= hi + 1e-9:
                vals.add(round(v, 4))
        k += 1
    return sorted(v for v in vals if abs(v - cur) > 1e-6)


def _plan(x: Conditions, u: Setting) -> list[tuple[str, float, Setting]]:
    try:
        return [(kn, v, with_value(kn, u, v)) for kn in KNOBS for v in knob_values(kn, x, u)]
    except (KeyError, ValueError):       # nozzle not in the catalog (mock modules)
        return []


def _adjustment(knob, x, u, value, gain, observed, target, drift0, drift1, gpa0, gpa1) -> Adjustment:
    cur = current_value(knob, x, u)
    lo, hi = _legal_range(knob, x, u)
    return Adjustment(knob, cur, float(value), float(gain),
                      float(observed + gain), float(100.0 * (drift1 - drift0)),
                      float(gpa1 - gpa0), bool(observed + gain >= target),
                      float(abs(value - cur) / max(hi - lo, 1e-9)))


def _rank(a: Adjustment) -> tuple:
    """Controls that reach the target first, the smallest move (share of its legal range) first
    among them, then the least added drift; the rest by predicted gain."""
    if a.reaches_target:
        return (0, a.move_frac, a.drift_change_pp)
    return (1, -a.gain_pp, a.move_frac)


def recommend_for_logs(model, logs: Sequence, c_req: float, margin_pp: float = 0.0,
                       derive_fn: Callable[[Conditions, Setting], Derived] | None = None,
                       observed: str = "measured") -> list[Recommendation]:
    """One Recommendation per SegmentLog. ``observed``: "measured" uses the card reading as the
    coverage to improve on, "predicted" the model's corrected prediction (no card needed).
    The target is ``c_req + margin_pp``."""
    if derive_fn is None:
        from coverageai import catalog
        derive_fn = catalog.derive
    target = float(c_req) + float(margin_pp)
    obs = [float(lg.meas.coverage_pct if observed == "measured" else lg.corrected_pred_cov)
           for lg in logs]
    short = [o < target for o in obs]
    plans = [(_plan(lg.x, lg.u) if short[i] else []) for i, lg in enumerate(logs)]
    xs, us, where = [], [], []
    for i, lg in enumerate(logs):
        if not plans[i]:
            continue
        where.append((i, len(us)))
        xs += [lg.x] * (len(plans[i]) + 1)
        us += [lg.u] + [alt for _, _, alt in plans[i]]
    preds = model.predict_batch(xs, us) if us else []
    # "unavailable": below target but this nozzle has no catalog ranges to search (mock modules)
    out = [Recommendation("unavailable" if short[i] and not plans[i] else "ok", obs[i], target)
           for i in range(len(logs))]
    pending = []                          # sections where no single control is enough
    for i, start in where:
        lg, plan = logs[i], plans[i]
        base = preds[start]
        gpa0 = derive_fn(lg.x, lg.u).app_rate_gpa
        best: dict[str, Adjustment] = {}
        for j, (knob, value, alt) in enumerate(plan):
            p = preds[start + 1 + j]
            a = _adjustment(knob, lg.x, lg.u, value, p.coverage_pct - base.coverage_pct, obs[i],
                            target, base.drift_frac, p.drift_frac, gpa0,
                            derive_fn(lg.x, alt).app_rate_gpa)
            b = best.get(knob)
            # smallest move that reaches the target; failing that, the largest gain
            key = (not a.reaches_target, abs(a.delta) if a.reaches_target else -a.gain_pp)
            if b is None or key < (not b.reaches_target,
                                   abs(b.delta) if b.reaches_target else -b.gain_pp):
                best[knob] = a
        opts = tuple(sorted((a for a in best.values() if a.gain_pp > 0), key=_rank))
        if opts and opts[0].reaches_target:
            out[i] = Recommendation("adjust", obs[i], target, opts,
                                    setting=with_value(opts[0].knob, lg.u, opts[0].proposed))
        else:
            out[i] = Recommendation("unreachable", obs[i], target, opts)
            pending.append((i, opts))
    _combine(model, logs, obs, target, derive_fn, pending, out)
    return out


def _combine(model, logs, obs, target, derive_fn, pending, out) -> None:
    """Joint moves for sections where no single control reaches the target: every combination of
    the helpful direction of each control (all step values from the current one to that control's
    best single value). Picks the combination that reaches the target with the smallest total
    relative move, else the one with the highest predicted coverage."""
    xs, us, spans = [], [], []
    for i, opts in pending:
        lg = logs[i]
        axes = []
        for a in opts:
            lo, hi = sorted((a.current, a.proposed))
            vals = [v for v in knob_values(a.knob, lg.x, lg.u) if lo - 1e-9 <= v <= hi + 1e-9]
            if len(vals) > MAX_JOINT_VALUES:      # keep both ends, thin the middle
                keep = np.unique(np.linspace(0, len(vals) - 1, MAX_JOINT_VALUES).round().astype(int))
                vals = [vals[j] for j in keep]
            axes.append([(a.knob, a.current)] + [(a.knob, v) for v in vals])
        combos = [c for c in itertools.product(*axes)] if axes else []
        sets = []
        for c in combos:
            u2 = lg.u
            for knob, v in c:
                if abs(v - current_value(knob, lg.x, lg.u)) > 1e-6:
                    u2 = with_value(knob, u2, v)
            sets.append(u2)
        spans.append((i, len(us), combos, sets))
        xs += [lg.x] * (len(sets) + 1)
        us += [lg.u] + sets
    preds = model.predict_batch(xs, us) if us else []
    for i, start, combos, sets in spans:
        if not combos:
            continue
        lg, base = logs[i], preds[start]
        gain = np.array([preds[start + 1 + j].coverage_pct - base.coverage_pct
                         for j in range(len(sets))])
        ranges = {}
        for kn in KNOBS:
            lo_, hi_ = _legal_range(kn, lg.x, lg.u)
            ranges[kn] = max(hi_ - lo_, 1e-9)
        move = np.array([sum(abs(v - current_value(kn, lg.x, lg.u)) / ranges[kn] for kn, v in c)
                         for c in combos])
        ok = obs[i] + gain >= target
        j = int(np.where(ok, move, np.inf).argmin()) if ok.any() else int(gain.argmax())
        if gain[j] <= 0:
            continue
        gpa0 = derive_fn(lg.x, lg.u).app_rate_gpa
        p = preds[start + 1 + j]
        parts = tuple(
            _adjustment(kn, lg.x, lg.u, v, gain[j], obs[i], target, base.drift_frac, p.drift_frac,
                        gpa0, derive_fn(lg.x, sets[j]).app_rate_gpa)
            for kn, v in combos[j] if abs(v - current_value(kn, lg.x, lg.u)) > 1e-6)
        out[i] = Recommendation("combine" if ok[j] else "unreachable", obs[i], target,
                                out[i].options, parts, float(obs[i] + gain[j]), sets[j])


def recommend(model, x: Conditions, u: Setting, observed_cov: float, c_req: float,
              margin_pp: float = 0.0, derive_fn=None) -> Recommendation:
    """Recommendation for a single (conditions, setting) with an observed coverage."""
    from types import SimpleNamespace
    lg = SimpleNamespace(x=x, u=u, meas=SimpleNamespace(coverage_pct=float(observed_cov)),
                         corrected_pred_cov=float(observed_cov))
    return recommend_for_logs(model, [lg], c_req, margin_pp, derive_fn)[0]
