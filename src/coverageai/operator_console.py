"""Operator console (the app's fourth view): what the product looks like for a spray operator.

The operator fits a nozzle (family, size), sets a label application rate in gal/acre (held
within +-``rate_control.tolerance_frac``) and a target travel speed. Nozzle, rate and speed fix
the starting pressure (``setup_for``). The nozzle stays on the boom for the whole field; the
console may change only pressure (the speed then follows so the rate is held) and, if allowed,
boom height, to keep card coverage at or above the fixed minimum (``MIN_COVERAGE``, the spec's
"Good" class limit, 30 %).

``run_console`` runs two passes on the same field and seed (paired):

- "Setting held": the operator's starting setting for every section (policy ``best_fixed`` with
  that one setting; boom height as in the field conditions).
- "Console adjusts": policy ``fast`` with the nozzle held, the rate locked (``rate_lock``) and
  boom-height control as chosen, with the app's card feedback (D31).

It then adds the rate-mode operator advice for the console run (``recommend_for_logs`` with
``rate_lock``; stored as the run's ``meta["recommendations"]`` so the replay shows the same
advice) and a model ranking of the other nozzles at the same rate in the same conditions
(``rate_control.alternative_nozzles``). Everything else here turns those results into numbers
and tables for ``app/operator_view.py``. Plain Python, no Streamlit.

Only ``app_logic.run_pass_wrapper`` touches the shared module stack; nothing here calls the
simulator. Simulation only: these are simulated passes, not field data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from coverageai import app_logic as al
from coverageai import catalog
from coverageai import rate_control as rc
from coverageai.config import spec
from coverageai.contracts import (Conditions, Setting, effective_boom_height_m,
                                  effective_speed_kmh)

HELD, CONSOLE = "held", "console"        # the two runs of a console result
TIERS = ("exact", "band", "unreachable")


def min_coverage() -> float:
    """The fixed coverage minimum: the lower limit of the "Good" class (spec, 30 %)."""
    return float(spec()["coverage_classes"]["acceptable_min"])


MIN_COVERAGE = min_coverage()


# ------------------------------------------------------------------ setup

@dataclass(frozen=True)
class Setup:
    """What nozzle + rate + speed imply before any spraying. ``pressure_psi`` is None when no
    pressure inside the nozzle's range gives the rate at the target speed."""
    family: str
    size: str
    gpa: float
    speed_kmh: float
    tol_frac: float
    pressure_psi: float | None
    droplet_class: str | None
    pressure_range_psi: tuple[float, float]
    speed_range_kmh: tuple[float, float] | None    # speeds that hold the rate, None if none does
    needed_psi: float | None                       # pressure the rate would need (may be out of range)
    n_candidates: int                              # iso-rate settings the console can choose from

    @property
    def ok(self) -> bool:
        return self.pressure_psi is not None and self.n_candidates > 0

    @property
    def rate_band(self) -> tuple[float, float]:
        return self.gpa * (1.0 - self.tol_frac), self.gpa * (1.0 + self.tol_frac)

    @property
    def setting(self) -> Setting | None:
        """The operator's starting setting (nozzle, starting pressure, target speed)."""
        if self.pressure_psi is None:
            return None
        return Setting(self.family, self.size, float(self.pressure_psi),
                       travel_speed_kmh=float(self.speed_kmh))

    def problem(self) -> str | None:
        """One or two plain sentences on why this setup cannot run, or None."""
        if self.ok:
            return None
        lo, hi = self.pressure_range_psi
        if self.speed_range_kmh is None:
            return (f"This nozzle cannot apply {self.gpa:g} gal/acre inside the travel-speed range "
                    f"at {lo:g} to {hi:g} psi. Choose another size or rate.")
        if self.pressure_psi is None:
            need = (f"It would need about {self.needed_psi:.0f} psi; " if self.needed_psi is not None
                    else "")
            a, b = self.speed_range_kmh
            return (f"No pressure from {lo:g} to {hi:g} psi gives {self.gpa:g} gal/acre at "
                    f"{self.speed_kmh:g} km/h. {need}set a speed from {a:.1f} to {b:.1f} km/h or "
                    f"choose another size.")
        return (f"The console has no pressure step that holds {self.gpa:g} gal/acre inside the "
                f"travel-speed range. Choose another size or rate.")


def setup_for(family: str, size: str, gpa: float, speed_kmh: float,
              tol_frac: float | None = None) -> Setup:
    """Starting pressure and droplet class implied by nozzle + rate + speed, and what else the
    console needs to know before it runs (see ``Setup``)."""
    tol = rc.tolerance_frac() if tol_frac is None else float(tol_frac)
    gpa, speed_kmh = float(gpa), float(speed_kmh)
    lo, hi = rc.pressure_bounds(family, size)
    p0 = rc.pressure_for_rate(family, size, gpa, speed_kmh) if gpa > 0 else None
    cls = catalog.flow_and_class(family, size, p0)[1] if p0 is not None else None
    s_lo, s_hi = (float(v) for v in spec()["ranges"]["travel_speed_kmh"])
    if gpa > 0:
        v_lo, v_hi = sorted((rc.speed_for_rate(family, size, lo, gpa),
                             rc.speed_for_rate(family, size, hi, gpa)))
    else:
        v_lo, v_hi = np.inf, -np.inf
    a, b = max(v_lo, s_lo), min(v_hi, s_hi)
    speeds = (float(a), float(b)) if a <= b + 1e-9 else None
    needed = None
    if gpa > 0:
        # flow follows sqrt(pressure) and the flow the rate needs grows with speed, so the
        # pressure needed is lo * (speed / speed that gives the rate at lo)^2
        v_at_lo = rc.speed_for_rate(family, size, lo, gpa)
        needed = float(lo * (speed_kmh / v_at_lo) ** 2) if v_at_lo > 0 else None
    n_c = len(rc.iso_rate_candidates(family, size, gpa, tol)) if gpa > 0 else 0
    return Setup(family, str(size), gpa, speed_kmh, tol, p0, cls, (float(lo), float(hi)), speeds,
                 needed, n_c)


# ------------------------------------------------------------------ the two runs

@dataclass
class ConsoleRun:
    """The two paired runs of one console press, with the advice and the nozzle ranking."""
    setup: Setup
    held: al.PassResult
    console: al.PassResult
    recs: list                       # recommend.Recommendation per section of the console run
    ranking: list[dict]              # every nozzle at this rate, best first (rank_key)
    boom_control: bool
    c_min: float = MIN_COVERAGE
    meta: dict = field(default_factory=dict)

    @property
    def nozzle(self) -> tuple[str, str]:
        return self.setup.family, self.setup.size

    @property
    def n(self) -> int:
        return self.console.n

    def own_rank(self) -> dict | None:
        """This nozzle's row in the ranking."""
        return next((r for r in self.ranking if (r["family"], r["size"]) == self.nozzle), None)

    def alternatives(self, top: int = 5) -> list[dict]:
        """The other nozzles, best first (``rank_key``)."""
        return [r for r in self.ranking if (r["family"], r["size"]) != self.nozzle][:max(int(top), 0)]


def run_console(modules, conds: Sequence[Conditions], seed: int, mismatch: str, family: str,
                size: str, gpa: float, speed_kmh: float, boom_control: bool = True,
                feedback_model=None, capture_cards: bool = False,
                progress: Callable[[int, int], None] | None = None) -> ConsoleRun:
    """Run "Setting held" and "Console adjusts" on the same field and seed (module docstring).
    Raises ValueError (before any run) if the setup cannot run (``Setup.problem``)."""
    setup = setup_for(family, size, gpa, speed_kmh)
    if not setup.ok:
        raise ValueError(setup.problem())
    c_min = MIN_COVERAGE
    n = len(conds)

    def tick(offset: int):
        if progress is None:
            return None
        return lambda done, total: progress(offset + done, 2 * n)

    held = al.run_pass_wrapper(modules, conds, "best_fixed", seed, mismatch, c_min,
                               fixed_setting=setup.setting, progress=tick(0))
    console = al.run_pass_wrapper(modules, conds, "fast", seed, mismatch, c_min,
                                  fixed_nozzle=(family, size), rate_lock=float(gpa),
                                  boom_control=bool(boom_control), feedback_model=feedback_model,
                                  capture_cards=capture_cards, progress=tick(n))
    from coverageai.recommend import recommend_for_logs
    recs = recommend_for_logs(console.modules.model, console.logs, c_min,
                              derive_fn=console.modules.derive_fn, rate_lock=float(gpa))
    console.meta["recommendations"] = recs      # the replay's advice (al.recommendations) is rate mode
    sel = getattr(modules.selector, "inner", modules.selector)
    ranking = sorted(rc.alternative_nozzles(list(conds), modules.model, c_min, float(gpa),
                                            top=10_000, selector=sel), key=rank_key)
    return ConsoleRun(setup, held, console, recs, ranking, bool(boom_control), c_min)


# ------------------------------------------------------------------ numbers

def rate_stats(res: al.PassResult, gpa: float, tol_frac: float) -> dict:
    """Applied rate over the run: mean, min, max, largest relative deviation, sections outside
    the band."""
    r = np.array([lg.derived.app_rate_gpa for lg in res.logs], float)
    dev = r / float(gpa) - 1.0
    out = np.abs(dev) > float(tol_frac) + 1e-9
    return {"mean": float(r.mean()), "min": float(r.min()), "max": float(r.max()),
            "max_dev_frac": float(np.abs(dev).max()), "n_out_of_band": int(out.sum()),
            "in_band": bool(not out.any())}


def tiers(res: al.PassResult) -> list[str | None]:
    """The selector's rate tier per section ("exact", "band", "unreachable"), None without one."""
    out = []
    for k in range(res.n):
        info = res.info_for(k) or {}
        out.append(info.get("rate_tier"))
    return out


def summary(run: ConsoleRun) -> dict:
    """Run totals for both runs: ``app_logic.compute_kpis`` (the app's one source of totals)
    plus the rate numbers. The console run also counts the sections the model said cannot
    reach the minimum at this rate."""
    s = run.setup
    out = {}
    for key, res in ((HELD, run.held), (CONSOLE, run.console)):
        kp = al.compute_kpis(res.logs, run.c_min)
        kp["rate"] = rate_stats(res, s.gpa, s.tol_frac)
        out[key] = kp
    t = tiers(run.console)
    out[CONSOLE]["tiers"] = {k: sum(1 for v in t if v == k) for k in TIERS}
    out[CONSOLE]["n_unreachable"] = out[CONSOLE]["tiers"]["unreachable"]
    return out


def section_frame(run: ConsoleRun) -> pd.DataFrame:
    """One row per section of the console run (internal column names; ``app/labels.py`` holds
    the display names). ``card_class`` is the class of the card reading; ``tier`` the selector's
    rate tier."""
    rows = []
    t = tiers(run.console)
    for k, lg in enumerate(run.console.logs):
        rows.append({
            "section": k + 1, "wind_ms": lg.x.wind_speed_ms, "pressure_psi": lg.u.pressure_psi,
            "speed_kmh": effective_speed_kmh(lg.x, lg.u),
            "boom_height_m": effective_boom_height_m(lg.x, lg.u),
            "rate_gpa": lg.derived.app_rate_gpa, "predicted_cov": lg.pred.coverage_pct,
            "card_cov": lg.meas.coverage_pct, "actual_cov": lg.outcome.coverage_pct,
            "card_class": lg.meas.coverage_class, "tier": t[k],
        })
    return pd.DataFrame(rows, columns=["section", "wind_ms", "pressure_psi", "speed_kmh",
                                       "boom_height_m", "rate_gpa", "predicted_cov", "card_cov",
                                       "actual_cov", "card_class", "tier"])


def advice_text(rec) -> str:
    """One sentence of rate-mode advice for a section (empty if nothing to change)."""
    if rec.status == "ok":
        return ""
    if rec.status == "adjust":
        return rec.best.describe()
    if rec.status == "unavailable":
        return "No advice: this nozzle is not in the catalog."
    joint = " and ".join(a.describe(effect=False) for a in rec.combined)
    if rec.status == "combine":
        return f"{joint}: coverage about {rec.combined_cov:.0f}%"
    if joint:
        return f"Best with this nozzle: {joint} (coverage about {rec.combined_cov:.0f}%)"
    return "No pressure or boom change raises coverage with this nozzle at this rate."


def advice_frame(run: ConsoleRun) -> pd.DataFrame:
    """Sections whose card read below the minimum, with what to change (internal names)."""
    rows = []
    for k, (lg, rec) in enumerate(zip(run.console.logs, run.recs)):
        if rec.status == "ok":
            continue
        rows.append({"section": k + 1, "card_cov": rec.observed_cov,
                     "short_pp": rec.shortfall_pp, "status": rec.status, "advice": advice_text(rec)})
    return pd.DataFrame(rows, columns=["section", "card_cov", "short_pp", "status", "advice"])


def first_move(rec) -> tuple[str, int] | None:
    """(control, +1 raise / -1 lower) of the main move a recommendation asks for, or None."""
    moves = list(rec.combined) if rec.status in ("combine", "unreachable") and rec.combined else list(rec.options[:1])
    if not moves:
        return None
    a = max(moves, key=lambda m: m.move_frac)
    return a.knob, (1 if a.delta > 0 else -1)


def advice_counts(run: ConsoleRun) -> dict:
    """Counts behind the "What to change" sentence: cards below the minimum, how many a change
    is predicted to fix, how many no change fixes, and the most common move (control, direction)
    over the sections that have one."""
    short = [r for r in run.recs if r.status != "ok"]
    fix = [r for r in short if r.status in ("adjust", "combine")]
    moves = [m for m in (first_move(r) for r in short) if m is not None]
    common = max(set(moves), key=moves.count) if moves else None
    return {"n": len(run.recs), "n_short": len(short), "n_fixable": len(fix),
            "n_unfixable": len(short) - len(fix), "common_move": common,
            "n_common": moves.count(common) if common else 0}


def alternatives_frame(run: ConsoleRun, top: int = 5) -> pd.DataFrame:
    """The ``top`` other nozzles at the same rate (internal names): share of sections the model
    predicts at or above the minimum, mean predicted coverage and drift, pressure and speed
    ranges the console would use. Model predictions with no card feedback."""
    rows = []
    for r in run.alternatives(top):
        rows.append({"family": r["family"], "size": r["size"],
                     "share_pct": 100.0 * r["share_meeting"], "mean_cov": r["mean_cov"],
                     "drift_pct": 100.0 * r["mean_drift"],
                     "pressure_psi": r["pressure_range_psi"], "speed_kmh": r["speed_range_kmh"],
                     "better": _better(r, run.own_rank())})
    return pd.DataFrame(rows, columns=["family", "size", "share_pct", "mean_cov", "drift_pct",
                                       "pressure_psi", "speed_kmh", "better"])


def rank_key(r: dict) -> tuple:
    """Order of the nozzle suggestions. First the share of sections the model predicts at or
    above the minimum (with the controller's safety margin), high first. Among nozzles with the
    same share: lower mean predicted drift when some sections reach the minimum (the
    ``alternative_nozzles`` order); higher mean predicted coverage when none does, because a
    nozzle that reaches no section should at least come closest."""
    share = round(float(r["share_meeting"]), 9)
    return (-share, float(r["mean_drift"]) if share > 0 else -float(r["mean_cov"]),
            r["family"], r["size"])


def _better(r: dict, own: dict | None) -> bool:
    """Whether nozzle row ``r`` ranks above this nozzle's row (``rank_key``)."""
    return own is None or rank_key(r) < rank_key(own)


def cfg(preset: str, seed: int, mismatch: str, family: str, size: str, gpa: float, speed_kmh: float,
        boom_control: bool, model: str, verifier: str, calibrate: bool, feedback: str) -> tuple:
    """What a console run depends on (the paper-card switch is a display choice, not part of it)."""
    return (preset, int(seed), mismatch, family, str(size), float(gpa), float(speed_kmh),
            bool(boom_control), model, verifier, bool(calibrate), feedback)
