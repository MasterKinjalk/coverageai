"""Fixed application rate (opt-in rate_lock) evaluation, simulation only.

The operator's nozzle stays on the boom and the rate (gal/acre) is held within +-5 %. The
controller moves only pressure (travel speed follows) and, optionally, boom height, to keep true
coverage >= 30 %. Policies, all paired on the same (preset, seed, setup):
  held            best_fixed with the operator's setting (nozzle, start pressure, speed) all field
  rate_open       open loop, nozzle_lock + rate_lock
  rate_fast       fast loop (rule feedback), same locks
  rate_fast_boom  as rate_fast plus boom-height control (app_logic.BOOM_HEIGHT_OPTIONS_M)
  free_fast       reference, not a peer: fast loop, free nozzle and free rate (can raise the rate)
Stack: build_modules("spray", "noisy"), evaluation seeds 0-19, 40 sections, mismatch med.

Usage: uv run --no-sync python experiments/eval_rate_lock.py [--seeds 20] [--workers 24]
       [--presets a,b] [--pilot]
Writes results/rate_lock_eval.{md,json} and experiments/figures/rate_lock_eval.png.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

C_REQ = 30.0
MISMATCH = "med"
N_SEC = 40
TOL = 0.05
# (nozzle family, size, rate gal/acre, operator speed km/h)
# Chosen by hand to span families and difficulty after a small pilot scan (seeds 0-1, two presets,
# rate_fast only) that showed which nozzle and rate pairs can reach 30 %: easy, middle, hard, never.
SETUPS = [("TP65", "06", 20.0, 16.0), ("3D", "04", 20.0, 12.0), ("ULD", "03", 15.0, 12.0),
          ("XR", "04", 15.0, 12.0), ("ULD", "04", 15.0, 16.0), ("DGX40", "04", 10.0, 20.0)]
POLICIES = ["held", "rate_open", "rate_fast", "rate_fast_boom", "free_fast"]
_MODS = None


def setup_name(s) -> str:
    return f"{s[0]}-{s[1]} {s[2]:g} gpa @ {s[3]:g} km/h"


def gname(g) -> str:
    return "pooled" if g is None else setup_name(SETUPS[g])


def _init():
    global _MODS
    from coverageai.integration import build_modules
    _MODS = build_modules("spray", "noisy", n_jobs=1)


def _run(policy, setup, conds, seed, pid):
    from coverageai import rate_control as rc
    from coverageai.app_logic import BOOM_HEIGHT_OPTIONS_M
    from coverageai.contracts import Setting
    from coverageai.loop import run_pass
    fam, size, gpa, speed = setup
    m = _MODS
    sel = m.selector
    kw: dict = {}
    saved = {a: getattr(sel, a) for a in ("nozzle_lock", "rate_lock", "boom_height_options_m")}
    try:
        if policy == "held":
            p0 = rc.pressure_for_rate(fam, size, gpa, speed)
            kw = dict(fixed_setting=Setting(fam, size, p0, travel_speed_kmh=speed))
            pol = "best_fixed"
        else:
            pol = "open" if policy == "rate_open" else "fast"
            if policy != "free_fast":
                sel.nozzle_lock, sel.rate_lock = (fam, size), float(gpa)
            if policy == "rate_fast_boom":
                sel.boom_height_options_m = list(BOOM_HEIGHT_OPTIONS_M)
        m.simulator.begin_pass(seed, MISMATCH)
        logs = run_pass(list(conds), pol, m, seed, MISMATCH, C_REQ, pid, render=False, **kw)
        # hidden wear flow factor, read from a copy of the plant (never the live one)
        ff = float(copy.deepcopy(m.simulator)._pass_factors[1])
    finally:
        for a, v in saved.items():
            setattr(sel, a, v)
    cov = np.array([lg.outcome.coverage_pct for lg in logs])
    gp = np.array([lg.derived.app_rate_gpa for lg in logs])
    return dict(
        hit=float((cov >= C_REQ).mean()), cov=float(cov.mean()),
        drift=float(np.mean([lg.outcome.drift_frac for lg in logs])),
        rate=float(gp.mean()), rate_ok=float((np.abs(gp / gpa - 1) <= TOL + 1e-9).mean()),
        true_rate=float((gp * ff).mean()),
        true_rate_ok=float((np.abs(gp * ff / gpa - 1) <= TOL + 1e-9).mean()),
        unreach=float(np.mean([bool(lg.selector_infeasible) for lg in logs])),
        speed=float(np.mean([lg.u.travel_speed_kmh if lg.u.travel_speed_kmh is not None
                             else lg.x.travel_speed_kmh for lg in logs])),
        pressure=float(np.mean([lg.u.pressure_psi for lg in logs])), flow_factor=ff)


def task(args):
    from coverageai.app_logic import make_conditions
    preset, seed = args
    conds = make_conditions(preset, seed, N_SEC)
    out = []
    for si, s in enumerate(SETUPS):
        for pol in POLICIES:
            r = _run(pol, s, conds, seed, f"{pol}_{preset}_{seed}_{si}")
            out.append(dict(preset=preset, seed=seed, setup=si, policy=pol, **r))
    return out


def per_seed(rows, setup, pol, key):
    """Mean over presets (and over setups when setup is None) per seed: the paired unit."""
    d: dict[int, list] = {}
    for r in rows:
        if r["policy"] == pol and (setup is None or r["setup"] == setup):
            d.setdefault(r["seed"], []).append(r[key])
    return np.array([np.mean(d[k]) for k in sorted(d)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--presets", default="")
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    from coverageai import rate_control as rc, scenarios
    presets = a.presets.split(",") if a.presets else scenarios.list_presets()
    seeds = list(range(a.seeds))
    p0s = [rc.pressure_for_rate(s[0], s[1], s[2], s[3]) for s in SETUPS]
    for s, p in zip(SETUPS, p0s):
        print(setup_name(s), "start pressure", None if p is None else round(p, 2), "psi")
        assert p is not None
    t0 = time.time()
    jobs = [(p, s) for p in presets for s in seeds]
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        rows = [r for part in ex.map(task, jobs) for r in part]
    rt = time.time() - t0
    rows.sort(key=lambda r: (r["setup"], r["policy"], r["preset"], r["seed"]))
    print("passes", len(rows), "runtime_s", round(rt, 1))
    report(rows, presets, seeds, p0s, rt, a.pilot)


def report(rows, presets, seeds, p0s, rt, pilot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from coverageai.stats import bootstrap_mean_ci, paired_bootstrap_mean_ci
    keys = ["hit", "drift", "rate", "rate_ok", "true_rate", "true_rate_ok", "unreach"]
    groups = list(range(len(SETUPS))) + [None]
    summ: dict = {}
    for g in groups:
        summ[gname(g)] = {pol: {k: list(bootstrap_mean_ci(per_seed(rows, g, pol, k), 4000, 0))
                                for k in keys} for pol in POLICIES}
    pairs = [("rate_fast", "held"), ("rate_fast_boom", "rate_fast"), ("rate_open", "rate_fast")]
    cis: dict = {}
    for g in groups:
        cis[gname(g)] = {f"{p} - {q}": {k: list(paired_bootstrap_mean_ci(
            per_seed(rows, g, p, k), per_seed(rows, g, q, k), 4000, 0))
            for k in ("hit", "drift", "rate_ok")} for p, q in pairs}
    ffs = [r["flow_factor"] for r in rows if r["policy"] == "held" and r["setup"] == 0]
    out = dict(config=dict(c_req=C_REQ, mismatch=MISMATCH, sections=N_SEC, seeds=seeds,
                           presets=presets, runtime_s=rt,
                           setups=[dict(nozzle=f"{s[0]}-{s[1]}", gpa=s[2], speed_kmh=s[3],
                                        start_pressure_psi=p) for s, p in zip(SETUPS, p0s)]),
               summary=summ, paired_ci=cis, per_pass=rows)
    if pilot:
        print(json.dumps(summ["pooled"], indent=1))
        return
    (ROOT / "results" / "rate_lock_eval.json").write_text(json.dumps(out, indent=1))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    w = 0.16
    cols = ["#888888", "#6baed6", "#2171b5", "#08306b", "#e08a1e"]
    for i, pol in enumerate(POLICIES):
        m = np.array([summ[gname(g)][pol]["hit"] for g in groups])
        ax.bar(np.arange(len(groups)) + (i - 2) * w, m[:, 0] * 100, w, label=pol, color=cols[i],
               yerr=[(m[:, 0] - m[:, 1]) * 100, (m[:, 2] - m[:, 0]) * 100], capsize=2)
    ax.set_xticks(range(len(groups)), [gname(g).replace(" @ ", "\n@ ") for g in groups], fontsize=8)
    ax.set_ylabel("sections with true coverage >= 30 % (%)")
    ax.set_title("Fixed application rate: share of sections at target (mean, 95 % CI; simulation)")
    ax.legend(fontsize=8, ncol=5, loc="upper center")
    ax.set_ylim(0, 115)
    fig.tight_layout()
    fig.savefig(ROOT / "experiments" / "figures" / "rate_lock_eval.png", dpi=140)

    def f(z, s=100, d=1):
        return f"{z[0]*s:.{d}f} [{z[1]*s:.{d}f}, {z[2]*s:.{d}f}]"

    L = ["# Fixed application rate: evaluation (simulation only)", "",
         "Everything here is simulated. Nothing is field validation.", "",
         "## Setup", "",
         f"- Seeds {seeds[0]} to {seeds[-1]} (evaluation seeds), {N_SEC} sections, mismatch `{MISMATCH}`, "
         f"target coverage {C_REQ:g} %.",
         f"- Scenario presets (all {len(presets)}): {', '.join(presets)}. The paired unit is the seed; "
         "each seed value is the mean over presets, so the CIs resample seeds.",
         "- Stack: `build_modules(\"spray\", \"noisy\")` (LightGBM v1, noisy reader, rule feedback, library defaults).",
         "- The six setups were chosen by hand after a small pilot scan (seeds 0 and 1, two presets, rate_fast "
         "only) to span families and difficulty. Those two seeds are also evaluation seeds, so the choice "
         "of setups is not independent of the evaluation data.",
         "- Metrics use the TRUE outcome of the simulator, not the card reading.",
         f"- Total passes: {len(rows)}. Runtime: {rt/60:.1f} min.", "",
         "Operator setups. The held setting uses the start pressure that gives the rate at the operator speed.", "",
         "| # | nozzle | rate gpa | speed km/h | start pressure psi |", "|---|---|---|---|---|"]
    for i, (s, p) in enumerate(zip(SETUPS, p0s)):
        L.append(f"| {i+1} | {s[0]}-{s[1]} | {s[2]:g} | {s[3]:g} | {p:.1f} |")
    L += ["", "Policies:", "",
          "- `held`: the operator's setting for the whole field.",
          "- `rate_open`: open loop, nozzle and rate locked, pressure (speed follows) chosen per section.",
          "- `rate_fast`: as `rate_open` with the fast card-feedback loop.",
          "- `rate_fast_boom`: as `rate_fast` plus boom-height control.",
          "- `free_fast`: REFERENCE, not a peer. Free nozzle and free rate, the existing controller. "
          "It can raise the rate, so it wins on coverage by construction.", "",
          "## Results", "", "Mean over seeds with 95 % bootstrap CI in brackets. Shares in percent.", ""]
    for g in groups:
        gk = gname(g)
        L += [f"### {'Pooled over the 6 setups' if g is None else 'Setup ' + str(g+1) + ': ' + gk}", "",
              "| policy | true cov >= 30 % (%) | mean drift (%) | set rate gpa | set rate within 5 % (%) | true rate within 5 % (%) | unreachable (%) |",
              "|---|---|---|---|---|---|---|"]
        for pol in POLICIES:
            s = summ[gk][pol]
            L.append(f"| {pol} | {f(s['hit'])} | {f(s['drift'])} | {f(s['rate'],1,2)} | "
                     f"{f(s['rate_ok'])} | {f(s['true_rate_ok'])} | {f(s['unreach'])} |")
        L.append("")
    L += ["## Paired differences", "",
          "Mean paired difference over seeds with 95 % CI, in percentage points (drift in points of emitted volume).", "",
          "| setup | comparison | d share >= 30 % | d drift | d set-rate-within-5 % |", "|---|---|---|---|---|"]
    for g in groups:
        gk = gname(g)
        for p, q in pairs:
            c = cis[gk][f"{p} - {q}"]
            L.append(f"| {gk} | {p} - {q} | {f(c['hit'])} | {f(c['drift'])} | {f(c['rate_ok'])} |")
    L += ["", "## Hidden wear", "",
          f"The simulator multiplies the flow by a hidden wear factor drawn once per pass "
          f"(here {min(ffs):.3f} to {max(ffs):.3f}). The true rate is the set rate times this factor, "
          "read from a deepcopy of the simulator after the pass (the live plant is not touched). "
          "The column \"true rate within 5 %\" shows how often the delivered rate stays in the band. "
          "The controller cannot see this factor, so it cannot hold the true rate. The band tier "
          "picks rates up to +5 % above target, and wear adds more on top, so the delivered rate is "
          "often outside the 5 % band even though the set rate is inside it.", ""]
    P = summ["pooled"]
    d1, d2, d3 = (cis["pooled"][f"{p} - {q}"]["hit"] for p, q in pairs)

    def sig(z):
        return "excludes 0" if (z[1] > 0 or z[2] < 0) else "includes 0"

    L += ["## Verdict", "",
          f"- Pooled share at target: held {P['held']['hit'][0]*100:.1f} %, rate_open {P['rate_open']['hit'][0]*100:.1f} %, "
          f"rate_fast {P['rate_fast']['hit'][0]*100:.1f} %, rate_fast_boom {P['rate_fast_boom']['hit'][0]*100:.1f} %, "
          f"free_fast (reference) {P['free_fast']['hit'][0]*100:.1f} %.",
          f"- rate_fast vs held: {d1[0]*100:+.1f} pp, CI [{d1[1]*100:+.1f}, {d1[2]*100:+.1f}], {sig(d1)}.",
          f"- rate_fast_boom vs rate_fast: {d2[0]*100:+.1f} pp, CI [{d2[1]*100:+.1f}, {d2[2]*100:+.1f}], {sig(d2)}.",
          f"- rate_open vs rate_fast: {d3[0]*100:+.1f} pp, CI [{d3[1]*100:+.1f}, {d3[2]*100:+.1f}], {sig(d3)}.",
          "- The locked modes hold the SET rate by construction. Where the nozzle at that rate cannot "
          "reach 30 % at any pressure, the selector flags it (unreachable column), and no policy that "
          "keeps the rate can fix that. It is a limit of the nozzle and rate, not of the controller.",
          "- Where free_fast beats the locked modes, the gap is the cost of keeping the rate fixed.", "",
          "## What was not tested", "",
          "- Image readers (B0, B1); only the noisy reader was used.",
          "- Trained card feedback (D29, D31) and the learning (multi-pass) strategy.",
          "- Targets other than 30 % coverage, other mismatch levels, other models (XGBoost v2, hybrid).",
          "- Wind forecast and speed-only control.",
          "- Other nozzle and rate combinations than the six above; they were chosen by hand.",
          "- Nothing here is field data.", ""]
    (ROOT / "results" / "rate_lock_eval.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
