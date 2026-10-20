"""Optional boom-height control (schema 1.2): selector with vs without height (and speed) in the decision.

Part A (open loop, no mismatch): same scenario sampler as eval_selector / eval_speed_control. The
selector uses a deterministic model (NominalPhysicsModel = simulator at mismatch none, mean spread
factor; plus the learned SprayModel when available). Scoring is DETERMINISTIC AND PAIRED: every
variant's chosen setting for scenario i is scored by simulator.simulate with
default_rng([seed, i]), so all variants see the same spread-factor draw (the speed-control
evaluation re-simulated with fresh noise and was noisy). Paired bootstrap CIs on per-scenario
differences vs 'off'.
Part B (closed loop): fast loop (BiasFeedback) and fair oracle, mismatch med/high, 4 presets x
seeds, oracle_nominal model, perfect verifier; paired by (preset, seed), bootstrap CIs vs 'off'.

Variants: off | height | speed | speed+height (band 0.25, min-overlap height per nozzle family).

Usage: uv run python experiments/eval_boom_control.py [--n 200] [--closed-seeds 10] [--workers 4]
       [--pilot] [--no-closed-loop]
Writes results/boom_control_eval.json and results/boom_control_eval.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from coverageai.config import spec                                    # noqa: E402
from coverageai.contracts import effective_boom_height_m, effective_speed_kmh  # noqa: E402
from coverageai.selector import ConstrainedSelector                   # noqa: E402

HEIGHTS = [0.30, 0.40, 0.50, 0.60, 0.75]          # m, coarse grid
SPEEDS = [float(v) for v in range(8, 31, 4)]       # km/h, coarse
BAND = 0.25
VARIANTS = {
    "off": {},
    "height": dict(boom_height_options_m=HEIGHTS, boom_height_band_frac=BAND),
    "speed": dict(speed_options_kmh=SPEEDS, speed_band_frac=BAND),
    "speed+height": dict(speed_options_kmh=SPEEDS, speed_band_frac=BAND,
                         boom_height_options_m=HEIGHTS, boom_height_band_frac=BAND),
}


def boot_ci(d: np.ndarray, B: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(axis=1)
    return {"mean": float(d.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5))}


def fmt_ci(z):
    return "-" if z is None else f"{z['mean']:+.4f} [{z['lo']:+.4f}, {z['hi']:+.4f}]"


# ------------------------------------------------------------------ part A
def score_variant(model, sim, scen, sel, c_req, seed):
    cov, drift, gpa, hr, sr = [], [], [], [], []
    for i, x in enumerate(scen):
        u, _ = sel.select(x, model, 0.0, c_req)
        o = sim.simulate(x, u, np.random.default_rng([seed, i]), "none")
        cov.append(o.coverage_pct)
        drift.append(o.drift_frac)
        gpa.append(sel._derive_fn(x, u).app_rate_gpa)
        hr.append(effective_boom_height_m(x, u) / x.boom_height_m)
        sr.append(effective_speed_kmh(x, u) / x.travel_speed_kmh)
    return {k: np.array(v) for k, v in dict(cov=cov, drift=drift, gpa=gpa, hr=hr, sr=sr).items()}


def part_a(model, sim, scen, c_req, seed):
    raw = {name: score_variant(model, sim, scen, ConstrainedSelector(**kw), c_req, seed)
           for name, kw in VARIANTS.items()}
    out = {}
    base = raw["off"]
    for name, r in raw.items():
        ok = (r["cov"] >= c_req).astype(float)
        d = {"success": float(ok.mean()), "mean_drift": float(r["drift"].mean()),
             "mean_gpa": float(r["gpa"].mean()),
             "frac_height_changed": float((np.abs(r["hr"] - 1) > 1e-9).mean()),
             "mean_height_ratio": float(r["hr"].mean()),
             "frac_speed_changed": float((np.abs(r["sr"] - 1) > 1e-9).mean())}
        if name != "off":
            d["d_success_vs_off"] = boot_ci(ok - (base["cov"] >= c_req).astype(float))
            d["d_drift_vs_off"] = boot_ci(r["drift"] - base["drift"])
            d["d_gpa_vs_off"] = boot_ci(r["gpa"] - base["gpa"])
        out[name] = d
    return out


# ------------------------------------------------------------------ part B
_MODS: dict = {}


def _modules(tag):
    if tag not in _MODS:
        from coverageai import integration
        mods = integration.build_modules("oracle_nominal", "perfect")
        mods.selector = ConstrainedSelector(candidates_fn=mods.selector._candidates_fn,
                                            derive_fn=mods.derive_fn, **VARIANTS[tag])
        _MODS[tag] = mods
    return _MODS[tag]


def run_one(task):
    tag, policy, mismatch, preset, s = task
    from coverageai import scenarios as sc
    from coverageai.feedback import BiasFeedback
    from coverageai.loop import run_pass, summarize
    c_req = float(spec()["c_req_pct"])
    conds = sc.generate(sc.get(preset), s)
    logs = run_pass(conds, policy, _modules(tag), s + 2018, mismatch, c_req, f"{tag}-{policy}-{preset}-{s}",
                    feedback=BiasFeedback() if policy == "fast" else None, render=False)
    r = summarize(logs, c_req)
    r["frac_height_changed"] = float(np.mean([l.u.boom_height_m is not None for l in logs]))
    return task, r


def part_b(seeds, workers, tags):
    presets = ("calm_morning", "gusty_afternoon", "hot_dry", "crosswind_shift")
    tasks = [(t, pol, mm, p, s) for t in tags for pol in ("fast", "oracle") for mm in ("med", "high")
             for p in presets for s in range(seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = dict(ex.map(run_one, tasks, chunksize=1))
    print(f"closed loop: {len(tasks)} passes in {time.time() - t0:.0f}s")
    keys = ("frac_met", "mean_shortfall_pp", "mean_volume_gpa", "mean_drift_frac", "frac_height_changed")
    out = {}
    unit = [(p, s) for p in presets for s in range(seeds)]
    for pol in ("fast", "oracle"):
        for mm in ("med", "high"):
            arr = {t: {k: np.array([res[(t, pol, mm, p, s)][k] for p, s in unit]) for k in keys} for t in tags}
            for t in tags:
                d = {k: float(arr[t][k].mean()) for k in keys}
                if t != "off":
                    for k in keys[:4]:
                        d[f"d_{k}_vs_off"] = boot_ci(arr[t][k] - arr["off"][k])
                out[f"{pol}/{mm}/{t}"] = d
    return out


# ------------------------------------------------------------------ report
def fmt_a(d):
    L = ["| variant | success | mean drift | mean GPA | height changed | h/h0 | speed changed | "
         "d drift vs off [95% CI] | d success vs off [95% CI] |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, r in d.items():
        L.append(f"| {k} | {r['success']:.3f} | {r['mean_drift']:.4f} | {r['mean_gpa']:.2f} | "
                 f"{r['frac_height_changed']:.2f} | {r['mean_height_ratio']:.2f} | {r['frac_speed_changed']:.2f} | "
                 f"{fmt_ci(r.get('d_drift_vs_off'))} | {fmt_ci(r.get('d_success_vs_off'))} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--closed-seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-closed-loop", action="store_true")
    ap.add_argument("--pilot", action="store_true", help="n=20, 1 closed-loop seed, writes nothing")
    args = ap.parse_args()
    if args.pilot:
        args.n, args.closed_seeds = 20, 1
    args.workers = min(args.workers, 4)
    c_req = float(spec()["c_req_pct"])
    from eval_selector import sample_conditions
    from coverageai import catalog
    from coverageai.integration import NominalPhysicsModel, find_spray_model_path
    from coverageai.simulator import SpraySimulator
    sim = SpraySimulator()
    scen = sample_conditions(args.n, args.seed)
    out = {"n": args.n, "seed": args.seed, "c_req": c_req, "heights_m": HEIGHTS, "speeds_kmh": SPEEDS,
           "band": BAND,
           "min_overlap_height_m": {f: catalog.min_boom_height_m(f) for f in spec()["nozzle_families"]},
           "open_loop": {}}
    t0 = time.time()
    out["open_loop"]["nominal_physics"] = part_a(NominalPhysicsModel(sim), sim, scen, c_req, args.seed)
    print(f"part A nominal: {time.time() - t0:.0f}s")
    try:
        path = find_spray_model_path()
        if path is not None:
            from coverageai.model.predictor import SprayModel
            out["open_loop"]["spray"] = part_a(SprayModel.load(path), sim, scen, c_req, args.seed)
    except Exception as e:
        print(f"spray model skipped: {e!r}")
    if not args.no_closed_loop:
        out["closed_loop"] = part_b(args.closed_seeds, args.workers, ["off", "height", "speed+height"])
    md = ["# Boom-height control evaluation", "",
          f"N={args.n} random no-mismatch scenarios, C_req={c_req:.0f}%. Heights {HEIGHTS} m and speeds {SPEEDS} km/h, "
          f"each within +-{BAND:.0%} of the scenario value, spec ranges, and (heights) above the family minimum "
          "overlap height H >= 1.3 * spacing / (2 tan(angle/2)): "
          + ", ".join(f"{f} {v:.2f} m" for f, v in out["min_overlap_height_m"].items()) + ". "
          "Scoring is paired: each variant's choice for scenario i is simulated with rng [seed, i]; CIs are "
          "paired bootstrap over scenarios of the per-scenario difference vs 'off'.", ""]
    for name, d in out["open_loop"].items():
        md += [f"## Open loop (selector model: {name})", "", fmt_a(d), ""]
    if "closed_loop" in out:
        md += [f"## Closed loop (oracle_nominal model, perfect verifier, 4 presets x {args.closed_seeds} seeds, "
               "paired by preset/seed; 95% paired-bootstrap CI of the difference vs off; 'oracle' = fair oracle)", "",
               "| policy/mismatch/variant | frac_met | shortfall pp | GPA | drift | height set | d frac_met | d drift | d GPA |",
               "|---|---|---|---|---|---|---|---|---|"]
        for k, r in out["closed_loop"].items():
            md.append(f"| {k} | {r['frac_met']:.3f} | {r['mean_shortfall_pp']:.2f} | {r['mean_volume_gpa']:.2f} | "
                      f"{r['mean_drift_frac']:.4f} | {r['frac_height_changed']:.2f} | "
                      f"{fmt_ci(r.get('d_frac_met_vs_off'))} | {fmt_ci(r.get('d_mean_drift_frac_vs_off'))} | "
                      f"{fmt_ci(r.get('d_mean_volume_gpa_vs_off'))} |")
        md.append("")
    print("\n".join(md))
    if not args.pilot:
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results" / "boom_control_eval.json").write_text(json.dumps(out, indent=2), encoding="utf8")
        (ROOT / "results" / "boom_control_eval.md").write_text("\n".join(md), encoding="utf8")


if __name__ == "__main__":
    main()
