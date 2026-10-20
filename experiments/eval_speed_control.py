"""Optional travel-speed control: selector with vs without speed in the decision (schema 1.1).

Part A (open-loop, no mismatch): same scenarios and scoring as eval_selector.py. Model =
oracle simulator (selector sees truth, isolates the selector) and, when available, the
learned SprayModel selecting while the simulator scores. Speed options are a grid inside
+-band of the scenario's speed and the spec range.
Part B (cheap closed-loop check): oracle_nominal model, perfect verifier, 'oracle' and
'fast' policies, mismatch 'med', all presets, a few seeds; selector with vs without speed.

Usage: uv run python experiments/eval_speed_control.py [--n 200] [--seed 0] [--no-closed-loop]
Writes results/speed_control_eval.json and results/speed_control_eval.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from coverageai.config import spec                                   # noqa: E402
from coverageai.selector import ConstrainedSelector                  # noqa: E402
from eval_selector import SimulatorOracle, evaluate, sample_conditions  # noqa: E402

SPEEDS = [float(v) for v in range(8, 31, 2)]     # km/h, 5-19 mph on a 2 km/h grid
BANDS = (0.15, 0.25, 0.40)


def speed_stats(model, scenarios, sel, c_req, truth):
    chosen = []
    for x in scenarios:
        u, _ = sel.select(x, model, 0.0, c_req)
        chosen.append(x.travel_speed_kmh if u.travel_speed_kmh is None else u.travel_speed_kmh)
    ratio = np.array(chosen) / np.array([x.travel_speed_kmh for x in scenarios])
    return {"mean_speed_ratio": float(ratio.mean()), "frac_speed_changed": float((np.abs(ratio - 1) > 1e-9).mean()),
            "min_ratio": float(ratio.min()), "max_ratio": float(ratio.max())}


def part_a(model, scenarios, c_req, truth=None):
    out = {"off": evaluate(model, scenarios, ConstrainedSelector(), c_req, truth=truth)}
    for band in BANDS:
        sel = ConstrainedSelector(speed_options_kmh=SPEEDS, speed_band_frac=band)
        r = evaluate(model, scenarios, sel, c_req, truth=truth)
        r.update(speed_stats(model, scenarios, sel, c_req, truth))
        out[f"on_band_{band}"] = r
    return out


def part_b(seeds: int):
    from coverageai import integration, scenarios as sc
    from coverageai.feedback import BiasFeedback
    from coverageai.loop import run_pass, summarize
    c_req = float(spec()["c_req_pct"])
    res = {}
    for tag, opts in (("off", []), ("on_band_0.25", SPEEDS)):
        mods = integration.build_modules("oracle_nominal", "perfect")
        mods.selector = ConstrainedSelector(candidates_fn=mods.selector._candidates_fn,
                                            derive_fn=mods.derive_fn, speed_options_kmh=opts,
                                            speed_band_frac=0.25)
        for policy in ("oracle", "fast"):
            rows = []
            for preset in ("calm_morning", "gusty_afternoon", "hot_dry", "crosswind_shift"):
                for s in range(seeds):
                    conds = sc.generate(sc.get(preset), s)
                    logs = run_pass(conds, policy, mods, s + 2018, "med", c_req, f"{tag}-{preset}-{s}",
                                    feedback=BiasFeedback() if policy == "fast" else None, render=False)
                    rows.append(summarize(logs, c_req))
            res[f"{tag}/{policy}"] = {k: float(np.mean([r[k] for r in rows]))
                                       for k in ("frac_met", "mean_shortfall_pp", "mean_volume_gpa",
                                                 "mean_drift_frac")}
    return res


def fmt_table(d):
    lines = ["| config | success | mean drift | mean GPA | infeasible | speed changed | mean v/v0 |",
             "|---|---|---|---|---|---|---|"]
    for k, r in d.items():
        lines.append(f"| {k} | {r['success']:.3f} | {r['mean_drift']:.4f} | {r['mean_gpa']:.2f} | "
                     f"{r['infeasible_frac']:.3f} | {r.get('frac_speed_changed', 0):.2f} | "
                     f"{r.get('mean_speed_ratio', 1):.2f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--closed-seeds", type=int, default=3)
    ap.add_argument("--no-closed-loop", action="store_true")
    args = ap.parse_args()
    c_req = float(spec()["c_req_pct"])
    scen = sample_conditions(args.n, args.seed)
    from coverageai.simulator import SpraySimulator
    oracle = SimulatorOracle(SpraySimulator())
    out = {"n": args.n, "seed": args.seed, "c_req": c_req, "speed_options_kmh": SPEEDS,
           "open_loop": {"oracle": part_a(oracle, scen, c_req)}}
    try:
        from coverageai import integration
        path = integration.find_spray_model_path()
        if path is not None:
            from coverageai.model.predictor import SprayModel
            out["open_loop"]["spray"] = part_a(SprayModel.load(path), scen, c_req, truth=oracle)
    except Exception as e:
        print(f"spray model skipped: {e!r}")
    if not args.no_closed_loop:
        out["closed_loop_oracle_nominal_med"] = part_b(args.closed_seeds)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "speed_control_eval.json").write_text(json.dumps(out, indent=2), encoding="utf8")

    md = ["# Travel-speed control evaluation", "",
          f"N={args.n} random no-mismatch scenarios, C_req={c_req:.0f}%, speed grid {SPEEDS[0]:.0f}-{SPEEDS[-1]:.0f} km/h "
          "step 2, clipped to the spec range and to +-band around the scenario speed. 'off' = v1.0 behaviour "
          "(speed uncontrolled). Selector model: oracle simulator (isolates the selector); 'spray': learned "
          "SprayModel selects, simulator scores.", ""]
    for name, d in out["open_loop"].items():
        md += [f"## Open loop ({name})", "", fmt_table(d), ""]
    if "closed_loop_oracle_nominal_med" in out:
        md += ["## Closed loop (oracle_nominal model, perfect verifier, mismatch med, 4 presets, "
               f"{args.closed_seeds} seeds, band 0.25)", "",
               "| policy | frac_met | shortfall pp | GPA | drift |", "|---|---|---|---|---|"]
        for k, r in out["closed_loop_oracle_nominal_med"].items():
            md.append(f"| {k} | {r['frac_met']:.3f} | {r['mean_shortfall_pp']:.2f} | "
                      f"{r['mean_volume_gpa']:.2f} | {r['mean_drift_frac']:.4f} |")
        md.append("")
    (ROOT / "results" / "speed_control_eval.md").write_text("\n".join(md), encoding="utf8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
