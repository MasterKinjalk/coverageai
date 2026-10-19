"""WP-11 selector evaluation on N random no-mismatch scenarios.

For each truth model (mock closed form; oracle simulator when importable) the selector
uses that same model (no model error), so this isolates selector behavior.
Reports: fraction meeting C_req, mean drift and volume (GPA) vs best_fixed, and a
fixed setting matched to the selector's success rate. Also a lambda sweep.

Usage: uv run python experiments/eval_selector.py [--n 200] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from coverageai.config import ROOT, spec
from coverageai.contracts import Conditions, Prediction
from coverageai.mocks import MockModel
from coverageai.selector import ConstrainedSelector, _stats_for_candidates, best_fixed_setting


class SimulatorOracle:
    """Model protocol wrapper around a Simulator with mismatch 'none' (truth)."""

    def __init__(self, sim, seed: int = 0):
        self.sim, self.rng = sim, np.random.default_rng(seed)

    def predict(self, x, u) -> Prediction:
        o = self.sim.simulate(x, u, self.rng, "none")
        return Prediction(o.coverage_pct, o.coverage_pct, o.coverage_pct, o.deposition_ul_cm2,
                          o.drift_frac, o.displacement_mm)

    def predict_batch(self, xs, us):
        return [self.predict(x, u) for x, u in zip(xs, us)]

    def fit(self, records) -> None:
        pass


def sample_conditions(n: int, seed: int) -> list[Conditions]:
    r = spec()["ranges"]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        v = {k: float(rng.uniform(*r[k])) for k in
             ["wind_speed_ms", "wind_dir_deg", "heading_deg", "temp_c", "rh_pct",
              "travel_speed_kmh", "boom_height_m"]}
        out.append(Conditions(**v))
    return out


def evaluate(model, scenarios, selector: ConstrainedSelector, c_req: float, bias: float = 0.0):
    """Run selector per scenario, score chosen setting under `model` as truth."""
    cov, drift, gpa, infeas = [], [], [], 0
    for x in scenarios:
        u, info = selector.select(x, model, bias, c_req)
        p = model.predict(x, u)
        cov.append(p.coverage_pct)
        drift.append(p.drift_frac)
        gpa.append(selector._derive_fn(x, u).app_rate_gpa)
        infeas += info["infeasible"]
    cov = np.array(cov)
    return {"success": float((cov >= c_req).mean()), "mean_drift": float(np.mean(drift)),
            "mean_gpa": float(np.mean(gpa)), "infeasible_frac": infeas / len(scenarios)}


def run_config(name, model, scenarios, c_req):
    sel = ConstrainedSelector()
    res = {"selector": evaluate(model, scenarios, sel, c_req)}
    cov_m, _, _ = _stats_for_candidates(scenarios, model, sel.candidates, sel._derive_fn)
    res["achievable_frac"] = float((cov_m.max(axis=0) >= c_req).mean())  # some candidate meets C_req
    bf, st = best_fixed_setting(scenarios, model, c_req, sel.candidates, sel._derive_fn, return_stats=True)
    j = st["index"]
    res["best_fixed"] = {"setting": vars(bf), "success": float(st["success"][j]),
                         "mean_drift": float(st["mean_drift"][j]), "mean_gpa": float(st["mean_gpa"][j])}
    # lowest-drift fixed setting whose success rate is >= the selector's
    ok = np.flatnonzero(st["success"] >= res["selector"]["success"] - 1e-12)
    if ok.size:
        k = int(ok[np.lexsort((st["mean_gpa"][ok], st["mean_drift"][ok]))[0]])
        res["fixed_matched_success"] = {"setting": vars(st["candidates"][k]),
                                        "success": float(st["success"][k]),
                                        "mean_drift": float(st["mean_drift"][k]),
                                        "mean_gpa": float(st["mean_gpa"][k])}
    else:
        res["fixed_matched_success"] = None
    m = res["fixed_matched_success"]
    res["selector_lower_drift_than_matched_fixed"] = (
        None if m is None else res["selector"]["mean_drift"] < m["mean_drift"])
    res["lambda_sweep"] = {}
    for lam in (0.0, 0.3, 0.6, 1.0):
        res["lambda_sweep"][str(lam)] = evaluate(model, scenarios, ConstrainedSelector(lam=lam), c_req)
    # runtime
    import time
    s = ConstrainedSelector()
    t = []
    for x in scenarios[:50]:
        t0 = time.perf_counter()
        s.select(x, model, 0.0, c_req)
        t.append((time.perf_counter() - t0, s.last_model_time_s))
    t = np.array(t)
    res["runtime_ms"] = {"total_mean": float(1e3 * t[:, 0].mean()), "model_mean": float(1e3 * t[:, 1].mean()),
                         "selector_only_mean": float(1e3 * (t[:, 0] - t[:, 1]).mean())}
    res["n_candidates"] = len(s.candidates)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    c_req = float(spec()["c_req_pct"])
    scenarios = sample_conditions(args.n, args.seed)

    models = {"mock": MockModel()}
    try:
        from coverageai.simulator import SpraySimulator
        models["oracle"] = SimulatorOracle(SpraySimulator())
    except Exception as e:  # WP-02 not merged yet
        print(f"oracle skipped: {e!r}")

    out = {"n": args.n, "seed": args.seed, "c_req": c_req, "configs": {}}
    for name, model in models.items():
        out["configs"][name] = run_config(name, model, scenarios, c_req)
        r = out["configs"][name]
        print(f"[{name}] selector success {r['selector']['success']:.3f} drift {r['selector']['mean_drift']:.4f} "
              f"gpa {r['selector']['mean_gpa']:.2f} | best_fixed success {r['best_fixed']['success']:.3f} "
              f"drift {r['best_fixed']['mean_drift']:.4f} gpa {r['best_fixed']['mean_gpa']:.2f} | "
              f"achievable {r['achievable_frac']:.3f} | runtime {r['runtime_ms']}")

    res_dir = ROOT / "results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / "selector_eval.json").write_text(json.dumps(out, indent=2), encoding="utf8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for name, r in out["configs"].items():
        sw = r["lambda_sweep"]
        xs = [sw[k]["mean_gpa"] for k in sw]
        ys = [sw[k]["mean_drift"] for k in sw]
        ax.plot(xs, ys, "o-", label=f"selector ({name})")
        for k, xv, yv in zip(sw, xs, ys):
            ax.annotate(f"$\\lambda$={k}", (xv, yv), textcoords="offset points", xytext=(4, 4), fontsize=8)
        bf = r["best_fixed"]
        ax.plot(bf["mean_gpa"], bf["mean_drift"], "s", label=f"best fixed ({name}, succ {bf['success']:.2f})")
    ax.set_xlabel("mean application volume (GPA)")
    ax.set_ylabel("mean drift fraction")
    ax.set_title(f"Selector trade-off (N={args.n}, C_req={c_req:.0f}%)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig_dir = ROOT / "experiments" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "selector_tradeoff.png", dpi=150)


if __name__ == "__main__":
    main()
