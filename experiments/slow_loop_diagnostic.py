"""Slow-loop diagnostic (WP-21 redesign): does the residual of the model fall over successive
passes, and does reliability improve?

For each (preset, mismatch, seed) it runs `n_passes` consecutive fast+slow passes
(loop.run_passes_slow, pass i hidden seed = seed + 1009 i) for several slow-loop variants,
with identical hidden realizations per pass across variants (paired):

  none             fast loop only, no slow update (the reference for "does the slow loop help")
  residual         redesigned slow loop: frozen base + per-regime empirical-Bayes offset
                   (feedback.ResidualCorrectedModel), the default of run_passes_slow
  residual_global  same with a single global offset (per_regime=False)
  legacy_refit_w30 the midterm slow loop: SprayModel.fit(new records), record weight 30
                   (only with --legacy and the spray model; LightGBM refits are slow)

Per pass it reports (mean over seeds, presets):
  raw residual       meas - model prediction in use (base + slow correction), before the
                     fast-loop bias: mean, variance (within pass, ddof 1) and MSE
  base residual      meas - frozen base prediction for the same segments (what the slow
                     loop is fitted to)
  ctrl residual      meas - (prediction + fast bias), what the controller acts on: var, MSE
  interval check     fraction of segments with true coverage below the lower bound the
                     selector used (pred lo + bias) and mean lower half-width (pred - lo)
  frac_met, volume   true outcome coverage >= C_req, mean GPA

Seeds default to 100.. (disjoint from the 0..19 used by experiments/closed_loop.py) so that
design choices made here are not tuned on the evaluation seeds.

Usage: uv run python experiments/slow_loop_diagnostic.py [--seeds 10] [--legacy] [--workers 6]
Writes results/closed_loop_slow_diagnostic.json and .md.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coverageai import integration, scenarios                     # noqa: E402
from coverageai.config import spec                                 # noqa: E402
from coverageai.feedback import ResidualCorrectedModel             # noqa: E402
from coverageai.loop import run_passes_slow                        # noqa: E402

RESULTS = ROOT / "results"
PRESETS = ("calm_morning", "gusty_afternoon", "hot_dry", "crosswind_shift")
VARIANTS = ("none", "residual", "residual_global", "legacy_refit_w30")


class _NoFit:
    """Model wrapper whose fit is a no-op (fast loop only across passes)."""

    def __init__(self, base):
        self.base = base

    def predict_batch(self, xs, us):
        return self.base.predict_batch(xs, us)

    def predict(self, x, u):
        return self.base.predict(x, u)

    def fit(self, records):
        pass


_W: dict = {}


def _init(model: str) -> None:
    mods = integration.build_modules(model, "noisy", n_jobs=2)
    _W.update(model=model, mods=mods, base=copy.deepcopy(mods.model))


def pass_stats(L, c_req: float) -> dict:
    meas = np.array([l.meas.coverage_pct for l in L])
    true = np.array([l.outcome.coverage_pct for l in L])
    pred = np.array([l.pred.coverage_pct for l in L])
    lo = np.array([l.pred.coverage_lo for l in L])
    bias = np.array([l.bias for l in L])
    corr = np.array([l.extra.get("slow_corr", 0.0) for l in L])
    raw = meas - pred
    base = raw + corr
    ctrl = meas - (pred + bias)
    return {
        "raw_resid_mean": raw.mean(), "raw_resid_var": raw.var(ddof=1), "raw_resid_mse": np.mean(raw ** 2),
        "base_resid_mean": base.mean(), "base_resid_var": base.var(ddof=1),
        "base_resid_mse": np.mean(base ** 2),
        "ctrl_resid_var": ctrl.var(ddof=1), "ctrl_resid_mse": np.mean(ctrl ** 2),
        "true_err_mae": np.mean(np.abs(true - (pred + bias))),
        "frac_below_lo": np.mean(true < lo + bias), "lower_halfwidth": np.mean(pred - lo),
        "slow_corr_mean": corr.mean(),
        "frac_met": np.mean(true >= c_req),
        "mean_volume_gpa": np.mean([l.derived.app_rate_gpa for l in L]),
    }


def cell(preset: str, mismatch: str, seed: int, variant: str, n_passes: int, c_req: float) -> list[dict]:
    mods = _W["mods"]
    conds = scenarios.generate(scenarios.get(preset), seed)
    base = _W["base"]
    saved = mods.model
    try:
        if variant == "none":
            mods.model = _NoFit(base)
            logs = run_passes_slow(conds, mods, seed, mismatch, c_req, n_passes, slow_mode="refit")
        elif variant == "legacy_refit_w30":
            mods.model = copy.deepcopy(base)
            mods.model.params["record_weight"] = 30.0
            logs = run_passes_slow(conds, mods, seed, mismatch, c_req, n_passes, slow_mode="refit")
        else:
            mods.model = base
            corr = ResidualCorrectedModel(base, mods.derive_fn, per_regime=(variant == "residual"))
            logs = run_passes_slow(conds, mods, seed, mismatch, c_req, n_passes,
                                   slow_mode="residual", correction=corr)
    finally:
        mods.model = saved
    n = len(conds)
    return [{"preset": preset, "mismatch": mismatch, "seed": seed, "variant": variant,
             "slow_pass": i, **pass_stats(logs[i * n:(i + 1) * n], c_req)} for i in range(n_passes)]


COLS = (("raw_resid_mean", "raw mean", 2), ("raw_resid_var", "raw var", 1),
        ("raw_resid_mse", "raw MSE", 1), ("ctrl_resid_mse", "ctrl MSE", 1),
        ("frac_below_lo", "true<lo", 3), ("lower_halfwidth", "lo half-w", 2),
        ("frac_met", "met", 3), ("mean_volume_gpa", "GPA", 2))


def markdown(df: pd.DataFrame, meta: dict) -> str:
    L = ["# Slow-loop diagnostic (redesigned slow loop)\n",
         f"Generated by `experiments/slow_loop_diagnostic.py`: model `{meta['model']}`, verifier noisy, "
         f"{meta['n_seeds']} dev seeds ({meta['seeds'][0]}..{meta['seeds'][-1]}, disjoint from the evaluation "
         f"seeds), presets {', '.join(meta['presets'])}, {meta['n_passes']} consecutive passes, C_req "
         f"{meta['c_req']:g}%. Mean over seeds and presets; +- is the 95% t-interval over seeds "
         "(seed-level means over presets). Raw residual = measured - model prediction in use "
         "(base + slow correction) before the fast bias; ctrl = measured - (prediction + fast bias); "
         "true<lo = share of segments whose true coverage is below the lower bound the selector used.\n"]
    from scipy import stats
    for mm in meta["mismatch"]:
        L.append(f"## mismatch {mm}\n")
        L.append("| variant | pass | " + " | ".join(c[1] for c in COLS) + " |")
        L.append("|---|---|" + "---|" * len(COLS))
        sub = df[df["mismatch"] == mm]
        for var in [v for v in VARIANTS if v in set(sub["variant"])]:
            for p in sorted(sub["slow_pass"].unique()):
                g = sub[(sub["variant"] == var) & (sub["slow_pass"] == p)]
                sl = g.groupby("seed")[[c[0] for c in COLS]].mean()
                cells = []
                for c, _, d in COLS:
                    x = sl[c].to_numpy()
                    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
                    cells.append(f"{x.mean():.{d}f} +- {h:.{d}f}")
                L.append(f"| {var} | {p + 1} | " + " | ".join(cells) + " |")
        L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="spray", choices=["spray", "oracle_nominal"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-offset", type=int, default=100)
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--mismatch", default="none,med,high")
    ap.add_argument("--presets", default=",".join(PRESETS))
    ap.add_argument("--legacy", action="store_true", help="include the midterm LightGBM-refit slow loop")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS / "closed_loop_slow_diagnostic"))
    args = ap.parse_args()
    c_req = float(spec()["c_req_pct"])
    seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))
    presets = args.presets.split(",")
    mism = args.mismatch.split(",")
    variants = [v for v in VARIANTS if v != "legacy_refit_w30" or (args.legacy and args.model == "spray")]
    workers = args.workers or max(1, min(8, (os.cpu_count() or 2) - 2))
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(args.model,)) as ex:
        futs = [ex.submit(cell, p, mm, s, v, args.passes, c_req)
                for v in variants for p in presets for mm in mism for s in seeds
                if v != "legacy_refit_w30" or p == "gusty_afternoon"]
        for i, f in enumerate(as_completed(futs), 1):
            rows += f.result()
            if i % 40 == 0 or i == len(futs):
                print(f"[{time.time() - t0:5.0f}s] {i}/{len(futs)}", flush=True)
    df = pd.DataFrame(rows)
    meta = {"model": args.model, "seeds": seeds, "n_seeds": len(seeds), "presets": presets,
            "mismatch": mism, "n_passes": args.passes, "c_req": c_req, "variants": variants,
            "wall_s": time.time() - t0,
            "note": "legacy_refit_w30 runs only on gusty_afternoon" if "legacy_refit_w30" in variants else ""}
    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps({**meta, "rows": df.to_dict("records")}), encoding="utf8")
    md = markdown(df, meta)
    if meta["note"]:
        md = md.replace("\n## mismatch", f"\nNote: {meta['note']} (its rows average that preset only).\n\n## mismatch", 1)
    out.with_suffix(".md").write_text(md, encoding="utf8")
    print(md)


if __name__ == "__main__":
    main()
