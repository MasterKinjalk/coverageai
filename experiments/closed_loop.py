"""WP-21 closed-loop experiments.

Grid: policies {open, fast, fast_slow, oracle, oracle_card, best_fixed} x mismatch
{none, low, med, high} x presets {calm_morning, gusty_afternoon, hot_dry, crosswind_shift}
x 20 seeds, in the physics-informed simulator (this is a test of the CONTROL LOGIC inside
our own simulator, not of field efficacy; see results/closed_loop_summary.md).

Oracles (see coverageai.loop):
  * oracle       FAIR oracle: knows the hidden mismatch realization of each segment (gust,
                 canopy, wear) but not the per-card WSP spread factor; same selector and
                 margin rule as the deployable policies, with the exact 10 % quantile of the
                 card noise as its lower bound. The proposal criterion is evaluated against it.
  * oracle_card  card-aware oracle (the midterm "oracle"): also sees the spread factor, so
                 it needs no margin. Reported for comparison; no controller can match it.

Protocol (important for pairing):
  * seed s in 0..19 selects the scenario conditions generate(preset, s).
  * the evaluation pass of EVERY policy uses hidden-mismatch seed E = s + 1009 * (P - 1)
    (P = SLOW_PASSES = 5, E = s + 4036), so open / fast / oracles / best_fixed / fast_slow
    all see the same realization (paired).
  * fast_slow = P consecutive passes with seeds s, s+1009, ..., E (run_passes_slow, residual
    slow loop: frozen model + per-regime empirical-Bayes offset refit on all verified cards
    after every pass, fast bias reset per pass); the last pass (seed E) is the one scored,
    and per-pass metrics of all P passes are kept to track the residual over passes.
  * metrics use the TRUE outcome (not the noisy measurement): fraction of segments with
    coverage >= C_req, mean shortfall (pp), mean volume (GPA), mean drift_frac, number of
    retreat flags (measurement based).
  * best_fixed = selector.best_fixed_setting over that scenario's conditions with the
    nominal model (what an operator would pick from the forecast, no adaptation).
  * gap recovery = (closed - open) / (oracle - open) on requirement-met fraction, from
    seed-level means; 95 % CI by paired bootstrap over seeds. Volume overhead vs oracle
    in % likewise. Both are computed against each oracle (column `reference`). Means
    +- 95 % CI over seeds use the t distribution.

Usage:
  uv run python experiments/closed_loop.py --model oracle_nominal
  uv run python experiments/closed_loop.py --model spray
  uv run python experiments/closed_loop.py --combine      # rebuild json/csv/figures/summary
Options: --seeds N (default 20), --workers N, --quick (2 seeds, 2 presets, fast smoke),
--no-ablation, --verifier {noisy,perfect,b0,b1}.
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
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coverageai import integration, scenarios                      # noqa: E402
from coverageai.config import spec                                  # noqa: E402
from coverageai.contracts import Setting                            # noqa: E402
from coverageai.feedback import (SLOW_PRIOR_SD_PP, BiasFeedback,    # noqa: E402
                                 ResidualCorrectedModel)
from coverageai.loop import run_pass, run_passes_slow, summarize    # noqa: E402
from coverageai.selector import best_fixed_setting                  # noqa: E402

RESULTS = ROOT / "results"
FIGS = ROOT / "experiments" / "figures"
POLICIES = ("open", "fast", "fast_slow", "oracle", "oracle_card", "best_fixed")
SINGLE_PASS = ("open", "fast", "oracle", "oracle_card", "best_fixed")
REFERENCES = ("oracle", "oracle_card")         # fair oracle first: the criterion reference
MISMATCH = ("none", "low", "med", "high")
PRESETS = ("calm_morning", "gusty_afternoon", "hot_dry", "crosswind_shift")
SLOW_PASSES = 5
EVAL_OFFSET = 1009 * (SLOW_PASSES - 1)          # evaluation seed E = s + 4036
ABL_PRESET, ABL_MISMATCH = "gusty_afternoon", "med"
B0_SEEDS = 5
METRICS = ("frac_met", "mean_shortfall_pp", "mean_volume_gpa", "mean_drift_frac", "n_retreat")
SLOW_COLS = ("raw_resid_var", "raw_resid_mse", "raw_resid_mean", "ctrl_resid_mse", "frac_met",
             "mean_volume_gpa", "slow_corr_mean")
POLICY_COLOR = {"open": "#7f7f7f", "fast": "#0072B2", "fast_slow": "#CC79A7",
                "oracle": "#009E73", "oracle_card": "#56B4E9", "best_fixed": "#D55E00"}
POLICY_LABEL = {"open": "open loop", "fast": "fast loop", "fast_slow": "fast + slow loop",
                "oracle": "fair oracle (knows mismatch, not card noise)",
                "oracle_card": "card-aware oracle (also knows card noise)",
                "best_fixed": "best fixed setting"}
REF_LABEL = {"oracle": "fair oracle", "oracle_card": "card-aware oracle"}
ORACLE_Q_SENS = (0.05, 0.2)                     # fair-oracle risk levels, sensitivity only

# ------------------------------------------------------------------ worker side
_W: dict = {}


def _init(model: str, verifier: str, model_path: str | None, record_weight: float,
          n_jobs: int) -> None:
    mods = integration.build_modules(model, verifier, model_path=model_path,
                                     record_weight=record_weight, n_jobs=n_jobs)
    _W.update(model=model, base_model=copy.deepcopy(mods.model), modules={verifier: mods},
              main_verifier=verifier, model_path=model_path)


def _mods(verifier: str | None = None):
    v = verifier or _W["main_verifier"]
    if v not in _W["modules"]:
        _W["modules"][v] = integration.build_modules(_W["model"], v, model_obj=_W["base_model"])
    return _W["modules"][v]


def pass_metrics(logs, c_req: float) -> dict:
    s = summarize(logs, c_req)
    raw = np.array([l.extra["raw_residual"] for l in logs], float)   # meas - model in use
    ctrl = np.array([l.residual for l in logs], float)                # meas - (model + bias)
    s["raw_resid_mean"] = float(raw.mean())
    s["raw_resid_var"] = float(raw.var(ddof=1))
    s["raw_resid_mse"] = float(np.mean(raw ** 2))
    s["ctrl_resid_mse"] = float(np.mean(ctrl ** 2))
    s["slow_corr_mean"] = float(np.mean([l.extra.get("slow_corr", 0.0) for l in logs]))
    s["frac_below_lo"] = float(np.mean([l.outcome.coverage_pct < l.pred.coverage_lo + l.bias
                                        for l in logs]))
    s["model_err_mean"] = float(np.mean([l.outcome.coverage_pct - l.pred.coverage_pct for l in logs]))
    s["meas_err_rmse"] = float(np.sqrt(np.mean([l.extra["true_cov_err"] ** 2 for l in logs])))
    s["final_bias"] = float(logs[-1].extra.get("bias_after", 0.0))
    return s


def trace(logs) -> dict:
    return {"k": [l.k for l in logs],
            "pred_raw": [l.pred.coverage_pct for l in logs],
            "pred_corrected": [l.corrected_pred_cov for l in logs],
            "measured": [l.meas.coverage_pct for l in logs],
            "true": [l.outcome.coverage_pct for l in logs],
            "bias_applied": [l.bias for l in logs],
            "bias_after": [l.extra.get("bias_after", 0.0) for l in logs],
            "gpa": [l.derived.app_rate_gpa for l in logs]}


def stage1_fixed(preset: str, s: int, c_req: float) -> tuple:
    mods = _mods()
    conds = scenarios.generate(scenarios.get(preset), s)
    u = best_fixed_setting(conds, mods.model, c_req, derive_fn=mods.derive_fn)
    return preset, s, (u.family, u.size, u.pressure_psi)


def stage2_cell(preset: str, mismatch: str, s: int, fixed: tuple, c_req: float,
                keep_trace: bool) -> dict:
    mods = _mods()
    E = s + EVAL_OFFSET
    conds = scenarios.generate(scenarios.get(preset), s)
    fixed_u = Setting(*fixed)
    out: dict = {"preset": preset, "mismatch": mismatch, "seed": s, "rows": [], "slow": [],
                 "trace": None}
    traces = {}
    for pol in SINGLE_PASS:
        logs = run_pass(conds, pol, mods, E, mismatch, c_req, f"{pol}_{preset}_{mismatch}_{s}",
                        fixed_setting=fixed_u)
        out["rows"].append({"policy": pol, **pass_metrics(logs, c_req)})
        traces[pol] = logs
    # slow loop (residual mode): the base model is frozen and modules.model is restored by
    # run_passes_slow, so no refit can leak into other cells.
    logs = run_passes_slow(conds, mods, s, mismatch, c_req, n_passes=SLOW_PASSES)
    n = len(conds)
    for i in range(SLOW_PASSES):
        m = pass_metrics(logs[i * n:(i + 1) * n], c_req)
        out["slow"].append({"slow_pass": i, **m})
        if i == SLOW_PASSES - 1:
            out["rows"].append({"policy": "fast_slow", **m})
            traces["fast_slow"] = logs[i * n:(i + 1) * n]
    if keep_trace:
        out["trace"] = {p: trace(traces[p]) for p in ("open", "fast", "fast_slow", "oracle")}
    return out


def ablation_cell(variant: dict, s: int, c_req: float) -> dict:
    mods = _mods(variant["verifier"])
    E = s + EVAL_OFFSET
    conds = scenarios.generate(scenarios.get(ABL_PRESET), s)
    if variant.get("slow"):       # slow-loop variant: score the last pass of a P-pass run
        saved = mods.model
        try:
            if variant["slow"] == "refit":          # legacy midterm slow loop
                mods.model = copy.deepcopy(_W["base_model"])
                mods.model.params["record_weight"] = variant["record_weight"]
                logs = run_passes_slow(conds, mods, s, ABL_MISMATCH, c_req, n_passes=SLOW_PASSES,
                                       slow_mode="refit")
            else:
                corr = ResidualCorrectedModel(mods.model, mods.derive_fn,
                                              per_regime=variant["per_regime"],
                                              prior_sd_pp=variant.get("prior_sd_pp", SLOW_PRIOR_SD_PP))
                logs = run_passes_slow(conds, mods, s, ABL_MISMATCH, c_req, n_passes=SLOW_PASSES,
                                       correction=corr)
        finally:
            mods.model = saved
        return {**variant, "seed": s, **pass_metrics(logs[-len(conds):], c_req)}
    fb = BiasFeedback(alpha=variant["alpha"], warmup_cards=variant["warmup"],
                      per_regime=variant["per_regime"])
    render = variant["verifier"] in ("b0", "b1")
    logs = run_pass(conds, "fast", mods, E, ABL_MISMATCH, c_req, f"abl_{variant['name']}_{s}",
                    feedback=fb, render=render)
    return {**variant, "seed": s, **pass_metrics(logs, c_req)}


def sensitivity_cell(preset: str, mismatch: str, s: int, q: float, c_req: float) -> dict:
    """Fair oracle at lower-bound quantile q (the main grid uses 0.1) on the evaluation pass."""
    mods = _mods()
    conds = scenarios.generate(scenarios.get(preset), s)
    logs = run_pass(conds, "oracle", mods, s + EVAL_OFFSET, mismatch, c_req,
                    f"oracle_q{q:g}_{preset}_{mismatch}_{s}", oracle_interval_q=q)
    return {"preset": preset, "mismatch": mismatch, "seed": s, "q": q, **pass_metrics(logs, c_req)}


# ------------------------------------------------------------------ statistics

def mean_ci(x) -> tuple[float, float]:
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float(x.mean()) if n else float("nan"), float("nan")
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(x.mean()), float(h)


def seed_level(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Average each metric over presets (and other columns not in `by`) per seed."""
    return df.groupby(by + ["seed"], as_index=False)[list(METRICS)].mean()


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Long table: mean and 95 % CI half-width of each metric per (policy, mismatch, preset),
    where preset == 'ALL' averages the four presets within each seed first."""
    rows = []
    sl = seed_level(df, ["policy", "mismatch"])
    for (pol, mm), g in sl.groupby(["policy", "mismatch"]):
        r = {"policy": pol, "mismatch": mm, "preset": "ALL", "n_seeds": len(g)}
        for m in METRICS:
            r[m], r[m + "_ci95"] = mean_ci(g[m])
        rows.append(r)
    for (pol, mm, pr), g in df.groupby(["policy", "mismatch", "preset"]):
        r = {"policy": pol, "mismatch": mm, "preset": pr, "n_seeds": len(g)}
        for m in METRICS:
            r[m], r[m + "_ci95"] = mean_ci(g[m])
        rows.append(r)
    return pd.DataFrame(rows)


def _boot(fn, arrays: list[np.ndarray], B: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    n = len(arrays[0])
    est = fn(*arrays)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (B, n))
    vals = np.array([fn(*[a[i] for a in arrays]) for i in idx])
    vals = vals[np.isfinite(vals)]
    if len(vals) < B // 2:
        return est, float("nan"), float("nan")
    return est, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _gap(o, c, r):
    d = r.mean() - o.mean()
    return (c.mean() - o.mean()) / d if abs(d) > 1e-9 else np.nan


def _overhead(c, r):
    return 100.0 * (c.mean() / r.mean() - 1.0)


def gap_table(df: pd.DataFrame, closed=("fast", "fast_slow"), cmp_extra=("best_fixed", "oracle"),
              groups: dict | None = None, refs=REFERENCES) -> pd.DataFrame:
    """Gap recovery and volume overhead per mismatch level, per preset and pooled, against
    each reference oracle in `refs` (column `reference`). `groups` maps a label to a list of
    mismatch levels pooled together (default: each level alone plus 'low+med+high')."""
    groups = groups or {**{m: [m] for m in MISMATCH}, "low+med+high": ["low", "med", "high"]}
    rows = []
    for ref in refs:
        for gname, levels in groups.items():
            for preset in ("ALL",) + tuple(sorted(df["preset"].unique())):
                sub = df[df["mismatch"].isin(levels)]
                if preset != "ALL":
                    sub = sub[sub["preset"] == preset]
                # per-seed mean over presets/levels in the group
                sl = sub.groupby(["policy", "seed"], as_index=False)[list(METRICS)].mean()
                piv = {m: sl.pivot(index="seed", columns="policy", values=m) for m in
                       ("frac_met", "mean_volume_gpa", "mean_shortfall_pp")}
                if "open" not in piv["frac_met"] or ref not in piv["frac_met"]:
                    continue
                o = piv["frac_met"]["open"].to_numpy()
                r = piv["frac_met"][ref].to_numpy()
                vr = piv["mean_volume_gpa"][ref].to_numpy()
                for pol in tuple(closed) + tuple(cmp_extra):
                    if pol not in piv["frac_met"] or pol == ref:
                        continue
                    c = piv["frac_met"][pol].to_numpy()
                    vc = piv["mean_volume_gpa"][pol].to_numpy()
                    g, glo, ghi = _boot(_gap, [o, c, r])
                    ov, olo, ohi = _boot(_overhead, [vc, vr])
                    rows.append({"reference": ref, "mismatch": gname, "preset": preset,
                                 "policy": pol, "n_seeds": len(o), "met_open": o.mean(),
                                 "met_policy": c.mean(), "met_oracle": r.mean(),
                                 "gap_pp": 100 * (r.mean() - o.mean()),
                                 "gap_recovery": g, "gap_recovery_lo": glo, "gap_recovery_hi": ghi,
                                 "vol_overhead_pct": ov, "vol_overhead_lo": olo,
                                 "vol_overhead_hi": ohi})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures

def _plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})
    return plt


def fig_met(agg_by_model: dict[str, pd.DataFrame], path: Path) -> None:
    plt = _plot_style()
    cols = ("ALL",) + PRESETS
    models = list(agg_by_model)
    fig, axes = plt.subplots(len(models), len(cols), figsize=(3.0 * len(cols), 2.9 * len(models)),
                             squeeze=False, sharey=True)
    for i, mdl in enumerate(models):
        agg = agg_by_model[mdl]
        for j, pr in enumerate(cols):
            ax = axes[i, j]
            for pol in POLICIES:
                g = agg[(agg["policy"] == pol) & (agg["preset"] == pr)].set_index("mismatch")
                if g.empty:
                    continue
                g = g.reindex(MISMATCH)
                xs = np.arange(len(MISMATCH)) + (list(POLICIES).index(pol) - 2) * 0.04
                ax.errorbar(xs, g["frac_met"], yerr=g["frac_met_ci95"], color=POLICY_COLOR[pol],
                            marker="o", ms=3.5, lw=1.4, capsize=2, label=POLICY_LABEL[pol])
            ax.set_xticks(range(len(MISMATCH)), MISMATCH)
            ax.set_ylim(0, 1.03)
            ax.set_title(("all presets" if pr == "ALL" else pr) + (f"  [{mdl}]" if j == 0 else ""),
                         fontsize=9)
            if j == 0:
                ax.set_ylabel("segments meeting C_req")
            if i == len(models) - 1:
                ax.set_xlabel("hidden mismatch level")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Requirement-met fraction (true outcome), mean and 95% CI over seeds", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_volume_drift(agg_by_model: dict[str, pd.DataFrame], path: Path) -> None:
    plt = _plot_style()
    models = list(agg_by_model)
    fig, axes = plt.subplots(len(models), 2, figsize=(8.0, 3.0 * len(models)), squeeze=False)
    for i, mdl in enumerate(models):
        agg = agg_by_model[mdl]
        for j, (col, lab) in enumerate((("mean_volume_gpa", "mean applied volume (GPA)"),
                                        ("mean_drift_frac", "mean drift fraction"))):
            ax = axes[i, j]
            for pol in POLICIES:
                g = agg[(agg["policy"] == pol) & (agg["preset"] == "ALL")].set_index("mismatch")
                if g.empty:
                    continue
                g = g.reindex(MISMATCH)
                xs = np.arange(len(MISMATCH)) + (list(POLICIES).index(pol) - 2) * 0.04
                ax.errorbar(xs, g[col], yerr=g[col + "_ci95"], color=POLICY_COLOR[pol], marker="o",
                            ms=3.5, lw=1.4, capsize=2, label=POLICY_LABEL[pol])
            ax.set_xticks(range(len(MISMATCH)), MISMATCH)
            ax.set_ylabel(lab)
            ax.set_title(f"{lab}  [{mdl}]", fontsize=9)
            if i == len(models) - 1:
                ax.set_xlabel("hidden mismatch level")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Cost of meeting the requirement (all presets, mean and 95% CI over seeds)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_bias_trace(trace_by_model: dict, c_req: float, path: Path) -> None:
    plt = _plot_style()
    models = list(trace_by_model)
    fig, axes = plt.subplots(2, len(models), figsize=(6.2 * len(models), 5.4), squeeze=False,
                             sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for j, mdl in enumerate(models):
        t = trace_by_model[mdl]
        f, o = t["fast"], t["open"]
        k = np.array(f["k"])
        ax = axes[0, j]
        ax.axhline(c_req, color="k", ls="--", lw=0.9, label=f"C_req = {c_req:g}%")
        ax.plot(k, f["pred_raw"], color="#E69F00", lw=1.4, label="predicted (raw model)")
        ax.plot(k, f["pred_corrected"], color="#0072B2", lw=1.6, label="corrected (raw + bias)")
        ax.plot(k, f["measured"], color="#CC79A7", lw=0, marker="o", ms=3.5,
                label="measured (card)")
        ax.plot(k, f["true"], color="#009E73", lw=1.2, label="true, fast loop")
        ax.plot(k, o["true"], color="#7f7f7f", lw=1.0, ls=":", label="true, open loop")
        if "fast_slow" in t:
            ax.plot(k, t["fast_slow"]["true"], color="#CC79A7", lw=1.0, ls="--",
                    label=f"true, fast + slow (pass {SLOW_PASSES})")
        ax.set_ylabel("coverage (% card area)")
        ax.set_title(f"example pass: gusty_afternoon, mismatch med, seed 0  [{mdl}]", fontsize=9)
        if j == 0:
            ax.legend(fontsize=7, ncol=2, loc="lower right")
        ax = axes[1, j]
        ax.step(k, f["bias_applied"], where="post", color="#0072B2", lw=1.5, label="bias applied")
        ax.plot(k, f["bias_after"], color="#D55E00", lw=1.0, ls="--", label="bias after update")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("segment k")
        ax.set_ylabel("bias (pp)")
        if j == 0:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_slow(slow_by_model: dict[str, pd.DataFrame], path: Path) -> None:
    plt = _plot_style()
    models = list(slow_by_model)
    panels = (("raw_resid_var", "residual variance (pp$^2$)"),
              ("raw_resid_mse", "residual MSE (pp$^2$)"),
              ("raw_resid_mean", "mean residual (pp)"),
              ("frac_met", "segments meeting C_req"),
              ("mean_volume_gpa", "mean volume (GPA)"))
    fig, axes = plt.subplots(len(models), len(panels), figsize=(3.3 * len(panels), 2.9 * len(models)),
                             squeeze=False)
    colors = {"low": "#56B4E9", "med": "#0072B2", "high": "#D55E00", "none": "#7f7f7f"}
    for i, mdl in enumerate(models):
        df = slow_by_model[mdl]
        for j, (col, lab) in enumerate(panels):
            ax = axes[i, j]
            for mm in MISMATCH:
                sl = df[df["mismatch"] == mm].groupby(["slow_pass", "seed"], as_index=False)[col].mean()
                ms = [mean_ci(g[col]) for _, g in sl.groupby("slow_pass")]
                ax.errorbar(np.arange(1, len(ms) + 1), [m for m, _ in ms], yerr=[c for _, c in ms],
                            color=colors[mm], marker="o", ms=3.5, lw=1.4, capsize=2, label=mm)
            ax.set_xticks(range(1, SLOW_PASSES + 1))
            ax.set_xlabel("slow-loop pass")
            ax.set_ylabel(lab)
            ax.set_title(f"{lab}  [{mdl}]" if j == 0 else lab, fontsize=9)
    axes[0, 0].legend(title="mismatch", fontsize=7)
    fig.suptitle("Slow loop over consecutive passes: residual = measured minus model in use "
                 "(base + slow correction, before fast bias); all presets, mean and 95% CI",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ run / combine

def run_experiment(args) -> Path:
    model, verifier = args.model, args.verifier
    c_req = float(spec()["c_req_pct"])
    seeds = list(range(args.seeds))
    presets = PRESETS[:2] if args.quick else PRESETS
    workers = args.workers or max(1, min(8, (os.cpu_count() or 2) - 2))
    t0 = time.time()
    initargs = (model, verifier, args.model_path, args.record_weight, args.n_jobs)
    meta_mods = integration.build_modules(model, verifier, model_path=args.model_path)
    noise = ({"sd_pp": meta_mods.verifier.sd_pp, "bias_pp": meta_mods.verifier.bias_pp,
              "source": meta_mods.verifier.source,
              "bands": [list(b) for b in (meta_mods.verifier.bands or ())]} if hasattr(meta_mods.verifier, "sd_pp") else None)
    del meta_mods

    fixed: dict[tuple, tuple] = {}
    rows, slow_rows, traces = [], [], {}
    with ProcessPoolExecutor(workers, initializer=_init, initargs=initargs) as ex:
        futs = [ex.submit(stage1_fixed, p, s, c_req) for p in presets for s in seeds]
        for f in as_completed(futs):
            p, s, u = f.result()
            fixed[(p, s)] = u
        print(f"[{time.time() - t0:5.0f}s] best_fixed settings done ({len(fixed)})", flush=True)

        futs = {}
        for p in presets:
            for mm in MISMATCH:
                for s in seeds:
                    keep = (p == ABL_PRESET and mm == ABL_MISMATCH and s == 0)
                    futs[ex.submit(stage2_cell, p, mm, s, fixed[(p, s)], c_req, keep)] = (p, mm, s)
        done = 0
        for f in as_completed(futs):
            r = f.result()
            done += 1
            base = {"preset": r["preset"], "mismatch": r["mismatch"], "seed": r["seed"]}
            rows += [{**base, **x} for x in r["rows"]]
            slow_rows += [{**base, **x} for x in r["slow"]]
            if r["trace"]:
                traces = r["trace"]
            if done % 20 == 0 or done == len(futs):
                print(f"[{time.time() - t0:5.0f}s] grid cells {done}/{len(futs)}", flush=True)

        abl_rows = []
        if not args.no_ablation:
            variants = ablation_variants(args)
            abl_futs = [ex.submit(ablation_cell, v, s, c_req) for v in variants
                        for s in seeds[: (B0_SEEDS if v["verifier"] in ("b0", "b1") else len(seeds))]]
            for f in as_completed(abl_futs):
                abl_rows.append(f.result())
            print(f"[{time.time() - t0:5.0f}s] ablations done ({len(abl_rows)} runs)", flush=True)

    df = pd.DataFrame(rows).sort_values(["policy", "mismatch", "preset", "seed"]).reset_index(drop=True)
    out = {
        "schema": "closed_loop/1", "model": model, "verifier": verifier, "c_req_pct": c_req,
        "seeds": seeds, "presets": list(presets), "mismatch": list(MISMATCH),
        "slow_passes": SLOW_PASSES, "eval_seed_offset": EVAL_OFFSET,
        "noise_model": noise, "record_weight": args.record_weight if model == "spray" else None,
        "wall_s": time.time() - t0, "workers": workers,
        "runs": df.to_dict("records"),
        "slow_runs": pd.DataFrame(slow_rows).to_dict("records"),
        "ablation_runs": abl_rows,
        "example_trace": {"preset": ABL_PRESET, "mismatch": ABL_MISMATCH, "seed": 0, **traces},
        "best_fixed": {f"{p}|{s}": list(u) for (p, s), u in sorted(fixed.items())},
    }
    path = RESULTS / f"closed_loop_{model}.json"
    path.write_text(json.dumps(out), encoding="utf8")
    print(f"wrote {path} ({out['wall_s']:.0f}s)")
    return path


def run_sensitivity(args) -> Path:
    """Re-run only the fair oracle at the risk levels ORACLE_Q_SENS (same evaluation passes)."""
    c_req = float(spec()["c_req_pct"])
    seeds = list(range(args.seeds))
    presets = PRESETS[:2] if args.quick else PRESETS
    workers = args.workers or max(1, min(8, (os.cpu_count() or 2) - 2))
    t0 = time.time()
    rows = []
    initargs = (args.model, args.verifier, args.model_path, args.record_weight, args.n_jobs)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=initargs) as ex:
        futs = [ex.submit(sensitivity_cell, p, mm, s, q, c_req) for q in ORACLE_Q_SENS
                for p in presets for mm in MISMATCH for s in seeds]
        for f in as_completed(futs):
            rows.append(f.result())
    path = RESULTS / f"closed_loop_oracle_q_{args.model}.json"
    path.write_text(json.dumps({"schema": "closed_loop_oracle_q/1", "model": args.model,
                                "qs": list(ORACLE_Q_SENS), "default_q": 0.1,
                                "wall_s": time.time() - t0, "runs": rows}), encoding="utf8")
    print(f"wrote {path} ({time.time() - t0:.0f}s)")
    return path


def sensitivity_table(res: dict, sens: dict | None) -> pd.DataFrame:
    """Pooled (low+med+high, all presets) criterion vs the fair oracle at each risk level q."""
    if not sens:
        return pd.DataFrame()
    df = pd.DataFrame(res["runs"])
    sd = pd.DataFrame(sens["runs"])
    variants = [(0.1, df[df["policy"] == "oracle"])] + [
        (q, g.drop(columns="q").assign(policy="oracle")) for q, g in sd.groupby("q")]
    out = []
    for q, orc in variants:
        dd = pd.concat([df[df["policy"] != "oracle"], orc], ignore_index=True)
        gt = gap_table(dd, cmp_extra=(), refs=("oracle",),
                       groups={"low+med+high": ["low", "med", "high"]})
        gt = gt[gt["preset"] == "ALL"].copy()
        gt.insert(0, "oracle_q", q)
        pooled = orc[orc["mismatch"].isin(["low", "med", "high"])]
        gt["oracle_volume_gpa"] = float(pooled["mean_volume_gpa"].mean())
        out.append(gt)
    return pd.concat(out, ignore_index=True).sort_values(["oracle_q", "policy"])


def ablation_variants(args) -> list[dict]:
    v = []
    base = dict(alpha=0.3, warmup=2, per_regime=True, verifier="noisy")

    def add(name, group, **kw):
        v.append({**base, "name": name, "group": group, **kw})

    for a in (0.1, 0.3, 0.6):
        add(f"alpha={a}", "alpha", alpha=a)
    for w in (0, 2, 4):
        add(f"warmup={w}", "warmup", warmup=w)
    add("per-regime bias", "bias scope", per_regime=True)
    add("global bias", "bias scope", per_regime=False)
    for vv in ("perfect", "noisy"):
        add(f"verifier={vv}", "verifier", verifier=vv)
    sp = f"{SLOW_PASSES} passes, final pass"
    add(f"slow loop: residual offset, per regime (default; {sp})", "slow loop",
        slow="residual", per_regime=True)
    add(f"slow loop: residual offset, global only ({sp})", "slow loop",
        slow="residual", per_regime=False)
    if args.model == "spray":
        for w in (1.0, 30.0):
            add(f"slow loop: legacy LightGBM refit, record weight {w:g} ({sp})", "slow loop",
                slow="refit", record_weight=w)
    for vv in ("b0", "b1"):
        if integration.image_verifier_available(vv) and not args.quick:
            add(f"verifier={vv} (rendered, {B0_SEEDS} seeds)", "verifier", verifier=vv)
        else:
            print(f"ablation: verifier {vv} not available, skipped")
    return v


def load_results() -> dict[str, dict]:
    out = {}
    for model in integration.MODEL_CHOICES:
        p = RESULTS / f"closed_loop_{model}.json"
        if p.is_file():
            out[model] = json.loads(p.read_text(encoding="utf8"))
    return out


def ablation_table(res: dict) -> pd.DataFrame:
    df = pd.DataFrame(res["runs"])
    ab = pd.DataFrame(res["ablation_runs"])
    if ab.empty:
        return ab
    ref = df[(df["preset"] == ABL_PRESET) & (df["mismatch"] == ABL_MISMATCH)]
    rows = []
    for (name, group), g in ab.groupby(["name", "group"], sort=False):
        seeds = sorted(g["seed"])
        o = ref[(ref["policy"] == "open") & ref["seed"].isin(seeds)].sort_values("seed")["frac_met"].to_numpy()
        r = ref[(ref["policy"] == "oracle") & ref["seed"].isin(seeds)].sort_values("seed")["frac_met"].to_numpy()
        vr = ref[(ref["policy"] == "oracle") & ref["seed"].isin(seeds)].sort_values("seed")["mean_volume_gpa"].to_numpy()
        g = g.sort_values("seed")
        c = g["frac_met"].to_numpy()
        gm, glo, ghi = _boot(_gap, [o, c, r])
        om, olo, ohi = _boot(_overhead, [g["mean_volume_gpa"].to_numpy(), vr])
        row = {"group": group, "variant": name, "n_seeds": len(g)}
        for m in METRICS:
            row[m], row[m + "_ci95"] = mean_ci(g[m])
        row.update(gap_recovery=gm, gap_recovery_lo=glo, gap_recovery_hi=ghi,
                   vol_overhead_pct=om, vol_overhead_lo=olo, vol_overhead_hi=ohi,
                   met_open=o.mean(), met_oracle=r.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def combine() -> None:
    results = load_results()
    if not results:
        raise SystemExit("no results/closed_loop_<model>.json found")
    FIGS.mkdir(parents=True, exist_ok=True)
    agg, gaps, slow, abl, sens, tables = {}, {}, {}, {}, {}, []
    for mdl, res in results.items():
        sp = RESULTS / f"closed_loop_oracle_q_{mdl}.json"
        sens[mdl] = sensitivity_table(res, json.loads(sp.read_text(encoding="utf8"))
                                      if sp.is_file() else None)
        df = pd.DataFrame(res["runs"])
        agg[mdl] = aggregate(df)
        gaps[mdl] = gap_table(df)
        slow[mdl] = pd.DataFrame(res["slow_runs"])
        abl[mdl] = ablation_table(res)
        t = agg[mdl].copy()
        t.insert(0, "model", mdl)
        tables.append(t)
    pd.concat(tables).to_csv(RESULTS / "closed_loop_table.csv", index=False)
    pd.concat([g.assign(model=m) for m, g in gaps.items()]).to_csv(
        RESULTS / "closed_loop_gap_recovery.csv", index=False)
    pd.concat([a.assign(model=m) for m, a in abl.items() if not a.empty] or [pd.DataFrame()]).to_csv(
        RESULTS / "closed_loop_ablation.csv", index=False)
    combined = {
        "schema": "closed_loop_combined/1", "c_req_pct": next(iter(results.values()))["c_req_pct"],
        "models": {m: {k: r[k] for k in ("model", "verifier", "seeds", "presets", "noise_model",
                                          "record_weight", "wall_s", "slow_passes", "eval_seed_offset")}
                   | {"aggregate": agg[m].to_dict("records"),
                      "gap_recovery": gaps[m].to_dict("records"),
                      "ablation": abl[m].to_dict("records"),
                      "fair_oracle_q_sensitivity": sens[m].to_dict("records"),
                      "slow_loop": slow[m].groupby(["mismatch", "slow_pass"])[
                          [c for c in SLOW_COLS if c in slow[m]]].mean().reset_index()
                      .to_dict("records"),
                      "best_fixed": r["best_fixed"], "example_trace": r["example_trace"]}
                   for m, r in results.items()},
    }
    (RESULTS / "closed_loop.json").write_text(json.dumps(combined, indent=1), encoding="utf8")
    c_req = combined["c_req_pct"]
    fig_met(agg, FIGS / "closed_loop_met.png")
    fig_volume_drift(agg, FIGS / "closed_loop_volume_drift.png")
    fig_bias_trace({m: r["example_trace"] for m, r in results.items()}, c_req,
                   FIGS / "bias_trace.png")
    fig_slow(slow, FIGS / "slow_loop_residuals.png")
    write_summary(results, agg, gaps, abl, slow, sens)
    print("combined outputs written to", RESULTS, "and", FIGS)


# ------------------------------------------------------------------ summary markdown

def _fmt_gap(r) -> str:
    if not np.isfinite(r["gap_recovery"]):
        return "n/a"
    lo = "n/a" if not np.isfinite(r["gap_recovery_lo"]) else f"{100 * r['gap_recovery_lo']:.0f}"
    hi = "n/a" if not np.isfinite(r["gap_recovery_hi"]) else f"{100 * r['gap_recovery_hi']:.0f}"
    return f"{100 * r['gap_recovery']:.0f}% [{lo}, {hi}]"


def _fmt_ov(r) -> str:
    return f"{r['vol_overhead_pct']:+.1f}% [{r['vol_overhead_lo']:+.1f}, {r['vol_overhead_hi']:+.1f}]"


def criterion_rows(g: pd.DataFrame, ref: str, pols=("fast", "fast_slow")) -> list[dict]:
    """Proposal criterion (pooled low+med+high, all presets) for each closed-loop policy."""
    out = []
    pooled = g[(g["reference"] == ref) & (g["preset"] == "ALL") & (g["mismatch"] == "low+med+high")]
    for pol in pols:
        r = pooled[pooled["policy"] == pol]
        if r.empty:
            continue
        r = r.iloc[0]
        rec_ok = bool(np.isfinite(r["gap_recovery"]) and r["gap_recovery"] >= 0.5)
        rec_conf = bool(np.isfinite(r["gap_recovery_lo"]) and r["gap_recovery_lo"] >= 0.5)
        vol_ok = bool(r["vol_overhead_pct"] <= 10.0)
        vol_conf = bool(r["vol_overhead_hi"] <= 10.0)
        out.append({"reference": ref, "policy": pol, "row": r, "rec_ok": rec_ok, "rec_conf": rec_conf,
                    "vol_ok": vol_ok, "vol_conf": vol_conf, "both": rec_ok and vol_ok,
                    "both_conf": rec_conf and vol_conf})
    return out


def _verdict(ok: bool, conf: bool) -> str:
    if ok and conf:
        return "met (95% CI clears the target)"
    if ok:
        return "met on the point estimate; CI crosses the target"
    return "missed"


def write_summary(results, agg, gaps, abl, slow, sens=None) -> None:
    L: list[str] = []
    L.append("# Closed-loop experiment summary (WP-21, post-midterm rerun)\n")
    L.append("Generated by `experiments/closed_loop.py --combine` from `results/closed_loop_<model>.json`. "
             "Do not edit the tables by hand; rerun the script. The headline section is hand-written "
             "in `results/closed_loop_notes.md` and inserted verbatim.\n")
    notes = RESULTS / "closed_loop_notes.md"
    if notes.is_file():
        L.append(notes.read_text(encoding="utf8").rstrip() + "\n")
    for mdl, res in results.items():
        n = len(res["seeds"])
        nz = res["noise_model"]
        P = res.get("slow_passes", SLOW_PASSES)
        L.append(f"## Model `{mdl}` ({n} seeds, verifier `{res['verifier']}`"
                 + (f", measurement noise from {nz['source']}: " + (
                     "; ".join(f"cov {lo:g}-{min(hi, 100):g}%: bias {b:+.2f}, sd {sd:.2f} pp" for lo, hi, sd, b in nz["bands"])
                     if nz.get("bands") else f"sd {nz['sd_pp']:.2f} pp, bias {nz['bias_pp']:+.2f} pp") if nz else "")
                 + f"; slow loop: residual offset over {P} passes, last pass scored)\n")
        g = gaps[mdl]
        L.append("### Proposal criterion (Table 1): gap recovery >= 50% and volume <= +10% vs oracle\n")
        L.append("Pooled over low, med and high mismatch and all presets (seed-level means over "
                 f"{n} seeds; brackets are 95% paired-bootstrap CIs). The criterion is evaluated against the "
                 "**fair oracle**; the card-aware oracle is shown for comparison with the midterm.\n")
        L.append("| reference | policy | met: open / policy / oracle | gap recovery (target >= 50%) | "
                 "volume vs oracle (target <= +10%) | recovery | volume | both |")
        L.append("|---|---|---|---|---|---|---|---|")
        for ref in REFERENCES:
            for c in criterion_rows(g, ref):
                r = c["row"]
                L.append(f"| {REF_LABEL[ref]} | {c['policy']} | {r['met_open']:.3f} / {r['met_policy']:.3f} / "
                         f"{r['met_oracle']:.3f} | {_fmt_gap(r)} | {_fmt_ov(r)} | "
                         f"{_verdict(c['rec_ok'], c['rec_conf'])} | {_verdict(c['vol_ok'], c['vol_conf'])} | "
                         f"**{'MET' if c['both'] else 'MISSED'}**{' (with CI)' if c['both_conf'] else ''} |")
        L.append("")
        st = (sens or {}).get(mdl)
        if st is not None and not st.empty:
            L.append("### Sensitivity of the verdict to the fair oracle's risk level\n")
            L.append("The fair oracle selects so that the q-quantile of the card noise meets C_req; the "
                     "default q = 0.1 matches the learned model's 80% CQR interval. Lower q = more "
                     "conservative oracle (more reliable, more volume). Same evaluation passes; pooled "
                     "low+med+high, all presets.\n")
            L.append("| oracle q | oracle met | oracle GPA | gap open-to-oracle (pp) | policy | gap recovery | "
                     "volume vs oracle | both |")
            L.append("|---|---|---|---|---|---|---|---|")
            for _, r in st.iterrows():
                both = bool(np.isfinite(r["gap_recovery"]) and r["gap_recovery"] >= 0.5
                            and r["vol_overhead_pct"] <= 10.0)
                verdict = ("n/a (gap < 2 pp: no gap to recover)" if r["gap_pp"] < 2.0
                           else ("met" if both else "missed"))
                L.append(f"| {r['oracle_q']:g} | {r['met_oracle']:.3f} | {r['oracle_volume_gpa']:.2f} | "
                         f"{r['gap_pp']:.1f} | {r['policy']} | {_fmt_gap(r)} | {_fmt_ov(r)} | {verdict} |")
            L.append("")
        for ref in REFERENCES:
            L.append(f"### By mismatch level, reference = {REF_LABEL[ref]}\n")
            L.append("| mismatch | policy | met: open / policy / oracle | gap (pp) | gap recovery | volume vs oracle |")
            L.append("|---|---|---|---|---|---|")
            pooled = g[(g["reference"] == ref) & (g["preset"] == "ALL")]
            for _, r in pooled.iterrows():
                if r["policy"] == "best_fixed":
                    continue
                note = " (gap < 2 pp: ratio not informative)" if r["gap_pp"] < 2.0 else ""
                L.append(f"| {r['mismatch']} | {r['policy']} | {r['met_open']:.3f} / {r['met_policy']:.3f} / "
                         f"{r['met_oracle']:.3f} | {r['gap_pp']:.1f} | {_fmt_gap(r)}{note} | {_fmt_ov(r)} |")
            L.append("")
        L.append("### Per preset (mismatch pooled over low, med, high; reference = fair oracle)\n")
        L.append("| preset | policy | met: open / policy / oracle | gap recovery | volume vs oracle |")
        L.append("|---|---|---|---|---|")
        pp = g[(g["reference"] == "oracle") & (g["mismatch"] == "low+med+high") & (g["preset"] != "ALL")
               & g["policy"].isin(["fast", "fast_slow"])]
        for _, r in pp.iterrows():
            L.append(f"| {r['preset']} | {r['policy']} | {r['met_open']:.3f} / {r['met_policy']:.3f} / "
                     f"{r['met_oracle']:.3f} | {_fmt_gap(r)} | {_fmt_ov(r)} |")
        L.append("")
        L.append("### Requirement-met fraction, mean +- 95% CI (all presets pooled per seed)\n")
        a = agg[mdl]
        a = a[a["preset"] == "ALL"]
        L.append("| policy | " + " | ".join(MISMATCH) + " |")
        L.append("|---|" + "---|" * len(MISMATCH))
        for pol in POLICIES:
            cells = []
            for mm in MISMATCH:
                r = a[(a["policy"] == pol) & (a["mismatch"] == mm)]
                cells.append(f"{r['frac_met'].iloc[0]:.3f} +- {r['frac_met_ci95'].iloc[0]:.3f}" if len(r) else "-")
            L.append(f"| {pol} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("### Mean volume (GPA), mean +- 95% CI (all presets pooled per seed)\n")
        L.append("| policy | " + " | ".join(MISMATCH) + " |")
        L.append("|---|" + "---|" * len(MISMATCH))
        for pol in POLICIES:
            cells = []
            for mm in MISMATCH:
                r = a[(a["policy"] == pol) & (a["mismatch"] == mm)]
                cells.append(f"{r['mean_volume_gpa'].iloc[0]:.2f} +- {r['mean_volume_gpa_ci95'].iloc[0]:.2f}" if len(r) else "-")
            L.append(f"| {pol} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("### Cost: mean volume (GPA) / mean drift_frac / mean shortfall (pp), mismatch med, all presets\n")
        L.append("| policy | volume GPA | drift_frac | shortfall pp | retreat flags / pass |")
        L.append("|---|---|---|---|---|")
        for pol in POLICIES:
            r = a[(a["policy"] == pol) & (a["mismatch"] == "med")]
            if len(r):
                r = r.iloc[0]
                L.append(f"| {pol} | {r['mean_volume_gpa']:.1f} +- {r['mean_volume_gpa_ci95']:.1f} | "
                         f"{r['mean_drift_frac']:.4f} +- {r['mean_drift_frac_ci95']:.4f} | "
                         f"{r['mean_shortfall_pp']:.2f} +- {r['mean_shortfall_pp_ci95']:.2f} | "
                         f"{r['n_retreat']:.2f} +- {r['n_retreat_ci95']:.2f} |")
        L.append("")
        ab = abl[mdl]
        if not ab.empty:
            L.append(f"### Ablations (preset {ABL_PRESET}, mismatch {ABL_MISMATCH}; fast loop unless "
                     "noted; reference = fair oracle)\n")
            L.append("| group | variant | seeds | met | gap recovery | volume vs oracle |")
            L.append("|---|---|---|---|---|---|")
            for _, r in ab.iterrows():
                L.append(f"| {r['group']} | {r['variant']} | {r['n_seeds']} | {r['frac_met']:.3f} +- "
                         f"{r['frac_met_ci95']:.3f} | {_fmt_gap(r)} | {_fmt_ov(r)} |")
            L.append("")
        s = slow[mdl]
        L.append(f"### Slow loop across {P} consecutive passes (all presets; mean +- 95% CI over seeds)\n")
        L.append("Residual = measured - model in use (frozen base + slow-loop correction refit after every "
                 "pass), before the fast-loop bias; var is the within-pass variance, MSE = mean^2 + var. "
                 "Pass 1 has no correction yet (identical to the fast loop). Each pass has a fresh hidden "
                 "realization (gust, canopy, wear), so only the cross-pass systematic part is learnable. The "
                 "`fast (eval pass)` row is the fast loop on the same hidden realization as the last pass.\n")
        cols = (("raw_resid_var", "resid var", 1), ("raw_resid_mse", "resid MSE", 1),
                ("raw_resid_mean", "resid mean", 2), ("slow_corr_mean", "correction", 2),
                ("frac_met", "met", 3), ("mean_volume_gpa", "GPA", 2))
        df_runs = pd.DataFrame(res["runs"])
        for mm in MISMATCH:
            L.append(f"**mismatch {mm}**\n")
            L.append("| pass | " + " | ".join(c[1] for c in cols) + " |")
            L.append("|---|" + "---|" * len(cols))
            sub = s[s["mismatch"] == mm]
            for i in range(P):
                sl = sub[sub["slow_pass"] == i].groupby("seed")[[c[0] for c in cols if c[0] in sub]].mean()
                cells = []
                for c, _, d in cols:
                    if c not in sl:
                        cells.append("-")
                        continue
                    m_, h = mean_ci(sl[c])
                    cells.append(f"{m_:.{d}f} +- {h:.{d}f}")
                L.append(f"| {i + 1} | " + " | ".join(cells) + " |")
            fr = df_runs[(df_runs["policy"] == "fast") & (df_runs["mismatch"] == mm)]
            if len(fr):
                sl = fr.groupby("seed")[[c[0] for c in cols if c[0] in fr]].mean()
                cells = []
                for c, _, d in cols:
                    if c not in sl:
                        cells.append("-")
                        continue
                    m_, h = mean_ci(sl[c])
                    cells.append(f"{m_:.{d}f} +- {h:.{d}f}")
                L.append("| fast (eval pass) | " + " | ".join(cells) + " |")
            L.append("")
    diag = RESULTS / "closed_loop_slow_diagnostic.md"
    if diag.is_file():
        L.append("## Slow-loop diagnostic\n")
        L.append("Per-pass interval calibration, residuals and the legacy (midterm) refit for comparison, on "
                 "dev seeds disjoint from the evaluation seeds: see `results/closed_loop_slow_diagnostic.md` "
                 "(generated by `experiments/slow_loop_diagnostic.py`).\n")
    L.append(LIMITATIONS)
    (RESULTS / "closed_loop_summary.md").write_text("\n".join(L) + "\n", encoding="utf8")


LIMITATIONS = """## Limitations

The evaluation is circular in an important sense. The plant, the model's training data, the
renderer and the hidden mismatch all come from one hand-built simulator (physics-informed, fit to
published trends, not a validated drift model), so the experiment measures whether the *control
logic* (selector + bias feedback + slow-loop correction) behaves as designed when the model is wrong
in a known, low-dimensional way. It does not measure field efficacy. The mismatch is injected by us
(gust multiplier, canopy interception, nozzle wear) and is therefore of the kind the bias term and
the slow-loop offset are built to absorb (a coverage offset that persists over a pass, with a mean
that persists across passes); unmodeled failure modes such as inversions, boom roll, nozzle
clogging or card placement error are not tested. Two oracles are reported. The fair oracle knows the
true hidden mismatch of each segment but not the random per-card WSP spread factor, and selects with
the same margin rule as the deployable policies; it is the reference for the proposal criterion,
because the spread factor is drawn when the card is sprayed and no controller can know it in
advance. The card-aware oracle also sees the spread factor and needs no margin, so its gap is an
upper bound that includes irreducible card noise. The fair oracle's lower bound is the exact 10%
quantile of the card noise, so it is not required to meet C_req on every segment and a deployable
policy with a wider margin can exceed its requirement-met share (at a volume cost). The default
verifier ("noisy") is perfect coverage plus Gaussian error with per-coverage-band sd and bias taken
from the synthetic-card evaluation of the classical vision module, which omits heavy-tailed failures,
real-card domain shift and saturation (only a 5-seed rendered check is included). Scenario conditions
are a synthetic generator, not weather records. The slow loop is evaluated on consecutive passes of
one scenario with a fresh hidden realization per pass, so only the part of the model error that is
systematic across passes can be learned. Results should be read as a controlled comparison of
policies inside the simulator, with the confidence intervals reflecting seed variability of that
simulator only.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, choices=["spray", "oracle_nominal", "mock"])
    ap.add_argument("--verifier", default="noisy", choices=["noisy", "perfect", "b0", "b1"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--record-weight", type=float, default=30.0,
                    help="SprayModel record weight for the LEGACY refit slow loop (ablation only)")
    ap.add_argument("--n-jobs", type=int, default=2, help="LightGBM threads per worker (refits)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--combine", action="store_true", help="only rebuild outputs from per-model json")
    ap.add_argument("--oracle-q-sensitivity", action="store_true",
                    help="only re-run the fair oracle at q in ORACLE_Q_SENS, then combine")
    ap.add_argument("--out-dir", default=None, help="write results/figures here (default: repo)")
    args = ap.parse_args()
    global RESULTS, FIGS
    if args.out_dir:
        RESULTS, FIGS = Path(args.out_dir) / "results", Path(args.out_dir) / "figures"
        RESULTS.mkdir(parents=True, exist_ok=True)
        FIGS.mkdir(parents=True, exist_ok=True)
    if args.combine:
        combine()
        return
    if args.model is None:
        args.model = "spray" if integration.spray_model_available() else "oracle_nominal"
    if args.quick:
        args.seeds = min(args.seeds, 3)
    if args.oracle_q_sensitivity:
        run_sensitivity(args)
    else:
        run_experiment(args)
    combine()


if __name__ == "__main__":
    main()
