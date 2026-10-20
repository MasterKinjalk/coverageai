"""Does a short-horizon wind forecaster help the closed loop? (coverageai.forecast)

Arms (forecaster given to run_pass / run_passes_slow; everything else identical):
  off           no forecast (the "nominal" forecaster: plans with x_k exactly like
                forecaster=None, verified bit-identical in tests/test_forecast.py, but logs
                the realized wind so the nominal forecast MAE can be reported)
  last          persistence of the last anemometer reading (gust ratio r_{k-1})
  ewma          EWMA of the gust ratio (beta 0.5)
  ar1           AR(1) on the gust ratio, fitted online (shrunk least squares)
  ar1+gust      ar1, selector plans for forecast + z sd (z = 1)
  nominal+gust  no forecast, but plans for nominal + z sd (uncertainty-only control: is a
                gain from prediction or just from planning for more wind / more volume?)
  gust_oracle   REFERENCE, not deployable: knows segment k's realized wind exactly. Upper
                bound on what any wind forecaster can gain in this simulator.
Loops: fast (one pass, hidden seed E = s + 4036, fresh forecaster) and fast_slow (5 passes,
seeds s, s+1009, ..., E; ONE forecaster carried over the passes so its gust model keeps
learning; last pass scored), exactly the protocol of experiments/closed_loop.py, so the off
arm reproduces the closed-loop fast / fast_slow rows. The fair oracle (policy "oracle") on
the same evaluation pass is reported for context.

Grid: mismatch {none, low, med, high} x presets {gusty_afternoon, crosswind_shift,
marginal_limits, calm_morning} x 20 seeds (0..19). Anemometer noise sd 0.3 m/s.
Metrics on the TRUE outcome: fraction of segments meeting C_req, mean shortfall (pp), mean
volume (GPA), mean drift fraction; forecast MAE (m/s) of the point forecast vs the realized
wind. Differences arm - off are paired by seed (seed-level means over presets); 95% CIs by
paired bootstrap over seeds (B = 2000, helper from closed_loop.py).

Usage:
  uv run python experiments/eval_forecast.py --quick          # pilot: 3 seeds, 2 presets
  uv run python experiments/eval_forecast.py --workers 4      # full grid
  uv run python experiments/eval_forecast.py --combine        # rebuild md/json from raw
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from closed_loop import EVAL_OFFSET, SLOW_PASSES, _boot, mean_ci   # noqa: E402
from coverageai import integration, scenarios                     # noqa: E402
from coverageai.config import spec                                 # noqa: E402
from coverageai.loop import run_pass, run_passes_slow, summarize   # noqa: E402

RESULTS = ROOT / "results"
FIGS = ROOT / "experiments" / "figures"
ARMS = ("off", "last", "ewma", "ar1", "ar1+gust", "nominal+gust", "gust_oracle")
ARM_FC = {"off": "nominal"}                       # arm -> forecaster name (default: same)
DEPLOYABLE = ("last", "ewma", "ar1", "ar1+gust", "nominal+gust")
LOOPS = ("fast", "fast_slow")
MISMATCH = ("none", "low", "med", "high")
PRESETS = ("gusty_afternoon", "crosswind_shift", "marginal_limits", "calm_morning")
METRICS = ("frac_met", "mean_shortfall_pp", "mean_volume_gpa", "mean_drift_frac", "fc_mae_ms",
           "nominal_mae_ms")
TRACE = ("gusty_afternoon", "high", 0)

_W: dict = {}


def _init(model: str, verifier: str) -> None:
    _W["mods"] = integration.build_modules(model, verifier)


def pass_metrics(logs, c_req: float) -> dict:
    s = summarize(logs, c_req)
    out = {k: s[k] for k in ("frac_met", "mean_shortfall_pp", "mean_volume_gpa",
                             "mean_drift_frac", "n_retreat")}
    err = [l.extra["fc_abs_err_ms"] for l in logs if l.extra.get("fc_abs_err_ms") is not None]
    nom = [l.extra["fc_nominal_abs_err_ms"] for l in logs
           if l.extra.get("fc_nominal_abs_err_ms") is not None]
    out["fc_mae_ms"] = float(np.mean(err)) if err else float("nan")
    out["nominal_mae_ms"] = float(np.mean(nom)) if nom else float("nan")
    out["fc_phi_end"] = float(logs[-1].extra.get("fc_phi", float("nan")))
    out["mean_plan_minus_nominal_ms"] = float(np.mean(
        [l.extra.get("fc_plan_wind_ms", l.x.wind_speed_ms) - l.x.wind_speed_ms for l in logs]))
    return out


def wind_sensitivity(sim, logs, factor: float = 1.3) -> dict:
    """Mismatch-free physics: mean change in coverage (pp) and drift fraction when the wind
    of each logged (x, u) is multiplied by `factor` (1.3 = a 1-sd gust at mismatch high).
    Bounds how much any wind forecast can matter for the settings actually chosen."""
    if not hasattr(sim, "nominal_arrays"):
        return {}
    xs = [l.x for l in logs]
    us = [l.u for l in logs]
    hi = [dataclasses.replace(x, wind_speed_ms=x.wind_speed_ms * factor) for x in xs]
    a, b = sim.nominal_arrays(xs, us), sim.nominal_arrays(hi, us)
    return {"sens_dcov_pp": float(np.mean(b["coverage_pct"] - a["coverage_pct"])),
            "sens_dcov_abs_max_pp": float(np.max(np.abs(b["coverage_pct"] - a["coverage_pct"]))),
            "sens_ddrift": float(np.mean(b["drift_frac"] - a["drift_frac"]))}


def wind_trace(logs) -> dict:
    return {k: [l.extra.get(f"fc_{k}") for l in logs]
            for k in ("nominal_wind_ms", "wind_ms", "plan_wind_ms", "realized_wind_ms",
                      "obs_wind_ms", "sd_wind_ms")} | {
        "true_cov": [l.outcome.coverage_pct for l in logs],
        "gpa": [l.derived.app_rate_gpa for l in logs]}


def cell(preset: str, mismatch: str, s: int, c_req: float, arms=ARMS, loops=LOOPS) -> dict:
    mods = _W["mods"]
    conds = scenarios.generate(scenarios.get(preset), s)
    E = s + EVAL_OFFSET
    n = len(conds)
    rows, traces = [], {}
    base = {"preset": preset, "mismatch": mismatch, "seed": s}
    logs = run_pass(conds, "oracle", mods, E, mismatch, c_req, "ref_oracle")
    rows.append({**base, "loop": "reference", "arm": "fair_oracle", **pass_metrics(logs, c_req)})
    for arm in arms:
        fc = ARM_FC.get(arm, arm)
        if "fast" in loops:
            logs = run_pass(conds, "fast", mods, E, mismatch, c_req, f"fast_{arm}", forecaster=fc)
            extra = wind_sensitivity(mods.simulator, logs) if arm == "off" else {}
            rows.append({**base, "loop": "fast", "arm": arm, **pass_metrics(logs, c_req), **extra})
            if (preset, mismatch, s) == TRACE and arm in ("off", "ar1", "ar1+gust", "gust_oracle"):
                traces[arm] = wind_trace(logs)
        if "fast_slow" in loops:
            logs = run_passes_slow(conds, mods, s, mismatch, c_req, n_passes=SLOW_PASSES,
                                   forecaster=fc)
            rows.append({**base, "loop": "fast_slow", "arm": arm,
                         **pass_metrics(logs[(SLOW_PASSES - 1) * n:], c_req)})
    return {"rows": rows, "traces": traces}


# ------------------------------------------------------------------ statistics

def _mdiff(a, b):
    return float(np.mean(a - b))


def _pct(a, b):
    return 100.0 * (a.mean() / b.mean() - 1.0)


def paired_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per loop x group (mismatch level / pooled) x preset ('ALL' = seed means over presets):
    each arm's mean and the paired difference vs 'off' with bootstrap 95% CIs."""
    groups = {**{m: [m] for m in MISMATCH}, "low+med+high": ["low", "med", "high"]}
    rows = []
    d = df[df["loop"] != "reference"]
    ref = df[df["loop"] == "reference"]
    for loop in LOOPS:
        for gname, levels in groups.items():
            for preset in ("ALL",) + tuple(sorted(df["preset"].unique())):
                sub = d[(d["loop"] == loop) & d["mismatch"].isin(levels)]
                rsub = ref[ref["mismatch"].isin(levels)]
                if preset != "ALL":
                    sub, rsub = sub[sub["preset"] == preset], rsub[rsub["preset"] == preset]
                if sub.empty:
                    continue
                sl = sub.groupby(["arm", "seed"], as_index=False)[list(METRICS)].mean()
                rsl = rsub.groupby("seed")[["frac_met", "mean_volume_gpa"]].mean()
                piv = {m: sl.pivot(index="seed", columns="arm", values=m) for m in METRICS}
                if "off" not in piv["frac_met"]:
                    continue
                off = {m: piv[m]["off"].to_numpy() for m in METRICS}
                orc = rsl.reindex(piv["frac_met"].index)
                for arm in ARMS:
                    if arm not in piv["frac_met"]:
                        continue
                    a = {m: piv[m][arm].to_numpy() for m in METRICS}
                    r = {"loop": loop, "mismatch": gname, "preset": preset, "arm": arm,
                         "n_seeds": len(a["frac_met"]),
                         "oracle_frac_met": float(orc["frac_met"].mean()),
                         "oracle_gpa": float(orc["mean_volume_gpa"].mean())}
                    for m in METRICS:
                        r[m], r[m + "_ci95"] = mean_ci(a[m])
                    if arm != "off":
                        for m, fn, key in (("frac_met", _mdiff, "d_met"),
                                           ("mean_shortfall_pp", _mdiff, "d_shortfall_pp"),
                                           ("mean_drift_frac", _mdiff, "d_drift"),
                                           ("fc_mae_ms", _mdiff, "d_mae_ms"),
                                           ("mean_volume_gpa", _pct, "d_volume_pct")):
                            est, lo, hi = _boot(fn, [a[m], off[m]])
                            r[key], r[key + "_lo"], r[key + "_hi"] = est, lo, hi
                    rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ output

def _ci(r, key, scale=1.0, fmt="+.1f", unit="") -> str:
    if key not in r or not np.isfinite(r.get(key, np.nan)):
        return "-"
    lo, hi = r[key + "_lo"] * scale, r[key + "_hi"] * scale
    sig = "" if (np.isfinite(lo) and lo <= 0 <= hi) else " *"
    return f"{r[key] * scale:{fmt}}{unit} [{lo:{fmt}}, {hi:{fmt}}]{sig}"


def fig(tab: pd.DataFrame, traces: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})
    fig_, axes = plt.subplots(1, 3, figsize=(14.5, 3.8), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    ax = axes[0]
    if traces:
        k = np.arange(len(traces["off"]["realized_wind_ms"]))
        t = traces["ar1"]
        ax.plot(k, t["realized_wind_ms"], color="k", lw=1.6, label="realized wind (hidden gust)")
        ax.plot(k, t["nominal_wind_ms"], color="#7f7f7f", lw=1.0, ls=":", label="nominal x_k (off)")
        ax.plot(k, t["obs_wind_ms"], color="#CC79A7", lw=0, marker=".", ms=4,
                label="anemometer reading (after the segment)")
        ax.plot(k, t["wind_ms"], color="#0072B2", lw=1.3, label="AR(1) forecast")
        ax.plot(k, traces["ar1+gust"]["plan_wind_ms"], color="#D55E00", lw=1.0, ls="--",
                label="gust-aware planning wind (z = 1)")
        ax.set_xlabel("segment k")
        ax.set_ylabel("wind speed (m/s)")
        ax.set_title("example: gusty_afternoon, mismatch high, seed 0 (fast loop)")
        ax.legend(fontsize=6.5, ncol=2, loc="upper left")
    colors = {"last": "#E69F00", "ewma": "#56B4E9", "ar1": "#0072B2", "ar1+gust": "#D55E00",
              "nominal+gust": "#7f7f7f", "gust_oracle": "#009E73"}
    for ax, (key, lab, sc) in zip(axes[1:], (("d_met", "change in segments meeting C_req (pp)", 100),
                                             ("d_volume_pct", "change in volume (%)", 1))):
        for j, loop in enumerate(LOOPS):
            for i, arm in enumerate(colors):
                g = tab[(tab["loop"] == loop) & (tab["preset"] == "ALL") & (tab["arm"] == arm)
                        & tab["mismatch"].isin(MISMATCH)].set_index("mismatch").reindex(MISMATCH)
                xs = np.arange(len(MISMATCH)) + (i - 2.5) * 0.11 + j * 0.0
                y = g[key] * sc
                yerr = np.vstack([(g[key] - g[key + "_lo"]) * sc, (g[key + "_hi"] - g[key]) * sc])
                ax.errorbar(xs + (0.04 if j else -0.04), y, yerr=yerr, color=colors[arm],
                            marker="o" if j == 0 else "s", ms=3.2, lw=0, elinewidth=1.0, capsize=1.5,
                            mfc=colors[arm] if j == 0 else "white",
                            label=f"{arm} ({'fast' if j == 0 else 'fast+slow'})")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(range(len(MISMATCH)), MISMATCH)
        ax.set_xlabel("hidden mismatch level")
        ax.set_title(f"{lab}\nvs forecaster off, paired 95% CI (all presets)")
    axes[1].legend(fontsize=5.8, ncol=2, loc="lower left")
    fig_.tight_layout()
    fig_.savefig(path)
    plt.close(fig_)


def write_md(raw: dict, tab: pd.DataFrame, path: Path) -> None:
    L = ["# Short-horizon wind forecaster: evaluation\n",
         "Generated by `experiments/eval_forecast.py` (raw runs and tables in "
         "`results/forecast_eval.json`). Model `" + raw["model"] + "`, verifier `" + raw["verifier"]
         + f"`, {len(raw['seeds'])} seeds, presets {', '.join(raw['presets'])}, anemometer noise sd "
         f"{raw['obs_sd_ms']} m/s, C_req = {raw['c_req_pct']:g}%. Protocol as in `closed_loop.py` "
         "(paired hidden seed E = s + 4036; fast + slow = 5 passes, last scored). Differences are "
         "arm minus forecaster-off on the same passes; brackets are 95% paired-bootstrap CIs over "
         "seeds; `*` = CI excludes 0.\n"]
    if raw.get("headline"):
        L.append(raw["headline"].rstrip() + "
")
    verdict = raw.get("verdict")
    if verdict:
        L.append(verdict.rstrip() + "\n")
    for loop in LOOPS:
        lab = "fast loop" if loop == "fast" else "fast + slow loop (5th pass)"
        L.append(f"## {lab}: all presets, by mismatch level\n")
        L.append("| mismatch | arm | met | d met (pp) | d shortfall (pp) | d volume (%) | d drift (x1e-3) "
                 "| forecast MAE (m/s) | d MAE (m/s) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for mm in MISMATCH + ("low+med+high",):
            g = tab[(tab["loop"] == loop) & (tab["preset"] == "ALL") & (tab["mismatch"] == mm)]
            for _, r in g.set_index("arm").reindex(ARMS).dropna(how="all").reset_index().iterrows():
                L.append(f"| {mm} | {r['arm']} | {r['frac_met']:.3f} | {_ci(r, 'd_met', 100)} | "
                         f"{_ci(r, 'd_shortfall_pp', 1, '+.2f')} | {_ci(r, 'd_volume_pct')} | "
                         f"{_ci(r, 'd_drift', 1000, '+.2f')} | {r['fc_mae_ms']:.3f} | "
                         f"{_ci(r, 'd_mae_ms', 1, '+.3f')} |")
            if len(g):
                L.append(f"| {mm} | *fair oracle (ref.)* | {g['oracle_frac_met'].iloc[0]:.3f} | | | | | | |")
        L.append("")
        L.append(f"### {lab}: per preset, mismatch pooled over low + med + high\n")
        L.append("| preset | arm | met | d met (pp) | d shortfall (pp) | d volume (%) | forecast MAE | d MAE |")
        L.append("|---|---|---|---|---|---|---|---|")
        g = tab[(tab["loop"] == loop) & (tab["preset"] != "ALL") & (tab["mismatch"] == "low+med+high")]
        for _, r in g.iterrows():
            if r["arm"] == "off":
                L.append(f"| {r['preset']} | off | {r['frac_met']:.3f} | | | | {r['fc_mae_ms']:.3f} | |")
                continue
            L.append(f"| {r['preset']} | {r['arm']} | {r['frac_met']:.3f} | {_ci(r, 'd_met', 100)} | "
                     f"{_ci(r, 'd_shortfall_pp', 1, '+.2f')} | {_ci(r, 'd_volume_pct')} | "
                     f"{r['fc_mae_ms']:.3f} | {_ci(r, 'd_mae_ms', 1, '+.3f')} |")
        L.append("")
    sens = raw.get("sensitivity")
    if sens:
        L.append("## Why the headroom is small: wind sensitivity of the chosen settings
")
        L.append("Mismatch-free physics at the (x, u) pairs the forecaster-off fast loop actually chose: "
                 "mean change when the wind is 30% stronger (a 1-sd gust at mismatch high). Mean over "
                 "seeds and mismatch levels.
")
        L.append("| preset | d coverage (pp) | max abs d coverage (pp) | d drift fraction (x1e-3) |")
        L.append("|---|---|---|---|")
        for r in sens:
            L.append(f"| {r['preset']} | {r['sens_dcov_pp']:+.3f} | {r['sens_dcov_abs_max_pp']:.3f} | "
                     f"{1000 * r['sens_ddrift']:+.2f} |")
        L.append("")
    L.append(NOTES)
    path.write_text("\n".join(L) + "\n", encoding="utf8")


NOTES = """## Method notes and caveats

* **What the forecaster can know.** x_k (read at the segment start) is given to every arm. The only
  wind information it adds is the hidden gust multiplier g_k (AR(1), phi = 0.7 per segment, sd =
  mismatch `gust_sigma`: 0 / 0.10 / 0.20 / 0.30), learned from noisy anemometer readings of the
  segments already sprayed. A one-step AR(1) forecast can explain at most phi^2 = 49% of the gust
  variance, so the best possible forecast error is about 70% of the nominal one, before sensor noise.
* **gust_oracle** knows g_k exactly and is not deployable; it is the ceiling for any wind forecaster in
  this simulator. If it barely moves the metrics, wind forecasting has little to offer here no matter
  how good the forecaster is.
* The fast-loop bias already tracks the persistent part of the gust through the card residuals (alpha
  0.3), and canopy interception and nozzle wear (which the wind forecaster cannot see) dominate the
  coverage error at med / high mismatch. The forecaster only addresses the wind part.
* Gust-aware planning (forecast + z sd, z = 1) buys reliability with volume; `nominal+gust` is the
  control that plans for extra wind without any forecast, so a gain of `ar1+gust` over `nominal+gust`
  is the value of the forecast itself.
* Defaults (beta, priors, z, window) were fixed before the evaluation grid from a 4-seed pilot on dev
  seeds 100-103; nothing was tuned on seeds 0-19.
* Simulator-internal evaluation (same caveats as `closed_loop_summary.md`): the gust model is
  multiplicative AR(1) on speed only (direction unaffected), the anemometer is unbiased Gaussian, and
  scenario wind x_k is assumed known at the segment start. Real gusts have heavier tails and
  sub-segment structure that a per-segment forecast cannot follow.
"""


def verdict_text(tab: pd.DataFrame) -> str:
    """Plain-English verdict generated from the pooled numbers (low+med+high, all presets)."""
    def row(loop, arm, mm="low+med+high", preset="ALL"):
        g = tab[(tab["loop"] == loop) & (tab["arm"] == arm) & (tab["mismatch"] == mm)
                & (tab["preset"] == preset)]
        return None if g.empty else g.iloc[0]

    def sig(r, key):
        return r is not None and np.isfinite(r[key + "_lo"]) and not (r[key + "_lo"] <= 0 <= r[key + "_hi"])

    L = ["## Verdict\n"]
    for loop in LOOPS:
        lab = "Fast loop" if loop == "fast" else "Fast + slow loop"
        parts = []
        for arm in ("ar1", "ar1+gust", "nominal+gust", "ewma", "last", "gust_oracle"):
            r = row(loop, arm)
            if r is None:
                continue
            parts.append(f"`{arm}` {100 * r['d_met']:+.1f} pp met [{100 * r['d_met_lo']:+.1f}, "
                         f"{100 * r['d_met_hi']:+.1f}]{' *' if sig(r, 'd_met') else ''}, volume "
                         f"{r['d_volume_pct']:+.1f}% [{r['d_volume_pct_lo']:+.1f}, {r['d_volume_pct_hi']:+.1f}], "
                         f"wind MAE {r['d_mae_ms']:+.2f} m/s")
        o = row(loop, "off")
        head = (f"* **{lab}** (pooled low+med+high, all presets; off meets C_req in "
                f"{o['frac_met']:.3f} of segments, fair oracle {o['oracle_frac_met']:.3f}): ") if o is not None else f"* **{lab}**: "
        L.append(head + "; ".join(parts) + ".")
    return "\n".join(L) + "\n"


def run(args) -> dict:
    c_req = float(spec()["c_req_pct"])
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    presets = PRESETS[:2] if args.quick else PRESETS
    mms = MISMATCH
    t0 = time.time()
    rows, traces = [], {}
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.model, args.verifier)) as ex:
        futs = [ex.submit(cell, p, mm, s, c_req) for p in presets for mm in mms for s in seeds]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            rows += r["rows"]
            if r["traces"]:
                traces = r["traces"]
            if i % 10 == 0 or i == len(futs):
                print(f"[{time.time() - t0:6.0f}s] cells {i}/{len(futs)}", flush=True)
    return {"schema": "forecast_eval/1", "model": args.model, "verifier": args.verifier,
            "c_req_pct": c_req, "seeds": seeds, "presets": list(presets), "mismatch": list(mms),
            "arms": list(ARMS), "loops": list(LOOPS), "slow_passes": SLOW_PASSES,
            "eval_seed_offset": EVAL_OFFSET, "obs_sd_ms": float(spec()["forecast"]["obs_sd_ms"]),
            "forecast_config": dict(spec()["forecast"]), "wall_s": time.time() - t0,
            "workers": args.workers, "runs": rows,
            "example_trace": {"preset": TRACE[0], "mismatch": TRACE[1], "seed": TRACE[2], **traces}}


def combine(raw: dict, out_stem: Path) -> None:
    df = pd.DataFrame(raw["runs"])
    tab = paired_table(df)
    raw["verdict"] = raw.get("verdict_override") or verdict_text(tab)
    raw["table"] = tab.replace({np.nan: None}).to_dict("records")
    if "sens_dcov_pp" in df:
        raw["sensitivity"] = (df.dropna(subset=["sens_dcov_pp"]).groupby("preset")[
            ["sens_dcov_pp", "sens_dcov_abs_max_pp", "sens_ddrift"]].mean().reset_index()
            .to_dict("records"))
    out_stem.with_suffix(".json").write_text(json.dumps(raw, default=float), encoding="utf8")
    write_md(raw, tab, out_stem.with_suffix(".md"))
    FIGS.mkdir(parents=True, exist_ok=True)
    fig(tab, {k: v for k, v in raw["example_trace"].items() if isinstance(v, dict)},
        FIGS / (out_stem.name + ".png"))
    print("wrote", out_stem.with_suffix(".json"), out_stem.with_suffix(".md"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="spray", choices=list(integration.MODEL_CHOICES))
    ap.add_argument("--verifier", default="noisy")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="pilot: 3 seeds, 2 presets")
    ap.add_argument("--out", default=None, help="output stem (default results/forecast_eval)")
    ap.add_argument("--combine", action="store_true", help="rebuild outputs from the json")
    ap.add_argument("--headline", default=None,
                    help="with --combine: markdown file with the hand-written verdict, stored in "
                         "the json ('headline') and printed at the top of the md")
    args = ap.parse_args()
    stem = Path(args.out) if args.out else RESULTS / "forecast_eval"
    if args.combine:
        raw = json.loads(stem.with_suffix(".json").read_text(encoding="utf8"))
        if args.headline:
            raw["headline"] = Path(args.headline).read_text(encoding="utf8")
        combine(raw, stem)
        return
    if args.quick:
        args.seeds = min(args.seeds, 3)
    combine(run(args), stem)


if __name__ == "__main__":
    main()
