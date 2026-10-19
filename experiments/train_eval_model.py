"""WP-10 model training and evaluation.

* Random 80/20 split and three grouped splits (hold out U > 5 m/s; hold out nozzle family
  3D; hold out hot-dry T > 35 C and RH < 30 %), for ridge, random forest and LightGBM.
* MAE / RMSE per output, 10-90 interval coverage, error of the random-split model on each
  regime, and the irreducible noise floor of coverage (spread factor is sampled per row).
* One-at-a-time sensitivity sweeps as pass/fail, versus the simulator.
* Writes results/model_v1.json, experiments/figures/model_parity.png and
  model_sensitivity.png, and saves the final model (all rows) to models/spray_model_v1.joblib.

Usage: uv run python experiments/make_dataset.py && uv run python experiments/train_eval_model.py
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from coverageai import catalog
from coverageai.config import ROOT, spec
from coverageai.contracts import Conditions, Setting
from coverageai.model import features as F
from coverageai.model.predictor import (DEFAULT_PATH, TARGETS, ForestBaseline, RidgeBaseline,
                                        SprayModel, _from_internal)
from coverageai.simulator import SpraySimulator

OUT_KEYS = ("coverage_pct", "deposition_ul_cm2", "drift_frac", "displacement_mm")
REGIMES = {
    "U_gt_5": lambda d: d["wind_speed_ms"] > 5.0,
    "family_3D": lambda d: d["family"] == "3D",
    "hot_dry": lambda d: (d["temp_c"] > 35.0) & (d["rh_pct"] < 30.0),
}


def metrics(pred: np.ndarray, Y: np.ndarray, exp_cov: np.ndarray | None = None) -> dict:
    e = pred - Y
    m = {k: {"mae": float(np.abs(e[:, j]).mean()), "rmse": float(np.sqrt((e[:, j] ** 2).mean()))}
         for j, k in enumerate(OUT_KEYS)}
    if exp_cov is not None:   # error against the spread-factor-averaged (noise-free) coverage
        m["coverage_pct"]["mae_vs_expected"] = float(np.abs(pred[:, 0] - exp_cov).mean())
    return m


def interval_stats(a: dict, y: np.ndarray) -> dict:
    lo, hi = a["coverage_lo"], a["coverage_hi"]
    return {"coverage_of_80pct_band": float(((y >= lo) & (y <= hi)).mean()),
            "frac_below": float((y < lo).mean()), "frac_above": float((y > hi).mean()),
            "mean_width_pp": float((hi - lo).mean())}


def predict_all(model, Fm: np.ndarray) -> np.ndarray:
    if isinstance(model, SprayModel):
        a = model.predict_features(Fm)
        return np.column_stack([a[k] for k in OUT_KEYS])
    return model.predict_arrays(Fm)


def run_split(name, df, Fall, Yall, test_mask, seed, exp_cov) -> tuple[dict, dict]:
    tr, te = ~test_mask, test_mask
    res: dict = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "models": {}}
    preds: dict[str, np.ndarray] = {}
    Ftr, Ytr, Fte, Yte = Fall[tr], Yall[tr], Fall[te], Yall[te]
    t = time.perf_counter()
    ridge = RidgeBaseline().fit_arrays(Ftr, Ytr)
    preds["ridge"] = ridge.predict_arrays(Fte)
    res["models"]["ridge"] = {**metrics(preds["ridge"], Yte, exp_cov[te]), "fit_s": time.perf_counter() - t}
    t = time.perf_counter()
    rf = ForestBaseline(n_estimators=60, seed=seed).fit_arrays(Ftr, Ytr)
    preds["rf"] = rf.predict_arrays(Fte)
    res["models"]["rf"] = {**metrics(preds["rf"], Yte, exp_cov[te]), "fit_s": time.perf_counter() - t}
    t = time.perf_counter()
    gbm = SprayModel(seed=seed).fit_frame(df[tr])
    a = gbm.predict_features(Fte)
    preds["lightgbm"] = np.column_stack([a[k] for k in OUT_KEYS])
    res["models"]["lightgbm"] = {**metrics(preds["lightgbm"], Yte, exp_cov[te]), "fit_s": time.perf_counter() - t,
                                 "interval": interval_stats(a, Yte[:, 0])}
    print(f"[{name}] n_test={te.sum()}  cov MAE ridge/rf/lgbm = "
          f"{res['models']['ridge']['coverage_pct']['mae']:.3f} / "
          f"{res['models']['rf']['coverage_pct']['mae']:.3f} / "
          f"{res['models']['lightgbm']['coverage_pct']['mae']:.3f}", flush=True)
    return res, {"pred": preds, "Yte": Yte, "model": gbm, "ridge": ridge, "rf": rf, "te": te}


def expected_coverage(df: pd.DataFrame) -> np.ndarray:
    """Spread-factor-averaged coverage (9-point midpoint rule over the SF range), mismatch none."""
    xs = [Conditions(*r) for r in df[list(F.X_FIELDS)].itertuples(index=False)]
    us = [Setting(r.family, r.size, float(r.pressure_psi)) for r in df.itertuples()]
    ders = [catalog.derive(x, u) for x, u in zip(xs, us)]
    lo, hi = spec()["physics"]["spread_factor"]
    grid = lo + (np.arange(9) + 0.5) / 9 * (hi - lo)
    return np.mean([SpraySimulator(physics_overrides={"spread_factor": [s, s]}).simulate_arrays(
        xs, us, np.random.default_rng(1), derived=ders)["coverage_pct"] for s in grid], axis=0)


# ---------------------------------------------------------------- sensitivity

BASE_X = Conditions(wind_speed_ms=3.0, wind_dir_deg=90.0, heading_deg=0.0, temp_c=25.0,
                    rh_pct=50.0, travel_speed_kmh=19.0, boom_height_m=0.5)
BASE_U = Setting("XR", "04", 40.0)


def _truth(sim: SpraySimulator, xs, us) -> dict:
    return sim.simulate_arrays(xs, us, np.random.default_rng(0))


def _mid_sim() -> SpraySimulator:
    lo, hi = spec()["physics"]["spread_factor"]
    return SpraySimulator(physics_overrides={"spread_factor": [(lo + hi) / 2] * 2})


def _replace(x: Conditions, **kw) -> Conditions:
    d = x.__dict__ | kw
    return Conditions(**d)


def sensitivity(model: SprayModel) -> tuple[dict, dict]:
    sim = _mid_sim()
    r = spec()["ranges"]
    out: dict = {}
    curves: dict = {}

    def sweep(label, field, grid, out_key, sign, base_x=BASE_X, base_u=BASE_U, min_frac=0.85):
        xs = [_replace(base_x, **{field: float(v)}) for v in grid]
        us = [base_u] * len(grid)
        pm = np.array([getattr(p, out_key) for p in model.predict_batch(xs, us)])
        tr = _truth(sim, xs, us)[out_key]
        d = np.diff(pm) * sign
        tol = 1e-3 * max(1.0, np.ptp(pm))
        frac = float((d >= -tol).mean())
        ok = bool(frac >= min_frac and (pm[-1] - pm[0]) * sign > 0)
        out[label] = {"passed": ok, "monotone_step_fraction": frac, "total_change": float(pm[-1] - pm[0]),
                      "truth_total_change": float(tr[-1] - tr[0]), "field": field, "output": out_key,
                      "expected_sign": sign}
        curves[label] = (np.asarray(grid), pm, tr)

    # dC/dGPA > 0 at fixed droplet spectrum: GPA falls with travel speed, so C should fall.
    sweep("dC/dGPA>0 (speed sweep)", "travel_speed_kmh", np.linspace(*r["travel_speed_kmh"], 25),
          "coverage_pct", -1)
    sweep("ddrift/dU>0", "wind_speed_ms", np.linspace(0.0, 7.0, 29), "drift_frac", +1)
    sweep("ddrift/dH>0", "boom_height_m", np.linspace(*r["boom_height_m"], 25), "drift_frac", +1)

    # dC/dGPA via nozzle size at fixed family and pressure (droplet class may also change).
    sizes = [s for s in catalog.sizes("XR") if catalog.pressure_range("XR", s)[0] <= 40 <= catalog.pressure_range("XR", s)[1]]
    us = [Setting("XR", s, 40.0) for s in sizes]
    gpa = np.array([catalog.derive(BASE_X, u).app_rate_gpa for u in us])
    pm = np.array([p.coverage_pct for p in model.predict_batch([BASE_X] * len(us), us)])
    rho = float(spearmanr(gpa, pm).statistic)
    out["dC/dGPA>0 (size sweep, rank corr)"] = {"passed": bool(rho > 0.8), "spearman": rho, "n": len(us)}

    # finer droplet class -> more drift, at several crosswind conditions.
    cands = catalog.candidates()
    vmd = np.array([catalog.derive(BASE_X, u).vmd_um for u in cands])
    cls = np.array([catalog.derive(BASE_X, u).droplet_class for u in cands])
    rhos, cls_ok = [], []
    for U in (2.0, 3.0, 4.5, 6.0):
        x = _replace(BASE_X, wind_speed_ms=U)
        d = np.array([p.drift_frac for p in model.predict_batch([x] * len(cands), cands)])
        rhos.append(float(spearmanr(vmd, d).statistic))
    x = _replace(BASE_X, wind_speed_ms=4.0)
    d = np.array([p.drift_frac for p in model.predict_batch([x] * len(cands), cands)])
    dt = _truth(sim, [x] * len(cands), cands)["drift_frac"]
    order = list(spec()["droplets"]["class_order"])
    by_cls = {c: float(d[cls == c].mean()) for c in order if (cls == c).any()}
    by_cls_t = {c: float(dt[cls == c].mean()) for c in order if (cls == c).any()}
    seq = list(by_cls.values())
    mono = all(b <= a + 0.01 for a, b in zip(seq, seq[1:]))
    out["finer class -> more drift"] = {"passed": bool(mono and np.mean(rhos) < -0.5),
                                        "spearman_vmd_vs_drift_by_U": rhos, "mean_drift_by_class": by_cls,
                                        "truth_mean_drift_by_class": by_cls_t}
    curves["_class"] = (order, by_cls, by_cls_t)
    return out, curves


def plot_sensitivity(curves: dict, path) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
    items = [("dC/dGPA>0 (speed sweep)", "Travel speed (km/h)  [GPA falls to the right]", "Coverage (%)"),
             ("ddrift/dU>0", "Wind speed (m/s)", "Drift fraction"),
             ("ddrift/dH>0", "Boom height (m)", "Drift fraction")]
    for a, (k, xl, yl) in zip(ax.ravel()[:3], items):
        g, pm, tr = curves[k]
        a.plot(g, tr, "k--", label="simulator")
        a.plot(g, pm, "C0-", label="model")
        a.set_xlabel(xl); a.set_ylabel(yl); a.set_title(k); a.legend(); a.grid(alpha=0.3)
    order, m, t = curves["_class"]
    a = ax[1, 1]
    labs = [c for c in order if c in m]
    a.plot(labs, [t[c] for c in labs], "k--o", label="simulator")
    a.plot(labs, [m[c] for c in labs], "C0-o", label="model")
    a.set_xlabel("Droplet class (fine to coarse)"); a.set_ylabel("Mean drift fraction")
    a.set_title("finer class -> more drift (U=4 m/s crosswind)"); a.legend(); a.grid(alpha=0.3)
    fig.suptitle("Model sensitivity sweeps around the base point (XR04 @ 40 psi, 3 m/s crosswind, H = 0.5 m)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_parity(rand: dict, grouped: dict, results: dict, path) -> None:
    fig, ax = plt.subplots(2, 4, figsize=(17, 8))
    Y, P = rand["Yte"], rand["pred"]["lightgbm"]
    lab = ["Coverage (%)", "Deposition (uL/cm2)", "Drift fraction", "Displacement (mm)"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Y), min(6000, len(Y)), replace=False)
    for j, a in enumerate(ax[0]):
        a.scatter(Y[idx, j], P[idx, j], s=3, alpha=0.3)
        lim = [min(Y[:, j].min(), P[:, j].min()), max(Y[:, j].max(), P[:, j].max())]
        a.plot(lim, lim, "k--", lw=1)
        m = results["splits"]["random"]["models"]["lightgbm"][OUT_KEYS[j]]
        a.set_title(f"random split: {lab[j]}\nMAE {m['mae']:.3g}, RMSE {m['rmse']:.3g}")
        a.set_xlabel("simulator"); a.set_ylabel("LightGBM")
    for a, (name, g) in zip(ax[1][:3], grouped.items()):
        Yg, Pg = g["Yte"][:, 0], g["pred"]["lightgbm"][:, 0]
        sel = rng.choice(len(Yg), min(5000, len(Yg)), replace=False)
        a.scatter(Yg[sel], Pg[sel], s=3, alpha=0.3, color="C1")
        a.plot([0, 100], [0, 100], "k--", lw=1)
        m = results["splits"][name]["models"]["lightgbm"]["coverage_pct"]
        a.set_title(f"held out {name}: coverage\nMAE {m['mae']:.2f} pp")
        a.set_xlabel("simulator"); a.set_ylabel("LightGBM")
    a = ax[1][3]
    names = list(results["splits"])
    w = 0.27
    for i, mdl in enumerate(("ridge", "rf", "lightgbm")):
        a.bar(np.arange(len(names)) + (i - 1) * w,
              [results["splits"][n]["models"][mdl]["coverage_pct"]["mae"] for n in names], w, label=mdl)
    a.set_xticks(np.arange(len(names)), names, rotation=15)
    a.set_ylabel("Coverage MAE (pp)"); a.axhline(3, color="k", ls=":", lw=1); a.legend()
    a.set_title("Coverage MAE by split (dotted: 3 pp target)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "datasets" / "v1.parquet"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = pd.read_parquet(a.data)
    Fall = F.featurize_frame(df)
    Yall = df[list(OUT_KEYS)].to_numpy(float)
    rng = np.random.default_rng(a.seed)
    results: dict = {"dataset": {"path": "data/datasets/v1.parquet", "rows": len(df)}, "splits": {}}

    splits = {"random": rng.random(len(df)) < 0.2,
              **{n: f(df).to_numpy() for n, f in REGIMES.items()}}
    store = {}
    exp_cov = expected_coverage(df)
    results["coverage_noise_floor"] = {
        "irreducible_mae_expected_vs_sampled": float(np.abs(exp_cov - Yall[:, 0]).mean()),
        "note": "coverage depends on a per-row random spread factor; no model can beat this MAE "
                "against sampled outcomes. mae_vs_expected compares predictions with the noise-free mean."}
    for name, mask in splits.items():
        results["splits"][name], store[name] = run_split(name, df, Fall, Yall, mask, a.seed, exp_cov)

    # random-split model evaluated on each regime (in-distribution training)
    r = store["random"]
    gbm_r: SprayModel = r["model"]
    reg = {}
    for n, f in REGIMES.items():
        m = f(df).to_numpy() & r["te"]
        pr = {k: predict_all(mdl, Fall[m]) for k, mdl in
              (("ridge", r["ridge"]), ("rf", r["rf"]), ("lightgbm", gbm_r))}
        reg[n] = {"n_test": int(m.sum()), **{k: metrics(v, Yall[m]) for k, v in pr.items()}}
    results["random_split_by_regime"] = reg

    # final model on all rows
    t = time.perf_counter()
    final = SprayModel(seed=a.seed).fit_frame(df)
    fit_s = time.perf_counter() - t
    path = final.save(DEFAULT_PATH)
    cands = catalog.candidates()
    xs = [BASE_X] * len(cands)
    final.predict_batch(xs, cands)
    ts = []
    for _ in range(100):
        t = time.perf_counter()
        final.predict_batch(xs, cands)
        ts.append(time.perf_counter() - t)
    results["timing"] = {"fit_50k_s": fit_s, "n_candidates": len(cands),
                         "predict_batch_median_ms": float(np.median(ts) * 1e3),
                         "predict_batch_p95_ms": float(np.percentile(ts, 95) * 1e3),
                         "model_path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)}

    sens, curves = sensitivity(final)
    results["sensitivity"] = sens
    plot_sensitivity(curves, ROOT / "experiments" / "figures" / "model_sensitivity.png")
    plot_parity(r, {n: store[n] for n in REGIMES}, results, ROOT / "experiments" / "figures" / "model_parity.png")

    cm = results["splits"]["random"]["models"]
    results["acceptance"] = {
        "coverage_mae_random_le_3pp": cm["lightgbm"]["coverage_pct"]["mae"] <= 3.0,
        "coverage_mae_below_ridge": cm["lightgbm"]["coverage_pct"]["mae"] < cm["ridge"]["coverage_pct"]["mae"],
        "predict_batch_le_20ms": results["timing"]["predict_batch_median_ms"] <= 20.0,
        "refit_le_30s": fit_s <= 30.0,
        "sensitivity_all_pass": all(v["passed"] for v in sens.values()),
        "interval_80pct_band_coverage": cm["lightgbm"]["interval"]["coverage_of_80pct_band"],
    }
    out = ROOT / "results" / "model_v1.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf8")
    print(json.dumps(results["acceptance"], indent=2))
    print(json.dumps(results["timing"], indent=2))
    print(json.dumps(results["coverage_noise_floor"], indent=2))
    for n, sp in results["splits"].items():
        print(n, {m: round(v["coverage_pct"]["mae_vs_expected"], 3) for m, v in sp["models"].items()})
    print({k: v["passed"] for k, v in sens.items()})


if __name__ == "__main__":
    main()
