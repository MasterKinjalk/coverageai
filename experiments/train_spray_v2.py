"""SprayModel v2: XGBoost on the GPU, more simulator rows, tuned depth and rounds (D18).

Usage:
  uv run --no-sync python experiments/train_spray_v2.py                       # full run (CUDA)
  uv run --no-sync python experiments/train_spray_v2.py --quick               # small smoke run

Protocol. Every arm is scored on the SAME held-out rows as v1 (`train_eval_model.py`):
  * test rows: dataset v1, random 80/20 split from default_rng(0); and the three held-out
    regimes (U > 5 m/s, family 3D, hot-dry), each trained without that regime.
  * extra rows: the same generator (`make_dataset.make_dataset`) with another seed
    (--extra-seed), cached under data/datasets/ (gitignored); independent of the v1 rows.
  * tuning: 10 % of the training pool is a validation split (never a test row). For every
    booster and every depth in --depths, a validation curve is recorded up to early stopping.
    Coverage is scored against the noise-free expected coverage (spread factor averaged), so
    the 2.76 pp card noise does not drive model selection; the quantile boosters use the
    pinball loss on sampled coverage. Per booster the cheapest point on any curve
    (rounds x depth, i.e. prediction cost) within --tol of the best validation error wins.
  * arms on the test rows: A = v1 (LightGBM, v1 config, v1 training rows, CPU);
    B = XGBoost tuned on the v1 training rows only (GPU); C = LightGBM v1 config on the big
    pool (CPU); D = XGBoost tuned on the big pool (GPU) = the v2 recipe. B and C separate the
    effect of the model from the effect of more data. Paired bootstrap 95 % CIs of D - A.
  * final v2: all v1 rows + all extra rows, saved WITHOUT its training data (size) to
    models/spray_model_v2.joblib (gitignored unless whitelisted).
Writes results/model_v2.json and results/model_v2.md. Never writes the v1 files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from coverageai import catalog                                              # noqa: E402
from coverageai.model import features as F                                  # noqa: E402
from coverageai.model.predictor import (TARGETS, SprayModel, _to_internal,  # noqa: E402
                                        xgb_train_params)
from make_dataset import make_dataset                                       # noqa: E402
from train_eval_model import (BASE_X, OUT_KEYS, REGIMES, expected_coverage,  # noqa: E402
                              interval_stats, metrics, sensitivity)

RESULTS = ROOT / "results"
DATASETS = ROOT / "data" / "datasets"
QUANTILES = ("cov_lo", "cov_hi")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ data

def load_data(extra_n: int, extra_seed: int):
    v1 = pd.read_parquet(DATASETS / "v1.parquet")
    if v1[list(F.X_FIELDS)].isna().any().any():
        raise SystemExit("data/datasets/v1.parquet has NaN conditions; regenerate it with "
                         "`python experiments/make_dataset.py` (fixed U_COLS)")
    test = np.random.default_rng(0).random(len(v1)) < 0.2     # same draw as train_eval_model.py
    cache = DATASETS / f"v2_extra_{extra_n}_{extra_seed}.parquet"
    if cache.exists():
        extra = pd.read_parquet(cache)
        log(f"extra rows from cache {cache.name}: {len(extra)}")
    else:
        t = time.perf_counter()
        extra = make_dataset(extra_n, extra_seed)
        extra.to_parquet(cache, index=False)
        log(f"generated {len(extra)} extra rows in {time.perf_counter() - t:.0f} s -> {cache.name}")
    return v1, test, extra


# ------------------------------------------------------------------ tuning

def tune(pool: pd.DataFrame, depths, lr: float, max_rounds: int, es: int, tol: float,
         device: str, seed: int, label: str, cost_cap: int | None = None) -> dict:
    """Per booster: validation curves for every depth; pick the cheapest point (rounds x
    depth) within `tol` of the best point whose cost is <= `cost_cap` (None = no cap)."""
    import xgboost as xgb
    val = np.random.default_rng(seed + 1).random(len(pool)) < 0.10
    Fm = F.featurize_frame(pool)
    Z = _to_internal(pool[list(TARGETS)].to_numpy(float), Fm)
    ev = expected_coverage(pool[val])
    p = SprayModel(learning_rate=lr, backend="xgboost", device=device, seed=seed).params
    n_jobs = os.cpu_count() or 1
    dtr = xgb.QuantileDMatrix(Fm[~val], label=Z[~val, 0])
    dva = xgb.QuantileDMatrix(Fm[val], label=ev, ref=dtr)
    n_rows = int((~val).sum())
    specs = [(t, j, "reg:squarederror", "mae", {}) for j, t in enumerate(TARGETS)]
    specs += [(q, 0, "reg:quantileerror", "quantile", {"quantile_alpha": p[k]})
              for q, k in zip(QUANTILES, ("q_lo", "q_hi"))]
    out = {}
    for name, j, obj, metric, extra in specs:
        dtr.set_label(Z[~val, j])
        dva.set_label(ev if name == "coverage_pct" else Z[val, j])
        rows, curves = [], {}
        for depth in depths:
            t = time.perf_counter()
            hist: dict = {}
            b = xgb.train(xgb_train_params(dict(p, max_depth=depth), n_jobs, n_rows, name,
                                           objective=obj, eval_metric=metric, **extra),
                          dtr, max_rounds, evals=[(dva, "val")], early_stopping_rounds=es,
                          evals_result=hist, verbose_eval=False)
            curves[depth] = np.asarray(hist["val"][metric], float)
            rows.append({"depth": int(depth), "rounds": int(b.best_iteration) + 1,
                         "val": float(b.best_score), "fit_s": time.perf_counter() - t})
            log(f"tune[{label}] {name:18s} depth {depth:2d}: rounds {rows[-1]['rounds']:5d} "
                f"val {metric} {rows[-1]['val']:.5f} ({rows[-1]['fit_s']:.0f} s)")
        # Cheapest point on any validation curve within tol of the best: prediction cost is
        # about rounds x depth, and the curves are flat near their minimum.
        def ok_len(depth):    # curve points affordable under the cap
            return len(curves[depth]) if cost_cap is None else min(len(curves[depth]), cost_cap // depth)
        best = min((float(curves[d][:ok_len(d)].min()) for d in curves if ok_len(d) > 0))
        pick = None
        for depth, c in curves.items():
            c = c[:ok_len(depth)]
            hit = c <= best * (1 + tol)
            if hit.any():
                i = int(np.argmax(hit))
                if pick is None or (i + 1) * depth < pick["rounds"] * pick["depth"]:
                    pick = {"depth": int(depth), "rounds": i + 1, "val": float(c[i])}
        out[name] = {"tried": rows, "picked": pick, "metric": metric, "best_val": best,
                     "best_val_uncapped": min(r["val"] for r in rows), "cost_cap": cost_cap,
                     "val_target": "expected coverage (noise-free)" if name == "coverage_pct" else "sampled"}
    return out


def config_from(grid: dict, lr: float, device: str, seed: int) -> dict:
    return dict(backend="xgboost", device=device, learning_rate=lr, seed=seed,
                max_depth={k: v["picked"]["depth"] for k, v in grid.items()},
                n_estimators={t: grid[t]["picked"]["rounds"] for t in TARGETS},
                quantile_n_estimators={q: grid[q]["picked"]["rounds"] for q in QUANTILES})


# ------------------------------------------------------------------ evaluation

def fit(cfg: dict | None, df: pd.DataFrame, label: str) -> SprayModel:
    t = time.perf_counter()
    m = SprayModel(**(cfg or {})).fit_frame(df)
    log(f"fit {label}: {len(df)} rows, {time.perf_counter() - t:.0f} s")
    return m


def score(model: SprayModel, Fte: np.ndarray, Yte: np.ndarray, exp_te: np.ndarray):
    a = model.predict_features(Fte)
    P = np.column_stack([a[k] for k in OUT_KEYS])
    res = {**metrics(P, Yte, exp_te), "interval": interval_stats(a, Yte[:, 0]),
           "fit_s": model.fit_seconds}
    return res, P


def abs_errors(P: np.ndarray, Y: np.ndarray, exp_cov: np.ndarray) -> dict[str, np.ndarray]:
    e = {k: np.abs(P[:, j] - Y[:, j]) for j, k in enumerate(OUT_KEYS)}
    e["coverage_vs_expected"] = np.abs(P[:, 0] - exp_cov)
    return e


def boot_ci(d: np.ndarray, B: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
    return {"mean": float(d.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5))}


def predict_timing(model: SprayModel, reps: int = 100) -> dict:
    cands = catalog.candidates()
    xs = [BASE_X] * len(cands)
    model.predict_batch(xs, cands)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        model.predict_batch(xs, cands)
        ts.append(time.perf_counter() - t)
    return {"n_candidates": len(cands), "median_ms": float(np.median(ts) * 1e3),
            "p95_ms": float(np.percentile(ts, 95) * 1e3)}


def gpu_vs_cpu(cfg: dict, df: pd.DataFrame) -> dict:
    """Fit time of the tuned coverage booster alone on the GPU and on the CPU (same rows)."""
    import xgboost as xgb
    Fm = F.featurize_frame(df)
    y = df["coverage_pct"].to_numpy(float)
    out = {}
    for dev in ("cuda", "cpu"):
        p = SprayModel(**{**cfg, "device": dev}).params
        d = xgb.QuantileDMatrix(Fm, label=y)
        t = time.perf_counter()
        xgb.train(xgb_train_params(p, os.cpu_count() or 1, len(df), "coverage_pct",
                                   objective="reg:squarederror"),
                  d, cfg["n_estimators"]["coverage_pct"])
        out[f"{dev}_s"] = time.perf_counter() - t
        log(f"coverage booster on {dev}: {out[f'{dev}_s']:.1f} s")
    out["rows"], out["rounds"] = len(df), cfg["n_estimators"]["coverage_pct"]
    out["depth"] = cfg["max_depth"]["coverage_pct"]
    return out


# ------------------------------------------------------------------ report

ARM_NAMES = {"A_v1_lgbm_small": "A: v1 (LightGBM, v1 config, v1 rows)",
             "B_xgb_tuned_small": "B: XGBoost tuned, v1 rows",
             "C_lgbm_v1cfg_big": "C: LightGBM v1 config, big pool",
             "D_xgb_tuned_big": "D: XGBoost tuned, big pool (v2 recipe)"}


def write_md(r: dict, path: Path) -> None:
    rs = r["random_split"]
    L = ["# SprayModel v2 (XGBoost on the GPU, more data)", "",
         f"Generated by `experiments/train_spray_v2.py`. Test rows are the v1 random-split test rows "
         f"({rs['n_test']} rows of `data/datasets/v1.parquet`, `default_rng(0)` < 0.2), identical for "
         f"every arm. Extra rows: {r['data']['extra_rows']} from the same generator, seed "
         f"{r['data']['extra_seed']}. Coverage noise floor (no model can beat it against sampled "
         f"outcomes): {r['coverage_noise_floor_test']:.3f} pp. GPU: {r['env']['gpu']}; xgboost "
         f"{r['env']['xgboost']}.", "",
         "## Arms on the v1 test rows", "",
         "| arm | train rows | coverage MAE (pp) | vs expected (pp) | drift MAE | displacement MAE (mm) "
         "| deposition MAE (uL/cm2) | 80% band coverage | band width (pp) | fit (s) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, name in ARM_NAMES.items():
        a = rs["arms"].get(k)
        if a is None:
            continue
        L.append(f"| {name} | {a['n_train']} | {a['coverage_pct']['mae']:.3f} | "
                 f"{a['coverage_pct']['mae_vs_expected']:.3f} | {a['drift_frac']['mae']:.5f} | "
                 f"{a['displacement_mm']['mae']:.3f} | {a['deposition_ul_cm2']['mae']:.5f} | "
                 f"{a['interval']['coverage_of_80pct_band']:.3f} | {a['interval']['mean_width_pp']:.2f} | "
                 f"{a['fit_s']:.0f} |")
    L += ["", "## D - A, paired bootstrap 95% CI over test rows (negative = v2 better)", "",
          "| error | v1 (A) | v2 recipe (D) | D - A [95% CI] |", "|---|---|---|---|"]
    for k, c in rs["paired_D_minus_A"].items():
        L.append(f"| {k} | {c['a']:.5g} | {c['d']:.5g} | {c['mean']:+.5g} [{c['lo']:+.5g}, {c['hi']:+.5g}] |")
    L += ["", "## Held-out regimes (trained without the regime)", "",
          "| regime | test rows | v1 coverage MAE | v2 coverage MAE | v1 vs expected | v2 vs expected "
          "| v1 drift MAE | v2 drift MAE | v1 displacement | v2 displacement |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n, g in r["regimes"].items():
        a, d = g["A"], g["D"]
        L.append(f"| {n} | {g['n_test']} | {a['coverage_pct']['mae']:.3f} | {d['coverage_pct']['mae']:.3f} | "
                 f"{a['coverage_pct']['mae_vs_expected']:.3f} | {d['coverage_pct']['mae_vs_expected']:.3f} | "
                 f"{a['drift_frac']['mae']:.5f} | {d['drift_frac']['mae']:.5f} | "
                 f"{a['displacement_mm']['mae']:.3f} | {d['displacement_mm']['mae']:.3f} |")
    f = r["final"]
    L += ["", "## Final v2 model", "",
          f"- File `{f['path']}` ({f['size_mb']:.1f} MB, saved without training data, so the legacy "
          f"`slow_mode=\"refit\"` cannot refit it; the default residual slow loop works). Trained on "
          f"{f['n_train']} rows in {f['fit_s']:.0f} s.",
          f"- Config: depth {f['config']['max_depth']}, rounds {f['config']['n_estimators']}, "
          f"quantile rounds {f['config']['quantile_n_estimators']}, learning rate {f['config']['learning_rate']}.",
          f"- `predict_batch` over {f['timing']['n_candidates']} candidates (one CPU thread): median "
          f"{f['timing']['median_ms']:.1f} ms, p95 {f['timing']['p95_ms']:.1f} ms (v1: "
          f"{r['v1_timing']['median_ms']:.1f} / {r['v1_timing']['p95_ms']:.1f} ms; target 20 ms).",
          f"- Sensitivity checks: {sum(v['passed'] for v in f['sensitivity'].values())}/"
          f"{len(f['sensitivity'])} pass ({', '.join(k for k, v in f['sensitivity'].items() if not v['passed']) or 'none failed'}).",
          ]
    if r.get("gpu_vs_cpu"):
        g = r["gpu_vs_cpu"]
        L.append(f"- GPU vs CPU, tuned coverage booster alone ({g['rows']} rows, {g['rounds']} rounds, "
                 f"depth {g['depth']}): {g['cuda_s']:.1f} s on the GPU vs {g['cpu_s']:.1f} s on "
                 f"{r['env']['cpu_threads']} CPU threads.")
    L += ["", "## Tuning (validation split of the big pool; picked = cheapest within tolerance)", "",
          "| booster | metric | depth: rounds / val | picked |", "|---|---|---|---|"]
    for name, g in r["tuning"]["big"].items():
        tried = "; ".join(f"{t['depth']}: {t['rounds']} / {t['val']:.4g}" for t in g["tried"])
        L.append(f"| {name} | {g['metric']} | {tried} | depth {g['picked']['depth']}, "
                 f"{g['picked']['rounds']} rounds, val {g['picked']['val']:.4g} |")
    path.write_text("\n".join(L) + "\n", encoding="utf8")


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-n", type=int, default=1_000_000)
    ap.add_argument("--extra-seed", type=int, default=20261019)
    ap.add_argument("--depths", type=int, nargs="+", default=[6, 8, 10, 12])
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--max-rounds", type=int, default=5000)
    ap.add_argument("--early-stop", type=int, default=100)
    ap.add_argument("--tol", type=float, default=0.02, help="relative tolerance for the cheapest config")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-regimes", action="store_true")
    ap.add_argument("--cost-cap", type=int, default=None,
                    help="max rounds x depth per booster (bounds predict_batch time); default none")
    ap.add_argument("--tag", default="", help="suffix for outputs, e.g. 'fast' -> model_v2_fast.*")
    ap.add_argument("--skip-small", action="store_true", help="skip arm B (tuning on v1 rows only)")
    ap.add_argument("--quick", action="store_true", help="smoke run: 50k extra rows, few rounds, no save")
    a = ap.parse_args()
    if a.quick:
        a.extra_n, a.depths, a.max_rounds, a.early_stop = 50_000, [6, 8], 300, 20
    tag = "_quick" if a.quick else (f"_{a.tag}" if a.tag else "")
    out_model = ROOT / "models" / f"spray_model_v2{tag}.joblib"
    import xgboost as xgb
    t_all = time.perf_counter()

    v1, test, extra = load_data(a.extra_n, a.extra_seed)
    small = v1[~test].reset_index(drop=True)
    big = pd.concat([small, extra], ignore_index=True)
    te = v1[test].reset_index(drop=True)
    exp_v1 = expected_coverage(v1)
    Fte, Yte, exp_te = F.featurize_frame(te), te[list(OUT_KEYS)].to_numpy(float), exp_v1[test]
    log(f"rows: v1 train {len(small)}, extra {len(extra)}, big pool {len(big)}, test {len(te)}")

    grid_big = tune(big, a.depths, a.lr, a.max_rounds, a.early_stop, a.tol, a.device, a.seed, "big",
                    a.cost_cap)
    cfg_big = config_from(grid_big, a.lr, a.device, a.seed)
    arms_cfg = {"A_v1_lgbm_small": (None, small)}
    grid_small = None
    if not a.skip_small:
        grid_small = tune(small, a.depths, a.lr, a.max_rounds, a.early_stop, a.tol, a.device, a.seed,
                          "small", a.cost_cap)
        arms_cfg["B_xgb_tuned_small"] = (config_from(grid_small, a.lr, a.device, a.seed), small)
    arms_cfg.update({"C_lgbm_v1cfg_big": (None, big), "D_xgb_tuned_big": (cfg_big, big)})
    arms, preds = {}, {}
    for k, (cfg, df) in arms_cfg.items():
        m = fit(cfg, df, k)
        arms[k], preds[k] = score(m, Fte, Yte, exp_te)
        arms[k]["n_train"] = len(df)
        log(f"{k}: coverage MAE {arms[k]['coverage_pct']['mae']:.4f}, vs expected "
            f"{arms[k]['coverage_pct']['mae_vs_expected']:.4f}, drift {arms[k]['drift_frac']['mae']:.5f}, "
            f"displacement {arms[k]['displacement_mm']['mae']:.3f}")
    ea = abs_errors(preds["A_v1_lgbm_small"], Yte, exp_te)
    ed = abs_errors(preds["D_xgb_tuned_big"], Yte, exp_te)
    paired = {k: {"a": float(ea[k].mean()), "d": float(ed[k].mean()), **boot_ci(ed[k] - ea[k])} for k in ea}

    regimes = {}
    if not a.no_regimes:
        for n, f in REGIMES.items():
            r1, rx = f(v1).to_numpy(), f(extra).to_numpy()
            Fr, Yr = F.featurize_frame(v1[r1]), v1[r1][list(OUT_KEYS)].to_numpy(float)
            mA = fit(None, v1[~r1], f"regime {n} A")
            mD = fit(cfg_big, pd.concat([v1[~r1], extra[~rx]], ignore_index=True), f"regime {n} D")
            regimes[n] = {"n_test": int(r1.sum()), "A": score(mA, Fr, Yr, exp_v1[r1])[0],
                          "D": score(mD, Fr, Yr, exp_v1[r1])[0]}

    gvc = gpu_vs_cpu(cfg_big, big) if a.device == "cuda" and not a.quick else None

    final_df = pd.concat([v1, extra], ignore_index=True)
    final = fit(cfg_big, final_df, "final v2")
    sens, _ = sensitivity(final)
    final_info = {"n_train": len(final_df), "fit_s": final.fit_seconds, "config": cfg_big,
                  "timing": predict_timing(final), "sensitivity": sens,
                  "cqr_offset": final.cqr_offset}
    if a.quick:
        final_info.update(path="(quick run: not saved)", size_mb=0.0)
    else:
        p = final.save(out_model, include_data=False)
        final_info.update(path=p.relative_to(ROOT).as_posix(), size_mb=p.stat().st_size / 2**20)

    import torch
    res = {
        "data": {"v1": "data/datasets/v1.parquet", "extra_rows": len(extra), "extra_seed": a.extra_seed,
                 "big_pool_rows": len(big), "final_rows": len(final_df)},
        "args": vars(a),
        "env": {"xgboost": xgb.__version__, "cpu_threads": os.cpu_count(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"},
        "coverage_noise_floor_test": float(np.abs(exp_te - Yte[:, 0]).mean()),
        "tuning": {"big": grid_big, "small": grid_small},
        "random_split": {"n_test": len(te), "arms": arms, "paired_D_minus_A": paired},
        "regimes": regimes, "gpu_vs_cpu": gvc, "final": final_info,
        "v1_timing": predict_timing(SprayModel.load()),
        "total_s": time.perf_counter() - t_all,
    }
    (RESULTS / f"model_v2{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf8")
    write_md(res, RESULTS / f"model_v2{tag}.md")
    log(f"done in {res['total_s'] / 60:.1f} min -> results/model_v2{tag}.md")


if __name__ == "__main__":
    main()
