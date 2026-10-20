"""Physics-first hybrid vs pure ML vs pure physics (model/hybrid.py).

Stages (run in order; each one updates results/hybrid_eval.json, `report` writes the .md):
  train     fit the final hybrids on all of dataset v1 and save models/hybrid_v1.joblib
            (exact physics base) and models/hybrid_misspec_v1.joblib (misspecified base).
  offline   accuracy on the random 80/20 split and the three grouped held-out regimes of
            train_eval_model.py (U > 5 m/s, family 3D, hot-dry): coverage MAE vs sampled and vs
            expected (spread-factor-averaged) coverage, 80 % interval coverage / width, drift,
            deposition and displacement MAE; plus a learning curve (training-set size).
  selector  200 no-mismatch scenarios (eval_selector.sample_conditions(200, 0)); every model
            selects with the same ConstrainedSelector; the choice is scored in the simulator
            (mismatch none) with the SAME rng per scenario for every model, and also by the exact
            success probability over the spread factor (41-point grid). Paired bootstrap vs spray.
  closed    closed loop (noisy verifier): policies open / fast / fast_slow (5 passes, last scored)
            per model, plus the fair oracle once per cell; mismatch med and high, 4 presets,
            20 seeds; evaluation seed s + 4036 for every policy (paired, as closed_loop.py).
  report    results/hybrid_eval.md.

Models: spray (pure ML, SprayModel), physics (nominal physics alone = oracle_nominal),
physics_misspec / physics_misspec_pess (misspecified physics alone), hybrid,
hybrid_misspec / hybrid_misspec_pess (physics base + learned residual + CQR).

CIRCULARITY: the data, the plant and the exact physics base all come from the same simulator.
The `physics` and `hybrid` rows are therefore an upper bound; the *_misspec rows are the
informative ones (a physics model that is wrong by realistic amounts).

Usage:
  uv run python experiments/eval_hybrid.py --stage train
  uv run python experiments/eval_hybrid.py --stage offline
  uv run python experiments/eval_hybrid.py --stage selector
  uv run python experiments/eval_hybrid.py --stage closed --workers 4 [--seeds 2 --pilot]
  uv run python experiments/eval_hybrid.py --stage report
"""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT / "experiments"))

from coverageai import catalog, integration, scenarios                     # noqa: E402
from coverageai.config import spec                                          # noqa: E402
from coverageai.contracts import Conditions, Setting                        # noqa: E402
from coverageai.model import features as F                                  # noqa: E402
from coverageai.model.hybrid import (DEFAULT_PATH, MISSPEC_PATH, MISSPEC_PRESETS,  # noqa: E402
                                     PhysicsBase, PhysicsOnlyModel, PhysicsResidualModel,
                                     load_hybrid_model)
from coverageai.model.predictor import SprayModel                           # noqa: E402

RESULTS = ROOT / "results"
JSON_PATH = RESULTS / "hybrid_eval.json"
MD_PATH = RESULTS / "hybrid_eval.md"
DATA = ROOT / "data" / "datasets" / "v1.parquet"
OUT_KEYS = ("coverage_pct", "deposition_ul_cm2", "drift_frac", "displacement_mm")
REGIMES = {
    "U_gt_5": lambda d: d["wind_speed_ms"] > 5.0,
    "family_3D": lambda d: d["family"] == "3D",
    "hot_dry": lambda d: (d["temp_c"] > 35.0) & (d["rh_pct"] < 30.0),
}
MISSPECS = {"exact": None, "misspec": "optimistic", "misspec_pess": "pessimistic"}
OFFLINE_MODELS = ("spray", "physics", "hybrid", "physics_misspec", "hybrid_misspec",
                  "physics_misspec_pess", "hybrid_misspec_pess")
SEL_MODELS = ("spray", "physics", "hybrid", "physics_misspec", "hybrid_misspec")
CL_MODELS = ("spray", "physics", "hybrid", "physics_misspec", "hybrid_misspec")
CL_MISMATCH = ("med", "high")
CL_PRESETS = ("calm_morning", "gusty_afternoon", "hot_dry", "crosswind_shift")
CL_POLICIES = ("open", "fast", "fast_slow")
LC_SIZES = (250, 1000, 4000, 16000)
SEED = 0


def load_json() -> dict:
    return json.loads(JSON_PATH.read_text(encoding="utf8")) if JSON_PATH.is_file() else {}


def save_json(key: str, value) -> None:
    d = load_json()
    d[key] = value
    d["schema"] = "hybrid_eval/1"
    d["misspec_presets"] = MISSPEC_PRESETS
    JSON_PATH.write_text(json.dumps(d, indent=1, default=float), encoding="utf8")


def load_data():
    df = pd.read_parquet(DATA)
    return df, F.featurize_frame(df), df[list(OUT_KEYS)].to_numpy(float)


def physics_matrices(df) -> dict[str, np.ndarray]:
    return {k: PhysicsBase(ms, catalog.derive).matrix_frame(df) for k, ms in MISSPECS.items()}


# ====================================================================== train

def stage_train(args) -> None:
    df, Fall, Y = load_data()
    out = {}
    for key, path in (("exact", DEFAULT_PATH), ("misspec", MISSPEC_PATH)):
        m = PhysicsResidualModel(misspec=MISSPECS[key], seed=SEED, n_jobs=args.n_jobs)
        t = time.perf_counter()
        m.fit_frame(df)
        fit_s = time.perf_counter() - t
        m.save(path)
        cands = catalog.candidates()
        x = Conditions(3.0, 90.0, 0.0, 25.0, 50.0, 19.0, 0.5)
        m.predict_batch([x] * len(cands), cands)
        ts = []
        for _ in range(30):
            t = time.perf_counter()
            m.predict_batch([x] * len(cands), cands)
            ts.append(time.perf_counter() - t)
        out[key] = {"path": path.relative_to(ROOT).as_posix(), "misspec": m.misspec, "n_train": m.n_train,
                    "fit_s": fit_s, "offset_lo": m.offset_lo, "offset_hi": m.offset_hi,
                    "predict_batch_median_ms": 1e3 * float(np.median(ts)),
                    "n_candidates": len(cands), "size_mb": path.stat().st_size / 1e6}
        print(key, out[key], flush=True)
    save_json("train", out)


# ====================================================================== offline

def expected_coverage(df) -> np.ndarray:
    from train_eval_model import expected_coverage as ec
    return ec(df)


def metrics(a: dict, Y: np.ndarray, exp_cov: np.ndarray) -> dict:
    c, y = a["coverage_pct"], Y[:, 0]
    lo, hi = a["coverage_lo"], a["coverage_hi"]
    return {"cov_mae": float(np.abs(c - y).mean()), "cov_rmse": float(np.sqrt(((c - y) ** 2).mean())),
            "cov_mae_vs_expected": float(np.abs(c - exp_cov).mean()),
            "cov_bias": float((c - y).mean()),
            "interval_coverage": float(((y >= lo) & (y <= hi)).mean()),
            "frac_below_lo": float((y < lo).mean()), "frac_above_hi": float((y > hi).mean()),
            "interval_width": float((hi - lo).mean()),
            "dep_mae": float(np.abs(a["deposition_ul_cm2"] - Y[:, 1]).mean()),
            "drift_mae": float(np.abs(a["drift_frac"] - Y[:, 2]).mean()),
            "disp_mae": float(np.abs(a["displacement_mm"] - Y[:, 3]).mean())}


def _phys_arrays(P: np.ndarray) -> dict:
    return {"coverage_pct": P[:, 0], "coverage_lo": P[:, 1], "coverage_hi": P[:, 2],
            "deposition_ul_cm2": P[:, 3], "drift_frac": P[:, 4], "displacement_mm": P[:, 5]}


def eval_split(df, Fall, Y, Pm, exp_cov, test, n_jobs, sub_train=None) -> dict:
    tr = ~test if sub_train is None else sub_train
    te = test
    res = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "models": {}}
    t = time.perf_counter()
    sm = SprayModel(seed=SEED, n_jobs=n_jobs).fit_frame(df[tr])
    a = sm.predict_features(Fall[te])
    res["models"]["spray"] = {**metrics(a, Y[te], exp_cov[te]), "fit_s": time.perf_counter() - t}
    for key, P in Pm.items():
        suf = "" if key == "exact" else "_" + key
        res["models"]["physics" + suf] = metrics(_phys_arrays(P[te]), Y[te], exp_cov[te])
        t = time.perf_counter()
        h = PhysicsResidualModel(misspec=MISSPECS[key], seed=SEED, n_jobs=n_jobs).fit_arrays(
            Fall[tr], P[tr], Y[tr])
        res["models"]["hybrid" + suf] = {**metrics(h.predict_from(Fall[te], P[te]), Y[te], exp_cov[te]),
                                         "fit_s": time.perf_counter() - t}
    mm = res["models"]
    for suf in [x for x in ("_misspec", "_misspec_pess") if "physics" + x in mm]:
        # share of the misspecification's excess error (vs exact physics) the residual removes
        ex = mm["physics" + suf]["cov_mae_vs_expected"] - mm["physics"]["cov_mae_vs_expected"]
        left = mm["hybrid" + suf]["cov_mae_vs_expected"] - mm["hybrid"]["cov_mae_vs_expected"]
        exd = mm["physics" + suf]["drift_mae"] - mm["physics"]["drift_mae"]
        leftd = mm["hybrid" + suf]["drift_mae"] - mm["hybrid"]["drift_mae"]
        res["recovery" + suf] = {"cov_excess_mae_removed": float(1 - left / ex) if ex > 1e-9 else None,
                                 "drift_excess_mae_removed": float(1 - leftd / exd) if exd > 1e-12 else None}
    return res


def stage_offline(args) -> None:
    df, Fall, Y = load_data()
    t0 = time.time()
    exp_cov = expected_coverage(df)
    Pm = physics_matrices(df)
    print(f"[{time.time() - t0:4.0f}s] physics + expected coverage done", flush=True)
    rng = np.random.default_rng(SEED)
    splits = {"random": rng.random(len(df)) < 0.2, **{n: f(df).to_numpy() for n, f in REGIMES.items()}}
    prev = load_json().get("offline", {})
    if args.resume and set(prev.get("splits", {})) == set(splits):
        out = {k: prev[k] for k in ("noise_floor_mae", "splits")}
        print("reusing saved split results", flush=True)
    else:
        out = {"noise_floor_mae": float(np.abs(exp_cov - Y[:, 0]).mean()), "splits": {}}
        for name, te in splits.items():
            out["splits"][name] = eval_split(df, Fall, Y, Pm, exp_cov, te, args.n_jobs)
            mm = out["splits"][name]["models"]
            print(f"[{time.time() - t0:4.0f}s] {name}: " + "  ".join(
                f"{k} {v['cov_mae']:.2f}/{v['cov_mae_vs_expected']:.2f}" for k, v in mm.items()), flush=True)
        save_json("offline", out)
    # learning curve on the random split: training subsets of the 80 % training part
    te = splits["random"]
    idx = np.flatnonzero(~te)
    lc = {}
    perm = np.random.default_rng(SEED + 1).permutation(idx)
    for n in LC_SIZES:
        sub = np.zeros(len(df), bool)
        sub[perm[:n]] = True
        r = eval_split(df, Fall, Y, {"exact": Pm["exact"], "misspec": Pm["misspec"]}, exp_cov, te,
                       args.n_jobs, sub_train=sub)
        lc[str(n)] = {k: {m: v[m] for m in ("cov_mae", "cov_mae_vs_expected", "interval_coverage",
                                            "frac_below_lo", "interval_width", "drift_mae")}
                      for k, v in r["models"].items()}
        print(f"[{time.time() - t0:4.0f}s] learning curve n={n}: " + "  ".join(
            f"{k} {v['cov_mae_vs_expected']:.2f}" for k, v in lc[str(n)].items()), flush=True)
    lc[str(int((~te).sum()))] = {k: {m: v[m] for m in ("cov_mae", "cov_mae_vs_expected",
                                                       "interval_coverage", "frac_below_lo",
                                                       "interval_width", "drift_mae")}
                                 for k, v in out["splits"]["random"]["models"].items()
                                 if k in ("spray", "physics", "hybrid", "physics_misspec", "hybrid_misspec")}
    out["learning_curve"] = lc
    out["wall_s"] = time.time() - t0
    save_json("offline", out)


# ====================================================================== models for selection

def make_model(key: str):
    if key == "spray":
        return integration.load_spray_model(n_jobs=1)
    if key == "physics":
        return PhysicsOnlyModel(None, catalog.derive)
    if key == "physics_misspec":
        return PhysicsOnlyModel(MISSPECS["misspec"], catalog.derive)
    if key == "hybrid":
        return load_hybrid_model(misspec=False, derive_fn=catalog.derive, n_jobs=1)
    if key == "hybrid_misspec":
        return load_hybrid_model(misspec=True, derive_fn=catalog.derive, n_jobs=1)
    raise ValueError(key)


# ====================================================================== selector

def _paired_boot(a: np.ndarray, b: np.ndarray, B: int = 4000, seed: int = 0, rel: bool = False):
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, (B, n))
    f = (lambda x, y: 100 * (x.mean() / y.mean() - 1)) if rel else (lambda x, y: x.mean() - y.mean())
    est = f(a, b)
    vals = np.array([f(a[i], b[i]) for i in idx])
    return float(est), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def stage_selector(args) -> None:
    from eval_selector import sample_conditions
    from coverageai.selector import ConstrainedSelector
    from coverageai.simulator import SpraySimulator
    c_req = float(spec()["c_req_pct"])
    scen = sample_conditions(args.n_scen, 0)
    sim = SpraySimulator(derive_fn=catalog.derive)
    lo, hi = spec()["physics"]["spread_factor"]
    grid = lo + (np.arange(41) + 0.5) / 41 * (hi - lo)
    sf_sims = [SpraySimulator(derive_fn=catalog.derive, physics_overrides={"spread_factor": [s, s]})
               for s in grid]
    sel = ConstrainedSelector(candidates_fn=catalog.candidates, derive_fn=catalog.derive)
    per: dict[str, dict] = {}
    out = {"n": len(scen), "c_req": c_req, "models": {}}
    for key in SEL_MODELS:
        mdl = make_model(key)
        us, ts, infeas, pred_lo = [], [], 0, []
        for x in scen:
            t = time.perf_counter()
            u, info = sel.select(x, mdl, 0.0, c_req)
            ts.append(time.perf_counter() - t)
            us.append(u)
            infeas += int(info["infeasible"])
        sampled = np.array([sim.simulate(x, u, np.random.default_rng([4242, i]), "none").coverage_pct
                            for i, (x, u) in enumerate(zip(scen, us))])
        ders = [catalog.derive(x, u) for x, u in zip(scen, us)]
        covs = np.array([s.simulate_arrays(scen, us, np.random.default_rng(0), "none", derived=ders)
                         ["coverage_pct"] for s in sf_sims])           # (41, n)
        p_ok = (covs >= c_req).mean(axis=0)
        drift = sf_sims[0].simulate_arrays(scen, us, np.random.default_rng(0), "none",
                                           derived=ders)["drift_frac"]
        gpa = np.array([d.app_rate_gpa for d in ders])
        per[key] = {"ok": (sampled >= c_req).astype(float), "p_ok": p_ok, "gpa": gpa, "drift": drift}
        out["models"][key] = {
            "success_sampled": float((sampled >= c_req).mean()),
            "success_expected": float(p_ok.mean()),
            "mean_true_cov": float(covs.mean()), "mean_drift": float(drift.mean()),
            "mean_gpa": float(gpa.mean()), "infeasible_frac": infeas / len(scen),
            "select_ms_median": 1e3 * float(np.median(ts)),
            "settings_head": [list(map(str, (u.family, u.size, u.pressure_psi))) for u in us[:5]]}
        print(key, {k: round(v, 4) if isinstance(v, float) else v
                    for k, v in out["models"][key].items() if k != "settings_head"}, flush=True)
    ref = per["spray"]
    for key in SEL_MODELS:
        if key == "spray":
            continue
        a = per[key]
        out["models"][key]["vs_spray"] = {
            "d_success_expected": _paired_boot(a["p_ok"], ref["p_ok"]),
            "d_success_sampled": _paired_boot(a["ok"], ref["ok"]),
            "d_volume_pct": _paired_boot(a["gpa"], ref["gpa"], rel=True),
            "d_drift": _paired_boot(a["drift"], ref["drift"])}
    save_json("selector", out)


# ====================================================================== closed loop

_W: dict = {}


def _init_cl(models: tuple[str, ...]) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mods = {}
    for key in models:
        base = "oracle_nominal" if key.startswith("physics") else ("spray" if key == "spray" else key)
        mods[key] = integration.build_modules(base, "noisy", model_obj=make_model(key))
    _W["mods"] = mods


def cl_cell(model: str, preset: str, mismatch: str, s: int, c_req: float) -> list[dict]:
    from closed_loop import EVAL_OFFSET, SLOW_PASSES, pass_metrics
    from coverageai.loop import run_pass, run_passes_slow
    mods = _W["mods"][model]
    conds = scenarios.generate(scenarios.get(preset), s)
    E = s + EVAL_OFFSET
    base = {"model": model, "preset": preset, "mismatch": mismatch, "seed": s}
    rows = []
    t = time.perf_counter()
    for pol in ("open", "fast"):
        logs = run_pass(conds, pol, mods, E, mismatch, c_req, f"h_{model}_{pol}_{preset}_{mismatch}_{s}")
        rows.append({**base, "policy": pol, **pass_metrics(logs, c_req)})
    logs = run_passes_slow(conds, mods, s, mismatch, c_req, n_passes=SLOW_PASSES)
    n = len(conds)
    rows.append({**base, "policy": "fast_slow", **pass_metrics(logs[-n:], c_req)})
    rows[-1]["cell_s"] = time.perf_counter() - t
    return rows


def oracle_cell(preset: str, mismatch: str, s: int, c_req: float) -> list[dict]:
    from closed_loop import EVAL_OFFSET, pass_metrics
    from coverageai.loop import run_pass
    mods = next(iter(_W["mods"].values()))
    conds = scenarios.generate(scenarios.get(preset), s)
    logs = run_pass(conds, "oracle", mods, s + EVAL_OFFSET, mismatch, c_req,
                    f"h_oracle_{preset}_{mismatch}_{s}")
    return [{"model": "oracle", "preset": preset, "mismatch": mismatch, "seed": s,
             "policy": "oracle", **pass_metrics(logs, c_req)}]


def stage_closed(args) -> None:
    c_req = float(spec()["c_req_pct"])
    seeds = list(range(args.seeds))
    models = tuple(args.models.split(",")) if args.models else CL_MODELS
    t0 = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(args.workers, initializer=_init_cl, initargs=(models,)) as ex:
        futs = [ex.submit(oracle_cell, p, mm, s, c_req) for p in CL_PRESETS for mm in CL_MISMATCH
                for s in seeds]
        futs += [ex.submit(cl_cell, m, p, mm, s, c_req) for s in seeds for m in models
                 for p in CL_PRESETS for mm in CL_MISMATCH]
        for i, f in enumerate(as_completed(futs), 1):
            rows += f.result()
            if i % 20 == 0 or i == len(futs):
                print(f"[{time.time() - t0:5.0f}s] {i}/{len(futs)} cells", flush=True)
    key = "closed_loop_pilot" if args.pilot else "closed_loop"
    save_json(key, {"seeds": seeds, "presets": list(CL_PRESETS), "mismatch": list(CL_MISMATCH),
                    "models": list(models), "verifier": "noisy", "wall_s": time.time() - t0,
                    "workers": args.workers, "runs": rows,
                    "summary": closed_summary(pd.DataFrame(rows))})
    print(f"done in {time.time() - t0:.0f}s")


def closed_summary(df: pd.DataFrame) -> dict:
    """Per model x policy x mismatch group: seed-level means (presets pooled) with t CIs, gap
    recovery and volume overhead vs the fair oracle, paired differences vs spray (bootstrap)."""
    from closed_loop import _boot, _gap, _overhead, mean_ci
    metrics_ = ("frac_met", "mean_volume_gpa", "mean_drift_frac", "mean_shortfall_pp",
                "frac_below_lo", "model_err_mean")
    out: dict = {}
    groups = {"med": ["med"], "high": ["high"], "med+high": ["med", "high"]}
    for gname, lv in groups.items():
        sub = df[df["mismatch"].isin(lv)]
        sl = sub.groupby(["model", "policy", "seed"], as_index=False)[list(metrics_)].mean()
        orc = sl[sl["policy"] == "oracle"].set_index("seed").sort_index()
        g: dict = {"oracle": {m: mean_ci(orc[m]) for m in metrics_}}
        for model in sorted(set(sl["model"]) - {"oracle"}):
            gm = {}
            ms = sl[sl["model"] == model]
            op = ms[ms["policy"] == "open"].set_index("seed").sort_index()
            for pol in CL_POLICIES:
                c = ms[ms["policy"] == pol].set_index("seed").sort_index()
                if c.empty:
                    continue
                r = {m: mean_ci(c[m]) for m in metrics_}
                if pol != "open":
                    r["gap_recovery"] = _boot(_gap, [op["frac_met"].to_numpy(), c["frac_met"].to_numpy(),
                                                     orc["frac_met"].to_numpy()])
                r["vol_overhead_pct"] = _boot(_overhead, [c["mean_volume_gpa"].to_numpy(),
                                                          orc["mean_volume_gpa"].to_numpy()])
                if model != "spray":
                    sp = sl[(sl["model"] == "spray") & (sl["policy"] == pol)].set_index("seed").sort_index()
                    if len(sp) == len(c):
                        r["d_met_vs_spray"] = _boot(lambda a, b: a.mean() - b.mean(),
                                                    [c["frac_met"].to_numpy(), sp["frac_met"].to_numpy()])
                        r["d_volume_pct_vs_spray"] = _boot(
                            lambda a, b: 100 * (a.mean() / b.mean() - 1),
                            [c["mean_volume_gpa"].to_numpy(), sp["mean_volume_gpa"].to_numpy()])
                gm[pol] = r
            g[model] = gm
        out[gname] = g
    return out


# ====================================================================== report

def _f(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def _ci(t, nd=1, pct=False, scale=1.0):
    if t is None:
        return "-"
    e, lo, hi = (x * scale if x is not None else None for x in t)
    s = "%" if pct else ""
    if lo is None or not np.isfinite(lo):
        return f"{e:+.{nd}f}{s}"
    return f"{e:+.{nd}f}{s} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def _mci(t, nd=3):
    return f"{t[0]:.{nd}f} ± {t[1]:.{nd}f}"


LABEL = {"spray": "SprayModel (pure ML)", "physics": "physics only (exact base)",
         "hybrid": "hybrid (exact base)", "physics_misspec": "physics only, misspecified (optimistic)",
         "hybrid_misspec": "hybrid, misspecified base (optimistic)",
         "physics_misspec_pess": "physics only, misspecified (pessimistic)",
         "hybrid_misspec_pess": "hybrid, misspecified base (pessimistic)"}


def stage_report(args) -> None:
    d = load_json()
    L = ["# Physics-first hybrid vs pure ML vs pure physics", "",
         "Generated by `experiments/eval_hybrid.py --stage report` from `results/hybrid_eval.json`. "
         "Model: `src/coverageai/model/hybrid.py` (`PhysicsResidualModel`).", ""]
    verdict = ROOT / "results" / "hybrid_eval_notes.md"
    if verdict.is_file():
        L += [verdict.read_text(encoding="utf8").strip(), ""]
    L += ["## Design", "",
          "* **Physics base**: the simulator's equations with no hidden mismatch and the mean WSP spread "
          "factor (identical to `oracle_nominal`); its own interval is the coverage at the 10 % / 90 % "
          "spread-factor quantiles.",
          "* **Residual**: LightGBM, 300 trees, 15 leaves, min 200 rows per leaf, L2 = 10, learning rate "
          "0.05 (strongly regularized), on the SprayModel features plus the physics outputs (coverage, its "
          "spread-factor band, retention, drift, displacement). Targets: coverage residual (pp), retention "
          "residual, drift residual, displacement residual.",
          "* **Interval**: quantile LightGBM (q = 0.1 / 0.9) on the coverage residual, conformalized on a 15 % "
          "calibration split with a separate one-sided offset per bound (each bound targets 10 % miss).",
          "* **Misspecified base** (`MISSPEC_PRESETS`), multiplicative errors on the true parameters: "
          + "; ".join(f"`{k}`: " + ", ".join(f"{a} x{b:g}" for a, b in v.items() if b != 1.0)
                      for k, v in MISSPEC_PRESETS.items()) + ".",
          "* **Slow loop**: `fit(records)` appends records and refits the residual (legacy refit mode); the "
          "closed-loop experiments wrap the model in `feedback.ResidualCorrectedModel`, like every other model.",
          ""]
    if "train" in d:
        t = d["train"]
        L += ["## Artifacts", "", "| artifact | rows | fit (s) | predict 547 candidates (ms) | size (MB) | "
              "conformal offsets lo / hi (pp) |", "|---|---|---|---|---|---|"]
        for k, v in t.items():
            L.append(f"| `{v['path']}` | {v['n_train']} | {v['fit_s']:.1f} | {v['predict_batch_median_ms']:.0f} | "
                     f"{v['size_mb']:.1f} | {v['offset_lo']:+.2f} / {v['offset_hi']:+.2f} |")
        L.append("")
    if "offline" in d:
        o = d["offline"]
        L += ["## Offline accuracy (dataset v1, no mismatch)", "",
              f"Noise floor: MAE of the expected (spread-factor-averaged) coverage against the sampled "
              f"coverage = {o['noise_floor_mae']:.2f} pp; no model can beat it against sampled outcomes. "
              "`MAE vs exp.` compares with the noise-free expectation. Interval = nominal 80 % band "
              "(10 % below, 10 % above).", ""]
        for name, sp in o["splits"].items():
            L += [f"### Split `{name}` (train {sp['n_train']}, test {sp['n_test']})", "",
                  "| model | cov MAE (pp) | cov MAE vs exp. | cov bias | in band | below lo | above hi | "
                  "width (pp) | drift MAE | dep MAE (uL/cm2) | disp MAE (mm) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for k in OFFLINE_MODELS:
                v = sp["models"][k]
                L.append(f"| {LABEL[k]} | {v['cov_mae']:.2f} | {v['cov_mae_vs_expected']:.2f} | "
                         f"{v['cov_bias']:+.2f} | {v['interval_coverage']:.3f} | {v['frac_below_lo']:.3f} | "
                         f"{v['frac_above_hi']:.3f} | {v['interval_width']:.1f} | {v['drift_mae']:.4f} | "
                         f"{v['dep_mae']:.4f} | {v['disp_mae']:.1f} |")
            rec = [f"{suf[1:]}: coverage {_f(sp['recovery' + suf]['cov_excess_mae_removed'] * 100, 0)}%, "
                   f"drift {_f(sp['recovery' + suf]['drift_excess_mae_removed'] * 100, 0)}%"
                   for suf in ("_misspec", "_misspec_pess") if sp["recovery" + suf]["cov_excess_mae_removed"] is not None]
            L += ["", "Share of the misspecification's excess MAE (vs expected) removed by the residual: "
                  + "; ".join(rec) + ".", ""]
        lc = o.get("learning_curve")
        if lc:
            L += ["### Learning curve (random split test set; training rows subsampled)", "",
                  "Coverage MAE vs expected (pp) / share below the lower bound.", "",
                  "| training rows | " + " | ".join(LABEL[k] for k in ("spray", "physics", "hybrid",
                                                                         "physics_misspec", "hybrid_misspec")) + " |",
                  "|---|---|---|---|---|---|"]
            for n, v in sorted(lc.items(), key=lambda kv: int(kv[0])):
                L.append(f"| {n} | " + " | ".join(
                    f"{v[k]['cov_mae_vs_expected']:.2f} / {v[k]['frac_below_lo']:.2f}" if k in v else "-"
                    for k in ("spray", "physics", "hybrid", "physics_misspec", "hybrid_misspec")) + " |")
            L.append("")
    if "selector" in d:
        s = d["selector"]
        L += ["## Selector on no-mismatch scenarios", "",
              f"{s['n']} scenarios (`eval_selector.sample_conditions({s['n']}, 0)`), C_req = {s['c_req']:g} %. "
              "Each model selects with the same `ConstrainedSelector`; the choice is scored in the simulator "
              "(mismatch none) with the same rng per scenario for every model (`sampled`) and by the exact "
              "probability of meeting C_req over the spread factor (`expected`). Differences vs SprayModel: "
              "paired bootstrap 95 % CI over scenarios.", "",
              "| model | success (sampled) | success (expected) | mean drift | mean GPA | select (ms) | "
              "Δ expected success vs spray (pp) | Δ volume vs spray | Δ drift vs spray |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k in SEL_MODELS:
            v = s["models"][k]
            vs = v.get("vs_spray", {})
            L.append(f"| {LABEL[k]} | {v['success_sampled']:.3f} | {v['success_expected']:.3f} | "
                     f"{v['mean_drift']:.4f} | {v['mean_gpa']:.1f} | {v['select_ms_median']:.0f} | "
                     f"{_ci(vs.get('d_success_expected'), 1, scale=100) if vs else '-'} | "
                     f"{_ci(vs.get('d_volume_pct'), 1, pct=True) if vs else '-'} | "
                     f"{_ci(vs.get('d_drift'), 4) if vs else '-'} |")
        L.append("")
    for key, title in (("closed_loop", "Closed loop"), ("closed_loop_pilot", "Closed loop (PILOT)")):
        if key not in d or (key == "closed_loop_pilot" and "closed_loop" in d):
            continue
        c = d[key]
        L += [f"## {title}", "",
              f"Noisy verifier, presets {', '.join(c['presets'])}, mismatch {', '.join(c['mismatch'])}, "
              f"{len(c['seeds'])} seeds, 40 segments per pass, C_req = 30 %. `fast_slow` = 5 passes with the "
              "residual slow loop (`ResidualCorrectedModel`), last pass scored. Every policy of every model is "
              "scored on the same hidden realization (seed s + 4036). Reference: fair oracle (knows the hidden "
              "mismatch, not the card noise). Means ± 95 % t-CI over seeds (presets pooled); gap recovery = "
              "(policy − open) / (oracle − open) with the model's own open loop; brackets = paired bootstrap 95 % CI.", ""]
        for gname in ("med+high", "med", "high"):
            g = c["summary"][gname]
            L += [f"### Mismatch {gname}", "",
                  f"Fair oracle: met {_mci(g['oracle']['frac_met'])}, volume {g['oracle']['mean_volume_gpa'][0]:.1f} GPA, "
                  f"drift {g['oracle']['mean_drift_frac'][0]:.4f}.", "",
                  "| model | policy | met | volume (GPA) | drift | below lower bound | gap recovery | "
                  "volume vs oracle | Δ met vs spray (pp) | Δ volume vs spray |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for model in [m for m in CL_MODELS if m in g]:
                for pol in CL_POLICIES:
                    r = g[model].get(pol)
                    if not r:
                        continue
                    L.append(f"| {LABEL[model]} | {pol} | {_mci(r['frac_met'])} | {r['mean_volume_gpa'][0]:.1f} | "
                             f"{r['mean_drift_frac'][0]:.4f} | {r['frac_below_lo'][0]:.3f} | "
                             f"{_ci(r.get('gap_recovery'), 0, pct=True, scale=100) if 'gap_recovery' in r else '-'} | "
                             f"{_ci(r['vol_overhead_pct'], 1, pct=True)} | "
                             f"{_ci(r.get('d_met_vs_spray'), 1, scale=100) if 'd_met_vs_spray' in r else '-'} | "
                             f"{_ci(r.get('d_volume_pct_vs_spray'), 1, pct=True) if 'd_volume_pct_vs_spray' in r else '-'} |")
            L.append("")
    MD_PATH.write_text("\n".join(L) + "\n", encoding="utf8")
    print(f"wrote {MD_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("train", "offline", "selector", "closed", "report"))
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n-scen", type=int, default=200)
    ap.add_argument("--models", default=None, help="comma list for --stage closed")
    ap.add_argument("--pilot", action="store_true", help="store closed-loop result as a pilot")
    ap.add_argument("--resume", action="store_true", help="offline: reuse saved split results")
    a = ap.parse_args()
    {"train": stage_train, "offline": stage_offline, "selector": stage_selector,
     "closed": stage_closed, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
