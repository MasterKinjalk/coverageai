"""Train the learned card-feedback models (decision D29, ``coverageai.feedback_learned``).

What it does
  1. Round 1 data: simulated passes through ``app_logic.run_pass_wrapper`` with behaviour policy
     open and fast (eq. 3 rule), all 6 presets, mismatch none / low / med / high, nozzle mode
     "auto" (one nozzle per field) and free, readers "perfect" and "noisy", and a target c_req
     drawn per cell from {20, 25, 30, 35, 40}. Rows come from ``feedback_learned.rows_from_logs``.
  2. Round 1 fit: LightGBM (small) and XGBoost on the GPU (a few depth / regularisation settings,
     early stopping), best by validation MAE.
  3. Round 2 data: the same training seeds with behaviour policy fast + the round-1 XGBoost
     feedback model (so the learned policy's own choices are in the training set), and the same
     for the validation seeds. Both models are refit on the union.
  4. Offline tables on held-out validation seeds, the two model files, the cached rows and the
     write-up ``results/feedback_model.md`` / ``.json``.

Seeds (a contract):
  * training scenario seeds 20000 .. 20000 + N - 1,
  * validation 21000 .. 21029: 21000 .. 21019 ("val-A") for early stopping and the choice of the
    configuration, 21020 .. 21029 ("val-B") for the offline tables only,
  * the evaluation seeds 0 .. 19 are NOT used here (see experiments/eval_feedback_model.py).
  The hidden-mismatch seed of a pass is s + 4036, as in experiments/closed_loop.py.

Everything is simulation: the models learn the simulator's hidden mismatch, not a real field.

Usage:
  uv run --no-sync python experiments/train_feedback_model.py --seeds 120 --workers 26
  uv run --no-sync python experiments/train_feedback_model.py --reuse      # refit / report from the cached rows
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coverageai import feedback_learned as fl                     # noqa: E402
from coverageai import scenarios                                  # noqa: E402
from coverageai.config import spec                                # noqa: E402

RESULTS = ROOT / "results"
FIGS = ROOT / "experiments" / "figures"
ROWS_PATH = ROOT / "data" / "datasets" / "feedback_v1.npz"
PRESETS = tuple(scenarios.list_presets())
MISMATCH = ("none", "low", "med", "high")
NOZZLE = ("auto", None)                  # code 0 = one nozzle per field (app default), 1 = free
NOZZLE_NAME = ("auto", "free")
READERS = ("perfect", "noisy")
POLICY_NAME = ("open", "fast_rule", "fast_learned_r1", "fast_learned_r2", "fast_learned_r3")
C_REQS = (20, 25, 30, 35, 40)
EVAL_OFFSET = 4036
TRAIN_SEED0 = 20000
VAL_A = tuple(range(21000, 21020))
VAL_B = tuple(range(21020, 21030))
CLIP = float(spec()["feedback"]["bias_clip_pp"])
NT = 26                                  # CPU threads for bulk prediction / LightGBM training
I_NC = fl.FEATURE_NAMES.index("n_cards")
I_RULE = fl.FEATURE_NAMES.index("rule_bias")
I_EMEAN = fl.FEATURE_NAMES.index("e_mean")

XGB_CONFIGS = [
    dict(name="xgb_d8_eta05", max_depth=8, learning_rate=0.05, extra=dict(min_child_weight=20, reg_lambda=1.0)),
    dict(name="xgb_d10_eta03", max_depth=10, learning_rate=0.03, extra=dict(min_child_weight=50, reg_lambda=5.0)),
    dict(name="xgb_d8_eta03_reg", max_depth=8, learning_rate=0.03,
         extra=dict(min_child_weight=200, reg_lambda=10.0, reg_alpha=1.0, colsample_bytree=0.6)),
    dict(name="xgb_d10_eta05_reg", max_depth=10, learning_rate=0.05,
         extra=dict(min_child_weight=500, reg_lambda=20.0, gamma=1.0, subsample=0.7)),
]
LGB_CONFIGS = [
    dict(name="lgbm_d6_lr05", max_depth=6, learning_rate=0.05, extra=dict()),
    dict(name="lgbm_d6_lr10", max_depth=6, learning_rate=0.10, extra=dict()),
]
XGB_ROUNDS, LGB_ROUNDS, ES_ROUNDS = 4000, 300, 100

# ------------------------------------------------------------------ worker side
_W: dict = {}


def _model_from_raw(raw: bytes) -> fl.FeedbackModel:
    import xgboost as xgb
    bst = xgb.Booster()
    bst.load_model(bytearray(raw))
    bst.set_param({"device": "cpu", "nthread": 1})
    return fl.FeedbackModel("xgboost", bst, {"trees": int(bst.num_boosted_rounds())})


def _init(fb_raw: bytes | None, calibrate: bool = False) -> None:
    """``calibrate`` (v2): the noisy reader is wrapped in the reader calibration (D30); the
    perfect reader is never wrapped. v1 runs with calibrate=False, i.e. exactly as before."""
    from coverageai import app_logic as al
    _W["al"] = al
    _W["mods"] = {r: (al.build_modules_adapter("spray", r, calibrate_reader=True) if calibrate and r == "noisy"
                      else al.build_modules_adapter("spray", r)) for r in READERS}
    _W["fb"] = _model_from_raw(fb_raw) if fb_raw is not None else None


def cell_c_req(seed: int, pi: int, mi: int, ni: int, ri: int) -> float:
    """Target of one cell, drawn from its own stream (no simulator stream is touched)."""
    return float(C_REQS[int(np.random.default_rng([seed, pi, mi, ni, ri, 77]).integers(len(C_REQS)))])


def gen_task(seed: int, pi: int, rnd: int) -> dict:
    """All passes of one (scenario seed, preset): round 1 = open + fast(rule), round 2 =
    fast(round-1 XGBoost). Returns the feature rows and per-row meta."""
    al, fb = _W["al"], _W["fb"]
    conds = al.make_conditions(PRESETS[pi], seed)
    E = seed + EVAL_OFFSET
    out = {k: [] for k in ("X", "y", "actual", "applied", "pred", "meta")}
    for mi, mm in enumerate(MISMATCH):
        for ni, nz in enumerate(NOZZLE):
            for ri, rd in enumerate(READERS):
                c_req = cell_c_req(seed, pi, mi, ni, ri)
                # round r >= 2: fast with the round-(r-1) XGBoost model, policy code r
                pols = ((0, "open", None), (1, "fast", None)) if rnd == 1 else ((rnd, "fast", fb),)
                for code, pol, fbm in pols:
                    res = al.run_pass_wrapper(_W["mods"][rd], conds, pol, E, mm, c_req, fixed_nozzle=nz,
                                              feedback_model=fbm, same_field=True)
                    X, y = fl.rows_from_logs(res.logs)
                    out["X"].append(X.astype(np.float32))
                    out["y"].append(y.astype(np.float32))
                    out["actual"].append(np.array([lg.outcome.coverage_pct - lg.pred.coverage_pct
                                                   for lg in res.logs], np.float32))
                    out["applied"].append(np.array([lg.bias for lg in res.logs], np.float32))
                    out["pred"].append(np.array([lg.pred.coverage_pct for lg in res.logs], np.float32))
                    n = len(res.logs)
                    out["meta"].append(np.column_stack([
                        np.full(n, seed), np.full(n, pi), np.full(n, mi), np.full(n, code), np.full(n, ni),
                        np.full(n, ri), np.arange(n), np.full(n, int(c_req))]).astype(np.int32))
    return {k: np.concatenate(v) for k, v in out.items()}


META_COLS = ("seed", "preset", "mismatch", "policy", "nozzle", "reader", "k", "c_req")


def generate(seeds, rnd: int, workers: int, fb_raw: bytes | None, label: str, calibrate: bool = False) -> dict:
    t0 = time.time()
    tasks = [(s, pi) for s in seeds for pi in range(len(PRESETS))]
    parts = []
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(fb_raw, calibrate)) as ex:
        futs = [ex.submit(gen_task, s, pi, rnd) for s, pi in tasks]
        step = max(1, len(futs) // 20)
        for i, f in enumerate(as_completed(futs), 1):
            parts.append(f.result())
            if i % step == 0 or i == len(futs):
                el = time.time() - t0
                print(f"[{label}] {i}/{len(futs)} tasks, {sum(len(p['y']) for p in parts):,} rows, "
                      f"{el:.0f}s, eta {el / i * (len(futs) - i):.0f}s", flush=True)
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.lexsort([d["meta"][:, c] for c in (6, 3, 5, 4, 2, 1, 0)])   # deterministic row order
    d = {k: v[order] for k, v in d.items()}
    n_pass = len(d["y"]) // 40
    print(f"[{label}] done: {len(d['y']):,} rows from {n_pass:,} passes in {time.time() - t0:.0f}s", flush=True)
    d["_wall_s"] = time.time() - t0
    return d


# ------------------------------------------------------------------ sanity checks

def sanity_checks(fb_model: fl.FeedbackModel | None, calibrate: bool = False) -> dict:
    """The rule_bias feature equals the correction BiasFeedback applied; with a learned model the
    correction applied in the loop equals clip(model(features from rows_from_logs))."""
    from coverageai import app_logic as al
    mods = al.build_modules_adapter("spray", "noisy", calibrate_reader=calibrate)
    conds = al.make_conditions("gusty_afternoon", 20000)
    res = al.run_pass_wrapper(mods, conds, "fast", 20000 + EVAL_OFFSET, "med", 30.0, fixed_nozzle=None,
                              same_field=True)
    X, _ = fl.rows_from_logs(res.logs)
    applied = np.array([lg.bias for lg in res.logs])
    out = {"rule_feature_max_abs_diff": float(np.max(np.abs(X[:, I_RULE] - applied)))}
    if fb_model is not None:
        res = al.run_pass_wrapper(mods, conds, "fast", 20000 + EVAL_OFFSET, "med", 30.0, fixed_nozzle=None,
                                  same_field=True, feedback_model=fb_model)
        X, _ = fl.rows_from_logs(res.logs)
        applied = np.array([lg.bias for lg in res.logs])
        p = np.clip(fb_model.predict(X), -CLIP, CLIP)
        out["learned_runtime_vs_rows_max_abs_diff"] = float(np.max(np.abs(p - applied)))
    print("sanity:", out, flush=True)
    return out


# ------------------------------------------------------------------ fitting

def predict_bulk(m: fl.FeedbackModel, X: np.ndarray) -> np.ndarray:
    if m.backend == "xgboost":
        m.booster.set_param({"nthread": NT})
        try:
            p = m.booster.inplace_predict(np.asarray(X, np.float32))
        finally:
            m.booster.set_param({"nthread": 1})
    else:
        p = m.booster.predict(X, num_iteration=m.meta.get("trees") or None, num_threads=NT)
    return np.clip(np.asarray(p, float), -CLIP, CLIP)


def fit_all(Xtr, ytr, Xva, yva, label: str) -> tuple[dict, dict, list[dict]]:
    """Fit every configuration; keep the best XGBoost and the best LightGBM by val-A MAE."""
    tried, best = [], {}
    for backend, cfgs in (("xgboost", XGB_CONFIGS), ("lightgbm", LGB_CONFIGS)):
        for cfg in cfgs:
            t0 = time.time()
            kw = dict(cfg["extra"])
            if backend == "xgboost":
                kw.update(eval_metric="mae")
                m = fl.FeedbackModel.train(Xtr, ytr, backend="xgboost", device="cuda", n_estimators=XGB_ROUNDS,
                                           max_depth=cfg["max_depth"], learning_rate=cfg["learning_rate"],
                                           X_val=Xva, y_val=yva, early_stopping_rounds=ES_ROUNDS, seed=0, **kw)
            else:
                kw.update(metric="l1", num_threads=NT)
                m = fl.FeedbackModel.train(Xtr, ytr, backend="lightgbm", n_estimators=LGB_ROUNDS,
                                           max_depth=cfg["max_depth"], learning_rate=cfg["learning_rate"],
                                           X_val=Xva, y_val=yva, early_stopping_rounds=ES_ROUNDS, seed=0, **kw)
            p = predict_bulk(m, Xva)
            r = {"round": label, "backend": backend, "name": cfg["name"], "max_depth": cfg["max_depth"],
                 "learning_rate": cfg["learning_rate"], "params": cfg["extra"], "trees": int(m.meta["trees"]),
                 "val_a_mae_card": float(np.mean(np.abs(yva - p))),
                 "val_a_rmse_card": float(np.sqrt(np.mean((yva - p) ** 2))), "fit_s": time.time() - t0}
            tried.append(r)
            print(f"[{label}] {cfg['name']}: trees {r['trees']}, val-A MAE {r['val_a_mae_card']:.4f}, "
                  f"RMSE {r['val_a_rmse_card']:.4f}, {r['fit_s']:.0f}s", flush=True)
            if backend not in best or r["val_a_mae_card"] < best[backend][1]["val_a_mae_card"]:
                best[backend] = (m, r)
    return best["xgboost"], best["lightgbm"], tried


# ------------------------------------------------------------------ offline tables

NC_BINS = ((0, 0, "0"), (1, 1, "1"), (2, 5, "2-5"), (6, 15, "6-15"), (16, 10 ** 9, "16+"))


def _metrics(e: np.ndarray) -> dict:
    return {"n": int(len(e)), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2)))}


def offline_tables(d: dict, corr: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    """MAE / RMSE of each correction against the card error and the actual error, overall and
    broken down by mismatch, number of earlier cards, behaviour policy, reader, nozzle mode."""
    meta = d["meta"][mask]
    y, act = d["y"][mask].astype(float), d["actual"][mask].astype(float)
    nc = d["X"][mask][:, I_NC]
    groups = {"all": {"all": np.ones(len(y), bool)}}
    groups["mismatch"] = {m: meta[:, 2] == i for i, m in enumerate(MISMATCH)}
    groups["n_cards"] = {lab: (nc >= lo) & (nc <= hi) for lo, hi, lab in NC_BINS}
    groups["policy"] = {p: meta[:, 3] == i for i, p in enumerate(POLICY_NAME) if np.any(meta[:, 3] == i)}
    groups["reader"] = {r: meta[:, 5] == i for i, r in enumerate(READERS)}
    groups["nozzle"] = {n: meta[:, 4] == i for i, n in enumerate(NOZZLE_NAME)}
    out = {}
    for gname, gs in groups.items():
        out[gname] = {}
        for lab, gm in gs.items():
            out[gname][lab] = {c: {"card": _metrics(y[gm] - v[mask][gm]), "actual": _metrics(act[gm] - v[mask][gm])}
                               for c, v in corr.items()}
    # the correction applied before any card: a learned prior about the simulator's hidden losses
    z = nc == 0
    prior = {}
    for i, m in enumerate(MISMATCH):
        g = z & (meta[:, 2] == i)
        prior[m] = {"n": int(g.sum()), "mean_actual_error": float(act[g].mean()),
                    "mean_card_error": float(y[g].mean()),
                    **{f"mean_corr_{c}": float(v[mask][g].mean()) for c, v in corr.items()}}
    out["zero_card_correction"] = prior
    return out


def importance_top(m: fl.FeedbackModel, n: int = 12) -> list[tuple[str, float]]:
    imp = m.importance()
    return sorted(imp.items(), key=lambda kv: -kv[1])[:n]


# ------------------------------------------------------------------ write-up

CORR_LABEL = {"none": "no correction", "rule": "rule (eq. 3)", "passmean": "pass mean of card errors",
              "lightgbm": "LightGBM", "xgboost": "XGBoost"}


def _tab(tbl: dict, gname: str, target: str, corr_names) -> list[str]:
    L = [f"| {gname} | n rows | " + " | ".join(f"{CORR_LABEL[c]} MAE / RMSE" for c in corr_names) + " |",
         "|---|---|" + "---|" * len(corr_names)]
    for lab, row in tbl[gname].items():
        L.append(f"| {lab} | {row[corr_names[0]][target]['n']:,} | " +
                 " | ".join(f"{row[c][target]['mae']:.2f} / {row[c][target]['rmse']:.2f}" for c in corr_names) + " |")
    return L


def write_md(res: dict) -> None:
    c = ("none", "rule", "passmean", "lightgbm", "xgboost")
    t2, t1 = res["offline_round2"], res["offline_round1"]
    L = ["# Learned card feedback: training and offline validation (D29)\n",
         "Generated by `experiments/train_feedback_model.py`. Do not edit by hand. Closed-loop results are in "
         "`results/feedback_model_closed_loop.md`.\n",
         "Everything here is simulation. The models learn the hidden mismatch of this simulator (canopy "
         "interception, nozzle wear, AR(1) gusts) and its card noise. They are not evidence about a real "
         "field. A real system would have to be trained on real verified cards.\n",
         "## Data\n"]
    dv = res["data"]
    L.append(f"- Training: scenario seeds {dv['train_seeds'][0]} to {dv['train_seeds'][-1]} "
             f"({len(dv['train_seeds'])} seeds), 6 presets, mismatch none / low / med / high, nozzle mode one-per-field "
             "and free, readers perfect and noisy, target c_req drawn per cell from {20, 25, 30, 35, 40}.")
    L.append(f"- Round 1 (behaviour open and fast with the rule): {dv['rows_round1_train']:,} training rows "
             f"({dv['passes_round1_train']:,} passes, {dv['gen_s_round1_train']:.0f} s).")
    L.append(f"- Round 2 (behaviour fast with the round-1 XGBoost feedback): {dv['rows_round2_train']:,} more rows "
             f"({dv['passes_round2_train']:,} passes, {dv['gen_s_round2_train']:.0f} s).")
    L.append(f"- Validation: seeds 21000 to 21029, same grid and both rounds, {dv['rows_val']:,} rows. Seeds "
             "21000 to 21019 (val-A) were used for early stopping and the choice of configuration; seeds 21020 "
             "to 21029 (val-B) only for the tables below.")
    L.append(f"- Total generation time {dv['gen_s_total'] / 60:.1f} min on {dv['workers']} worker processes. "
             "One row per section (40 sections per pass); the target is the card error "
             "(card reading - raw model prediction).\n")
    L.append("## Models\n")
    L.append("| round | backend | configuration | trees | val-A MAE (card) | val-A RMSE (card) | fit time (s) |")
    L.append("|---|---|---|---|---|---|---|")
    for r in res["configs_tried"]:
        L.append(f"| {r['round']} | {r['backend']} | {r['name']} (depth {r['max_depth']}, lr {r['learning_rate']}, "
                 f"{json.dumps(r['params'])}) | {r['trees']} | {r['val_a_mae_card']:.4f} | "
                 f"{r['val_a_rmse_card']:.4f} | {r['fit_s']:.0f} |")
    L.append("")
    for b in ("xgboost", "lightgbm"):
        ch = res["chosen"][b]
        L.append(f"- Chosen {b}: `{ch['name']}` with {ch['trees']} trees (lowest val-A MAE in round 2). "
                 f"File `{ch['file']}`, {ch['file_bytes'] / 1e6:.2f} MB.")
    L.append("")
    L.append("## Offline accuracy on val-B (round-2 models)\n")
    L.append("Each correction is compared with the card error it is trained on and with the actual error "
             "(true coverage - raw model prediction), which is what matters for the field. The rule column is "
             "exactly the correction the eq. 3 rule would apply at that point of the pass. The pass-mean column is a "
             "diagnostic, not a policy that was run: the plain average of all card errors read so far in the pass "
             "(0 before the first card), the feature the models rely on most. Model corrections are "
             f"clipped to +-{CLIP:g} pp, like the rule's. Behaviour policies: open, fast with the rule, fast with "
             "the round-1 XGBoost model.\n")
    for target, title in (("card", "Against the card error"), ("actual", "Against the actual error")):
        L.append(f"### {title}\n")
        for g in ("all", "mismatch", "n_cards", "policy", "reader", "nozzle"):
            L += _tab(t2, g, target, c)
            L.append("")
    L.append("### Round 1 against round 2 (val-B, all rows)\n")
    L.append("| models | target | " + " | ".join(CORR_LABEL[x] for x in c) + " |")
    L.append("|---|---|" + "---|" * len(c))
    for lab, t in (("round 1", t1), ("round 2", t2)):
        for target in ("card", "actual"):
            L.append(f"| {lab} | {target} MAE | " + " | ".join(f"{t['all']['all'][x][target]['mae']:.3f}" for x in c) + " |")
    L.append("")
    L.append("### Correction before the first card (a learned prior)\n")
    L.append("With no card read yet the rule applies 0. A model applies a fixed guess that depends only on the "
             "section. That guess is a learned prior about the simulator's average hidden losses, not feedback. "
             "Mean over val-B rows with zero earlier cards:\n")
    L.append("| mismatch | rows | mean actual error | mean card error | LightGBM applies | XGBoost applies |")
    L.append("|---|---|---|---|---|---|")
    for m, r in t2["zero_card_correction"].items():
        L.append(f"| {m} | {r['n']:,} | {r['mean_actual_error']:+.2f} | {r['mean_card_error']:+.2f} | "
                 f"{r['mean_corr_lightgbm']:+.2f} | {r['mean_corr_xgboost']:+.2f} |")
    L.append("")
    L.append("## Feature importance (share of total split gain, top 12)\n")
    L.append("| rank | XGBoost feature | share | LightGBM feature | share |")
    L.append("|---|---|---|---|---|")
    for i, ((fx, vx), (fg, vg)) in enumerate(zip(res["importance"]["xgboost"], res["importance"]["lightgbm"]), 1):
        L.append(f"| {i} | {fx} | {vx:.3f} | {fg} | {vg:.3f} |")
    L.append("")
    L.append("Figures: `experiments/figures/feedback_model_offline_mae.png`, "
             "`experiments/figures/feedback_model_importance.png`.\n")
    L.append("## Checks\n")
    s = res["sanity"]
    L.append(f"- The `rule_bias` feature matches the correction the rule applied in a fast pass: max difference "
             f"{s['rule_feature_max_abs_diff']:.2e} pp.")
    if "learned_runtime_vs_rows_max_abs_diff" in s:
        L.append(f"- The correction the loop applied with the final XGBoost model equals the model applied to the "
                 f"rows rebuilt by `rows_from_logs`: max difference {s['learned_runtime_vs_rows_max_abs_diff']:.2e} pp.")
    L.append("")
    L.append("## What was not tested\n")
    L.append("- Real cards and real fields. The hidden mismatch is the simulator's own, of a few known kinds; a "
             "model trained on it has learned those kinds and nothing else.")
    L.append("- The image readers (b0, b1) in the training data: rows come from the perfect and the noisy "
             "(image-free) readers only.")
    L.append("- Other targets than {20, 25, 30, 35, 40} %, other lambda values, speed and boom-height control, "
             "the wind forecaster, and the slow loop (fast_slow) with a learned fast loop.")
    L.append("- The round-2 validation rows were produced with the round-1 XGBoost model, not with the final one.")
    L.append("- val-A was used for early stopping and for the choice of configuration, so its numbers are "
             "optimistic; the tables use val-B only. The offline gain says nothing yet about the closed loop.")
    (RESULTS / "feedback_model.md").write_text("\n".join(L) + "\n", encoding="utf8")


def figures(res: dict, prefix: str = "feedback_model_", offline_key: str = "offline_round2") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})
    col = {"none": "#7f7f7f", "rule": "#0072B2", "lightgbm": "#E69F00", "xgboost": "#009E73"}
    t = res[offline_key]["n_cards"]
    labs = list(t)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, target in zip(axes, ("card", "actual")):
        for j, c in enumerate(("none", "rule", "lightgbm", "xgboost")):
            ax.bar(np.arange(len(labs)) + (j - 1.5) * 0.2, [t[lab][c][target]["mae"] for lab in labs], 0.2,
                   color=col[c], label=CORR_LABEL[c])
        ax.set_xticks(range(len(labs)), labs)
        ax.set_xlabel("cards read earlier in the pass")
        ax.set_title(f"MAE against the {target} error (val-B)")
    axes[0].set_ylabel("mean absolute error (pp)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / f"{prefix}offline_mae.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, b in zip(axes, ("xgboost", "lightgbm")):
        items = res["importance"][b][::-1]
        ax.barh([f for f, _ in items], [v for _, v in items], color=col[b])
        ax.set_title(f"{b}: share of split gain (top 12)")
    fig.tight_layout()
    fig.savefig(FIGS / f"{prefix}importance.png")
    plt.close(fig)


# ------------------------------------------------------------------ main

def _cat(a: dict, b: dict) -> dict:
    return {k: np.concatenate([a[k], b[k]]) for k in ("X", "y", "actual", "applied", "pred", "meta")}


def _sel(d: dict, m: np.ndarray) -> dict:
    return {k: d[k][m] for k in ("X", "y", "actual", "applied", "pred", "meta")}


# ------------------------------------------------------------------ v2 (calibrated reader, D30)

V2_ROWS = ROOT / "data" / "datasets" / "feedback_v2.npz"
V2_PATHS = dict(fl.MODEL_PATHS[2])
V2_ROUNDS = 3
ONPOL_THRESH = 0.5
K_BINS = ((0, 0, "0"), (1, 5, "1-5"), (6, 15, "6-15"), (16, 10 ** 9, "16+"))


def onpolicy_task(seed: int, pi: int) -> list[dict]:
    """Closed loop on one validation (seed, preset) with the worker's model and the rule:
    one nozzle per field x mismatch low/med/high x c_req 20/30/40, plus free nozzle at c_req 30.
    Returns per-section signed error after correction (true - model - correction)."""
    al, fb = _W["al"], _W["fb"]
    conds = al.make_conditions(PRESETS[pi], seed)
    E = seed + EVAL_OFFSET
    cfgs = [("auto", mm, c) for mm in ("low", "med", "high") for c in (20.0, 30.0, 40.0)]
    cfgs += [("free", mm, 30.0) for mm in ("low", "med", "high")]
    out = []
    for nz, mm, c in cfgs:
        for pol, fbm in (("rule", None), ("learned", fb)):
            res = al.run_pass_wrapper(_W["mods"]["noisy"], conds, "fast", E, mm, c,
                                      fixed_nozzle="auto" if nz == "auto" else None, feedback_model=fbm,
                                      same_field=True)
            err = [lg.outcome.coverage_pct - lg.corrected_pred_cov for lg in res.logs]
            out.append({"seed": seed, "preset": PRESETS[pi], "mismatch": mm, "nozzle": nz, "c_req": c, "policy": pol,
                        "err": [float(e) for e in err]})
    return out


def onpolicy_check(fb_raw: bytes, workers: int, label: str) -> dict:
    """Signed and absolute error after correction on the validation seeds 21000-21029 (calibrated
    noisy reader), per (nozzle, c_req), overall and by section index; seed-level bootstrap CI."""
    from coverageai.stats import bootstrap_mean_ci
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(fb_raw, True)) as ex:
        futs = [ex.submit(onpolicy_task, s, pi) for s in VAL_A + VAL_B for pi in range(len(PRESETS))]
        for f in as_completed(futs):
            rows += f.result()
    out = {"label": label, "wall_s": time.time() - t0, "n_passes": len(rows), "cells": {}}
    for nz, c in (("auto", 20.0), ("auto", 30.0), ("auto", 40.0), ("free", 30.0)):
        key = f"{nz}|{c:g}"
        out["cells"][key] = {}
        for pol in ("rule", "learned"):
            sub = [r for r in rows if r["nozzle"] == nz and r["c_req"] == c and r["policy"] == pol]
            E_ = np.array([r["err"] for r in sub])                  # passes x sections
            seeds = np.array([r["seed"] for r in sub])
            k = np.arange(E_.shape[1])
            cell = {}
            for lab, sel in [("all", np.ones(len(k), bool))] + [(b[2], (k >= b[0]) & (k <= b[1])) for b in K_BINS]:
                e = E_[:, sel]
                sm = np.array([e[seeds == s].mean() for s in np.unique(seeds)])
                sa = np.array([np.abs(e[seeds == s]).mean() for s in np.unique(seeds)])
                m, lo, hi = bootstrap_mean_ci(sm, rng=0)
                cell[lab] = {"signed": m, "signed_lo": lo, "signed_hi": hi, "abs": float(sa.mean()),
                             "n_sections": int(e.size)}
            out["cells"][key][pol] = cell
    worst = max(abs(v["learned"]["all"]["signed"]) for v in out["cells"].values())
    out["worst_abs_signed_learned"] = float(worst)
    out["within_threshold"] = bool(worst <= ONPOL_THRESH)
    print(f"[on-policy {label}] " + "; ".join(
        f"{k}: rule {v['rule']['all']['signed']:+.2f}, learned {v['learned']['all']['signed']:+.2f} "
        f"(|e| {v['rule']['all']['abs']:.2f} / {v['learned']['all']['abs']:.2f})" for k, v in out["cells"].items())
        + f"; {time.time() - t0:.0f}s", flush=True)
    return out


def _save_v2_rows(tr: dict, va: dict, raws: dict, gen_s: dict) -> None:
    rows = None
    for r in sorted(tr):
        part = _cat(tr[r], va[r])
        rows = part if rows is None else _cat(rows, part)
    V2_ROWS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(V2_ROWS, **rows, feature_names=np.array(fl.FEATURE_NAMES), meta_cols=np.array(META_COLS),
             **{f"r{r}_xgb_raw": np.frombuffer(b, np.uint8) for r, b in raws.items()},
             gen_s=json.dumps(gen_s), feature_version=fl.FEATURE_VERSION, calibrated_reader=True)
    print(f"wrote {V2_ROWS} ({V2_ROWS.stat().st_size / 1e6:.0f} MB)", flush=True)


def _onpol_md(chk: dict) -> list[str]:
    L = [f"On-policy check `{chk['label']}`: {chk['n_passes']:,} passes, {chk['wall_s']:.0f} s. Signed error after "
         "correction = mean of (true coverage - model - correction); brackets are 95 % bootstrap CIs over the 30 "
         "validation seeds; |error| is the mean absolute value. Pooled over mismatch low, med and high and the 6 "
         "presets; calibrated noisy reader.\n",
         "| nozzle, c_req | policy | signed, all sections | \\|error\\| all | signed, section 0 | 1-5 | 6-15 | 16+ |",
         "|---|---|---|---|---|---|---|---|"]
    for key, v in chk["cells"].items():
        nz, c = key.split("|")
        for pol, lab in (("rule", "rule (eq. 3)"), ("learned", "XGBoost v2")):
            cc = v[pol]
            L.append(f"| {nz}, {c} % | {lab} | {cc['all']['signed']:+.2f} [{cc['all']['signed_lo']:+.2f}, "
                     f"{cc['all']['signed_hi']:+.2f}] | {cc['all']['abs']:.2f} | "
                     + " | ".join(f"{cc[b]['signed']:+.2f} (\\|e\\| {cc[b]['abs']:.2f})" for b in ("0", "1-5", "6-15", "16+"))
                     + " |")
    L.append("")
    L.append(f"Largest |signed error| of the model over the four targets: {chk['worst_abs_signed_learned']:.2f} pp "
             f"({'within' if chk['within_threshold'] else 'outside'} the +-{ONPOL_THRESH:g} pp threshold).\n")
    return L


def write_md_v2(res: dict) -> None:
    c = ("none", "rule", "passmean", "lightgbm", "xgboost")
    fin = res["offline_final"]
    dv = res["data"]
    L = ["# Learned card feedback v2: trained on calibrated cards (D29, D30)\n",
         "Generated by `experiments/train_feedback_model.py --version 2`. Do not edit by hand. v1 is unchanged "
         "(`results/feedback_model.md`, `models/feedback_*_v1.joblib`). Closed-loop evaluation of v2 on fresh seeds: "
         "`results/feedback_model_v2_closed_loop.md`.\n",
         "Everything here is simulation. The models learn the hidden mismatch of this simulator and its card "
         "noise. The reader calibration (D30) subtracts the reader's mean error per coverage band, taken from the "
         "reader's own validation on synthetic cards with known truth; a real reader would need annotated real "
         "cards, and a real feedback model real verified cards.\n",
         "## What changed from v1\n",
         "- The noisy reader is wrapped in the reader calibration (`calibrate_reader=True`); the perfect reader is "
         "unchanged. Same features (version "
         f"{res['feature_version']}), same seeds, same grid.",
         f"- {res['n_rounds']} rounds instead of 2: round 1 open + rule; round r >= 2 fast with the round-(r-1) "
         "XGBoost model. The final models are fit on the union of all rounds.\n",
         "## Data\n",
         f"- Training seeds {dv['train_seeds'][0]} to {dv['train_seeds'][-1]} ({len(dv['train_seeds'])}); validation "
         "21000 to 21029 (val-A 21000-21019: early stopping and configuration; val-B 21020-21029: tables only).",
         ]
    for r in range(1, res["n_rounds"] + 1):
        L.append(f"- Round {r}: {dv['rows_train'][str(r)]:,} training rows, {dv['rows_val'][str(r)]:,} validation rows, "
                 f"generation {dv['gen_s'][str(r)]:.0f} s.")
    L.append(f"- Total {dv['rows_total']:,} rows, generation {dv['gen_s_total'] / 60:.1f} min on {dv['workers']} workers.\n")
    L.append("## Models\n")
    L.append("| round | backend | configuration | trees | val-A MAE (card) | val-A RMSE (card) | fit s |")
    L.append("|---|---|---|---|---|---|---|")
    for r in res["configs_tried"]:
        L.append(f"| {r['round']} | {r['backend']} | {r['name']} | {r['trees']} | {r['val_a_mae_card']:.4f} | "
                 f"{r['val_a_rmse_card']:.4f} | {r['fit_s']:.0f} |")
    L.append("")
    for b in ("xgboost", "lightgbm"):
        ch = res["chosen"][b]
        L.append(f"- Final {b}: `{ch['name']}`, {ch['trees']} trees, file `{ch['file']}`, {ch['file_bytes'] / 1e6:.2f} MB.")
    L.append("")
    L.append("## On-policy calibration check (validation seeds, closed loop with the final XGBoost)\n")
    for chk in res["onpolicy"]:
        L += _onpol_md(chk)
    L.append("## Offline accuracy on val-B (final models)\n")
    L.append("Same tables as v1. The rule column is the correction the eq. 3 rule would apply; the pass-mean column "
             "is a diagnostic (mean of all card errors so far, 0 before the first card). Model corrections clipped "
             f"to +-{CLIP:g} pp. Card errors here are against calibrated readings for the noisy reader.\n")
    for target, title in (("card", "Against the card error"), ("actual", "Against the actual error")):
        L.append(f"### {title}\n")
        for g in ("all", "mismatch", "n_cards", "policy", "reader", "nozzle"):
            L += _tab(fin, g, target, c)
            L.append("")
    L.append("### Models of each round (val-B, all rows)\n")
    L.append("| models | target | " + " | ".join(CORR_LABEL[x] for x in c) + " |")
    L.append("|---|---|" + "---|" * len(c))
    for r, t in sorted(res["offline_by_round"].items()):
        for target in ("card", "actual"):
            L.append(f"| round {r} | {target} MAE | " + " | ".join(f"{t['all']['all'][x][target]['mae']:.3f}" for x in c) + " |")
    L.append("")
    L.append("### Correction before the first card (learned prior)\n")
    L.append("| mismatch | rows | mean actual error | mean card error | LightGBM applies | XGBoost applies |")
    L.append("|---|---|---|---|---|---|")
    for m, r in fin["zero_card_correction"].items():
        L.append(f"| {m} | {r['n']:,} | {r['mean_actual_error']:+.2f} | {r['mean_card_error']:+.2f} | "
                 f"{r['mean_corr_lightgbm']:+.2f} | {r['mean_corr_xgboost']:+.2f} |")
    L.append("")
    L.append("## Feature importance (share of total split gain, top 12)\n")
    L.append("| rank | XGBoost feature | share | LightGBM feature | share |")
    L.append("|---|---|---|---|---|")
    for i, ((fx, vx), (fg, vg)) in enumerate(zip(res["importance"]["xgboost"], res["importance"]["lightgbm"]), 1):
        L.append(f"| {i} | {fx} | {vx:.3f} | {fg} | {vg:.3f} |")
    L.append("")
    s_ = res["sanity"]
    L.append("## Checks\n")
    L.append(f"- `rule_bias` feature vs the rule's applied correction (calibrated reader): max difference "
             f"{s_['rule_feature_max_abs_diff']:.2e} pp; loop correction vs model on rebuilt rows: "
             f"{s_.get('learned_runtime_vs_rows_max_abs_diff', float('nan')):.2e} pp.\n")
    L.append("Figures: `experiments/figures/feedback_model_v2_offline_mae.png`, "
             "`experiments/figures/feedback_model_v2_importance.png`.\n")
    L.append("## What was not tested\n")
    L += ["- Real cards, real readers and real fields. The calibration table and the hidden mismatch are the "
          "simulator's own.",
          "- The image readers in the training data (rows come from the perfect and the calibrated noisy reader).",
          "- A calibration table that is itself wrong (for example a reader whose bias drifts with lighting or "
          "card batch). The model was trained with the exact table the simulated reader follows.",
          "- The on-policy check uses the validation seeds that also served for early stopping (val-A) and the "
          "tables (val-B); the closed-loop evaluation on fresh seeds is in the closed-loop write-up."]
    (RESULTS / "feedback_model_v2.md").write_text("\n".join(L) + "\n", encoding="utf8")


def main_v2(args) -> None:
    workers = min(args.workers, 26)
    train_seeds = list(range(TRAIN_SEED0, TRAIN_SEED0 + args.seeds))
    val_seeds = list(VAL_A + VAL_B)
    t_all = time.time()
    FIGS.mkdir(parents=True, exist_ok=True)
    tr, va, raws, gen_s, tried, fits = {}, {}, {}, {}, [], {}
    if args.reuse and V2_ROWS.is_file():
        z = np.load(V2_ROWS)
        allrows = {k: z[k] for k in ("X", "y", "actual", "applied", "pred", "meta")}
        gen_s = {int(k): v for k, v in json.loads(str(z["gen_s"])).items()}
        raws = {int(k[1:].split("_")[0]): bytes(z[k]) for k in z.files if k.endswith("_xgb_raw")}
        is_val = np.isin(allrows["meta"][:, 0], val_seeds)
        code = allrows["meta"][:, 3]
        for r in sorted(gen_s):
            m = (code <= 1) if r == 1 else (code == r)
            tr[r], va[r] = _sel(allrows, ~is_val & m), _sel(allrows, is_val & m)
        train_seeds = sorted(set(int(s) for s in tr[1]["meta"][:, 0]))
        print(f"reusing {len(allrows['y']):,} rows (rounds {sorted(tr)}) from {V2_ROWS}", flush=True)
    sanity0 = sanity_checks(None, calibrate=True)

    def split_val(d):
        a = np.isin(d["meta"][:, 0], VAL_A)
        return _sel(d, a), _sel(d, ~a)

    def union(dd, upto):
        out = None
        for r in range(1, upto + 1):
            out = dd[r] if out is None else _cat(out, dd[r])
        return out

    def fit_round(r):
        trU, vaU = union(tr, r), union(va, r)
        va_a, _ = split_val(vaU)
        bx, bl, t_ = fit_all(trU["X"], trU["y"], va_a["X"], va_a["y"], f"v2 round {r}")
        for x in t_:
            x["round"] = r
        tried.extend(t_)
        fits[r] = (bx, bl)
        raws.setdefault(r, bytes(bx[0].booster.save_raw("ubj")))
        return bx, bl

    n_rounds = V2_ROUNDS
    for r in range(1, n_rounds + 1):
        if r not in tr:
            prev = raws.get(r - 1) if r > 1 else None
            if r > 1:
                sanity_checks(_model_from_raw(prev), calibrate=True)
            va[r] = generate(val_seeds, r, workers, prev, f"v2 val r{r}", calibrate=True)
            tr[r] = generate(train_seeds, r, workers, prev, f"v2 train r{r}", calibrate=True)
            gen_s[r] = va[r].pop("_wall_s") + tr[r].pop("_wall_s")
        fit_round(r)
        _save_v2_rows({q: tr[q] for q in tr}, {q: va[q] for q in va}, raws, {str(k): v for k, v in gen_s.items()})

    onpol = [onpolicy_check(raws[n_rounds], workers, f"after round {n_rounds}")]
    extra_round = False
    if not onpol[-1]["within_threshold"]:
        extra_round = True
        r = n_rounds + 1
        if r not in tr:
            va[r] = generate(val_seeds, r, workers, raws[n_rounds], f"v2 val r{r}", calibrate=True)
            tr[r] = generate(train_seeds, r, workers, raws[n_rounds], f"v2 train r{r}", calibrate=True)
            gen_s[r] = va[r].pop("_wall_s") + tr[r].pop("_wall_s")
        n_rounds = r
        fit_round(r)
        _save_v2_rows(tr, va, raws, {str(k): v for k, v in gen_s.items()})
        onpol.append(onpolicy_check(raws[n_rounds], workers, f"after round {n_rounds} (extra round)"))

    (x2, x2r), (g2, g2r) = fits[n_rounds]
    trU, vaU = union(tr, n_rounds), union(va, n_rounds)
    vb_mask = np.isin(vaU["meta"][:, 0], VAL_B)
    nc_va = vaU["X"][:, I_NC]
    base = {"none": np.zeros(len(vaU["y"])), "rule": vaU["X"][:, I_RULE].astype(float),
            "passmean": np.where(nc_va > 0, np.nan_to_num(vaU["X"][:, I_EMEAN].astype(float)), 0.0)}
    off_by_round = {}
    for r in range(1, n_rounds + 1):
        (xr, _), (gr, _) = fits[r]
        off_by_round[r] = offline_tables(vaU, {**base, "lightgbm": predict_bulk(gr, vaU["X"]),
                                               "xgboost": predict_bulk(xr, vaU["X"])}, vb_mask)
    chosen = {}
    for b, m, r in (("xgboost", x2, x2r), ("lightgbm", g2, g2r)):
        m.meta.update(config=r["name"], train_rows=int(len(trU["y"])), train_seeds=[train_seeds[0], train_seeds[-1]],
                      val_a_mae_card=r["val_a_mae_card"], version=2, calibrated_reader=True, rounds=n_rounds)
        pth = m.save(V2_PATHS[b])
        chosen[b] = {**r, "file": str(pth.relative_to(ROOT)).replace("\\", "/"), "file_bytes": pth.stat().st_size}
        print(f"saved {pth} ({pth.stat().st_size / 1e6:.2f} MB)", flush=True)
    sanity = {**sanity0, **sanity_checks(fl.FeedbackModel.load(V2_PATHS["xgboost"]), calibrate=True)}
    res = {"schema": "feedback_model_v2/1", "version": 2, "feature_version": fl.FEATURE_VERSION,
           "features": list(fl.FEATURE_NAMES), "clip_pp": CLIP, "calibrated_reader": True, "n_rounds": n_rounds,
           "extra_round": extra_round,
           "data": {"train_seeds": train_seeds, "val_a_seeds": list(VAL_A), "val_b_seeds": list(VAL_B),
                    "workers": workers,
                    "rows_train": {str(r): int(len(tr[r]["y"])) for r in tr},
                    "rows_val": {str(r): int(len(va[r]["y"])) for r in va},
                    "rows_total": int(len(trU["y"]) + len(vaU["y"])),
                    "gen_s": {str(r): float(v) for r, v in gen_s.items()}, "gen_s_total": float(sum(gen_s.values()))},
           "configs_tried": tried, "chosen": chosen, "onpolicy": onpol,
           "offline_final": off_by_round[n_rounds], "offline_by_round": {str(r): v for r, v in off_by_round.items()},
           "offline_round2": off_by_round[n_rounds],
           "importance": {"xgboost": importance_top(x2), "lightgbm": importance_top(g2)},
           "importance_all": {"xgboost": x2.importance(), "lightgbm": g2.importance()},
           "sanity": sanity, "wall_s": time.time() - t_all}
    (RESULTS / "feedback_model_v2.json").write_text(json.dumps(res, indent=1), encoding="utf8")
    write_md_v2(res)
    figures(res, prefix="feedback_model_v2_", offline_key="offline_final")
    print(f"wrote results/feedback_model_v2.md/.json; total {time.time() - t_all:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=120, help="number of training scenario seeds")
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--reuse", action="store_true", help="reuse the cached rows (v1: feedback_v1.npz, v2: feedback_v2.npz)")
    ap.add_argument("--version", type=int, default=1, choices=(1, 2),
                    help="1: the original models (raw reader, 2 rounds); 2: calibrated noisy reader, 3 rounds")
    args = ap.parse_args()
    if args.version == 2:
        main_v2(args)
        return
    workers = min(args.workers, 26)
    train_seeds = list(range(TRAIN_SEED0, TRAIN_SEED0 + args.seeds))
    val_seeds = list(VAL_A + VAL_B)
    t_all = time.time()
    FIGS.mkdir(parents=True, exist_ok=True)
    gen_s = {}

    if args.reuse and ROWS_PATH.is_file():
        z = np.load(ROWS_PATH)
        allrows = {k: z[k] for k in ("X", "y", "actual", "applied", "pred", "meta")}
        gen_s = json.loads(str(z["gen_s"]))
        train_seeds = sorted(set(int(s) for s in allrows["meta"][:, 0]) - set(val_seeds))
        r1_raw = bytes(z["r1_xgb_raw"])
        if list(z["feature_names"]) != list(fl.FEATURE_NAMES):
            raise SystemExit("cached rows were built with another feature list; run without --reuse")
        print(f"reusing {len(allrows['y']):,} rows from {ROWS_PATH}", flush=True)
        is_val = np.isin(allrows["meta"][:, 0], val_seeds)
        r1 = allrows["meta"][:, 3] < 2
        tr1, va1 = _sel(allrows, ~is_val & r1), _sel(allrows, is_val & r1)
        tr2, va2 = _sel(allrows, ~is_val & ~r1), _sel(allrows, is_val & ~r1)
    else:
        sanity_checks(None)
        print(f"training seeds {train_seeds[0]}..{train_seeds[-1]} ({len(train_seeds)}), "
              f"{len(train_seeds) * 6 * 32:,} round-1 passes", flush=True)
        va1 = generate(val_seeds, 1, workers, None, "val r1")
        tr1 = generate(train_seeds, 1, workers, None, "train r1")
        gen_s.update(val_r1=va1.pop("_wall_s"), train_r1=tr1.pop("_wall_s"))
        r1_raw = None

    def split_val(d):
        a = np.isin(d["meta"][:, 0], VAL_A)
        return _sel(d, a), _sel(d, ~a)

    # ---- round 1
    va1_a, _ = split_val(va1)
    (x1, x1r), (g1, g1r), tried1 = fit_all(tr1["X"], tr1["y"], va1_a["X"], va1_a["y"], "round 1")
    for r in tried1:
        r["round"] = 1
    if r1_raw is None:
        r1_raw = bytes(x1.booster.save_raw("ubj"))
        sanity_checks(_model_from_raw(r1_raw))      # runtime features == training features
        va2 = generate(val_seeds, 2, workers, r1_raw, "val r2")
        tr2 = generate(train_seeds, 2, workers, r1_raw, "train r2")
        gen_s.update(val_r2=va2.pop("_wall_s"), train_r2=tr2.pop("_wall_s"))
        allrows = _cat(_cat(tr1, tr2), _cat(va1, va2))
        ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(ROWS_PATH, **allrows, feature_names=np.array(fl.FEATURE_NAMES), meta_cols=np.array(META_COLS),
                 r1_xgb_raw=np.frombuffer(r1_raw, np.uint8), gen_s=json.dumps(gen_s),
                 feature_version=fl.FEATURE_VERSION)
        print(f"wrote {ROWS_PATH} ({ROWS_PATH.stat().st_size / 1e6:.0f} MB)", flush=True)

    # ---- round 2: refit on the union
    tr = _cat(tr1, tr2)
    va = _cat(va1, va2)
    va_a, va_b = split_val(va)
    (x2, x2r), (g2, g2r), tried2 = fit_all(tr["X"], tr["y"], va_a["X"], va_a["y"], "round 2")
    for r in tried2:
        r["round"] = 2

    # ---- offline tables on val-B
    vb_mask = np.isin(va["meta"][:, 0], VAL_B)
    nc_va = va["X"][:, I_NC]
    base = {"none": np.zeros(len(va["y"])), "rule": va["X"][:, I_RULE].astype(float),
            # diagnostic: the plain mean of all card errors so far in the pass (0 before the first card)
            "passmean": np.where(nc_va > 0, np.nan_to_num(va["X"][:, I_EMEAN].astype(float)), 0.0)}
    off1 = offline_tables(va, {**base, "lightgbm": predict_bulk(g1, va["X"]), "xgboost": predict_bulk(x1, va["X"])},
                          vb_mask)
    off2 = offline_tables(va, {**base, "lightgbm": predict_bulk(g2, va["X"]), "xgboost": predict_bulk(x2, va["X"])},
                          vb_mask)

    # ---- save the models
    chosen = {}
    for b, m, r in (("xgboost", x2, x2r), ("lightgbm", g2, g2r)):
        m.meta.update(config=r["name"], train_rows=int(len(tr["y"])), train_seeds=[train_seeds[0], train_seeds[-1]],
                      val_a_mae_card=r["val_a_mae_card"])
        p = m.save(fl.MODEL_PATHS[1][b])
        rel = p.relative_to(ROOT) if p.is_relative_to(ROOT) else p
        chosen[b] = {**r, "file": str(rel).replace("\\", "/"), "file_bytes": p.stat().st_size}
        print(f"saved {p} ({p.stat().st_size / 1e6:.2f} MB)", flush=True)
    sanity = sanity_checks(fl.FeedbackModel.load(fl.MODEL_PATHS[1]["xgboost"]))

    passes = lambda d: int(len(d["y"]) // 40)  # noqa: E731
    res = {
        "schema": "feedback_model/1", "feature_version": fl.FEATURE_VERSION, "features": list(fl.FEATURE_NAMES),
        "clip_pp": CLIP, "eval_offset": EVAL_OFFSET,
        "data": {"train_seeds": train_seeds,
                 "val_a_seeds": list(VAL_A), "val_b_seeds": list(VAL_B), "workers": workers,
                 "rows_round1_train": int(len(tr1["y"])), "rows_round2_train": int(len(tr2["y"])),
                 "rows_val": int(len(va["y"])), "rows_total": int(len(tr["y"]) + len(va["y"])),
                 "passes_round1_train": passes(tr1), "passes_round2_train": passes(tr2),
                 "gen_s_round1_train": gen_s.get("train_r1", float("nan")),
                 "gen_s_round2_train": gen_s.get("train_r2", float("nan")),
                 "gen_s_val": gen_s.get("val_r1", 0.0) + gen_s.get("val_r2", 0.0),
                 "gen_s_total": float(sum(gen_s.values())), "c_req_choices": list(C_REQS),
                 "presets": list(PRESETS), "mismatch": list(MISMATCH), "readers": list(READERS),
                 "nozzle_modes": list(NOZZLE_NAME), "policies": list(POLICY_NAME)},
        "configs_tried": tried1 + tried2, "chosen": chosen,
        "round1_chosen": {"xgboost": x1r, "lightgbm": g1r},
        "offline_round1": off1, "offline_round2": off2,
        "importance": {"xgboost": importance_top(x2), "lightgbm": importance_top(g2)},
        "importance_all": {"xgboost": x2.importance(), "lightgbm": g2.importance()},
        "sanity": sanity, "wall_s": time.time() - t_all,
    }
    (RESULTS / "feedback_model.json").write_text(json.dumps(res, indent=1), encoding="utf8")
    write_md(res)
    figures(res)
    print(f"wrote results/feedback_model.md/.json; total {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
