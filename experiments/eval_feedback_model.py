"""Closed-loop evaluation of the learned card feedback (decision D29) on the evaluation seeds.

Grid: scenario seeds 0..19 x 6 presets x mismatch none / low / med / high x reader noisy (main)
and perfect x nozzle mode "auto" (one nozzle per field, the app default) and free, at c_req 30
and the default lambda. Policies per cell: open; fast with the eq. 3 rule; fast with the
LightGBM feedback model; fast with the XGBoost feedback model; the fair oracle (reference).

Protocol (as experiments/closed_loop.py): conditions al.make_conditions(preset, s); the pass of
EVERY policy runs with hidden-mismatch seed E = s + 4036, so all policies see the same gusts,
canopy, wear and card noise (paired). Metrics use the TRUE outcome. The models are loaded from
models/feedback_{xgb,lgbm}_v1.joblib and are frozen before this script runs; nothing here feeds
back into training.

Per pass: share of sections with coverage >= c_req, mean shortfall (pp), mean volume (gal/acre),
mean drift (%), and the accuracy of the feedback: mean |true coverage - prediction after
feedback|, the same against the card reading, the mean correction, and the floor: mean |true
coverage - fair-oracle prediction| for the settings the policy chose (the fair oracle knows the
hidden mismatch but not the per-card spread; no feedback can do better on average). The floor is
computed on a deep copy of the simulator replayed from the start of the pass (CLAUDE.md rule 3).

Statistics: metrics are averaged over presets within a seed; means over the 20 seeds with
bootstrap 95 % CIs; paired differences (learned - rule) with paired-bootstrap 95 % CIs
(coverageai.stats); gap recovery (policy - open) / (oracle - open) on sections meeting target with
a bootstrap CI over seeds, as in closed_loop.py.

Optional check (--b0, on by default, capped at --b0-cap-s seconds): mismatch med with the
classic image reader "b0" on seeds 0..4, open / rule / XGBoost / oracle.

Usage:
  uv run --no-sync python experiments/eval_feedback_model.py --workers 26
  uv run --no-sync python experiments/eval_feedback_model.py --report-only [--verdict TEXT]   # rebuild md/figures
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coverageai import feedback_learned as _fl_top                 # noqa: E402
from coverageai import scenarios                                   # noqa: E402
from coverageai.stats import bootstrap_mean_ci, paired_bootstrap_mean_ci   # noqa: E402

RESULTS = ROOT / "results"
FIGS = ROOT / "experiments" / "figures"
OUT_JSON = RESULTS / "feedback_model_closed_loop.json"
SEEDS = tuple(range(20))
B0_SEEDS = tuple(range(5))
PRESETS = tuple(scenarios.list_presets())
MISMATCH = ("none", "low", "med", "high")
READERS = ("noisy", "perfect")
NOZZLE = {"auto": "auto", "free": None}
EVAL_OFFSET = 4036
C_REQ = 30.0
# name -> (loop policy, feedback model)
POLICIES = {"open": ("open", None), "rule": ("fast", None), "lightgbm": ("fast", "lightgbm"),
            "xgboost": ("fast", "xgboost"), "oracle": ("oracle", None)}
LEARNED = ("lightgbm", "xgboost")
CONTROLS = ("rule_shifted", "pass_mean")       # post-hoc diagnostics (--controls)
POLICIES_CTRL = {"rule_shifted": ("fast", "rule_shifted"), "pass_mean": ("fast", "pass_mean")}
LABEL = {"open": "open loop", "rule": "fast, rule (eq. 3)", "lightgbm": "fast, LightGBM feedback",
         "xgboost": "fast, XGBoost feedback", "oracle": "fair oracle",
         "rule_shifted": "control: rule + constant", "pass_mean": "control: pass mean + learned prior"}
ROWS_PATH = ROOT / "data" / "datasets" / "feedback_v1.npz"
VAL_B = tuple(range(21020, 21030))
METRICS = ("frac_met", "mean_shortfall_pp", "mean_volume_gpa", "mean_drift_pct", "fb_err_true", "fb_err_card",
           "fb_bias_true", "mean_corr", "floor_err", "raw_err_true", "n_retreat")
GROUPS = {"none": ["none"], "low": ["low"], "med": ["med"], "high": ["high"],
          "low+med+high": ["low", "med", "high"], "all": list(MISMATCH)}

# ------------------------------------------------------------------ worker side
_W: dict = {}


class ControlFeedback:
    """A fixed correction rule with the ``FeedbackModel`` interface (``predict``, ``backend``,
    ``meta``), so ``run_pass_wrapper(feedback_model=...)`` runs it through ``LearnedFeedback`` with
    the same features and clipping as the trained models.

    * ``rule_shifted``: the eq. 3 rule's correction plus a constant ``offset``.
    * ``pass_mean``: the mean of all card errors read so far in the pass; ``prior`` before the
      first card."""

    backend = "lightgbm"

    def __init__(self, kind: str, offset: float = 0.0, prior: float = 0.0):
        from coverageai import feedback_learned as fl
        self.kind, self.offset, self.prior = kind, float(offset), float(prior)
        self.meta: dict = {}
        self._i_rule = fl.FEATURE_NAMES.index("rule_bias")
        self._i_mean = fl.FEATURE_NAMES.index("e_mean")
        self._i_n = fl.FEATURE_NAMES.index("n_cards")

    def predict(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        if self.kind == "rule_shifted":
            return np.nan_to_num(X[:, self._i_rule]) + self.offset
        has = X[:, self._i_n] > 0
        return np.where(has, np.nan_to_num(X[:, self._i_mean]), self.prior)


def _init(ctrl: dict | None = None) -> None:
    from coverageai import app_logic as al
    from coverageai import feedback_learned as fl
    _W["al"] = al
    _W["mods"] = {}
    _W["fb"] = {b: fl.FeedbackModel.load(fl.MODEL_PATHS[1][b]) for b in LEARNED}
    if ctrl:
        _W["fb"]["rule_shifted"] = ControlFeedback("rule_shifted", offset=ctrl["offset"])
        _W["fb"]["pass_mean"] = ControlFeedback("pass_mean", prior=ctrl["prior"])


def _mods(reader: str):
    if reader not in _W["mods"]:
        _W["mods"][reader] = _W["al"].build_modules_adapter("spray", reader)
    return _W["mods"][reader]


def floor_errors(mods, logs, E: int, mismatch: str) -> np.ndarray:
    """|true coverage - fair-oracle prediction| per section for the logged settings, on a deep
    copy of the simulator replayed from the start of the pass (hidden draws in section order)."""
    from coverageai.loop import FairOracleModel
    sim = copy.deepcopy(mods.simulator)
    if hasattr(sim, "begin_pass"):
        sim.begin_pass(E, mismatch)
    orc = FairOracleModel(sim, mismatch, E)
    out = []
    for lg in logs:
        orc.k = lg.k
        out.append(abs(lg.outcome.coverage_pct - orc.predict(lg.x, lg.u).coverage_pct))
    return np.array(out)


def pass_metrics(logs, c_req: float) -> dict:
    cov = np.array([lg.outcome.coverage_pct for lg in logs])
    corr = np.array([lg.corrected_pred_cov for lg in logs])
    return {"frac_met": float(np.mean(cov >= c_req)),
            "mean_shortfall_pp": float(np.mean(np.maximum(c_req - cov, 0.0))),
            "mean_volume_gpa": float(np.mean([lg.derived.app_rate_gpa for lg in logs])),
            "mean_drift_pct": 100.0 * float(np.mean([lg.outcome.drift_frac for lg in logs])),
            "fb_err_true": float(np.mean(np.abs(cov - corr))),
            "fb_err_card": float(np.mean(np.abs(np.array([lg.meas.coverage_pct for lg in logs]) - corr))),
            "fb_bias_true": float(np.mean(cov - corr)),
            "mean_corr": float(np.mean([lg.bias for lg in logs])),
            "raw_err_true": float(np.mean(np.abs(cov - np.array([lg.pred.coverage_pct for lg in logs])))),
            "n_retreat": int(sum(lg.flagged_retreat for lg in logs))}


def cell(seed: int, preset: str, mismatch: str, reader: str, nozzle: str, policies: tuple[str, ...]) -> list[dict]:
    al = _W["al"]
    mods = _mods(reader)
    conds = al.make_conditions(preset, seed)
    E = seed + EVAL_OFFSET
    rows = []
    for name in policies:
        pol, fbm = {**POLICIES, **POLICIES_CTRL}[name]
        t0 = time.perf_counter()
        res = al.run_pass_wrapper(mods, conds, pol, E, mismatch, C_REQ, None, fixed_nozzle=NOZZLE[nozzle],
                                  feedback_model=_W["fb"][fbm] if fbm else None, same_field=True)
        m = pass_metrics(res.logs, C_REQ)
        m["floor_err"] = float(floor_errors(res.modules, res.logs, E, mismatch).mean())
        nz = res.meta.get("nozzle", {})
        rows.append({"seed": seed, "preset": preset, "mismatch": mismatch, "reader": reader, "nozzle": nozzle,
                     "policy": name, "nozzle_held": f"{nz.get('family')}-{nz.get('size')}" if nz.get("family") else None,
                     "feedback": res.meta.get("feedback", {}).get("kind"), "pass_s": time.perf_counter() - t0, **m})
    return rows


# ------------------------------------------------------------------ statistics

def seed_level(df: pd.DataFrame, levels: list[str]) -> pd.DataFrame:
    """Per (policy, seed): mean over presets (and over the mismatch levels of the group)."""
    sub = df[df["mismatch"].isin(levels)]
    return sub.groupby(["policy", "seed"], as_index=False)[list(METRICS)].mean()


def _boot(fn, arrays, B: int = 2000, seed: int = 0):
    n = len(arrays[0])
    est = fn(*arrays)
    idx = np.random.default_rng(seed).integers(0, n, (B, n))
    vals = np.array([fn(*[a[i] for a in arrays]) for i in idx])
    vals = vals[np.isfinite(vals)]
    if len(vals) < B // 2:
        return float(est), float("nan"), float("nan")
    return float(est), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _gap(o, c, r):
    d = r.mean() - o.mean()
    return (c.mean() - o.mean()) / d if abs(d) > 1e-9 else np.nan


def summarize(df: pd.DataFrame, policies=tuple(POLICIES)) -> dict:
    """Means, paired differences and gap recovery for every (reader, nozzle, mismatch group)."""
    out = {}
    for (reader, nozzle), d in df.groupby(["reader", "nozzle"]):
        for g, levels in GROUPS.items():
            sl = seed_level(d, levels)
            if sl.empty:
                continue
            piv = {m: sl.pivot(index="seed", columns="policy", values=m).sort_index() for m in METRICS}
            key = f"{reader}|{nozzle}|{g}"
            r = {"reader": reader, "nozzle": nozzle, "mismatch": g, "n_seeds": int(len(piv["frac_met"])),
                 "means": {}, "diff_vs_rule": {}, "gap_recovery": {}, "gap_diff_vs_rule": {}}
            pols = [p for p in policies if p in piv["frac_met"]]
            for p in pols:
                r["means"][p] = {m: dict(zip(("mean", "lo", "hi"), bootstrap_mean_ci(piv[m][p].to_numpy(), rng=0)))
                                 for m in METRICS}
            if "rule" in pols:
                for p in pols:
                    if p == "rule":
                        continue
                    r["diff_vs_rule"][p] = {}
                    for m in METRICS:
                        est, lo, hi = paired_bootstrap_mean_ci(piv[m][p].to_numpy(), piv[m]["rule"].to_numpy(), rng=0)
                        r["diff_vs_rule"][p][m] = {"mean": est, "lo": lo, "hi": hi,
                                                   "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0))}
            if "open" in pols and "oracle" in pols:
                o = piv["frac_met"]["open"].to_numpy()
                ref = piv["frac_met"]["oracle"].to_numpy()
                r["gap_pp"] = float(100 * (ref.mean() - o.mean()))
                for p in pols:
                    if p in ("open", "oracle"):
                        continue
                    c = piv["frac_met"][p].to_numpy()
                    r["gap_recovery"][p] = dict(zip(("mean", "lo", "hi"), _boot(_gap, [o, c, ref])))
                    if p != "rule" and "rule" in pols:
                        cr = piv["frac_met"]["rule"].to_numpy()
                        est, lo, hi = _boot(lambda a, b, e, f: _gap(a, b, f) - _gap(a, e, f), [o, c, cr, ref])
                        r["gap_diff_vs_rule"][p] = {"mean": est, "lo": lo, "hi": hi,
                                                    "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0))}
            out[key] = r
    return out


# ------------------------------------------------------------------ write-up

def _ci(x: dict, d: int = 3, scale: float = 1.0, signed: bool = False) -> str:
    f = f"{{:{'+' if signed else ''}.{d}f}}"
    return f"{f.format(scale * x['mean'])} [{f.format(scale * x['lo'])}, {f.format(scale * x['hi'])}]"


def _yn(x: dict) -> str:
    return "yes" if x["excludes_zero"] else "no"


COLS = (("frac_met", "on target", 3, 1.0), ("mean_shortfall_pp", "shortfall pp", 2, 1.0),
        ("mean_volume_gpa", "volume GPA", 2, 1.0), ("mean_drift_pct", "drift %", 3, 1.0),
        ("fb_err_true", "|true - corrected| pp", 2, 1.0), ("fb_err_card", "|card - corrected| pp", 2, 1.0),
        ("mean_corr", "mean correction pp", 2, 1.0), ("floor_err", "floor pp", 2, 1.0))


def _means_table(r: dict) -> list[str]:
    L = ["| policy | " + " | ".join(c[1] for c in COLS) + " | gap recovery |",
         "|---|" + "---|" * (len(COLS) + 1)]
    for p, mm in r["means"].items():
        cells = []
        for m, _, d, s in COLS:
            if p == "oracle" and m in ("fb_err_true", "fb_err_card", "mean_corr"):
                cells.append("-")
            else:
                cells.append(_ci(mm[m], d, s, signed=(m == "mean_corr")))
        gr = r["gap_recovery"].get(p)
        g = "-" if gr is None else (f"{100 * gr['mean']:.0f}% [{100 * gr['lo']:.0f}, {100 * gr['hi']:.0f}]"
                                    if np.isfinite(gr["lo"]) else (f"{100 * gr['mean']:.0f}% [n/a]" if np.isfinite(gr["mean"]) else "n/a"))
        L.append(f"| {LABEL[p]} | " + " | ".join(cells) + f" | {g} |")
    return L


def _diff_table(r: dict, pols=LEARNED) -> list[str]:
    L = ["| learned - rule | " + " | ".join(c[1] for c in COLS[:7]) + " | gap recovery |",
         "|---|" + "---|" * 8]
    for p in pols:
        if p not in r["diff_vs_rule"]:
            continue
        dd = r["diff_vs_rule"][p]
        cells = [f"{_ci(dd[m], d + (1 if m == 'frac_met' else 0), s, signed=True)} ({'CI excl. 0' if dd[m]['excludes_zero'] else 'CI incl. 0'})"
                 for m, _, d, s in COLS[:7]]
        gd = r["gap_diff_vs_rule"].get(p)
        g = "-" if gd is None or not np.isfinite(gd["lo"]) else (
            f"{100 * gd['mean']:+.0f} pts [{100 * gd['lo']:+.0f}, {100 * gd['hi']:+.0f}] "
            f"({'CI excl. 0' if gd['excludes_zero'] else 'CI incl. 0'})")
        L.append(f"| {LABEL[p]} | " + " | ".join(cells) + f" | {g} |")
    return L


CTRL_COLS = (("frac_met", "on target", 4), ("mean_volume_gpa", "volume GPA", 2),
             ("mean_drift_pct", "drift %", 3), ("fb_err_true", "|true - corrected| pp", 2))


def _controls_md(C: dict) -> list[str]:
    p = C["params"]
    L = ["## Controls: accuracy or caution (post-hoc diagnostic on the evaluation seeds)\n",
         "Added after the headline results above had been seen, to separate two explanations of the trained "
         "model's gain: a more accurate correction, or a correction that is more negative on average and so "
         "sprays more. These are diagnostics, not a model selection: the trained models and the tables above "
         "were not changed because of them. Both controls run through the same `LearnedFeedback` adapter "
         f"(same features, same +-{p.get('clip_pp', 15):g} pp clip), {C['n_passes']:,} new passes on the noisy-reader "
         "grid of the headline (seeds 0 to 19, hidden seed s + 4036, 6 presets, 4 mismatch levels, 2 nozzle modes). Open, rule, "
         f"XGBoost and oracle passes are {C['reuse']}.\n",
         f"- **rule + constant** (`rule_shifted`): the eq. 3 correction plus {p['offset']:+.3f} pp, the mean of "
         f"(XGBoost correction - rule correction) over {p['offset_rows']:,} validation rows (val-B, seeds 21020 to "
         "21029, round-2 behaviour policy). As cautious as the trained model on average, with no better information.",
         f"- **pass mean + learned prior** (`pass_mean`): the plain mean of all card errors read so far in the pass; "
         f"before the first card {p['prior']:+.3f} pp, the training mean of the card error at the first section "
         f"({p['prior_rows']:,} training rows). The simplest strategy built on the feature the trained model uses most.\n"]
    for nz, txt in (("auto", "one nozzle per field"), ("free", "free nozzle choice")):
        for g in ("low+med+high", "none"):
            r = C["summary"][f"noisy|{nz}|{g}"]
            L.append(f"### {txt}, mismatch {g} (noisy reader)\n")
            L += _means_table(r)
            L.append("")
            L.append("| XGBoost minus | " + " | ".join(c[1] for c in CTRL_COLS) + " |")
            L.append("|---|" + "---|" * len(CTRL_COLS))
            for c in CONTROLS:
                dd = r["xgb_minus"][c]
                L.append(f"| {LABEL[c]} | " + " | ".join(
                    f"{_ci(dd[m], d, 1.0, signed=True)} ({'CI excl. 0' if dd[m]['excludes_zero'] else 'CI incl. 0'})"
                    for m, _, d in CTRL_COLS) + " |")
            L.append("")
    if C.get("sentences"):
        L.append(C["sentences"] + "\n")
    return L


def write_md(res: dict) -> None:
    S = res["summary"]
    L = ["# Learned card feedback in the closed loop (D29, evaluation seeds 0 to 19)\n",
         "Generated by `experiments/eval_feedback_model.py` from `results/feedback_model_closed_loop.json`. "
         "Do not edit by hand. Training and offline validation: `results/feedback_model.md`.\n",
         "Everything here is simulation. The feedback models were trained on this simulator's hidden mismatch "
         "and card noise (training seeds 20000 and up, validation 21000 to 21029) and are evaluated here on "
         "the same simulator with new seeds. This tests whether the learned correction helps the control loop "
         "inside the simulator. It says nothing about a real field; a real system would have to be trained on "
         "real verified cards.\n",
         f"Grid: {len(SEEDS)} seeds x {len(PRESETS)} presets x 4 mismatch levels x 2 readers x 2 nozzle modes x "
         f"{len(POLICIES)} policies = {res['n_passes']:,} passes, c_req {C_REQ:g} %, default lambda, "
         f"{res['wall_s'] / 60:.1f} min on {res['workers']} workers. All policies of a cell share the hidden "
         "realization (seed s + 4036). "
         f"Each metric is averaged over the {len(PRESETS)} presets within a seed; brackets are "
         f"95 % bootstrap CIs over the {len(SEEDS)} seeds (paired for differences). \"Corrected\" is the raw model "
         "prediction plus the feedback correction for the section, before its card. \"Floor\" is the mean "
         "|true - fair-oracle prediction| for the settings the policy chose: the error left by per-card spread "
         "alone, which no feedback can remove; for the oracle row it is the oracle's own prediction error. Gap "
         "recovery = (policy - open) / (oracle - open) on sections on target.\n"]
    L.append("## Headline: noisy reader, mismatch low + med + high pooled\n")
    for nz, txt in (("auto", "one nozzle per field (app default)"), ("free", "free nozzle choice")):
        r = S[f"noisy|{nz}|low+med+high"]
        L.append(f"### {txt}\n")
        L += _means_table(r)
        L.append("")
        L += _diff_table(r)
        L.append("")
    L.append("## No hidden mismatch (noisy reader)\n")
    L.append("With mismatch none the model's only error is its own fit error plus card noise, and the right "
             "correction is close to zero. A learned prior about hidden losses can only hurt here.\n")
    for nz in ("auto", "free"):
        r = S[f"noisy|{nz}|none"]
        L.append(f"### nozzle {nz}\n")
        L += _means_table(r)
        L.append("")
        L += _diff_table(r)
        L.append("")
    L.append("## Per mismatch level\n")
    for reader in READERS:
        for nz in ("auto", "free"):
            L.append(f"### reader {reader}, nozzle {nz}\n")
            for g in ("low", "med", "high", "all"):
                r = S.get(f"{reader}|{nz}|{g}")
                if r is None:
                    continue
                L.append(f"**mismatch {g}** (open-to-oracle gap {r.get('gap_pp', float('nan')):.1f} pp of sections)\n")
                L += _means_table(r)
                L.append("")
                L += _diff_table(r)
                L.append("")
    b0 = res.get("b0")
    L.append("## Check with the classic image reader (b0), mismatch med, seeds 0 to 4\n")
    if not b0 or not b0.get("summary"):
        L.append(f"Skipped: {b0.get('note') if b0 else 'not run'}.\n")
    else:
        L.append(f"{b0['n_passes']} passes, {b0['wall_s'] / 60:.1f} min. {b0.get('note', '')} "
                 f"{len(b0.get('seeds_complete', B0_SEEDS))} seeds only: wide CIs.\n")
        for key, r in b0["summary"].items():
            L.append(f"### {key.replace('|', ', ')}\n")
            L += _means_table(r)
            L.append("")
            L += _diff_table(r, pols=("xgboost",))
            L.append("")
    C = res.get("controls")
    if C:
        L += _controls_md(C)
    if res.get("adoption_check"):
        L += _adoption_md(res["adoption_check"])
    L.append("## Verdict\n")
    L.append(res.get("verdict", "(verdict written after the run)") + "\n")
    L.append("## What was not tested\n")
    L += ["- Real cards or real fields. The models learned the simulator's own hidden losses (canopy interception, "
          "nozzle wear, AR(1) gusts), which are of a few known kinds.",
          "- Other lambda values. Targets other than 20, 30 and 40 %. The adoption-check options (fast_slow, speed "
          "and boom-height control, the ar1 forecaster) were tested one at a time with the noisy reader and one "
          "nozzle per field only: not combined with each other, not with free nozzles, the perfect reader or the "
          "image readers, and not with other forecasters.",
          "- The image readers beyond the 5-seed b0 check (b1 not run). The training rows came from the perfect "
          "and noisy readers only.",
          "- Kinds of mismatch outside the training distribution (for example a stronger canopy loss than the "
          "'high' level, or a reader with a different bias). A trained model can fail there in ways the rule does not.",
          "- The two controls (rule + constant, pass mean + learned prior) were run only after the headline had "
          "been seen, on the evaluation seeds, with the noisy reader only (not perfect, not b0), and their constants "
          "were not tuned; they are a diagnostic, not a validated alternative. A pass-mean rule chosen on "
          "validation seeds and evaluated on fresh seeds would be needed before preferring it to the trained model.",
          "- The run time cost in the app: a learned correction is one tree-ensemble prediction per selection "
          "(pass times are in the json, `pass_s`)."]
    (RESULTS / "feedback_model_closed_loop.md").write_text("\n".join(L) + "\n", encoding="utf8")


def figures(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})
    col = {"open": "#7f7f7f", "rule": "#0072B2", "lightgbm": "#E69F00", "xgboost": "#009E73", "oracle": "#CC79A7"}
    S = res["summary"]
    panels = (("frac_met", "sections on target"), ("mean_volume_gpa", "volume (GPA)"),
              ("fb_err_true", "|true - corrected| (pp)"))
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for i, nz in enumerate(("auto", "free")):
        for j, (m, lab) in enumerate(panels):
            ax = axes[i, j]
            for pi, p in enumerate(POLICIES):
                if m == "fb_err_true" and p == "oracle":
                    vals = [S[f"noisy|{nz}|{g}"]["means"][p]["floor_err"] for g in MISMATCH]
                    lab_p = "floor (fair oracle)"
                else:
                    vals = [S[f"noisy|{nz}|{g}"]["means"][p][m] for g in MISMATCH]
                    lab_p = LABEL[p]
                xs = np.arange(4) + (pi - 2) * 0.07
                ax.errorbar(xs, [v["mean"] for v in vals],
                            yerr=[[v["mean"] - v["lo"] for v in vals], [v["hi"] - v["mean"] for v in vals]],
                            color=col[p], marker="o", ms=3.5, lw=1.3, capsize=2, label=lab_p)
            ax.set_xticks(range(4), MISMATCH)
            ax.set_title(f"{lab}, noisy reader, nozzle {nz}", fontsize=9)
            if i == 1:
                ax.set_xlabel("hidden mismatch level")
    axes[0, 0].legend(fontsize=7)
    axes[0, 2].legend(fontsize=7)
    fig.suptitle("Learned vs rule feedback, evaluation seeds 0-19 (mean and 95% CI over seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "feedback_model_closed_loop.png")
    plt.close(fig)

    dpan = (("frac_met", "sections on target"), ("mean_volume_gpa", "volume (GPA)"),
            ("fb_err_true", "|true - corrected| (pp)"), ("mean_drift_pct", "drift (%)"))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for j, (m, lab) in enumerate(dpan):
        ax = axes[j]
        k = 0
        for nz in ("auto", "free"):
            for p in LEARNED:
                vals = [S[f"noisy|{nz}|{g}"]["diff_vs_rule"][p][m] for g in MISMATCH]
                xs = np.arange(4) + (k - 1.5) * 0.12
                ax.errorbar(xs, [v["mean"] for v in vals],
                            yerr=[[v["mean"] - v["lo"] for v in vals], [v["hi"] - v["mean"] for v in vals]],
                            color=col[p], marker="o" if nz == "auto" else "s", ms=3.5, lw=0, elinewidth=1.3,
                            capsize=2, label=f"{p}, nozzle {nz}")
                k += 1
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(range(4), MISMATCH)
        ax.set_title(f"{lab}: learned - rule", fontsize=9)
        ax.set_xlabel("hidden mismatch level")
    axes[0].legend(fontsize=7)
    fig.suptitle("Paired difference to the rule, noisy reader (mean and 95% paired-bootstrap CI over 20 seeds)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "feedback_model_closed_loop_diff.png")
    plt.close(fig)


# ------------------------------------------------------------------ main

def run(workers: int, do_b0: bool, b0_cap_s: float) -> dict:
    t0 = time.time()
    cells = [(s, p, mm, rd, nz) for s in SEEDS for p in PRESETS for mm in MISMATCH for rd in READERS for nz in NOZZLE]
    rows: list[dict] = []
    b0 = {"note": "not requested"} if not do_b0 else {}
    with cf.ProcessPoolExecutor(workers, initializer=_init) as ex:
        futs = [ex.submit(cell, *c, tuple(POLICIES)) for c in cells]
        step = max(1, len(futs) // 20)
        for i, f in enumerate(cf.as_completed(futs), 1):
            rows += f.result()
            if i % step == 0 or i == len(futs):
                el = time.time() - t0
                print(f"[grid] {i}/{len(futs)} cells, {el:.0f}s, eta {el / i * (len(futs) - i):.0f}s", flush=True)
        wall = time.time() - t0
        if do_b0:
            t1 = time.time()
            bc = [(s, p, "med", "b0", nz) for s in B0_SEEDS for p in PRESETS for nz in NOZZLE]
            bf = [ex.submit(cell, *c, ("open", "rule", "xgboost", "oracle")) for c in bc]
            brows, done = [], 0
            try:
                for f in cf.as_completed(bf, timeout=b0_cap_s):
                    brows += f.result()
                    done += 1
                    if done % 6 == 0 or done == len(bf):
                        print(f"[b0] {done}/{len(bf)} cells, {time.time() - t1:.0f}s", flush=True)
            except cf.TimeoutError:
                ex.shutdown(wait=False, cancel_futures=True)
                print(f"[b0] cap of {b0_cap_s:.0f}s reached after {done}/{len(bf)} cells", flush=True)
            b0 = {"rows": brows, "n_passes": len(brows), "wall_s": time.time() - t1, "cells_done": done,
                  "cells_planned": len(bf)}
            if done < len(bf):
                b0["note"] = (f"only {done} of {len(bf)} cells finished within the {b0_cap_s / 60:.0f}-minute cap; "
                              "the partial result is not reported")
                b0["rows"] = []
    return {"rows": rows, "wall_s": wall, "workers": workers, "b0": b0}


def control_params() -> dict:
    """Constants of the two controls, from the cached TRAINING / VALIDATION rows only:
    offset = mean(XGBoost correction - rule correction) on val-B rows of the round-2 policy;
    prior = training mean of the target (card error) at the first section of a pass."""
    from coverageai import feedback_learned as fl
    from coverageai.config import spec
    z = np.load(ROWS_PATH)
    X, y, meta = z["X"], z["y"], z["meta"]
    i_rule, i_n = fl.FEATURE_NAMES.index("rule_bias"), fl.FEATURE_NAMES.index("n_cards")
    clip = float(spec()["feedback"]["bias_clip_pp"])
    vb = np.isin(meta[:, 0], VAL_B) & (meta[:, 3] == 2)
    m = fl.FeedbackModel.load(fl.MODEL_PATHS[1]["xgboost"])
    m.booster.set_param({"nthread": 26})
    px = np.clip(np.asarray(m.booster.inplace_predict(X[vb]), float), -clip, clip)
    offset = float(np.mean(px - X[vb][:, i_rule]))
    tr = (meta[:, 0] < 21000) & (X[:, i_n] == 0)
    prior = float(np.mean(y[tr]))
    out = {"offset": offset, "offset_rows": int(vb.sum()), "offset_source": "val-B (seeds 21020-21029), "
           "behaviour policy fast + round-1 XGBoost; mean of clipped XGBoost correction - rule correction",
           "prior": prior, "prior_rows": int(tr.sum()),
           "prior_source": "training rows (seeds 20000-20119), mean card error with zero earlier cards"}
    print(f"controls: offset {offset:+.4f} pp ({out['offset_rows']:,} val-B rows), "
          f"prior {prior:+.4f} pp ({out['prior_rows']:,} training rows)", flush=True)
    return out


def run_controls(workers: int, ctrl: dict) -> list[dict]:
    t0 = time.time()
    cells = [(s, p, mm, "noisy", nz) for s in SEEDS for p in PRESETS for mm in MISMATCH for nz in NOZZLE]
    rows: list[dict] = []
    with cf.ProcessPoolExecutor(workers, initializer=_init, initargs=(ctrl,)) as ex:
        futs = [ex.submit(cell, *c, CONTROLS) for c in cells]
        step = max(1, len(futs) // 10)
        for i, f in enumerate(cf.as_completed(futs), 1):
            rows += f.result()
            if i % step == 0 or i == len(futs):
                print(f"[controls] {i}/{len(futs)} cells, {time.time() - t0:.0f}s", flush=True)
    return rows


def control_summary(runs: list[dict], ctrl_rows: list[dict]) -> dict:
    """Noisy reader: means for every policy incl. the controls, and XGBoost - control paired
    differences, per nozzle mode, for mismatch low+med+high and none."""
    df = pd.DataFrame([r for r in runs if r["reader"] == "noisy"] + ctrl_rows)
    pols = ("open", "rule", "rule_shifted", "pass_mean", "xgboost", "oracle")
    sm = summarize(df, policies=pols)
    out = {}
    for nz in NOZZLE:
        for g in ("low+med+high", "none"):
            key = f"noisy|{nz}|{g}"
            r = sm[key]
            sl = seed_level(df[df["nozzle"] == nz], GROUPS[g])
            piv = {m: sl.pivot(index="seed", columns="policy", values=m).sort_index() for m in METRICS}
            r["xgb_minus"] = {}
            for c in CONTROLS:
                r["xgb_minus"][c] = {}
                for m in METRICS:
                    est, lo, hi = paired_bootstrap_mean_ci(piv[m]["xgboost"].to_numpy(), piv[m][c].to_numpy(), rng=0)
                    r["xgb_minus"][c][m] = {"mean": est, "lo": lo, "hi": hi,
                                            "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0))}
            out[key] = r
    return out


# ------------------------------------------------------------------ adoption check (post-hoc)
# name -> (loop policy, extra run_pass_wrapper kwargs, c_req); reader noisy, nozzle auto, default lam
ADOPT_VARIANTS = {
    "A_fast_slow": ("fast_slow", {}, 30.0),
    "B_speed": ("fast", {"speed_control": True}, 30.0),
    "C_boom": ("fast", {"boom_control": True}, 30.0),
    "D_forecast_ar1": ("fast", {"forecaster": "ar1"}, 30.0),
    "E_creq20": ("fast", {}, 20.0),
    "E_creq40": ("fast", {}, 40.0),
}
ADOPT_LABEL = {"baseline": "baseline: fast, c_req 30 (headline passes)", "A_fast_slow": "A: fast_slow (slow loop)",
               "B_speed": "B: fast + speed control", "C_boom": "C: fast + boom-height control",
               "D_forecast_ar1": "D: fast + wind forecaster ar1", "E_creq20": "E: fast, c_req 20",
               "E_creq40": "E: fast, c_req 40"}
ADOPT_METRICS = ("frac_met", "mean_shortfall_pp", "mean_volume_gpa", "mean_drift_pct", "fb_err_true", "mean_corr",
                 "slow_corr", "total_corr", "pass_s", "floor_err", "fb_bias_true")
ADOPT_GROUPS = ("low+med+high", "none")


def adopt_cell(seed: int, preset: str, mismatch: str, variant: str) -> list[dict]:
    """Rule and XGBoost feedback on one cell of one adoption variant (noisy reader, nozzle auto)."""
    al = _W["al"]
    mods = _mods("noisy")
    pol, kw, c_req = ADOPT_VARIANTS[variant]
    conds = al.make_conditions(preset, seed)
    E = seed + EVAL_OFFSET
    rows = []
    for name, fbm in (("rule", None), ("xgboost", _W["fb"]["xgboost"])):
        t0 = time.perf_counter()
        res = al.run_pass_wrapper(mods, conds, pol, E, mismatch, c_req, None, fixed_nozzle="auto",
                                  feedback_model=fbm, same_field=True, **kw)
        dt = time.perf_counter() - t0
        m = pass_metrics(res.logs, c_req)
        m["floor_err"] = float(floor_errors(res.modules, res.logs, res.seed, mismatch).mean())
        m["slow_corr"] = float(np.mean([lg.extra.get("slow_corr", 0.0) for lg in res.logs]))
        m["total_corr"] = m["mean_corr"] + m["slow_corr"]
        rows.append({"seed": seed, "preset": preset, "mismatch": mismatch, "variant": variant, "policy": name,
                     "displayed_seed": int(res.seed), "c_req": c_req, "pass_s": dt, **m})
    return rows


def run_adoption(workers: int) -> tuple[list[dict], float]:
    t0 = time.time()
    tasks = [(s, p, mm, v) for v in ADOPT_VARIANTS for s in SEEDS for p in PRESETS for mm in MISMATCH]
    rows: list[dict] = []
    with cf.ProcessPoolExecutor(workers, initializer=_init) as ex:
        futs = [ex.submit(adopt_cell, *t) for t in tasks]
        step = max(1, len(futs) // 20)
        for i, f in enumerate(cf.as_completed(futs), 1):
            rows += f.result()
            if i % step == 0 or i == len(futs):
                el = time.time() - t0
                print(f"[adoption] {i}/{len(futs)} cells, {el:.0f}s, eta {el / i * (len(futs) - i):.0f}s", flush=True)
    return rows, time.time() - t0


def _seed_means(df: pd.DataFrame, levels) -> pd.DataFrame:
    return (df[df["mismatch"].isin(levels)].groupby("seed")[list(ADOPT_METRICS)].mean().sort_index())


def _pdiff(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    out = {}
    for m in ADOPT_METRICS:
        est, lo, hi = paired_bootstrap_mean_ci(a[m].to_numpy(), b[m].to_numpy(), rng=0)
        out[m] = {"mean": est, "lo": lo, "hi": hi, "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0))}
    return out


def adoption_summary(runs: list[dict], arows: list[dict]) -> dict:
    """Per variant and mismatch group: seed-level means (bootstrap CI) of rule and XGBoost, XGBoost - rule; for A
    also fast_slow vs plain fast (the headline c_req-30 passes, noisy reader, nozzle auto)."""
    base = pd.DataFrame([r for r in runs if r["reader"] == "noisy" and r["nozzle"] == "auto"
                         and r["policy"] in ("rule", "xgboost")])
    base = base.assign(variant="baseline", slow_corr=0.0, total_corr=base["mean_corr"])
    df = pd.concat([base, pd.DataFrame(arows)], ignore_index=True)
    out = {}
    for v in ("baseline",) + tuple(ADOPT_VARIANTS):
        out[v] = {}
        for g in ADOPT_GROUPS:
            sm = {pol: _seed_means(df[(df["variant"] == v) & (df["policy"] == pol)], GROUPS[g])
                  for pol in ("rule", "xgboost")}
            r = {"n_seeds": int(len(sm["rule"])),
                 "means": {pol: {m: dict(zip(("mean", "lo", "hi"), bootstrap_mean_ci(sm[pol][m].to_numpy(), rng=0)))
                                 for m in ADOPT_METRICS} for pol in sm},
                 "xgb_minus_rule": _pdiff(sm["xgboost"], sm["rule"])}
            if v == "A_fast_slow":
                bs = {pol: _seed_means(df[(df["variant"] == "baseline") & (df["policy"] == pol)], GROUPS[g])
                      for pol in ("rule", "xgboost")}
                r["fast_slow_xgb_minus_fast_xgb"] = _pdiff(sm["xgboost"], bs["xgboost"])
                r["fast_slow_rule_minus_fast_rule"] = _pdiff(sm["rule"], bs["rule"])
            out[v][g] = r
    return out


ADOPT_COLS = (("frac_met", "on target", 3), ("mean_shortfall_pp", "shortfall pp", 2),
              ("mean_volume_gpa", "volume GPA", 2), ("mean_drift_pct", "drift %", 3),
              ("fb_err_true", "|true - corrected| pp", 2), ("mean_corr", "fast correction pp", 2),
              ("total_corr", "total correction pp", 2), ("pass_s", "s / pass", 2))


def _adoption_md(A: dict) -> list[str]:
    L = ["## Adoption check: other app options (post-hoc, evaluation seeds, frozen models)\n",
         "Run after the owner decided to make the trained feedback the app default, to check the option "
         "combinations the earlier sections did not cover. Post-hoc on the evaluation seeds with the frozen models; "
         "nothing was changed because of it. Grid per variant: seeds 0 to 19, hidden seed s + 4036, 6 presets, "
         "mismatch none / low / med / high, noisy reader, one nozzle per field (`fixed_nozzle=\"auto\"`, the app "
         "default), default lambda, `same_field=True`; each cell runs the rule (`feedback_model=None`) and the "
         f"XGBoost model. {A['n_passes']:,} new passes in {A['wall_s'] / 60:.1f} min on {A['workers']} workers. "
         "The baseline row reuses the headline `fast` passes at c_req 30 (not re-run). For fast_slow, `n_learn` is "
         "the default 3, the learning passes use the same feedback as the displayed pass, \"fast correction\" is the "
         "fast loop's term and \"total correction\" adds the slow loop's offset (for the other variants the two are "
         "equal). Seconds per pass were measured on a loaded machine (26 workers) and include the fast_slow learning "
         "passes; the baseline's come from the headline run. Brackets: 95 % bootstrap CIs over 20 seeds (paired for "
         "differences); 'excl. 0' marks a CI that excludes zero.\n"]
    for v in ("baseline",) + tuple(ADOPT_VARIANTS):
        L.append(f"### {ADOPT_LABEL[v]}\n")
        for g in ADOPT_GROUPS:
            r = A["summary"][v][g]
            L.append(f"**mismatch {g}**\n")
            L.append("| row | " + " | ".join(c[1] for c in ADOPT_COLS) + " |")
            L.append("|---|" + "---|" * len(ADOPT_COLS))
            for pol, lab in (("rule", "rule (eq. 3)"), ("xgboost", "XGBoost feedback")):
                mm = r["means"][pol]
                L.append(f"| {lab} | " + " | ".join(_ci(mm[m], d, 1.0, signed=(m.endswith("corr")))
                                                   for m, _, d in ADOPT_COLS) + " |")
            diffs = [("XGBoost - rule", r["xgb_minus_rule"])]
            if "fast_slow_xgb_minus_fast_xgb" in r:
                diffs += [("fast_slow + XGBoost - fast + XGBoost", r["fast_slow_xgb_minus_fast_xgb"]),
                          ("fast_slow + rule - fast + rule", r["fast_slow_rule_minus_fast_rule"])]
            for lab, dd in diffs:
                L.append(f"| {lab} | " + " | ".join(
                    f"{_ci(dd[m], d + (1 if m == 'frac_met' else 0), 1.0, signed=True)}"
                    f"{' (excl. 0)' if dd[m]['excludes_zero'] else ''}" for m, _, d in ADOPT_COLS) + " |")
            L.append("")
    if A.get("sentences"):
        L.append(A["sentences"] + "\n")
    return L


B0_ONLY_POLICIES = ("rule", "xgboost")


def run_b0_only(workers: int, cap_s: float) -> dict:
    """The reduced b0 check (rule vs XGBoost, nozzle auto, mismatch med, seeds B0_SEEDS). On the
    cap, only seeds whose 6 presets all finished are kept (so the pairing and the preset average
    stay intact)."""
    t1 = time.time()
    bc = [(s, p, "med", "b0", "auto") for s in B0_SEEDS for p in PRESETS]
    brows, done = [], 0
    ex = cf.ProcessPoolExecutor(workers, initializer=_init)
    try:
        bf = [ex.submit(cell, *c, B0_ONLY_POLICIES) for c in bc]
        try:
            for f in cf.as_completed(bf, timeout=cap_s):
                brows += f.result()
                done += 1
                print(f"[b0] {done}/{len(bf)} cells, {time.time() - t1:.0f}s", flush=True)
        except cf.TimeoutError:
            print(f"[b0] cap of {cap_s:.0f}s reached after {done}/{len(bf)} cells", flush=True)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    df = pd.DataFrame(brows)
    full = [s for s, g in df.groupby("seed") if len(g) == len(PRESETS) * len(B0_ONLY_POLICIES)] if len(df) else []
    rows = [r for r in brows if r["seed"] in full]
    return {"rows": rows, "n_passes": len(rows), "wall_s": time.time() - t1, "cells_done": done,
            "cells_planned": len(bc), "seeds_complete": [int(s) for s in full],
            "policies": list(B0_ONLY_POLICIES), "nozzle": "auto"}

# ================================================================== v2 evaluation (calibrated reader, D30)
V2_JSON = RESULTS / "feedback_model_v2_closed_loop.json"
V2_MD = RESULTS / "feedback_model_v2_closed_loop.md"
FRESH_SEEDS = tuple(range(100, 120))       # never used anywhere in the project before this run
DESIGN_SEEDS = tuple(range(20))            # seeds that informed the design (adoption check -> D30, v2)
V2_B0_SEEDS = tuple(range(100, 105))
# name -> (loop policy, feedback model key, calibrated reader)
STRATS = {"open": ("open", None, False), "rule_raw": ("fast", None, False), "rule_cal": ("fast", None, True),
          "v1x_raw": ("fast", "v1x", False), "v1x_cal": ("fast", "v1x", True), "v2x_cal": ("fast", "v2x", True),
          "v2l_cal": ("fast", "v2l", True), "oracle": ("oracle", None, False), "v2x_raw": ("fast", "v2x", False)}
S_LABEL = {"v2x_raw": "v2 XGBoost, raw reader", "open": "open loop", "rule_raw": "rule, raw reader (old default)", "rule_cal": "rule, calibrated reader",
           "v1x_raw": "v1 XGBoost, raw reader", "v1x_cal": "v1 XGBoost, calibrated reader",
           "v2x_cal": "v2 XGBoost, calibrated reader (new default)", "v2l_cal": "v2 LightGBM, calibrated reader",
           "oracle": "fair oracle"}
MAIN5 = ("open", "rule_raw", "rule_cal", "v1x_cal", "v2x_cal", "oracle")


def _blk(name, seeds, reader, nozzle, c_req, strats, kwargs=None, policy=None, mismatch=MISMATCH, cal_mode=None):
    """``cal_mode`` replaces the calibration of the strategies that use a calibrated reader: None = True
    (the default table: the noisy reader's own bands, an image reader's loop-measured table), or
    "synthetic" (an image reader's synthetic-card table, kept to reproduce the first b0 check)."""
    return {"name": name, "seeds": tuple(seeds), "reader": reader, "nozzle": nozzle, "c_req": float(c_req),
            "strats": tuple(strats), "kwargs": dict(kwargs or {}), "policy": policy, "mismatch": tuple(mismatch),
            "cal_mode": cal_mode}


V2_BLOCKS = [
    _blk("main_c30", FRESH_SEEDS, "noisy", "auto", 30, tuple(STRATS)),
    _blk("main_c20", FRESH_SEEDS, "noisy", "auto", 20, MAIN5),
    _blk("main_c40", FRESH_SEEDS, "noisy", "auto", 40, MAIN5),
    _blk("free_c30", FRESH_SEEDS, "noisy", "free", 30, ("open", "rule_raw", "rule_cal", "v2x_cal", "oracle")),
    _blk("var_speed", FRESH_SEEDS, "noisy", "auto", 30, ("rule_cal", "v2x_cal"), {"speed_control": True}),
    _blk("var_boom", FRESH_SEEDS, "noisy", "auto", 30, ("rule_cal", "v2x_cal"), {"boom_control": True}),
    _blk("var_ar1", FRESH_SEEDS, "noisy", "auto", 30, ("rule_cal", "v2x_cal"), {"forecaster": "ar1"}),
    _blk("var_fast_slow", FRESH_SEEDS, "noisy", "auto", 30, ("rule_cal", "v2x_cal"), policy="fast_slow"),
    _blk("perfect_c30", FRESH_SEEDS, "perfect", "auto", 30, ("open", "rule_raw", "v2x_cal", "oracle")),
    _blk("perfect_c40", FRESH_SEEDS, "perfect", "auto", 40, ("open", "rule_raw", "v2x_cal", "oracle")),
    _blk("design_c20", DESIGN_SEEDS, "noisy", "auto", 20, MAIN5),
    _blk("design_c30", DESIGN_SEEDS, "noisy", "auto", 30, MAIN5),
    _blk("design_c40", DESIGN_SEEDS, "noisy", "auto", 40, MAIN5),
]
V2_B0_BLOCKS = [_blk(f"b0_c{c}", V2_B0_SEEDS, "b0", "auto", c, ("rule_raw", "rule_cal", "v2x_cal"), mismatch=("med",),
                     cal_mode="synthetic") for c in (30, 40)]
V2_B0_LOOP_SEEDS = tuple(range(105, 110))     # not used with b0 before
V2_B0_LOOP_BLOCKS = [_blk(f"b0loop_c{c}", V2_B0_LOOP_SEEDS, "b0", "auto", c, ("rule_raw", "rule_cal", "v2x_raw", "v2x_cal"),
                          mismatch=("med",), cal_mode=True) for c in (30, 40)]
V2_METRICS = ("frac_met", "mean_shortfall_pp", "mean_volume_gpa", "mean_drift_pct", "fb_err_true", "fb_bias_true",
              "mean_corr", "slow_corr", "total_corr", "card_bias", "floor_err", "pass_s")
V2_COMPS = (("v2x_cal", "rule_raw", "new default - old default"), ("v2x_cal", "rule_cal", "v2 - rule (both calibrated)"),
            ("v2x_cal", "v1x_cal", "v2 - v1 (both calibrated)"), ("rule_cal", "rule_raw", "calibration alone (rule)"),
            ("v1x_raw", "rule_raw", "v1 - rule (both raw, earlier headline)"),
            ("v2l_cal", "v2x_cal", "v2 LightGBM - v2 XGBoost"),
            ("v2x_raw", "rule_raw", "v2 - rule (both raw reader)"),
            ("v2x_cal", "v2x_raw", "v2: calibrated - raw reader"))
_WV2: dict = {}
V2_MODEL_FILES = {"v2x": _fl_top.MODEL_PATHS[2]["xgboost"], "v2l": _fl_top.MODEL_PATHS[2]["lightgbm"]}


def _init_v2() -> None:
    from coverageai import app_logic as al
    from coverageai import feedback_learned as fl
    _WV2["al"] = al
    _WV2["stacks"] = {}
    _WV2["fb"] = {"v1x": fl.FeedbackModel.load(fl.MODEL_PATHS[1]["xgboost"]),
                  "v2x": fl.FeedbackModel.load(V2_MODEL_FILES["v2x"]),
                  "v2l": fl.FeedbackModel.load(V2_MODEL_FILES["v2l"])}


def _stack(reader: str, cal):
    key = (reader, cal)
    if key not in _WV2["stacks"]:
        _WV2["stacks"][key] = _WV2["al"].build_modules_adapter("spray", reader, calibrate_reader=cal)
    return _WV2["stacks"][key]


def v2_cell(blk: dict, seed: int, preset: str, mismatch: str) -> list[dict]:
    al = _WV2["al"]
    conds = al.make_conditions(preset, seed)
    E = seed + EVAL_OFFSET
    rows = []
    for name in blk["strats"]:
        pol, fbk, cal = STRATS[name]
        if cal and blk.get("cal_mode") is not None:
            cal = blk["cal_mode"]
        if blk["policy"] and pol == "fast":
            pol = blk["policy"]
        mods = _stack(blk["reader"], cal)
        t0 = time.perf_counter()
        if pol == "fast_slow" and fbk is not None:
            # the library routes fast_slow to the rule (TRAINED_FEEDBACK_POLICIES, D31); to measure what the
            # trained model does there, replay the wrapper's fast_slow path with the model plugged in
            logs, run_seed = _fast_slow_direct(mods, conds, E, mismatch, blk["c_req"], _WV2["fb"][fbk])
            sim_mods, how = mods, "direct"
        else:
            res = al.run_pass_wrapper(mods, conds, pol, E, mismatch, blk["c_req"], None,
                                      fixed_nozzle=NOZZLE[blk["nozzle"]],
                                      feedback_model=_WV2["fb"][fbk] if fbk else None,
                                      same_field=True, **blk["kwargs"])
            logs, run_seed, sim_mods, how = res.logs, res.seed, res.modules, res.meta.get("feedback", {}).get("kind")
        dt = time.perf_counter() - t0
        m = pass_metrics(logs, blk["c_req"])
        m["floor_err"] = float(floor_errors(sim_mods, logs, run_seed, mismatch).mean())
        m["slow_corr"] = float(np.mean([lg.extra.get("slow_corr", 0.0) for lg in logs]))
        m["total_corr"] = m["mean_corr"] + m["slow_corr"]
        m["card_bias"] = float(np.mean([lg.meas.coverage_pct - lg.outcome.coverage_pct for lg in logs]))
        rows.append({"block": blk["name"], "seed": seed, "preset": preset, "mismatch": mismatch, "strategy": name,
                     "feedback_kind": how, "pass_s": dt, **m})
    return rows


def _fast_slow_direct(mods, conds, E: int, mismatch: str, c_req: float, fb_model):
    """fast_slow exactly as ``run_pass_wrapper(policy="fast_slow", fixed_nozzle="auto", same_field=True)``
    runs it (nozzle chosen by ``best_nozzle``, ``n_learn`` learning passes through the wrapper's own
    ``_learn_between_passes``, fast loop reset, displayed pass on seed E), but with the trained model
    as the fast loop in the learning passes and the displayed pass. ``fb_model`` None = the rule. The
    shared selector's nozzle lock is restored in ``finally``; ``mods.model`` is never mutated (the
    slow loop works on a private copy of the Modules record)."""
    import dataclasses
    from coverageai.feedback_learned import LearnedFeedback
    from coverageai.loop import run_pass
    al = _WV2["al"]
    learned = LearnedFeedback(fb_model) if fb_model is not None else None
    sel = mods.selector
    old_lock = sel.nozzle_lock
    m2 = dataclasses.replace(mods)
    try:
        lock, _ = sel.best_nozzle(list(conds), mods.model, float(c_req))
        sel.nozzle_lock = lock
        run_seed, _ = al._learn_between_passes(m2, list(conds), int(E), mismatch, float(c_req), al.N_LEARN_PASSES,
                                               same_field=True, feedback=learned)
        if learned is not None:
            learned.reset()
        if hasattr(m2.simulator, "begin_pass"):
            m2.simulator.begin_pass(run_seed, mismatch)
        logs = run_pass(list(conds), "fast_slow", m2, run_seed, mismatch, float(c_req), f"fast_slow_s{run_seed}",
                        render=False, feedback=learned)
    finally:
        sel.nozzle_lock = old_lock
    return logs, run_seed


def run_v2(blocks, workers: int, cap_s: float | None = None) -> tuple[list[dict], float, int, int]:
    t0 = time.time()
    tasks = [(b, s, p, mm) for b in blocks for s in b["seeds"] for p in PRESETS for mm in b["mismatch"]]
    tasks.sort(key=lambda t: t[1])          # seed-major: under a time cap, whole seeds finish first
    rows: list[dict] = []
    done = 0
    ex = cf.ProcessPoolExecutor(workers, initializer=_init_v2)
    try:
        futs = [ex.submit(v2_cell, *t) for t in tasks]
        step = max(1, len(futs) // 25)
        try:
            for f in cf.as_completed(futs, timeout=cap_s):
                rows += f.result()
                done += 1
                if done % step == 0 or done == len(futs):
                    el = time.time() - t0
                    print(f"[v2] {done}/{len(futs)} cells, {el:.0f}s, eta {el / done * (len(futs) - done):.0f}s",
                          flush=True)
        except cf.TimeoutError:
            print(f"[v2] cap of {cap_s:.0f}s reached after {done}/{len(futs)} cells", flush=True)
    finally:
        ex.shutdown(wait=cap_s is None, cancel_futures=True)
    return rows, time.time() - t0, done, len(tasks)


def _sl(df: pd.DataFrame, strat: str, levels) -> pd.DataFrame:
    sub = df[(df["strategy"] == strat) & df["mismatch"].isin(levels)]
    return sub.groupby("seed")[list(V2_METRICS)].mean().sort_index()


def _pd(a: pd.DataFrame, b: pd.DataFrame, metrics=V2_METRICS) -> dict:
    out = {}
    for m in metrics:
        est, lo, hi = paired_bootstrap_mean_ci(a[m].to_numpy(), b[m].to_numpy(), rng=0)
        out[m] = {"mean": est, "lo": lo, "hi": hi, "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0))}
    return out


def v2_summary(df: pd.DataFrame, blocks) -> dict:
    out = {}
    for b in blocks:
        d = df[df["block"] == b["name"]]
        if d.empty:
            continue
        groups = {"low+med+high": ["low", "med", "high"], "none": ["none"]} if len(b["mismatch"]) > 1 else \
            {b["mismatch"][0]: list(b["mismatch"])}
        out[b["name"]] = {}
        for g, levels in groups.items():
            sl = {st: _sl(d, st, levels) for st in b["strats"]}
            r = {"n_seeds": int(len(next(iter(sl.values())))), "means": {}, "comps": {}, "vs_oracle": {},
                 "gap_recovery": {}}
            for st, t in sl.items():
                r["means"][st] = {m: dict(zip(("mean", "lo", "hi"), bootstrap_mean_ci(t[m].to_numpy(), rng=0)))
                                  for m in V2_METRICS}
            for a, c, lab in V2_COMPS + (("v2x_cal", "rule_cal", "v2 - rule (both calibrated)"),):
                if a in sl and c in sl and f"{a}-{c}" not in r["comps"]:
                    r["comps"][f"{a}-{c}"] = {"label": lab, **_pd(sl[a], sl[c])}
            if "v2x_cal" in sl and "rule_raw" in sl and b["reader"] == "perfect":
                r["comps"]["v2x_cal-rule_raw"]["label"] = "v2 - rule (perfect reader; calibration has no effect)"
            if "oracle" in sl:
                for st in sl:
                    if st != "oracle":
                        r["vs_oracle"][st] = _pd(sl[st], sl["oracle"], ("mean_volume_gpa", "mean_drift_pct"))
            if "open" in sl and "oracle" in sl:
                o, ref = sl["open"]["frac_met"].to_numpy(), sl["oracle"]["frac_met"].to_numpy()
                r["gap_pp"] = float(100 * (ref.mean() - o.mean()))
                for st in sl:
                    if st not in ("open", "oracle"):
                        r["gap_recovery"][st] = dict(zip(("mean", "lo", "hi"),
                                                         _boot(_gap, [o, sl[st]["frac_met"].to_numpy(), ref])))
            if b["name"] == "var_fast_slow":
                m30 = df[df["block"] == "main_c30"]
                for st in ("rule_cal", "v2x_cal"):
                    r["comps"][f"fast_slow_{st}-fast_{st}"] = {"label": f"fast_slow - fast ({S_LABEL[st]})",
                                                               **_pd(sl[st], _sl(m30, st, levels))}
            out[b["name"]][g] = r
    return out


V2_COLS = (("frac_met", "on target", 3), ("mean_shortfall_pp", "shortfall pp", 2), ("mean_volume_gpa", "volume GPA", 2),
           ("mean_drift_pct", "drift %", 3), ("fb_err_true", "abs err pp", 2), ("fb_bias_true", "signed err pp", 2),
           ("total_corr", "correction pp", 2), ("card_bias", "card - true pp", 2), ("pass_s", "s/pass", 2))


def _v2_means(r: dict, labels: dict | None = None) -> list[str]:
    lab_ = {**S_LABEL, **(labels or {})}
    L = ["| strategy | " + " | ".join(c[1] for c in V2_COLS) + " | gap recovery |",
         "|---|" + "---|" * (len(V2_COLS) + 1)]
    for st, mm in r["means"].items():
        cells = []
        for m, _, d in V2_COLS:
            if st in ("oracle", "open") and m in ("fb_err_true", "fb_bias_true", "total_corr"):
                cells.append("-")
            else:
                cells.append(_ci(mm[m], d, 1.0, signed=m in ("fb_bias_true", "total_corr", "card_bias")))
        gr = r["gap_recovery"].get(st)
        g = "-" if gr is None else (f"{100 * gr['mean']:.0f}% [{100 * gr['lo']:.0f}, {100 * gr['hi']:.0f}]"
                                    if np.isfinite(gr["lo"]) else "n/a")
        L.append(f"| {lab_[st]} | " + " | ".join(cells) + f" | {g} |")
    return L


CMP_COLS = (("frac_met", "on target", 4), ("mean_shortfall_pp", "shortfall pp", 2),
            ("mean_volume_gpa", "volume GPA", 2), ("mean_drift_pct", "drift %", 3),
            ("fb_err_true", "abs err pp", 2), ("fb_bias_true", "signed err pp", 2), ("pass_s", "s/pass", 2))


def _v2_comps(r: dict, keys=None, comp_labels: dict | None = None) -> list[str]:
    L = ["| comparison | " + " | ".join(c[1] for c in CMP_COLS) + " |", "|---|" + "---|" * len(CMP_COLS)]
    for k, dd in r["comps"].items():
        if keys is not None and k not in keys:
            continue
        L.append(f"| {(comp_labels or {}).get(k, dd['label'])} | " + " | ".join(
            f"{_ci(dd[m], d, 1.0, signed=True)}{' *' if dd[m]['excludes_zero'] else ''}" for m, _, d in CMP_COLS) + " |")
    return L


def _v2_vs_oracle(r: dict) -> list[str]:
    if not r["vs_oracle"]:
        return []
    L = ["| strategy - fair oracle | volume GPA | drift % |", "|---|---|---|"]
    for st, dd in r["vs_oracle"].items():
        L.append(f"| {S_LABEL[st]} | " + " | ".join(
            f"{_ci(dd[m], d, 1.0, signed=True)}{' *' if dd[m]['excludes_zero'] else ''}"
            for m, d in (("mean_volume_gpa", 2), ("mean_drift_pct", 3))) + " |")
    return L


def _v2_block_md(S: dict, name: str, title: str, vs_oracle: bool = True, labels: dict | None = None,
                 comp_labels: dict | None = None) -> list[str]:
    L = [f"### {title}\n"]
    for g, r in S[name].items():
        L.append(f"**mismatch {g}**" + (f" (open-to-oracle gap {r['gap_pp']:.1f} pp of sections)" if "gap_pp" in r else "")
                 + "\n")
        L += _v2_means(r, labels)
        L.append("")
        if r["comps"]:
            L += _v2_comps(r, comp_labels=comp_labels)
            L.append("")
        if vs_oracle:
            vo = _v2_vs_oracle(r)
            if vo:
                L += vo
                L.append("")
    return L


def write_v2_md(res: dict) -> None:
    S = res["summary"]
    L = ["# Learned card feedback v2 with reader calibration: closed loop on fresh seeds (D29, D30)\n",
         "Generated by `experiments/eval_feedback_model.py --v2` from `results/feedback_model_v2_closed_loop.json`. "
         "Do not edit by hand. Training of v2 and its on-policy check: `results/feedback_model_v2.md`. The earlier v1 "
         "evaluation: `results/feedback_model_closed_loop.md`.\n",
         "Everything here is simulation. The feedback models learned this simulator's hidden mismatch and card noise. "
         "The reader calibration subtracts the reader's mean error per coverage band, measured in the reader's own "
         "validation on synthetic cards with known truth; a real reader would need annotated real cards, and a "
         "real feedback model real verified cards.\n",
         "Protocol. Fresh scenario seeds 100 to 119, never used before in this project, run once after v2 was frozen; "
         "hidden-mismatch seed s + 4036 for every strategy (paired). 6 presets, mismatch none / low / med / high, one "
         "nozzle per field (`fixed_nozzle=\"auto\"`) unless stated, noisy reader unless stated, default lambda. Metrics "
         "use the true outcome. Each metric is averaged over the presets within a seed; brackets are 95 % bootstrap "
         "CIs over the 20 seeds, paired for differences; `*` marks a difference whose CI excludes zero. abs err = mean "
         "|true - model - correction|; signed err = mean (true - model - correction), positive = the loop expected "
         "less coverage than it got (over-applied correction); card - true = mean reading error of the reader in use. "
         "Gap recovery = (strategy - open) / (oracle - open) on sections on target. Seconds per pass on a loaded "
         f"machine ({res['workers']} workers). {res['n_passes']:,} passes in {res['wall_s'] / 60:.1f} min.\n"]
    L.append("## Headline: new default (v2, calibrated) against old default (rule, raw), fresh seeds, c_req 30\n")
    L += _v2_block_md(S, "main_c30", "c_req 30, one nozzle per field, noisy reader")
    L.append("## Per target (fresh seeds)\n")
    for c in (20, 40):
        L += _v2_block_md(S, f"main_c{c}", f"c_req {c}, one nozzle per field, noisy reader")
    L.append("## Free nozzle choice, c_req 30 (fresh seeds)\n")
    L += _v2_block_md(S, "free_c30", "free nozzle, noisy reader")
    L.append("## App options, c_req 30, calibrated reader: rule against v2 XGBoost (fresh seeds)\n")
    for name, title in (("var_speed", "speed control"), ("var_boom", "boom-height control"),
                        ("var_ar1", "wind forecaster ar1"), ("var_fast_slow", "fast_slow (slow loop, n_learn 3)")):
        L += _v2_block_md(S, name, title, vs_oracle=False)
    L.append("## Perfect reader (fresh seeds): calibration has no effect, so this shows the model is not relying on "
             "reader bias\n")
    for c in (30, 40):
        L += _v2_block_md(S, f"perfect_c{c}", f"perfect reader, c_req {c}")
    L.append("## Seeds that informed the design (0 to 19)\n")
    L.append("These seeds were used for the v1 evaluation and the adoption check that led to the reader calibration "
             "and to v2. They cannot confirm v2; they are shown for continuity with the earlier results only.\n")
    for c in (20, 30, 40):
        L += _v2_block_md(S, f"design_c{c}", f"seeds 0-19, c_req {c}, one nozzle per field, noisy reader")
    b0 = res.get("b0")
    syn_lab = {"rule_cal": "rule, synthetic-card table", "v2x_cal": "v2 XGBoost, synthetic-card table"}
    syn_cmp = {"v2x_cal-rule_raw": "v2 (synthetic-card table) - rule raw",
               "v2x_cal-rule_cal": "v2 - rule (both synthetic-card table)",
               "rule_cal-rule_raw": "synthetic-card table alone (rule)"}
    L.append("## Classic image reader b0 with the synthetic-card table (seeds 100 to 104, mismatch med)\n")
    if not b0 or not b0.get("summary"):
        L.append(f"{(b0 or {}).get('note', 'Not run.')}\n")
    else:
        L.append(f"{b0['n_passes']} passes in {b0['wall_s'] / 60:.1f} min, seeds {b0['seeds_complete']}. "
                 f"{b0.get('note', '')} Five seeds: wide CIs. The calibrated strategies here use the table measured "
                 "on the reader's synthetic validation cards (`results/b0_synthetic.json`; reproduce with "
                 "`calibrate_reader=\"synthetic\"`). 'card - true' is the mean reading error of the image reader in "
                 "the loop: raw for the rule with the raw reader, after the synthetic-card table for the other two.\n")
        for c in (30, 40):
            if f"b0_c{c}" in b0["summary"]:
                L += _v2_block_md(b0["summary"], f"b0_c{c}", f"b0, c_req {c}, synthetic-card table", vs_oracle=False,
                                  labels=syn_lab, comp_labels=syn_cmp)
    bl = res.get("b0_loop")
    L.append("## Classic image reader b0 with the loop-measured table (seeds 105 to 109)\n")
    if not bl or not bl.get("summary"):
        L.append("Not run.\n")
    else:
        L.append(f"Mismatch med, one nozzle per field, {bl['n_passes']} passes in {bl['wall_s'] / 60:.1f} min, seeds "
                 f"{bl['seeds_complete']} (not used with b0 before). The calibrated strategies use the table measured "
                 "on closed-loop cards of development seeds 20000-20005 (`results/b0_loop_calibration.json`, "
                 "`experiments/calibrate_reader_loop.py`). Five seeds: wide CIs. 'card - true' is the mean reading "
                 "error of the reader in use, after calibration for the calibrated strategies.\n")
        for c in (30, 40):
            if f"b0loop_c{c}" in bl["summary"]:
                L += _v2_block_md(bl["summary"], f"b0loop_c{c}", f"b0, c_req {c}, loop-measured table", vs_oracle=False,
                                  labels={"rule_cal": "rule, loop table", "v2x_cal": "v2 XGBoost, loop table (new default)"},
                                  comp_labels={"v2x_cal-rule_cal": "v2 - rule (both loop table)",
                                               "rule_cal-rule_raw": "loop table alone (rule)",
                                               "v2x_cal-v2x_raw": "v2: loop table - raw reader"})
    L.append("## Verdict\n")
    L.append(res.get("verdict", "(verdict written after the run)") + "\n")
    L.append("## What was not tested\n")
    L += res.get("not_tested") or [
        "- Real cards, real readers and real fields. The calibration table and the hidden mismatch are the "
        "simulator's own; the noisy reader follows the table exactly, so calibration works perfectly by construction "
        "for it. The b0 check is the only test against a reader that does not follow the table by construction.",
        "- A calibration table that is wrong or out of date (reader bias that drifts with lighting, card batch or "
        "droplet size).",
        "- The b1 (U-Net) reader; other lambda values; targets other than 20, 30 and 40 %; the options combined with "
        "each other or with free nozzles; other forecasters.",
        "- Free nozzle at c_req 20 and 40; the v1 models at c_req 20/40 with the raw reader on fresh seeds."]
    V2_MD.write_text("\n".join(L) + "\n", encoding="utf8")


def v2_figure(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})
    S = res["summary"]
    keys = (("v2x_cal-rule_raw", "new - old default", "#009E73"), ("v2x_cal-rule_cal", "v2 - rule (calibrated)", "#0072B2"),
            ("v2x_cal-v1x_cal", "v2 - v1 (calibrated)", "#E69F00"), ("rule_cal-rule_raw", "calibration alone", "#CC79A7"))
    panels = (("frac_met", "sections on target"), ("mean_volume_gpa", "volume (GPA)"), ("mean_drift_pct", "drift (%)"),
              ("fb_bias_true", "signed error after correction (pp)"))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for j, (m, lab) in enumerate(panels):
        ax = axes[j]
        for i, (k, kl, col) in enumerate(keys):
            vals = [S[f"main_c{c}"]["low+med+high"]["comps"][k][m] for c in (20, 30, 40)]
            xs = np.arange(3) + (i - 1.5) * 0.15
            ax.errorbar(xs, [v["mean"] for v in vals], yerr=[[v["mean"] - v["lo"] for v in vals],
                                                            [v["hi"] - v["mean"] for v in vals]],
                        color=col, marker="o", ms=3.5, lw=0, elinewidth=1.3, capsize=2, label=kl)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(range(3), ["c_req 20", "c_req 30", "c_req 40"])
        ax.set_title(lab, fontsize=9)
    axes[0].legend(fontsize=7)
    fig.suptitle("Fresh seeds 100-119, mismatch low+med+high, one nozzle per field: paired differences "
                 "(mean and 95% CI over seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "feedback_model_v2_closed_loop_diff.png")
    plt.close(fig)


def main_v2(args) -> None:
    w = min(args.workers, 26)
    if args.v2_report_only:
        res = json.loads(V2_JSON.read_text(encoding="utf8"))
        if args.verdict:
            res["verdict"] = args.verdict
        if args.not_tested:
            res["not_tested"] = json.loads(args.not_tested)
    elif args.v2_b0_loop:
        from coverageai import integration
        res = json.loads(V2_JSON.read_text(encoding="utf8"))
        tab = integration.loop_calibration_bands("b0")
        if tab is None:
            raise SystemExit("results/b0_loop_calibration.json not found: run experiments/calibrate_reader_loop.py")
        rows, wall, done, total = run_v2(V2_B0_LOOP_BLOCKS, w)
        df = pd.DataFrame(rows)
        need = len(PRESETS) * 4 * len(V2_B0_LOOP_BLOCKS)
        full = [int(s_) for s_, g in df.groupby("seed") if len(g) == need] if len(df) else []
        res["b0_loop"] = {"rows": rows, "n_passes": len(rows), "wall_s": wall, "cells_done": done,
                          "cells_planned": total, "seeds_complete": full,
                          "table_source": tab[1], "table_bands": [list(b) for b in tab[0]],
                          "summary": v2_summary(df[df["seed"].isin(full)], V2_B0_LOOP_BLOCKS) if full else {}}
    elif args.v2_b0:
        res = json.loads(V2_JSON.read_text(encoding="utf8"))
        rows, wall, done, total = run_v2(V2_B0_BLOCKS, w, cap_s=args.b0_cap_s)
        df = pd.DataFrame(rows)
        full = []
        if len(df):
            need = len(PRESETS) * 3 * len(V2_B0_BLOCKS)
            full = [int(s_) for s_, g in df.groupby("seed") if len(g) == need]
        rows = [r for r in rows if r["seed"] in full]
        b0 = {"rows": rows, "n_passes": len(rows), "wall_s": wall, "cells_done": done, "cells_planned": total,
              "seeds_complete": full}
        if done < total:
            b0["note"] = (f"The {args.b0_cap_s / 60:.0f}-minute cap was reached after {done} of {total} cells; only "
                          "seeds with all cells finished are reported.")
        if rows:
            b0["summary"] = v2_summary(pd.DataFrame(rows), V2_B0_BLOCKS)
        else:
            b0["note"] = b0.get("note", "") + " No seed finished; nothing is reported."
        res["b0"] = b0
    else:
        from coverageai import feedback_learned as fl
        rows, wall, done, total = run_v2(V2_BLOCKS, w)
        df = pd.DataFrame(rows)
        res = {"schema": "feedback_model_v2_closed_loop/1", "fresh_seeds": list(FRESH_SEEDS),
               "design_seeds": list(DESIGN_SEEDS), "eval_offset": EVAL_OFFSET, "presets": list(PRESETS),
               "blocks": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in b.items()} for b in V2_BLOCKS],
               "strategies": {k: list(v) for k, v in STRATS.items()},
               "models": {k: {"file": str(pth.relative_to(ROOT) if pth.is_relative_to(ROOT) else pth).replace("\\", "/"),
                              "bytes": pth.stat().st_size}
                          for k, pth in (("v1x", fl.MODEL_PATHS[1]["xgboost"]),
                                         ("v2x", V2_MODEL_FILES["v2x"]), ("v2l", V2_MODEL_FILES["v2l"]))},
               "n_passes": len(rows), "wall_s": wall, "workers": w, "runs": rows,
               "summary": v2_summary(df, V2_BLOCKS)}
    V2_JSON.write_text(json.dumps(res, indent=1), encoding="utf8")
    write_v2_md(res)
    v2_figure(res)
    print("wrote", V2_JSON, "and", V2_MD, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--no-b0", action="store_true")
    ap.add_argument("--b0-cap-s", type=float, default=900.0)
    ap.add_argument("--report-only", action="store_true", help="rebuild md/figures from the json")
    ap.add_argument("--verdict", default=None, help="with --report-only: the hand-written verdict to store")
    ap.add_argument("--controls", action="store_true",
                    help="post-hoc: run the two control strategies on the noisy grid and merge them into the json")
    ap.add_argument("--controls-text", default=None, help="with --report-only: sentences for the controls section")
    ap.add_argument("--adoption", action="store_true",
                    help="post-hoc: run the adoption check (other app options) and merge it into the json")
    ap.add_argument("--adoption-text", default=None, help="with --report-only: sentences for the adoption check")
    ap.add_argument("--v2", action="store_true", help="v2 evaluation on fresh seeds (writes feedback_model_v2_closed_loop.*)")
    ap.add_argument("--v2-b0", action="store_true", help="v2: the b0 check (merged into the v2 json)")
    ap.add_argument("--v2-b0-loop", action="store_true",
                    help="v2: b0 with the loop-measured calibration table, seeds 105-109 (merged into the v2 json)")
    ap.add_argument("--not-tested", default=None, help="with --v2-report-only: JSON list of 'what was not tested' lines")
    ap.add_argument("--v2-report-only", action="store_true", help="v2: rebuild md/figure from the v2 json")
    ap.add_argument("--b0-only", action="store_true",
                    help="run only the reduced b0 check (rule vs XGBoost, nozzle auto) and merge it into the json")
    args = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    if args.v2 or args.v2_b0 or args.v2_report_only or args.v2_b0_loop:
        main_v2(args)
        return
    if args.adoption:
        res = json.loads(OUT_JSON.read_text(encoding="utf8"))
        w = min(args.workers, 26)
        arows, wall = run_adoption(w)
        res["adoption_check"] = {"variants": {k: {"policy": v[0], "kwargs": v[1], "c_req": v[2]}
                                              for k, v in ADOPT_VARIANTS.items()},
                                 "reader": "noisy", "nozzle": "auto", "seeds": list(SEEDS), "n_passes": len(arows),
                                 "wall_s": wall, "workers": w,
                                 "baseline": "headline fast passes (rule, xgboost) at c_req 30, reused, not re-run",
                                 "runs": arows, "summary": adoption_summary(res["runs"], arows)}
    elif args.controls:
        res = json.loads(OUT_JSON.read_text(encoding="utf8"))
        from coverageai.config import spec
        t0 = time.time()
        ctrl = control_params()
        ctrl["clip_pp"] = float(spec()["feedback"]["bias_clip_pp"])
        crow = run_controls(min(args.workers, 26), ctrl)
        res["controls"] = {"params": ctrl, "runs": crow, "n_passes": len(crow), "wall_s": time.time() - t0,
                           "reuse": "reused from the headline run (not re-run)",
                           "summary": control_summary(res["runs"], crow)}
    elif args.b0_only:
        res = json.loads(OUT_JSON.read_text(encoding="utf8"))
        old = res.get("b0", {})
        res["b0_first_attempt"] = {k: v for k, v in old.items() if k not in ("rows", "summary")}
        b0 = run_b0_only(min(args.workers, 26), args.b0_cap_s)
        if b0["rows"]:
            sm = summarize(pd.DataFrame(b0["rows"]), policies=B0_ONLY_POLICIES)
            b0["summary"] = {k: v for k, v in sm.items() if k.endswith("|med")}
        b0["note"] = (f"Reduced rerun: rule vs XGBoost only, nozzle auto, seeds {b0['seeds_complete']}. The first "
                      f"attempt (open, rule, XGBoost and oracle, both nozzle modes) finished "
                      f"{old.get('cells_done', '?')} of {old.get('cells_planned', '?')} cells in its "
                      f"{old.get('wall_s', 0) / 60:.0f} minutes and was discarded unread.")
        res["b0"] = b0
    elif args.report_only:
        res = json.loads(OUT_JSON.read_text(encoding="utf8"))
        if args.verdict:
            res["verdict"] = args.verdict
        if args.controls_text and res.get("controls"):
            res["controls"]["sentences"] = args.controls_text
        if args.adoption_text and res.get("adoption_check"):
            res["adoption_check"]["sentences"] = args.adoption_text
    else:
        from coverageai import feedback_learned as fl
        out = run(min(args.workers, 26), not args.no_b0, args.b0_cap_s)
        df = pd.DataFrame(out["rows"])
        res = {"schema": "feedback_model_closed_loop/1", "c_req_pct": C_REQ, "seeds": list(SEEDS),
               "presets": list(PRESETS), "mismatch": list(MISMATCH), "readers": list(READERS),
               "nozzle_modes": list(NOZZLE), "policies": list(POLICIES), "eval_offset": EVAL_OFFSET,
               "models": {b: {"file": str(fl.MODEL_PATHS[1][b].relative_to(ROOT)).replace("\\", "/"),
                              "bytes": fl.MODEL_PATHS[1][b].stat().st_size} for b in LEARNED},
               "n_passes": int(len(df)), "wall_s": out["wall_s"], "workers": out["workers"],
               "summary": summarize(df), "runs": out["rows"],
               "pass_s_mean": df.groupby("policy")["pass_s"].mean().to_dict()}
        b0 = out["b0"]
        if b0.get("rows"):
            bdf = pd.DataFrame(b0["rows"])
            b0["summary"] = summarize(bdf, policies=("open", "rule", "xgboost", "oracle"))
            b0["summary"] = {k: v for k, v in b0["summary"].items() if k.endswith("|med")}
        res["b0"] = b0
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf8")
    write_md(res)
    figures(res)
    print("wrote", OUT_JSON, "and results/feedback_model_closed_loop.md", flush=True)


if __name__ == "__main__":
    main()
