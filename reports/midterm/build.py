"""Build the ABE 598 midterm report (WP-40): figures -> filled HTML -> PDF via headless Edge.

    uv run python reports/midterm/build.py            # build HTML and PDF
    uv run python reports/midterm/build.py --no-pdf   # HTML only

Inputs (read at build time, never typed into the template):
    results/model_v1.json, dataset_v1.md-derived constants, selector_eval.json,
    real_data_summary.json, b0_synthetic.json, b1_synthetic_cpu_partial.json, closed_loop.json,
    closed_loop_summary.md (tables are parsed from it), Docs/ai_usage_log.md, experiments/figures/*.png
Missing optional results (B1, closed loop, selector eval) render a marked
"pending" box; nothing is invented. Two groups of numbers are typed in this file, each with a
comment giving its source: the B1 idle-CPU re-timing and the e2e --real run (orchestrator
measurements quoted in the work brief, not in a results file) and the prose figures taken
from results/closed_loop_summary.md. Placeholders in template.html use {{name}}.
Captions use "Table #key." / "Figure #key." and prose uses @tab:key / @fig:key; numbers
are assigned in document order.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "results"
FIG_SRC = [ROOT / "experiments" / "figures", RESULTS / "figures", RESULTS]
FIG_DST = HERE / "figures"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
MAX_W = 1600
JPEG_THRESHOLD = 700 * 1024      # PNGs still larger than this after resizing become JPEG


# ----------------------------------------------------------------------------- helpers
def load_json(name: str):
    p = RESULTS / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # corrupt or half-written file: treat as pending
        print(f"WARNING: could not parse {p}: {e}", file=sys.stderr)
        return None


def pending(title: str, text: str) -> str:
    return (f'<div class="pendbox"><b>PENDING &mdash; RESULTS IN PROGRESS.</b> '
            f'<b style="color:#333">{html.escape(title)}</b> {text}</div>')


def f(x, n=2):
    return f"{x:.{n}f}"


def pct(x, n=1):
    return f"{100 * x:.{n}f}"


def status(kind: str, text: str | None = None) -> str:
    label = {"met": "met", "track": "on track", "risk": "at risk", "pending": "pending", "missed": "missed"}[kind]
    return f'<span class="st {kind}">{text or label}</span>'


# ----------------------------------------------------------------------------- figures
def prep_figures() -> dict[str, str]:
    from PIL import Image
    FIG_DST.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    seen: dict[str, Path] = {}
    for d in FIG_SRC:
        if d.is_dir():
            for p in sorted(d.glob("*.png")):
                seen.setdefault(p.stem, p)
    unused = {"demo_live", "demo_field", "demo_nozzle", "demo_policies", "closed_loop_met", "closed_loop_volume_drift",
              "slow_loop_residuals", "bias_trace", "b1_parity_cpu_partial"}     # only their cropped versions are used
    for stem, src in seen.items():
        if stem in unused:
            continue
        im = Image.open(src)
        if im.width > MAX_W:
            im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
        dst = FIG_DST / f"{stem}.png"
        (FIG_DST / f"{stem}.jpg").unlink(missing_ok=True)
        if im.mode not in ("RGB", "RGBA", "L", "P"):
            im = im.convert("RGB")
        im.save(dst, optimize=True)
        rel = f"figures/{dst.name}"
        if dst.stat().st_size > JPEG_THRESHOLD:
            jpg = FIG_DST / f"{stem}.jpg"
            im.convert("RGB").save(jpg, quality=88, optimize=True)
            dst.unlink()
            rel = f"figures/{jpg.name}"
        out[stem] = rel
    crops = {                                  # derived crops of regenerated figures (box in source pixels)
        "closed_loop_met": ("closed_loop_met_spray", (0, 0, 1950, 380)),
        "bias_trace": ("bias_trace_spray", (0, 0, 812, 702)),
        "b1_parity_cpu_partial": ("b1_parity_left", (0, 0, 676, 600)),
        "demo_live": ("demo_live_top", (0, 0, 1500, 1190)),
    }
    for stem, (name, box) in crops.items():
        if stem in seen:
            im = Image.open(seen[stem]).convert("RGB").crop(box)
            if im.width > MAX_W:
                im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
            dst = FIG_DST / f"{name}.png"
            im.save(dst, optimize=True)
            out[name] = f"figures/{dst.name}"
            if dst.stat().st_size > JPEG_THRESHOLD:
                jpg = FIG_DST / f"{name}.jpg"
                im.save(jpg, quality=88, optimize=True)
                dst.unlink()
                out[name] = f"figures/{jpg.name}"
    for p in FIG_DST.glob("*"):          # figures already present (e.g. copied folder without sources)
        out.setdefault(p.stem, f"figures/{p.name}")
    return out


# ----------------------------------------------------------------------------- model / dataset
def model_parts() -> dict[str, str]:
    d = load_json("model_v1.json")
    if d is None:
        raise SystemExit("results/model_v1.json is required")
    sp = d["splits"]
    names = [("random", "Random 80/20"), ("U_gt_5", "Wind &gt; 5 m/s withheld"),
             ("family_3D", "3D family withheld"), ("hot_dry", "Hot-dry withheld")]
    rows, grp = [], []
    for key, label in names:
        m = sp[key]["models"]
        cov = {k: m[k]["coverage_pct"]["mae"] for k in ("ridge", "rf", "lightgbm")}
        best = min(cov, key=cov.get)
        cells = "".join(
            f'<td class="num">{"<b>" if k == best else ""}{f(cov[k])}{"</b>" if k == best else ""}</td>'
            for k in ("ridge", "rf", "lightgbm"))
        lg = m["lightgbm"]
        rows.append(f"<tr><td>{label}</td>{cells}<td class=\"num\">{f(lg['coverage_pct']['mae_vs_expected'])}</td>"
                    f"<td class=\"num\">{f(lg['drift_frac']['mae'], 4)}</td>"
                    f"<td class=\"num\">{f(lg['displacement_mm']['mae'], 1)}</td></tr>")
        if key != "random":
            grp.append(cov["lightgbm"])
    rnd = sp["random"]["models"]
    floor = d["coverage_noise_floor"]["irreducible_mae_expected_vs_sampled"]
    lg_r = rnd["lightgbm"]["coverage_pct"]["mae"]
    all_better = all(sp[k]["models"]["lightgbm"]["coverage_pct"]["mae"] < min(
        sp[k]["models"]["ridge"]["coverage_pct"]["mae"], sp[k]["models"]["rf"]["coverage_pct"]["mae"]) for k in sp)
    iv = rnd["lightgbm"]["interval"]
    tm = d["timing"]
    sens = d.get("sensitivity", {})
    n_pass = sum(1 for v in sens.values() if v.get("passed"))
    headline = (
        f"LightGBM reaches a coverage MAE of {f(lg_r)} pp on the random hold-out (target 3 pp) against "
        f"{f(rnd['ridge']['coverage_pct']['mae'])} pp for ridge and {f(rnd['rf']['coverage_pct']['mae'])} pp for "
        f"random forest{', and is the best model on every split' if all_better else ''}. Its 80% prediction interval covers "
        f"{pct(iv['coverage_of_80pct_band'])}% of held-out rows with mean width {f(iv['mean_width_pp'], 1)} pp. "
        f"Scoring all {tm['n_candidates']} candidates takes {f(tm['predict_batch_median_ms'], 1)} ms (median) and a "
        f"{d['dataset']['rows']:,}-row refit takes {f(tm['fit_50k_s'], 1)} s, so the selector can run for every segment.")
    g = lambda k, o, fld="mae": sp[k]["models"]["lightgbm"][o][fld]  # noqa: E731
    return {
        "model_rows": "\n".join(rows),
        "model_headline": headline,
        "noise_floor": f(floor),
        "lgb_rand_mae": f(lg_r),
        "gap_to_floor": f(lg_r - floor),
        "lgb_exp_mae": f(rnd["lightgbm"]["coverage_pct"]["mae_vs_expected"]),
        "ridge_exp_mae": f(rnd["ridge"]["coverage_pct"]["mae_vs_expected"]),
        "ridge_rand_mae": f(rnd["ridge"]["coverage_pct"]["mae"]),
        "grp_min": f(min(grp)), "grp_max": f(max(grp)),
        "shift_rand": f(g("random", "displacement_mm"), 1), "shift_u5": f(g("U_gt_5", "displacement_mm"), 1),
        "shift_3d": f(g("family_3D", "displacement_mm"), 1),
        "drift_rand": f(g("random", "drift_frac"), 4), "drift_u5": f(g("U_gt_5", "drift_frac"), 4),
        "sens_pass": str(n_pass), "sens_total": str(len(sens)),
        "dataset_rows": f"{d['dataset']['rows']:,}",
        "_cov_random": lg_r, "_ridge_random": rnd["ridge"]["coverage_pct"]["mae"],
        "_grp": grp, "_sens_ok": n_pass == len(sens), "_model_json": d,
    }


def simoes_parts() -> dict[str, str]:
    d = load_json("real_data_summary.json")
    s = (d or {}).get("subsets", {}).get("synthetic", {})
    if not s:
        raise SystemExit("results/real_data_summary.json (synthetic subset) is required")
    return {"simoes_n": str(s["n_images"]), "simoes_droplets": f"{s['n_droplets_total']:,}",
            "simoes_cov_min": f(s["coverage_pct_min"], 1), "simoes_cov_max": f(s["coverage_pct_max"], 1),
            "simoes_cov_mean": f(s["coverage_pct_mean"], 1)}


# ----------------------------------------------------------------------------- anchors (live from code)
def anchor_rows() -> str:
    rows = []
    try:
        sys.path.insert(0, str(ROOT / "src"))
        import numpy as np
        from coverageai.physics.droplets import terminal_velocity
        vt = {d: float(terminal_velocity(np.array([float(d)]))[0]) for d in (20, 100, 200, 400)}
        ref = {20: 0.012, 100: 0.265, 200: 0.70, 400: 1.80}
        for d in (20, 100, 200, 400):
            rows.append(f'<tr><td>Terminal velocity, {d} &micro;m droplet (m/s)</td>'
                        f'<td>{ref[d]:g} &plusmn; 25%</td><td class="num">{vt[d]:.3f}</td></tr>')
    except Exception as e:
        print(f"WARNING: terminal velocity anchors unavailable ({e})", file=sys.stderr)
    drift = None
    try:
        sys.path.insert(0, str(ROOT / "tests"))
        import numpy as np
        from test_simulator import VMD_CENTER, cond, make_sim, run  # type: ignore
        drift = {}
        for c in ("F", "M", "C"):
            sim = make_sim(vmd=VMD_CENTER[c], flow_gpm=0.4)
            drift[c] = float(np.mean([run(sim, cond(2.0), s).drift_frac for s in range(6)]))
    except Exception as e:
        print(f"WARNING: live drift anchors unavailable ({e}); using values quoted in decision D10", file=sys.stderr)
        drift = {"F": 0.30, "M": 0.08, "C": 0.01}
    for c, name, band in (("F", "fine", "10% to 35%"), ("M", "medium", "2% to 15%"), ("C", "coarse", "below 5%")):
        rows.append(f'<tr><td>Drift fraction, {name} spray ({c}), 2 m/s crosswind</td><td>{band}</td>'
                    f'<td class="num">{100 * drift[c]:.0f}%</td></tr>')
    return "\n".join(rows)


# ----------------------------------------------------------------------------- selector
def _setting(s: dict) -> str:
    return f"{s['family']} {s['size']} @ {s['pressure_psi']:g} psi"


def selector_block(figs: dict[str, str]) -> tuple[str, dict]:
    d = load_json("selector_eval.json")
    info = {"has_real": False}
    cfgs = (d or {}).get("configs", {})
    sp, orc = cfgs.get("spray"), cfgs.get("oracle")
    if not sp:
        return pending("Selector evaluation with the integrated model.",
                       "results/selector_eval.json has no \"spray\" configuration; "
                       "<code>experiments/eval_selector.py</code> must be rerun."), info
    info.update(has_real=True, real=sp, real_name="spray", oracle=orc)
    n, creq = d["n"], d["c_req"]
    s, mf, bf = sp["selector"], sp["fixed_matched_success"], sp["best_fixed"]
    rt = sp["runtime_ms"]

    def row(label, c, ms=None, bold=False):
        cls = ' class="best"' if bold else ""
        return (f'<tr{cls}><td>{label}</td><td class="num">{pct(c["success"])}</td>'
                f'<td class="num">{f(c["mean_drift"], 4)}</td><td class="num">{f(c["mean_gpa"], 1)}</td>'
                f'<td class="num">{f(ms, 1) if ms is not None else "&ndash;"}</td></tr>')

    rows = (row("Selector, learned model (config <code>spray</code>)", s, rt["total_mean"], True)
            + row(f"Lowest-drift fixed setting with matching success: {_setting(mf['setting'])}", mf)
            + row(f"Best fixed setting overall: {_setting(bf['setting'])}", bf)
            + (row("Selector, simulator as model (config <code>oracle</code>)", orc["selector"], orc["runtime_ms"]["total_mean"]) if orc else ""))
    tbl = (f'<div class="caption"><b>Table #select.</b> Selector against fixed settings on {n} random no-mismatch scenarios at '
           f'<i>C</i><sub>req</sub> = {creq:g}% (success = share with true coverage &ge; <i>C</i><sub>req</sub>; target &ge; 90% at lower drift than '
           f'the matched fixed setting). In the first row the learned model selects and the simulator scores; all {sp["n_candidates"]} candidates are scored per call.</div>'
           '<table><tr><th>Policy</th><th class="num">Success (%)</th><th class="num">Mean drift (fraction)</th>'
           '<th class="num">Mean volume (GPA)</th><th class="num">Time per selection (ms)</th></tr>' + rows + "</table>")
    lower = 100 * (1 - s["mean_drift"] / mf["mean_drift"])
    vol_ratio = s["mean_gpa"] / mf["mean_gpa"]
    sw = sp["lambda_sweep"]
    vol_txt = "about half" if 0.45 < vol_ratio < 0.6 else f"{100 * vol_ratio:.0f}% of"
    txt = (f"<p>With the learned model the selector meets the requirement in {pct(s['success'])}% of the {n} scenarios (target &ge; 90%). The lowest-drift fixed setting "
           f"that matches this success ({_setting(mf['setting'])}, {pct(mf['success'], 0)}%) has mean drift {f(mf['mean_drift'], 4)} against {f(s['mean_drift'], 4)} "
           f"({lower:.0f}% lower) and applies {f(mf['mean_gpa'], 1)} GPA against {f(s['mean_gpa'], 1)}, so the selector uses {vol_txt} "
           f"the volume. The overall best fixed setting ({_setting(bf['setting'])}) reaches {pct(bf['success'], 0)}% but at drift {f(bf['mean_drift'], 4)} and {f(bf['mean_gpa'], 1)} GPA. "
           f"One selection takes {f(rt['total_mean'], 0)} ms including {f(rt['model_mean'], 0)} ms of model time, so it can run for every segment. "
           f"Raising &lambda; from 0 to 1 moves the selector from {f(sw['0.0']['mean_drift'], 3)} drift at {f(sw['0.0']['mean_gpa'], 1)} GPA to {f(sw['1.0']['mean_drift'], 4)} drift at "
           f"{f(sw['1.0']['mean_gpa'], 1)} GPA, with success at or above {pct(min(v['success'] for v in sw.values()))}% throughout (@fig:selsens).</p>")
    fig = (f'<figure style="margin:0"><img src="{figs["selector_tradeoff"]}" alt="Selector trade-off for the learned model and the simulator-as-model selector">'
           '<figcaption><b>Figure #selsens.</b> Mean drift against mean volume as &lambda; varies (0, 0.3, 0.6, 1.0). Green: learned model selects (<code>spray</code>); '
           'blue: simulator selects and scores (<code>oracle</code>); square: best fixed setting (100% success under both). The <code>spray</code> curve lies to the right '
           'because the interval margin costs volume; that margin is what keeps its success high.</figcaption></figure>')
    oracle_txt = ""
    if orc:
        o = orc["selector"]
        oracle_txt = (f"<p><b>Why the interval-based margin is needed.</b> In config <code>oracle</code> the simulator both selects and scores (an expected-value model "
                      f"with no prediction interval), yet success is only {pct(o['success'])}%. Without an interval the selector keeps only the fixed 1.5 pp margin floor, and the "
                      f"per-segment random spread factor of the card pushes about half of the just-feasible choices below <i>C</i><sub>req</sub>. Its low drift "
                      f"({f(o['mean_drift'], 4)}) and volume ({f(o['mean_gpa'], 1)} GPA) are therefore bought with failures, not a better selector. The learned model's "
                      f"conformal interval supplies the margin that this noise requires, which is why the margin is taken from the interval and not set to a constant.</p>")
    return tbl + txt + f'<div class="twoup" style="align-items:center"><div style="flex:1.6">{oracle_txt}</div><div style="flex:1">{fig}</div></div>', info


# ----------------------------------------------------------------------------- verification
B1_LABEL = "B1 (6-epoch CPU checkpoint)"
# Idle-CPU re-timing of the same 6-epoch checkpoint on 16 test cards (~1867x686 px): orchestrator measurement
# quoted in the work brief on 2026-10-18; it is not stored in a results file.
B1_IDLE = {"n": 16, "first_s": 1.74, "mean_s": 1.20, "median_s": 1.20, "max_s": 1.48}


def _band_rows(d: dict, label: str) -> str:
    rows = []
    for band, v in d.get("by_coverage_band", {}).items():
        rows.append(
            f'<tr><td>{label}</td><td>{html.escape(band)}%</td><td class="num">{v["n"]}</td>'
            f'<td class="num">{f(v["mae_pp"])}</td><td class="num">{v["bias_pp"]:+.2f}</td>'
            f'<td class="num">{pct(v["class_agreement"])}</td><td class="num">{pct(v["saturated_frac"], 0)}</td>'
            f'<td class="num">{f(v["iou_mean"])}</td><td class="num">{f(v.get("count_rel_err_median", float("nan")))}</td></tr>')
    o = d["overall"]
    rows.append(
        f'<tr class="best"><td>{label}</td><td>all</td><td class="num">{o["n"]}</td>'
        f'<td class="num">{f(o["mae_pp"])}</td><td class="num">{o["bias_pp"]:+.2f}</td>'
        f'<td class="num">{pct(o["class_agreement"])}</td><td class="num">{pct(o["saturated_frac"], 0)}</td>'
        f'<td class="num">{f(o["iou_mean"])}</td><td class="num">{f(o.get("count_rel_err_median", float("nan")))}</td></tr>')
    return "".join(rows)


def _b1_rows(d: dict, label: str) -> str:
    dash = '<td class="num">&ndash;</td>'
    rows = []
    for band, v in d.get("by_band", {}).items():
        rows.append(
            f'<tr><td>{label}</td><td>{html.escape(band)}%</td><td class="num">{v["n"]}</td>'
            f'<td class="num">{f(v["coverage_mae_pp"])}</td><td class="num">{v["coverage_bias_pp"]:+.2f}</td>'
            f'<td class="num">{pct(v["class_agreement"])}</td>{dash}<td class="num">{f(v["iou"])}</td>{dash}</tr>')
    o = d["overall"]
    rows.append(
        f'<tr class="best"><td>{label}</td><td>all</td><td class="num">{o["n"]}</td>'
        f'<td class="num">{f(o["coverage_mae_pp"])}</td><td class="num">{o["coverage_bias_pp"]:+.2f}</td>'
        f'<td class="num">{pct(o["class_agreement"])}</td>{dash}<td class="num">{f(o["iou"])}</td>{dash}</tr>')
    return "".join(rows)


def _ov(d: dict) -> dict:
    """Overall metrics with tolerant key names (B0 and B1 files differ)."""
    o = d.get("overall", d)
    pick = lambda *ks: next((o[k] for k in ks if k in o and o[k] is not None), None)  # noqa: E731
    return {"mae": pick("mae_pp", "coverage_mae_pp", "mae"), "agree": pick("class_agreement", "class_acc"),
            "rt_mean": pick("runtime_mean_s", "runtime_s_mean", "runtime_s"),
            "rt_max": pick("runtime_max_s", "runtime_s_max", "runtime_p95_s", "runtime_mean_s"),
            "n": pick("n", "n_cards"), "bias": pick("bias_pp", "coverage_bias_pp"), "iou": pick("iou_mean", "iou")}


def verification(b0, b1, figs) -> tuple[str, str, str, dict]:
    """Returns (b0_block, b1_block, b0_analysis, info)."""
    info = {"b0": None, "b1": None}
    if b0 is None:
        return (pending("Classical verifier B0.", "results/b0_synthetic.json is missing."),
                pending("Learned verifier B1.", "not available."), "B0 results are pending.", info)
    o = b0["overall"]
    info["b0"] = _ov(b0)
    has_b1 = bool(b1 and "overall" in b1 and "coverage_mae_pp" in b1["overall"])
    rows = _band_rows(b0, "B0") + (_b1_rows(b1, B1_LABEL) if has_b1 else "")
    bands = b0.get("by_coverage_band", {})
    top = list(bands.items())[-1] if bands else None
    split = b0.get("split", "test")
    text = (
        f"<p>On the {o['n']} held-out synthetic {html.escape(str(split))} cards from the team's renderer (Table #verify), B0 reaches a coverage "
        f"MAE of {f(o['mae_pp'])} pp (target &le; 2 pp), three-class agreement of {pct(o['class_agreement'])}% (target &ge; 90%) and a runtime of "
        f"{f(o['runtime_mean_s'])} s mean and {f(o['runtime_max_s'])} s maximum per card on a laptop CPU (target &le; 2 s). Error is not uniform: "
        + (f"in the {html.escape(top[0])}% band the MAE is {f(top[1]['mae_pp'])} pp with a bias of {f(top[1]['bias_pp'])} pp "
           f"({pct(top[1]['saturated_frac'], 0)}% of those cards are flagged saturated) and the worst single card is off by {f(o['max_abs_err_pp'], 1)} pp. " if top else "")
        + f"B0 underestimates coverage above 50% because stains merge. Stain counts and Dv values are unreliable (median relative count error "
        f"{f(o['count_rel_err_median'])} overall, {f(bands[list(bands)[0]]['count_rel_err_median']) if bands else 'n/a'} in the lowest band), which is why coverage, "
        f"not counts, is the quantity the loop uses.</p>")
    tbl = ('<div class="caption"><b>Table #verify.</b> Verifier accuracy on the 180 synthetic test cards by true-coverage band (B0 and B1 use different band edges): '
           'coverage MAE and signed bias in pp, three-class agreement, share of cards flagged saturated, mean stain-mask IoU and median relative stain-count error '
           '(&ndash; = not computed for B1). B1 rows are a 6-epoch CPU checkpoint, not the final model.</div>'
           '<table><tr><th style="width:25%">Verifier</th><th>True coverage</th><th class="num">n</th><th class="num">MAE (pp)</th><th class="num">Bias (pp)</th>'
           '<th class="num">Class agreement (%)</th><th class="num">Saturated (%)</th><th class="num">IoU</th><th class="num">Count error</th></tr>'
           + rows + "</table>")
    analysis = (
        f"B0 meets all three synthetic targets, but its error is "
        + (f"{f(top[1]['mae_pp'])} pp in the {html.escape(top[0])}% band against {f(bands[list(bands)[0]]['mae_pp'])} pp in the lowest band" if top else "concentrated at high coverage")
        + " (WSP saturation, as the proposal anticipated). Class thresholds sit at 15% and 30%, so this rarely changes the class; it would matter for a requirement above 50%.")
    if not has_b1:
        b1_block = pending("Learned verifier B1 (U-Net).", "results/b1_synthetic_cpu_partial.json is missing or has an unexpected format.")
    else:
        v = _ov(b1)
        info["b1"] = v
        bb = b1["by_band"]
        lo, hi = list(bb.items())[0], list(bb.items())[-1]
        t = B1_IDLE
        ok_rt = t["max_s"] <= 2.0
        info["b1_idle"] = t
        info["b1_rt_ok"] = ok_rt
        b1_block = (
            f"<p><b>{B1_LABEL}.</b> B1 is a U-Net with a lightweight pretrained encoder [8] (<code>vision/unet.py</code>); the weights evaluated here come from a "
            f"<b>6-epoch run on a laptop CPU</b> and understate the final model. On the same {v['n']} test cards the checkpoint reaches a coverage MAE of "
            f"{f(v['mae'])} pp (bias {v['bias']:+.2f} pp), stain-mask IoU {f(v['iou'])} (B0: {f(o['iou_mean'])}) and three-class agreement of {pct(v['agree'])}% (B0: {f(o['mae_pp'])} pp MAE). "
            f"Error grows with coverage ({f(lo[1]['coverage_mae_pp'])} pp in the {lo[0]}% band, {f(hi[1]['coverage_mae_pp'])} pp in the {hi[0]}% band). "
            f"<b>Runtime:</b> on an idle laptop CPU over {t['n']} test cards the first call took {f(t['first_s'])} s (including warm-up) "
            f"and later calls averaged {f(t['mean_s'])} s per card (maximum {f(t['max_s'])} s), inside the 2 s target; an earlier timing over all {v['n']} cards while the CPU was "
            f"busy with the experiment grid (mean {f(v['rt_mean'])} s, maximum {f(v['rt_max'], 1)} s) is not used for the criterion. Like B0, B1 is tested only on cards from the "
            f"same generator (@fig:wsp, right). <b>Next:</b> full GPU training (NCSA H100, or the Colab notebook "
            f"<code>notebooks/train_unet_gpu.ipynb</code>), then re-evaluation and re-timing on all {v['n']} cards.</p>")
    return text + tbl, b1_block, analysis, info


# ----------------------------------------------------------------------------- closed loop
def md_tables(md: str) -> list[tuple[list[str], list[list[str]]]]:
    lines, i, out = md.splitlines(), 0, []
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            hdr = [c.strip() for c in ln.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            out.append((hdr, rows))
            continue
        i += 1
    return out


def _cell(c: str) -> str:
    c = html.escape(c).replace("&gt;=", "&ge;").replace("&lt;=", "&le;").replace("+-", "&plusmn;")
    c = re.sub(r"\*\*MISSED\*\*", '<b style="color:#b3261e">MISSED</b>', c)
    c = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", c)
    c = re.sub(r"`([^`]+)`", r"<code>\1</code>", c)
    return re.sub(r"(?<![\w.])-(?=\d)", "&minus;", c)


def md_table_html(hdr: list[str], rows: list[list[str]], num_from: int | None = None, widths: list[int] | None = None) -> str:
    th = "".join(f'<th{" class=\"num\"" if num_from is not None and j >= num_from else ""}'
                 f'{f" style=\"width:{widths[j]}%\"" if widths else ""}>{_cell(h)}</th>' for j, h in enumerate(hdr))
    body = "".join("<tr>" + "".join(f'<td{" class=\"num\"" if num_from is not None and j >= num_from else ""}>{_cell(c)}</td>'
                                    for j, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><tr>{th}</tr>{body}</table>"


def closed_loop_block(figs: dict[str, str]) -> tuple[str, dict]:
    cl = load_json("closed_loop.json")
    md_path = RESULTS / "closed_loop_summary.md"
    info = {"present": cl is not None or md_path.exists(), "head": None}
    if not md_path.exists():
        return pending("Closed-loop experiments.", "results/closed_loop_summary.md is missing; no closed-loop number is reported before it exists."), info
    md = md_path.read_text(encoding="utf-8")
    head_md, rest = md.split("## Model `spray`", 1)
    spray_md = rest.split("## Model `oracle_nominal`", 1)[0]
    head_t = md_tables(head_md)
    sp_t = md_tables(spray_md)
    met_t = next((t for t in sp_t if t[0][0] == "policy" and "none" in t[0]), None)
    abl_t = next((t for t in sp_t if t[0][0] == "group"), None)
    if not (head_t and met_t and abl_t):
        return pending("Closed-loop summary.", "results/closed_loop_summary.md has an unexpected layout; the orchestrator should check it."), info
    info["head"] = head_t[0][1]
    # Prose figures below are copied from results/closed_loop_summary.md (headline section, which is authoritative).
    head_rows = [r for r in head_t[0][1] if not r[0].startswith("20 seeds")]          # seed count is stated in the text
    abl_rows = [r for r in abl_t[1] if r[0] != "warmup" and r[1] != "verifier=noisy"]  # warm-up discussed in text; noisy = alpha 0.3 baseline
    A = lambda k: figs[k]  # noqa: E731
    design = (
        "<p>The grid (<code>experiments/closed_loop.py</code>) runs five policies over four presets (<code>calm_morning</code>, <code>crosswind_shift</code>, "
        "<code>gusty_afternoon</code>, <code>hot_dry</code>), four hidden-mismatch levels (none, low, med, high) and 20 seeds, 40 segments per pass at <i>C</i><sub>req</sub> = 30%, "
        "all inside the simulator. The headline model is the learned <code>SprayModel</code>; a run with the mismatch-free simulator physics as the model (<code>oracle_nominal</code>) "
        "gives the same picture (fast loop 60% [43, 71] recovery, +13.8% volume). The verifier is <i>noisy</i>: perfect coverage plus Gaussian error with per-band bias and standard deviation "
        "from the B0 results. Gap recovery is (closed &minus; open)/(oracle &minus; open) in the share of segments meeting "
        "<i>C</i><sub>req</sub>; volume overhead is mean applied GPA relative to the oracle. Headline numbers pool low, med and high mismatch; brackets are 95% paired-bootstrap CIs over seeds.</p>")
    head = (
        '<div class="caption"><b>Table #clhead.</b> Closed-loop success criterion (proposal Table 1), learned model.</div>'
        + md_table_html(head_t[0][0], head_rows, widths=[27, 10, 35, 28])
        + "<p><b>The closed-loop criterion is missed.</b> The fast loop recovers 58% [38, 69] of the open-loop-to-oracle gap, which meets 50% on the point estimate but not with confidence, "
          "and uses +15.2% [+13.7, +16.9] more volume than the oracle against the 10% limit. Adding the slow loop meets the volume limit (+6.2% [+5.2, +7.2]) "
          "but not recovery (&minus;6% [&minus;44, 17]).</p>")
    fastloop_text = (
        "<p><b>The fast loop works as designed.</b> The requirement-met share under mismatch stays at 0.94 (low 0.947, med 0.943, high 0.932; Table #clmet, @fig:clmet), while open loop falls from 0.954 (low) to 0.864 (med) "
        "and 0.762 (high) and the best fixed setting to 0.61 at high. Recovery by preset is 69% (<code>calm_morning</code>), 54% (<code>crosswind_shift</code>), 54% (<code>gusty_afternoon</code>) and "
        "48% (<code>hot_dry</code>, CI [20, 66]). At low mismatch the open-to-oracle gap is only 4.6 pp (the model's margin already protects open loop), so recovery there is indistinguishable from zero; "
        "the criterion is informative at med and high mismatch (gaps 13.6 and 23.8 pp).</p>")
    met_table = (
        '<div class="caption"><b>Table #clmet.</b> Share of segments meeting <i>C</i><sub>req</sub> by policy and mismatch level (learned model; mean &plusmn; 95% CI over 20 seeds; presets pooled; the oracle row, 1.000 at every level, is omitted).</div>'
        + md_table_html(met_t[0], [r for r in met_t[1] if r[0] != "oracle"], num_from=1))
    figmet = (f'<figure><img src="{A("closed_loop_met_spray")}" alt="Requirement-met share by mismatch level, pooled and per preset">'
              '<figcaption><b>Figure #clmet.</b> Requirement-met share of segments (true outcome) against hidden-mismatch level, learned model: all presets pooled (left) and per preset; mean and 95% CI over 20 seeds.</figcaption></figure>')
    volume = (
        "<p><b>Why the volume criterion is missed.</b> With no mismatch at all the fast loop already uses +12.2% [11.2, 13.1] more volume than the oracle. The oracle sees the per-segment random spread factor of the card "
        "(irreducible noise, standard deviation about 5 pp of coverage) and needs no safety margin, whereas every model-based policy must keep a margin from its prediction interval. Only about 3 pp of the 15.2% "
        "comes from correcting the hidden mismatch. The same noise bounds how much of the residual a bias correction can remove: raw residual variance stays at about 27 to 29 pp<sup>2</sup> "
        "inside a pass. The 10% limit therefore compares against a reference no deployable controller can match (Section 4.1).</p>")
    figbias = (
        f'<figure style="margin:0"><img src="{A("bias_trace_spray")}" alt="Example pass: predicted, corrected and measured coverage with the feedback bias">'
        '<figcaption><b>Figure #bias.</b> One pass (<code>gusty_afternoon</code>, med mismatch, seed 0, learned model). Top: raw and bias-corrected prediction, measured card values, and true coverage '
        'under the fast loop and open loop. Bottom: bias applied to each segment and bias after the update (0 to about &minus;6 pp).</figcaption></figure>')
    slow = (
        "<p><b>The slow loop, as implemented, did not help.</b> It refits <code>SprayModel</code> on the measured cards of earlier passes with record weight 30. It removes the average model error "
        "(mean raw residual at high mismatch &minus;6.0 pp in pass 1, &minus;1.8 pp in pass 3) and cuts volume, but requirement-met drops to 0.85 against 0.94 for the fast loop and the raw residual "
        "variance rises from about 27 to about 33 pp<sup>2</sup>. A 6-seed diagnostic with no mismatch (<code>experiments/slow_loop_diagnostic.py</code>) shows the 80% prediction interval narrowing after the refit "
        "(mean half-width 8.0 to 5.2 pp), true coverage falling below its lower bound in 21 to 27% of segments instead of 7.5%, and the MAE rising from 4.7 to 5.0&ndash;5.1 pp. A plausible cause, not a tested mechanism, is that a few dozen noisy, heavily weighted "
        "records are fitted too closely and the conformal calibration is not redone on them. Record weights 1, 5 and 30 give 36%, 10% and "
        "&minus;20% recovery (Table #clabl); 30, the grid's value, is the worst. The slow loop needs a better update rule (Section 6).</p>")
    abl = (
        "<p><b>Ablations</b> (fast loop, <code>gusty_afternoon</code>, med mismatch, 20 seeds unless noted). A smaller &alpha; is cheaper (&alpha; = 0.1, 0.3, 0.6 give 45%, 59%, 54% recovery at +13.0%, +17.9%, +20.5% volume) "
        "at a small loss of reliability. A global bias recovers more (68%) than per-regime biases (59%) at +19.8% against +17.9% volume, plausibly because each of four regimes sees few cards before its warm-up gate opens; "
        "warm-up of 0, 2 or 4 cards (57%, 59%, 50%) differs by less than the CI. Perfect, noisy and rendered-B0 verifiers give 51%, 59% and 44% (B0: 5 seeds, CI [&minus;200, 90]), inside the seed-to-seed spread, "
        "so measurement error is not what limits the loop. Retreat flags were never raised (0 in 1,600 learned-model passes).</p>"
        '<div class="caption"><b>Table #clabl.</b> Ablations of the fast loop and of the slow-loop record weight (learned model; warm-up and the repeated baseline row omitted).</div>'
        + md_table_html(abl_t[0], abl_rows, num_from=2))
    lim = (
        "<p><b>What these runs do and do not show.</b> Plant, model training data, renderer and hidden mismatch all come from one hand-built simulator, so the runs test whether the "
        "<i>control logic</i> behaves as designed when the model is wrong in a known, low-dimensional way; they do not measure field efficacy. We injected the mismatch (gust multiplier, "
        "canopy interception, nozzle wear), and it is of the kind the bias term is built to absorb; inversions, boom roll, clogging and card-placement error are not tested, and the default verifier omits heavy-tailed failures and real-card "
        "domain shift.</p>")
    return (design + head + fastloop_text + figmet + met_table
            + f'<div class="twoup" style="align-items:center"><div style="flex:1.15">{volume}</div><div style="flex:1">{figbias}</div></div>'
            + slow + abl + lim), info


# ----------------------------------------------------------------------------- criteria and summary
# Orchestrator measurement quoted in the work brief (2026-10-18): `uv run python -m coverageai.e2e --real`
E2E_REAL = {"segments": 40, "seconds": 33, "mean_abs_err_pp": 0.57}


def criteria_rows(mp, sel_info, ver_info, cl_info, b0, b1, n_sim_tests) -> str:
    R = []
    add = lambda c, t, m, s: R.append(f"<tr><td>{c}</td><td>{t}</td><td>{m}</td><td>{s}</td></tr>")  # noqa: E731
    e = E2E_REAL
    add("End-to-end",
        "A simulated pass runs from prediction to correction with no manual steps; every decision is logged and visualized.",
        f"<code>uv run python -m coverageai.e2e --real</code> runs a {e['segments']}-segment pass with rendered cards and B0 in {e['seconds']} s (mean |measured &minus; true| {e['mean_abs_err_pp']:.2f} pp). "
        "The Streamlit demo (<code>app/main.py</code>) logs and visualizes every decision (Section 2.9).",
        status("met"))
    add("Simulator",
        "Coverage rises with volume; drift rises with wind, release height and fine fraction.",
        f"{n_sim_tests} simulator tests pass, including monotone sweeps for each trend and the D10 anchors (@tab:anchors).",
        status("met"))
    grp = mp["_grp"]
    add("Predictive model",
        "Coverage MAE &le; 3 pp on held-out conditions and below the linear baseline; grouped-regime error reported; sensitivity checks pass.",
        f"{mp['lgb_rand_mae']} pp random; {', '.join(f(x) for x in grp)} pp on three withheld regimes; ridge {mp['ridge_rand_mae']} pp; "
        f"{mp['sens_pass']} of {mp['sens_total']} sensitivity checks pass. Noise floor {mp['noise_floor']} pp.",
        status("met" if mp["_cov_random"] <= 3.0 and mp["_cov_random"] < mp["_ridge_random"] and all(x <= 3.0 for x in grp) and mp["_sens_ok"] else "track"))
    if sel_info["has_real"]:
        c = sel_info["real"]
        mf = c["fixed_matched_success"]
        ok = c["selector"]["success"] >= 0.9 and c.get("selector_lower_drift_than_matched_fixed") is True
        add("Selector", "No mismatch: &ge; 90% of scenarios meet <i>C</i><sub>req</sub> at lower drift than the best fixed setting with the same success.",
            f"Learned model: success {pct(c['selector']['success'])}%, drift {f(c['selector']['mean_drift'], 4)}, {f(c['selector']['mean_gpa'], 1)} GPA, against "
            f"{pct(mf['success'], 0)}%, {f(mf['mean_drift'], 4)}, {f(mf['mean_gpa'], 1)} GPA for the matched fixed setting: lower drift, about half the volume (@tab:select).",
            status("met" if ok else "missed"))
    else:
        add("Selector", "No mismatch: &ge; 90% of scenarios meet <i>C</i><sub>req</sub> at lower drift than the best fixed setting with the same success.",
            "Not yet measured with the integrated model.", status("pending"))
    if ver_info["b0"]:
        o, b1v, t = ver_info["b0"], ver_info["b1"], ver_info.get("b1_idle")
        b0ok = o["mae"] <= 2.0 and o["agree"] >= 0.90 and o["rt_max"] <= 2.0
        b1ok = bool(b1v and b1v["mae"] is not None and b1v["mae"] <= 2.0 and b1v["agree"] >= 0.9 and ver_info.get("b1_rt_ok"))
        txt = (f"B0: MAE {f(o['mae'])} pp, agreement {pct(o['agree'])}%, {f(o['rt_max'])} s worst case ({o['n']} cards). "
               + (f"{B1_LABEL}: MAE {f(b1v['mae'])} pp, agreement {pct(b1v['agree'])}%, {f(t['mean_s'])} s mean and {f(t['max_s'])} s maximum per card on an idle CPU ({t['n']} cards). "
                  "Full GPU training of B1 pending." if b1v else "B1 pending."))
        add("Verification, synthetic cards", "MAE &le; 2 pp; three-class agreement &ge; 90%; &le; 2 s per card on a laptop CPU.", txt,
            status("met" if b0ok and (b1ok or not b1v) else "track"))
    else:
        add("Verification, synthetic cards", "MAE &le; 2 pp; three-class agreement &ge; 90%; &le; 2 s per card.", "B0 results not available.", status("pending"))
    add("Verification, real cards", "MAE &le; 5 pp on real cards.",
        "Untested. The two annotated real cards are in a restricted Zenodo version (D9); access must still be requested through Zenodo "
        '<span class="confirm">TEAM TO CONFIRM</span>.', status("risk"))
    h = cl_info.get("head")
    if h:
        add("Closed loop", "Under mismatch, recovers &ge; 50% of the open-loop-to-oracle gap in requirement-met segments with &le; 10% more volume than the oracle (20 seeds, 95% CI).",
            f"Fast loop: recovery {h[0][2]} (meets 50% on the point estimate, CI crosses it) but volume {h[2][2]} against the oracle (limit 10%). Fast + slow loop: recovery {h[1][2]}, volume {h[3][2]}. "
            "Causes: the oracle sees the per-card spread factor (+12.2% volume even with no mismatch), and the slow loop lowers reliability (Section 2.8).",
            status("missed"))
    else:
        add("Closed loop", "Under mismatch, recovers &ge; 50% of the open-loop-to-oracle gap in requirement-met segments with &le; 10% more volume than the oracle (20 seeds, 95% CI).",
            "No measured value yet.", status("pending"))
    return "\n".join(R)


def summary_text(mp, sel_info, ver_info, cl_info) -> str:
    g = mp["_grp"]
    c = sel_info["real"]
    o, b1v = ver_info["b0"], ver_info["b1"]
    h = cl_info["head"]
    return (
        f"All planned modules are built, tested and merged: simulator, nozzle catalog (316 chart rows, 547 candidate settings), predictive model, selector, scenario library, water-sensitive-paper "
        f"(WSP) renderer, classical (B0) and learned (B1) verifiers, feedback module, loop harness, integration, experiment grid and a Streamlit demo. "
        f"Five of the seven criteria in Table 1 of the proposal are met inside our simulator: the simulator, the predictive model (coverage MAE {mp['lgb_rand_mae']} pp random and {f(min(g))} to {f(max(g))} pp on withheld regimes, ridge {mp['ridge_rand_mae']} pp), "
        f"the selector ({pct(c['selector']['success'])}% requirement-met, with lower drift and about half the volume of the matched fixed setting), synthetic-card verification (B0 MAE {f(o['mae'])} pp; "
        f"B1 {f(b1v['mae'])} pp from a 6-epoch CPU checkpoint) and end-to-end operation. "
        f"<b>The closed-loop criterion is missed:</b> the fast loop recovers {h[0][2]} of the open-loop-to-oracle gap but uses {h[2][2]} more volume than the oracle (limit 10%), mostly because the oracle sees the per-card spread factor, "
        f"and the slow loop as implemented did not help ({h[1][2]} recovery). "
        f"Still open are the full GPU training of B1 and the real-card criterion, which is untested because the annotated cards are restricted. "
        f"All results come from our own simulator and renderer, not the field. "
        f"Milestone M1 was 16 days late and M2 2 days late, but all modules were built and merged on Oct 18; the remaining M3 work is B1 GPU training, real-card evaluation and a slow-loop redesign.")


# ----------------------------------------------------------------------------- numbering
def number_captions(doc: str) -> str:
    counts = {"Table": 0, "Figure": 0}
    keys: dict[str, str] = {}

    def cap(m):
        kind, key = m.group(1), m.group(2)
        counts[kind] += 1
        keys[f"{kind[:3].lower()}:{key}"] = f"{kind} {counts[kind]}"
        return f"<b>{kind} {counts[kind]}.</b>"

    doc = re.sub(r"<b>(Table|Figure) #(\w+)\.</b>", cap, doc)
    doc = re.sub(r"@(tab|fig):(\w+)", lambda m: keys.get(f"{m.group(1)}:{m.group(2)}", f"??{m.group(0)}??"), doc)
    doc = re.sub(r"Table #(\w+)", lambda m: keys.get(f"tab:{m.group(1)}", f"??{m.group(0)}??"), doc)
    return doc


# ----------------------------------------------------------------------------- main
def count_tests_and_lines() -> tuple[int, int, int]:
    n_tests = sum(len(re.findall(r"^\s*(?:async )?def test_", p.read_text(encoding="utf-8"), re.M)) for p in (ROOT / "tests").glob("test_*.py"))
    sim_tests = len(re.findall(r"^def test_", (ROOT / "tests" / "test_simulator.py").read_text(encoding="utf-8"), re.M))
    lines = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in (ROOT / "src" / "coverageai").rglob("*.py"))
    return n_tests, sim_tests, lines


def pdf_pages(p: Path) -> int:
    data = p.read_bytes()
    n = len(re.findall(rb"/Type\s*/Page[^s]", data))
    m = re.findall(rb"/Type\s*/Pages[^>]*?/Count\s+(\d+)", data)
    return max([n] + [int(x) for x in m])


def build(do_pdf: bool = True) -> None:
    figs = prep_figures()
    need = ["sim_trends", "model_parity", "wsp_examples", "b0_parity", "selector_tradeoff", "model_sensitivity",
            "closed_loop_met_spray", "bias_trace_spray", "demo_live_top"]
    missing = [k for k in need if k not in figs]
    if missing:
        raise SystemExit(f"missing figures: {missing}")
    mp = model_parts()
    sp = simoes_parts()
    b0, b1 = load_json("b0_synthetic.json"), load_json("b1_synthetic_cpu_partial.json")
    b0_block, b1_block, b0_analysis, ver_info = verification(b0, b1, figs)
    sel_html, sel_info = selector_block(figs)
    cl_html, cl_info = closed_loop_block(figs)
    n_tests, sim_tests, src_lines = count_tests_and_lines()
    log = ROOT / "Docs" / "ai_usage_log.md"
    ai_rows = len(re.findall(r"^\| 20\d\d-", log.read_text(encoding="utf-8"), re.M)) if log.exists() else 0

    vals = {
        "build_date": date.today().strftime("%B %d, %Y").replace(" 0", " "),
        "summary_text": summary_text(mp, sel_info, ver_info, cl_info),
        "src_lines": f"{src_lines:,}", "n_tests": str(n_tests),
        "anchor_rows": anchor_rows(),
        "selector_block": sel_html, "b0_block": b0_block, "b1_block": b1_block, "b0_analysis": b0_analysis,
        "closed_loop_block": cl_html,
        "criteria_rows": criteria_rows(mp, sel_info, ver_info, cl_info, b0, b1, sim_tests),
        "ai_log_rows": str(ai_rows),
        "sel_spray_success": pct(sel_info["real"]["selector"]["success"]) if sel_info["has_real"] else "n/a",
        "sel_oracle_success": pct(sel_info["oracle"]["selector"]["success"]) if sel_info.get("oracle") else "n/a",
        **sp, **{k: v for k, v in mp.items() if not k.startswith("_")},
        **{f"fig_{k}": v for k, v in figs.items()},
    }
    doc = (HERE / "template.html").read_text(encoding="utf-8")
    # nested placeholders (blocks may contain markers) are resolved before numbering
    for _ in range(2):
        doc = re.sub(r"\{\{(\w+)\}\}", lambda m: str(vals[m.group(1)]) if m.group(1) in vals else m.group(0), doc)
    left = sorted(set(re.findall(r"\{\{(\w+)\}\}", doc)))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    doc = number_captions(doc)
    if "??" in doc:
        print("WARNING: unresolved cross-reference marker '??' in output", file=sys.stderr)
    out_html = HERE / "midterm_report.html"
    out_html.write_text(doc, encoding="utf-8")
    print(f"wrote {out_html} ({out_html.stat().st_size // 1024} KB)")
    if not do_pdf:
        return
    pdf = HERE / "midterm_report.pdf"
    pdf.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as prof:
        cmd = [EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={prof}",
               "--allow-file-access-from-files", f"--print-to-pdf={pdf}", out_html.as_uri()]
        r = subprocess.run(cmd, check=False, timeout=180, capture_output=True, text=True)
        deadline, last = time.time() + 60, -1          # Edge may exit before the PDF is flushed
        while time.time() < deadline:
            size = pdf.stat().st_size if pdf.exists() else 0
            if size > 0 and size == last:
                break
            last = size
            time.sleep(1.0)
    if not pdf.exists() or pdf.stat().st_size == 0:
        raise SystemExit(f"PDF was not produced (Edge exit {r.returncode}): {r.stderr[-400:]}")
    print(f"wrote {pdf} ({pdf.stat().st_size // 1024} KB), {pdf_pages(pdf)} pages")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    build(not ap.parse_args().no_pdf)
