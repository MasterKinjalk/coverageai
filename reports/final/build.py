"""Build the ABE 598 final report and the final scorecard: results -> filled HTML -> PDF via headless Edge.

    uv run --no-sync python reports/final/build.py            # HTML, PDF and results/scorecard.md
    uv run --no-sync python reports/final/build.py --no-pdf   # HTML and scorecard only

Derived from reports/midterm/build.py (which is left unchanged). Every number in the report and in
results/scorecard.md is read from a results file at build time:
    results/model_v1.json, selector_eval.json, real_data_summary.json, verification_eval.json,
    closed_loop_summary.md (tables parsed), boom_control_eval.md, speed_control_eval.md,
    forecast_eval.json, hybrid_eval.md, Docs/ai_usage_log.md, Docs/decisions.md, experiments/figures/*.png
Typed numbers are limited to the constants below, each with a comment giving its source.

Two inputs from the orchestrating session (both received 2026-10-19); if either is set to None the
report shows a dashed PENDING box and the scorecard says "pending":
    APP_CHECK       (a) result of the Streamlit app browser check (HTML-safe text)
    B1_IDLE_RETIME  (b) B0/B1 per-card CPU time re-measured on a lightly loaded desktop CPU,
                        read from results/b1_idle_retime.json
Placeholders in template.html use {{name}}. Captions use "Table #key." / "Figure #key." and prose uses
@tab:key / @fig:key; numbers are assigned in document order.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "results"
FIG_SRC = [ROOT / "experiments" / "figures"]
FIG_DST = HERE / "figures"
SCORECARD_MD = RESULTS / "scorecard.md"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
MAX_W = 1600
JPEG_THRESHOLD = 700 * 1024      # PNGs still larger than this after resizing become JPEG

# ----------------------------------------------------------------------------- orchestrator inputs
# (a) Streamlit app browser check, summarized from the browser-check agent's report to the
#     orchestrator (2026-10-19; commit 15229a8 "App: fix browser-check bugs, add opt-in
#     speed/boom/forecast controls"). The 30 s / 40-card U-Net figure is that agent's measurement.
APP_CHECK: str | None = (
    "Checked on 2026-10-19 with headless Chrome over CDP against the live app: all six strategies and the perfect, "
    "classic and noisy readers ran; play/pause, speeds, scrubber, hover and click-to-seek and the end-of-run summary work; "
    "<code>fast_slow</code> shows the learning-between-passes table and the card replay is consistent for every strategy at "
    "med and high mismatch; the Compare tab, Spray lab and nozzle explorer render. A full 40-card pass with the U-Net reader "
    "takes about 30 s on the idle desktop CPU. Five display bugs found in the check were fixed; the app tests pass.")
# (b) Per-card reader timing re-measured on 2026-10-19 (same protocol as results/verification_eval.json:
#     30 renderer test cards, warm-up, min of 3 calls, B1 on CPU). Read from the results file.
B1_IDLE_RETIME: dict | None = None     # filled in build() by idle_retime()

# ----------------------------------------------------------------------------- typed constants (with sources)
# Orchestrator measurement quoted in the midterm work brief (2026-10-18), also typed in
# reports/midterm/build.py: `uv run python -m coverageai.e2e --real`.
E2E_REAL = {"segments": 40, "seconds": 33, "mean_abs_err_pp": 0.57}
# Environment facts reported by the orchestrating session on 2026-10-19 (full suite run on the
# new machine: 32 logical cores, RTX 5070 Ti, CUDA torch 2.11+cu128).
SUITE = {"tests": 394, "machine": "32 logical cores, RTX 5070 Ti"}


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


def read_md(name: str) -> str:
    p = RESULTS / name
    if not p.exists():
        raise SystemExit(f"results/{name} is required")
    return p.read_text(encoding="utf-8")


def idle_retime() -> dict | None:
    """results/b1_idle_retime.json -> {"cpu", "load", "b0", "b1", "b1_single"} or None."""
    d = load_json("b1_idle_retime.json")
    if not d:
        return None
    return {"cpu": d["cpu"], "load": d.get("cpu_load_note", ""), "b0": d["min_of_3"]["b0"], "b1": d["min_of_3"]["b1"],
            "b1_single": d.get("single_call", {}).get("b1")}


def pending(title: str, text: str) -> str:
    return (f'<div class="pendbox"><b>PENDING &mdash; INPUT NOT YET RECEIVED.</b> '
            f'<b style="color:#333">{html.escape(title)}</b> {text}</div>')


def f(x, n=2):
    return f"{x:.{n}f}"


def pct(x, n=1):
    return f"{100 * x:.{n}f}"


def sgn(x, n=1):
    return f"{x:+.{n}f}"


STATUS_LABEL = {"met": "met", "missed": "missed", "partial": "partial", "blocked": "blocked",
                "neg": "done-negative", "pending": "pending"}


def status(kind: str, text: str | None = None) -> str:
    return f'<span class="st {kind}">{text or STATUS_LABEL[kind]}</span>'


# ----------------------------------------------------------------------------- markdown tables
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


def dicts(t) -> list[dict]:
    hdr, rows = t
    return [dict(zip(hdr, r)) for r in rows]


def pick(rows: list[dict], **kw) -> dict:
    for r in rows:
        if all(r.get(k.replace("__", " ")) == v for k, v in kw.items()):
            return r
    raise SystemExit(f"row not found: {kw}")


def section(md: str, start: str, end: str | None = None) -> str:
    if start not in md:
        raise SystemExit(f"section {start!r} not found")
    s = md.split(start, 1)[1]
    return s.split(end, 1)[0] if end and end in s else s


def _cell(c: str) -> str:
    c = html.escape(c).replace("&gt;=", "&ge;").replace("&lt;=", "&le;").replace("+-", "&plusmn;")
    c = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", c)
    c = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<i>\1</i>", c)
    c = re.sub(r"`([^`]+)`", r"<code>\1</code>", c)
    c = c.replace("C_req", "<i>C</i><sub>req</sub>").replace("pp^2", "pp<sup>2</sup>")
    return re.sub(r"(?<![\w.])-(?=\d)", "&minus;", c)


def md_table_html(hdr, rows, num_from=None, widths=None, cls="") -> str:
    th = "".join(f'<th{" class=\"num\"" if num_from is not None and j >= num_from else ""}'
                 f'{f" style=\"width:{widths[j]}%\"" if widths else ""}>{_cell(h)}</th>' for j, h in enumerate(hdr))
    body = "".join("<tr>" + "".join(f'<td{" class=\"num\"" if num_from is not None and j >= num_from else ""}>{_cell(c)}</td>'
                                    for j, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<table{f" class=\"{cls}\"" if cls else ""}><tr>{th}</tr>{body}</table>'


def lo_hi(s: str) -> tuple[float, float, float]:
    """'+4.1% [+2.8, +5.5]' -> (4.1, 2.8, 5.5)."""
    m = re.match(r"\s*([+-]?[\d.]+)\s*%?\s*\[\s*([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)\s*\]", s.replace(" %", "%"))
    if not m:
        raise SystemExit(f"cannot parse CI from {s!r}")
    return tuple(float(g) for g in m.groups())  # type: ignore[return-value]


def ci_pp(s: str) -> str:
    """'-0.0287 [-0.0362, -0.0206]' (fraction) -> '-2.9 pp [-3.6, -2.1]'."""
    v, lo, hi = lo_hi(s)
    return f"{100 * v:+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}]"


def ci_r(s: str, n: int = 2) -> str:
    """'-2.5927 [-3.0943, -2.1093]' -> '-2.59 [-3.09, -2.11]'."""
    v, lo, hi = lo_hi(s)
    return f"{v:+.{n}f} [{lo:+.{n}f}, {hi:+.{n}f}]"


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
    used = {"sim_trends", "model_parity", "wsp_examples", "b0_parity", "selector_tradeoff", "forecast_eval",
            "closed_loop_volume_drift"}

    def save(im, name):
        if im.width > MAX_W:
            im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
        if im.mode not in ("RGB", "RGBA", "L", "P"):
            im = im.convert("RGB")
        dst = FIG_DST / f"{name}.png"
        (FIG_DST / f"{name}.jpg").unlink(missing_ok=True)
        im.save(dst, optimize=True)
        out[name] = f"figures/{dst.name}"
        if dst.stat().st_size > JPEG_THRESHOLD:
            jpg = FIG_DST / f"{name}.jpg"
            im.convert("RGB").save(jpg, quality=88, optimize=True)
            dst.unlink()
            out[name] = f"figures/{jpg.name}"

    for stem in used:
        if stem in seen:
            save(Image.open(seen[stem]), stem)
    crops = {                                  # derived crops (box in source pixels)
        "closed_loop_met": ("closed_loop_met_spray", (0, 0, 1950, 380)),     # top row: learned model
        "slow_loop_residuals": ("slow_loop_spray", (0, 0, 2145, 380)),       # top row: learned model
        "bias_trace": ("bias_trace_spray", (0, 0, 812, 702)),                # left panel: learned model
        "b1_parity": ("b1_parity_left", (0, 0, 676, 600)),                   # parity panel only
        "demo_live": ("demo_live_top", (0, 0, 1500, 1190)),
    }
    for stem, (name, box) in crops.items():
        if stem in seen:
            save(Image.open(seen[stem]).convert("RGB").crop(box), name)
    return out


# ----------------------------------------------------------------------------- model / dataset (as midterm)
def model_parts() -> dict:
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
    g = lambda k, o: sp[k]["models"]["lightgbm"][o]["mae"]  # noqa: E731
    return {
        "model_rows": "\n".join(rows), "model_headline": headline,
        "noise_floor": f(floor), "lgb_rand_mae": f(lg_r), "gap_to_floor": f(lg_r - floor),
        "lgb_exp_mae": f(rnd["lightgbm"]["coverage_pct"]["mae_vs_expected"]),
        "ridge_exp_mae": f(rnd["ridge"]["coverage_pct"]["mae_vs_expected"]),
        "ridge_rand_mae": f(rnd["ridge"]["coverage_pct"]["mae"]),
        "grp_min": f(min(grp)), "grp_max": f(max(grp)),
        "shift_rand": f(g("random", "displacement_mm"), 1), "shift_u5": f(g("U_gt_5", "displacement_mm"), 1),
        "shift_3d": f(g("family_3D", "displacement_mm"), 1),
        "drift_rand": f(g("random", "drift_frac"), 4), "drift_u5": f(g("U_gt_5", "drift_frac"), 4),
        "sens_pass": str(n_pass), "sens_total": str(len(sens)),
        "dataset_rows": f"{d['dataset']['rows']:,}",
        "_cov_random": lg_r, "_ridge_random": rnd["ridge"]["coverage_pct"]["mae"], "_grp": grp,
        "_sens_ok": n_pass == len(sens),
    }


def simoes_parts() -> dict[str, str]:
    d = load_json("real_data_summary.json")
    s = (d or {}).get("subsets", {}).get("synthetic", {})
    if not s:
        raise SystemExit("results/real_data_summary.json (synthetic subset) is required")
    return {"simoes_n": str(s["n_images"]), "simoes_droplets": f"{s['n_droplets_total']:,}",
            "simoes_cov_min": f(s["coverage_pct_min"], 1), "simoes_cov_max": f(s["coverage_pct_max"], 1),
            "simoes_cov_mean": f(s["coverage_pct_mean"], 1)}


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
    try:
        sys.path.insert(0, str(ROOT / "tests"))
        import numpy as np
        from test_simulator import VMD_CENTER, cond, make_sim, run  # type: ignore
        drift = {}
        for c in ("F", "M", "C"):
            sim = make_sim(vmd=VMD_CENTER[c], flow_gpm=0.4)
            drift[c] = float(np.mean([run(sim, cond(2.0), s).drift_frac for s in range(6)]))
    except Exception as e:
        raise SystemExit(f"live drift anchors unavailable ({e})")
    for c, name, band in (("F", "fine", "10% to 35%"), ("M", "medium", "2% to 15%"), ("C", "coarse", "below 5%")):
        rows.append(f'<tr><td>Drift fraction, {name} spray ({c}), 2 m/s crosswind</td><td>{band}</td>'
                    f'<td class="num">{100 * drift[c]:.0f}%</td></tr>')
    return "\n".join(rows)


# ----------------------------------------------------------------------------- selector
def _setting(s: dict) -> str:
    return f"{s['family']} {s['size']} @ {s['pressure_psi']:g} psi"


def selector_block(figs) -> tuple[str, dict]:
    d = load_json("selector_eval.json")
    cfgs = (d or {}).get("configs", {})
    sp, orc = cfgs.get("spray"), cfgs.get("oracle")
    if not sp:
        raise SystemExit("results/selector_eval.json has no 'spray' configuration")
    n, creq = d["n"], d["c_req"]
    s, mf, bf = sp["selector"], sp["fixed_matched_success"], sp["best_fixed"]
    rt = sp["runtime_ms"]

    def row(label, c, ms=None, bold=False):
        cls = ' class="best"' if bold else ""
        return (f'<tr{cls}><td>{label}</td><td class="num">{pct(c["success"])}</td>'
                f'<td class="num">{f(c["mean_drift"], 4)}</td><td class="num">{f(c["mean_gpa"], 1)}</td>'
                f'<td class="num">{f(ms, 1) if ms is not None else "&ndash;"}</td></tr>')

    rows = (row("Selector, learned model (<code>spray</code>)", s, rt["total_mean"], True)
            + row(f"Lowest-drift fixed setting with matching success: {_setting(mf['setting'])}", mf)
            + row(f"Best fixed setting overall: {_setting(bf['setting'])}", bf)
            + (row("Selector, simulator as model, no interval (<code>oracle</code>)", orc["selector"], orc["runtime_ms"]["total_mean"]) if orc else ""))
    tbl = (f'<div class="caption"><b>Table #select.</b> Selector against fixed settings on {n} random no-mismatch scenarios at '
           f'<i>C</i><sub>req</sub> = {creq:g}% (success = share with true coverage &ge; <i>C</i><sub>req</sub>). The learned model selects and the simulator scores; '
           f'all {sp["n_candidates"]} candidates are scored per call. Single draw per scenario, no CI.</div>'
           '<table><tr><th>Policy</th><th class="num">Success (%)</th><th class="num">Mean drift (fraction)</th>'
           '<th class="num">Mean volume (GPA)</th><th class="num">Time per selection (ms)</th></tr>' + rows + "</table>")
    lower = 100 * (1 - s["mean_drift"] / mf["mean_drift"])
    txt = (f"<p>The selector meets the requirement in {pct(s['success'])}% of the {n} scenarios (target &ge; 90%). The lowest-drift fixed setting "
           f"with matching success ({_setting(mf['setting'])}, {pct(mf['success'], 0)}%) has drift {f(mf['mean_drift'], 4)} against {f(s['mean_drift'], 4)} "
           f"({lower:.0f}% lower for the selector) and applies {f(mf['mean_gpa'], 1)} GPA against {f(s['mean_gpa'], 1)}. One selection takes "
           f"{f(rt['total_mean'], 0)} ms. Without a prediction interval (simulator as model, 1.5 pp margin floor only) success falls to "
           f"{pct(orc['selector']['success']) if orc else 'n/a'}%: the per-card spread factor is noise no model can predict, so the margin must come from a "
           f"calibrated interval (@fig:selsens).</p>")
    fig = (f'<figure style="margin:0"><img src="{figs["selector_tradeoff"]}" alt="Selector trade-off">'
           '<figcaption><b>Figure #selsens.</b> Mean drift against mean volume as &lambda; varies (0, 0.3, 0.6, 1.0). Green: learned model selects; '
           'blue: simulator selects and scores; square: best fixed setting.</figcaption></figure>')
    info = {"real": sp, "oracle": orc, "n": n}
    return tbl + f'<div class="twoup" style="align-items:center"><div style="flex:1.6">{txt}</div><div style="flex:1">{fig}</div></div>', info


# ----------------------------------------------------------------------------- verification
def verification_block(figs) -> tuple[str, dict]:
    v = load_json("verification_eval.json")
    if v is None:
        raise SystemExit("results/verification_eval.json is required")
    R = v["results"]
    tm = v["timing"]
    info = {
        "b0r": R["b0"]["renderer"]["overall"], "b1r": R["b1"]["renderer"]["overall"],
        "b0s": R["b0"]["simoes"]["overall"], "b1s": R["b1"]["simoes"]["overall"],
        "b1s_bands": R["b1"]["simoes"]["by_coverage_band"], "b0s_bands": R["b0"]["simoes"]["by_coverage_band"],
        "t_b0": tm["b0"], "t_b1": tm["b1"], "b1_status": v["meta"].get("b1_status", ""),
        "real_blocked": "blocked" in R["b0"]["real"],
    }
    rows = []
    for ver, lab in (("b0", "B0 classical"), ("b1", "B1 U-Net")):
        for src, slab in (("renderer", "own renderer"), ("simoes", "Sim&otilde;es synthetic")):
            r = R[ver][src]
            for band, b in r["by_coverage_band"].items():
                if not b.get("n"):
                    continue
                rows.append(f'<tr><td>{lab}</td><td>{slab}</td><td>{html.escape(band)}%</td><td class="num">{b["n"]}</td>'
                            f'<td class="num">{f(b["mae_pp"])}</td><td class="num">{sgn(b["bias_pp"], 2)}</td>'
                            f'<td class="num">{pct(b["class_agreement"])}</td><td class="num">{f(b["iou_mean"], 3)}</td></tr>')
            o = r["overall"]
            rows.append(f'<tr class="best"><td>{lab}</td><td>{slab}</td><td>all</td><td class="num">{o["n"]}</td>'
                        f'<td class="num">{f(o["mae_pp"])}</td><td class="num">{sgn(o["bias_pp"], 2)}</td>'
                        f'<td class="num">{pct(o["class_agreement"])}</td><td class="num">{f(o["iou_mean"], 3)}</td></tr>')
    tbl = ('<div class="caption"><b>Table #verify.</b> Card-reading accuracy by true-coverage band on the 180 held-out cards of our renderer and on the '
           '300 Sim&otilde;es et al. synthetic cards (out-of-generator; no card above 50%). Coverage MAE and signed bias in pp, three-class agreement, '
           'mean stain-mask IoU. Targets: MAE &le; 2 pp, agreement &ge; 90%. Source: <code>results/verification_eval.json</code>.</div>'
           '<table class="big"><tr><th>Verifier</th><th>Cards</th><th>True coverage</th><th class="num">n</th><th class="num">MAE (pp)</th>'
           '<th class="num">Bias (pp)</th><th class="num">Class agreement (%)</th><th class="num">IoU</th></tr>' + "".join(rows) + "</table>")
    b0r, b1r, b0s, b1s = info["b0r"], info["b1r"], info["b0s"], info["b1s"]
    sb = info["b1s_bands"]
    text = (
        f"<p><b>B1 is the full model now.</b> The U-Net was trained on a GPU ({html.escape(info['b1_status'])}); the midterm's 6-epoch CPU checkpoint is retired. "
        f"On our renderer's test cards B1 reaches {f(b1r['mae_pp'])} pp MAE and {pct(b1r['class_agreement'])}% agreement against {f(b0r['mae_pp'])} pp and "
        f"{pct(b0r['class_agreement'])}% for B0. B0's error concentrates above 50% coverage, where stains merge "
        f"({f(R['b0']['renderer']['by_coverage_band']['50+']['mae_pp'])} pp, bias {sgn(R['b0']['renderer']['by_coverage_band']['50+']['bias_pp'], 2)} pp); B1 stays at or below "
        f"{f(max(b['mae_pp'] for b in R['b1']['renderer']['by_coverage_band'].values()))} pp in every band.</p>"
        f"<p><b>Outside the generator both readers lose accuracy, and B1 loses more than its own-card numbers suggest.</b> On the Sim&otilde;es cards B1 has "
        f"{f(b1s['mae_pp'])} pp MAE (bias {sgn(b1s['bias_pp'], 2)} pp, every band underread) and B0 {f(b0s['mae_pp'])} pp. Class agreement is "
        f"{pct(b1s['class_agreement'])}% (B1) and {pct(b0s['class_agreement'])}% (B0), below the 90% target. The misses sit at 15 to 50% coverage: in the 30&ndash;50% band "
        f"({sb['30-50']['n']} cards) B1 underreads by {f(-sb['30-50']['bias_pp'])} pp and agrees on the class for {pct(sb['30-50']['class_agreement'], 0)}% of cards; in the 15&ndash;30% band "
        f"({sb['15-30']['n']} cards) by {f(-sb['15-30']['bias_pp'])} pp with {pct(sb['15-30']['class_agreement'], 0)}% agreement. Underreading near a class limit reports a card as worse "
        f"than it is, which is the safe direction for re-treatment but would make the fast loop spray more. The Sim&otilde;es cards use an inferred pixel scale (no dpi is published) "
        f"and contain no card above {f(load_json('real_data_summary.json')['subsets']['synthetic']['coverage_pct_max'], 0)}% coverage.</p>")
    t0, t1 = info["t_b0"], info["t_b1"]
    timing = (f"<p><b>Runtime.</b> Controlled timing ({t0['n']} renderer cards, minimum of {t0['repeats']} calls after warm-up, laptop CPU): B0 median "
              f"{f(t0['median_s'])} s, p95 {f(t0['p95_s'])} s; B1 median {f(t1['median_s'])} s, p95 {f(t1['p95_s'])} s, maximum {f(t1['max_s'])} s. Both are inside "
              f"the 2 s target, but the B1 numbers were taken while other experiments loaded the CPU.</p>")
    if B1_IDLE_RETIME:
        t = B1_IDLE_RETIME
        b0i, b1i, b1s1 = t["b0"], t["b1"], t.get("b1_single")
        timing = (f"<p><b>Runtime: met.</b> Controlled timing on the project laptop ({t0['n']} renderer cards, minimum of {t0['repeats']} calls after warm-up) "
                  f"gave B0 median {f(t0['median_s'])} s, p95 {f(t0['p95_s'])} s, and B1 median {f(t1['median_s'])} s, p95 {f(t1['p95_s'])} s, maximum {f(t1['max_s'])} s, "
                  "while other experiments loaded the CPU. Contention can only slow a reader down, so these are upper bounds for that laptop and already inside the "
                  f"2 s target. The same protocol on the final desktop ({html.escape(t['cpu'])}, lightly loaded, B1 forced to CPU) gives B0 median {f(b0i['median_s'])} s, "
                  f"p95 {f(b0i['p95_s'])} s, and B1 median {f(b1i['median_s'])} s, p95 {f(b1i['p95_s'])} s, maximum {f(b1i['max_s'])} s"
                  + (f" (single call without the minimum of 3: B1 p95 {f(b1s1['p95_s'])} s, maximum {f(b1s1['max_s'])} s)" if b1s1 else "")
                  + ". This shows the readers' cost without contention; it is a desktop, not a laptop number. Source: <code>results/b1_idle_retime.json</code>.</p>")
    else:
        timing += pending("B1 idle-CPU timing.", "The B1 per-card CPU time is being re-measured on an idle CPU (target &le; 2 s on a laptop CPU; the 1.09 s p95 "
                                                  "above was measured on a busy CPU). Fill <code>B1_IDLE_RETIME</code> in <code>reports/final/build.py</code>.")
    real = ("<p><b>Real annotated cards: blocked.</b> The two annotated real cards (2,754 droplets) and the pre-annotated real squares are in Zenodo record 14010262, "
            "whose files are restricted (D9). The tile evaluation is implemented and runs as soon as the files are placed under "
            "<code>data/external/simoes2025/real/</code>; access has to be requested through Zenodo (draft email: <code>reports/email_draft_simoes.md</code>) "
            '<span class="confirm">TEAM TO CONFIRM whether the request was sent</span>. The real-card criterion (&le; 5 pp) is therefore untested.</p>')
    fig = (f'<figure><div class="twoup" style="align-items:center">'
           f'<div style="flex:1.5"><img src="{figs["wsp_examples"]}" alt="Synthetic WSP cards"></div>'
           f'<div style="flex:1"><img src="{figs["b0_parity"]}" alt="B0 parity"></div>'
           f'<div style="flex:1"><img src="{figs["b1_parity_left"]}" alt="B1 parity"></div></div>'
           '<figcaption><b>Figure #wsp.</b> Left: synthetic cards from our renderer, 5% to 76% true coverage, scanner and phone profiles. Middle: B0 coverage against true '
           'coverage on the 180 held-out renderer cards (shaded: &plusmn;2 pp). Right: B1 (GPU-trained) on the same cards; this panel comes from the GPU-host evaluation '
           '(<code>results/b1_synthetic.json</code>, MAE ' + f(load_json("b1_synthetic.json")["overall"]["coverage_mae_pp"]) +
           ' pp), the table from the local re-evaluation. Lines mark the 15% and 30% class limits.</figcaption></figure>')
    return tbl + text + fig + timing + real, info


# ----------------------------------------------------------------------------- closed loop
def closed_loop_parts(figs) -> tuple[str, dict]:
    md = read_md("closed_loop_summary.md")
    head = md.split("## Model `spray`", 1)[0]
    spray = section(md, "## Model `spray`", "## Model `oracle_nominal`")
    nom = section(md, "## Model `oracle_nominal`", "## Slow-loop diagnostic")
    T = md_tables(spray)
    TN = md_tables(nom)
    crit = dicts(next(t for t in T if t[0][0] == "reference"))
    critn = dicts(next(t for t in TN if t[0][0] == "reference"))
    qsens = dicts(next(t for t in T if t[0][0] == "oracle q"))
    bym = [t for t in T if t[0][0] == "mismatch" and "gap (pp)" in t[0]]
    fair_m, card_m = dicts(bym[0]), dicts(bym[1])
    preset = dicts(next(t for t in T if t[0][0] == "preset"))
    polt = [t for t in T if t[0][0] == "policy" and "none" in t[0]]
    met_t, gpa_t = polt[0], polt[1]
    abl = next(t for t in T if t[0][0] == "group")
    passes = [t for t in T if t[0][0] == "pass"]
    resid_head = dicts(next(t for t in md_tables(head) if t[0][:2] == ["mismatch", "metric"]))

    g = lambda ref, pol, col: pick(crit, reference=ref, policy=pol)[col]  # noqa: E731
    REC, VOL = "gap recovery (target >= 50%)", "volume vs oracle (target <= +10%)"
    c = {
        "f_rec": g("fair oracle", "fast", REC), "f_vol": g("fair oracle", "fast", VOL),
        "fs_rec": g("fair oracle", "fast_slow", REC), "fs_vol": g("fair oracle", "fast_slow", VOL),
        "fs_met": g("fair oracle", "fast_slow", "met: open / policy / oracle"),
        "f_met": g("fair oracle", "fast", "met: open / policy / oracle"),
        "c_f_rec": g("card-aware oracle", "fast", REC), "c_f_vol": g("card-aware oracle", "fast", VOL),
        "c_fs_rec": g("card-aware oracle", "fast_slow", REC), "c_fs_vol": g("card-aware oracle", "fast_slow", VOL),
        "fair_over_card": pick(card_m, mismatch="low+med+high", policy="oracle")["volume vs oracle"],
        "fair_over_card_none": pick(card_m, mismatch="none", policy="oracle")["volume vs oracle"],
        "n_f_rec": pick(critn, reference="fair oracle", policy="fast")[REC],
        "n_fs_rec": pick(critn, reference="fair oracle", policy="fast_slow")[REC],
        "n_f_vol": pick(critn, reference="fair oracle", policy="fast")[VOL],
        "n_fs_vol": pick(critn, reference="fair oracle", policy="fast_slow")[VOL],
        "gap_all": pick(fair_m, mismatch="low+med+high", policy="fast")["gap (pp)"],
        "gap_none": pick(fair_m, mismatch="none", policy="fast")["gap (pp)"],
        "gap_low": pick(fair_m, mismatch="low", policy="fast")["gap (pp)"],
        "gap_med": pick(fair_m, mismatch="med", policy="fast")["gap (pp)"],
        "gap_high": pick(fair_m, mismatch="high", policy="fast")["gap (pp)"],
        "fs_rec_med": pick(fair_m, mismatch="med", policy="fast_slow")["gap recovery"],
        "fs_rec_high": pick(fair_m, mismatch="high", policy="fast_slow")["gap recovery"],
        "f_rec_med": pick(fair_m, mismatch="med", policy="fast")["gap recovery"],
        "f_rec_high": pick(fair_m, mismatch="high", policy="fast")["gap recovery"],
        "fs_vol_high": pick(fair_m, mismatch="high", policy="fast_slow")["volume vs oracle"],
        "hot_f": pick(preset, preset="hot_dry", policy="fast")["gap recovery"],
        "q05_f": pick(qsens, oracle__q="0.05", policy="fast")["gap recovery"],
        "q05_fs": pick(qsens, oracle__q="0.05", policy="fast_slow")["gap recovery"],
        "q05_fs_vol": pick(qsens, oracle__q="0.05", policy="fast_slow")["volume vs oracle"],
        "q05_met": pick(qsens, oracle__q="0.05", policy="fast")["oracle met"],
    }
    preset_fs = [pick(preset, preset=p, policy="fast_slow") for p in ("calm_morning", "crosswind_shift", "gusty_afternoon", "hot_dry")]
    c["preset_fs_min"] = min(preset_fs, key=lambda r: lo_hi(r["gap recovery"])[0])["gap recovery"].split(" [")[0]
    c["preset_fs_max"] = max(preset_fs, key=lambda r: lo_hi(r["gap recovery"])[0])["gap recovery"].split(" [")[0]
    hv = pick(resid_head, mismatch="high", metric="residual variance (pp^2)")
    hm = pick(resid_head, mismatch="high", metric="mean residual (pp)")
    hmse = pick(resid_head, mismatch="high", metric="residual MSE (pp^2)")
    c.update(var_p1=hv["pass 1"], var_p5=hv["pass 5"], mean_p1=hm["pass 1"], mean_p2=hm["pass 2"], mean_p5=hm["pass 5"],
             mse_p1=hmse["pass 1"], mse_p2=hmse["pass 2"])
    allvar = [float(r[k]) for r in resid_head if r["metric"].startswith("residual variance") for k in r if k.startswith("pass")]
    c["var_min"], c["var_max"] = f"{min(allvar):.1f}", f"{max(allvar):.1f}"
    A = dicts(abl)
    sl = lambda key: next(r for r in A if r["group"] == "slow loop" and key in r["variant"])  # noqa: E731
    c["abl_res"], c["abl_w30"], c["abl_w1"] = sl("per regime"), sl("record weight 30"), sl("record weight 1")
    c["abl_b0"] = next(r for r in A if r["variant"].startswith("verifier=b0"))
    c["abl_b1"] = next(r for r in A if r["variant"].startswith("verifier=b1"))
    c["abl_alpha"] = [pick(A, variant=f"alpha={a}") for a in ("0.1", "0.3", "0.6")]
    c["abl_global"] = pick(A, variant="global bias")
    c["abl_regime"] = pick(A, variant="per-regime bias")

    design = (
        "<p><b>Setup.</b> The grid (<code>experiments/closed_loop.py</code>) runs six policies (<code>open</code>, <code>fast</code>, <code>fast_slow</code>, fair "
        "<code>oracle</code>, card-aware <code>oracle_card</code>, <code>best_fixed</code>) over four presets (<code>calm_morning</code>, <code>crosswind_shift</code>, "
        "<code>gusty_afternoon</code>, <code>hot_dry</code>), four hidden-mismatch levels (none, low, med, high) and 20 seeds, 40 segments per pass at "
        "<i>C</i><sub>req</sub> = 30%. The model is the learned <code>SprayModel</code>; the verifier is <i>noisy</i> (true coverage plus Gaussian error with the per-band "
        "bias and spread of B0). <code>fast_slow</code> runs five consecutive passes and the fifth is scored. Every policy sees the same hidden realization, so all "
        "comparisons are paired. Gap recovery is (policy &minus; open)/(oracle &minus; open) in the share of segments meeting <i>C</i><sub>req</sub>; volume overhead is "
        "mean GPA relative to the oracle. Criterion numbers pool low, med and high mismatch; brackets are 95% paired-bootstrap CIs over seeds.</p>"
        "<p><b>Two changes since the midterm, both fixed before the evaluation grid ran.</b> (1) <i>Fair oracle</i> (D15). The midterm oracle also saw the WSP spread factor "
        "of each card, which is drawn when the card is sprayed and which no controller can know; it needed no safety margin. The fair oracle (<code>loop.FairOracleModel</code>) "
        "knows the hidden mismatch of each segment exactly (gust state, canopy interception, nozzle wear) but not the spread factor, predicts at the mean spread factor, "
        "takes the exact 10% quantile as its lower bound, and selects with the same margin rule as the learned model. It measures perfect knowledge of the mismatch "
        f"under the same risk policy. The midterm oracle is kept as <code>oracle_card</code>, an unreachable bound: the fair oracle itself uses {c['fair_over_card']} more volume "
        f"than <code>oracle_card</code> ({c['fair_over_card_none']} with no mismatch). (2) <i>Slow loop redesign</i> (D16). The midterm slow loop refit LightGBM on the measured "
        "cards with record weight 30 and made things worse. The new slow loop (<code>feedback.ResidualCorrectedModel</code>) freezes the offline model and refits, after "
        "every pass, an additive coverage offset on all verified cards so far, per fast-loop regime and shrunk by empirical Bayes toward a global offset and toward 0 "
        "(prior sd 5 pp). The interval is shifted, never narrowed. Slow-loop variants were compared only on dev seeds 100&ndash;109, disjoint from the evaluation seeds 0&ndash;19.</p>")
    hdr = ["Criterion", "Reference", "Policy", "Recovery (95% CI)", "Volume vs oracle (95% CI)", "Verdict"]
    rows = [
        [">= 50% gap recovery, <= +10% volume", "fair oracle", "fast + slow", c["fs_rec"], c["fs_vol"], "**met** (both CIs clear)"],
        ["", "fair oracle", "fast", c["f_rec"], c["f_vol"], "met on point estimate; recovery CI crosses 50%"],
        ["", "card-aware oracle (bound)", "fast + slow", c["c_fs_rec"], c["c_fs_vol"], "missed on volume"],
        ["", "card-aware oracle (bound)", "fast", c["c_f_rec"], c["c_f_vol"], "missed"],
    ]
    head_tbl = ('<div class="caption"><b>Table #clhead.</b> Closed-loop criterion (proposal Table 1), learned model, 20 seeds, low + med + high mismatch pooled. '
                'Source: <code>results/closed_loop_summary.md</code>.</div>' + md_table_html(hdr, rows, widths=[22, 15, 10, 15, 17, 21]))
    verdict = (
        f"<p><b>The closed-loop criterion is met against the fair oracle.</b> The fast + slow loop recovers {c['fs_rec']} of the open-to-oracle gap at {c['fs_vol']} volume; "
        f"both CIs clear their targets. The fast loop alone recovers {c['f_rec']} at {c['f_vol']}, which meets 50% only on the point estimate; its weakest preset is "
        f"<code>hot_dry</code> ({c['hot_f']}). With the mismatch-free physics as the model (<code>oracle_nominal</code>) the picture is the same: fast {c['n_f_rec']} at {c['n_f_vol']}, "
        f"fast + slow {c['n_fs_rec']} at {c['n_fs_vol']}. Against <code>oracle_card</code> the volume half stays missed ({c['c_fs_vol']}), for the structural reason above.</p>"
        f"<p><b>Caveats that bound the result.</b> The pooled gap is only {c['gap_all']} pp (none {c['gap_none']}, low {c['gap_low']}, med {c['gap_med']}, high {c['gap_high']} pp): the open "
        f"loop's interval margin already protects it at low mismatch, so the criterion is informative only at med and high, where fast + slow recovers {c['fs_rec_med']} and {c['fs_rec_high']}. "
        "Recovery above 100% means the policy met <i>C</i><sub>req</sub> slightly more often than the fair oracle, paid for with volume "
        f"({c['fs_vol_high']} at high mismatch); the oracle limits information, not reliability. The verdict depends on the oracle's risk level: with a stricter fair oracle "
        f"(q = 0.05, met {c['q05_met']}) recovery falls to {c['q05_f']} (fast) and {c['q05_fs']} (fast + slow, volume {c['q05_fs_vol']}), still meeting the criterion with less margin. "
        f"Per preset, fast + slow recovers {c['preset_fs_min']} to {c['preset_fs_max']}.</p>")
    met_table = ('<div class="caption"><b>Table #clmet.</b> Share of segments meeting <i>C</i><sub>req</sub> by policy and hidden-mismatch level (learned model; mean &plusmn; 95% CI '
                 'over 20 seeds, presets pooled).</div>' + md_table_html(met_t[0], met_t[1], num_from=1))
    figmet = (f'<figure><img src="{figs["closed_loop_met_spray"]}" alt="Requirement-met share by mismatch level">'
              '<figcaption><b>Figure #clmet.</b> Share of segments meeting <i>C</i><sub>req</sub> (true outcome) against hidden-mismatch level, learned model: all presets pooled '
              '(left) and per preset; mean and 95% CI over 20 seeds. Green: fair oracle; light blue: card-aware oracle (unreachable bound).</figcaption></figure>')
    figvol = (f'<figure class="w70"><img src="{figs["closed_loop_volume_drift"]}" alt="Volume and drift by mismatch level">'
              '<figcaption><b>Figure #clvol.</b> Mean applied volume and drift fraction by mismatch level (top: learned model; bottom: <code>oracle_nominal</code>). '
              'The card-aware oracle (light blue) sprays less than any policy that cannot see the card noise, including the fair oracle (green).</figcaption></figure>')
    r_res, r_w30, r_w1 = c["abl_res"], c["abl_w30"], c["abl_w1"]
    slow = (
        f"<p><b>The slow loop removes the systematic error but not the variance.</b> At high mismatch the mean residual (measured minus model in use) goes from "
        f"{c['mean_p1']} pp in pass 1 to {c['mean_p2']} pp in pass 2, and the residual MSE from {c['mse_p1']} to {c['mse_p2']} pp<sup>2</sup>. The within-pass residual "
        f"variance does not fall: it stays between {c['var_min']} and {c['var_max']} pp<sup>2</sup> in every pass at every mismatch level, including none "
        f"(high: {c['var_p1']} in pass 1, {c['var_p5']} in pass 5; @fig:slow). It is set by the per-card spread factor and the per-segment gust, which no offset can remove. "
        "The proposal expected it to shrink pass after pass; it does not, and we report that as a negative result. "
        f"In the ablation cell (<code>gusty_afternoon</code>, med, 20 seeds) the residual offset meets {r_res['met'].split(' +-')[0]} with {r_res['gap recovery']} recovery; the midterm refit "
        f"meets {r_w30['met'].split(' +-')[0]} with {r_w30['gap recovery']} (record weight 30) and {r_w1['met'].split(' +-')[0]} with {r_w1['gap recovery']} (weight 1).</p>")
    figslow = (f'<figure><img src="{figs["slow_loop_spray"]}" alt="Slow loop over passes">'
               '<figcaption><b>Figure #slow.</b> Slow loop over five consecutive passes, learned model, all presets: residual variance, residual MSE, mean residual, '
               'share meeting <i>C</i><sub>req</sub> and volume per pass, by mismatch level (mean and 95% CI over 20 seeds). Pass 1 has no offset yet.</figcaption></figure>')
    a1, a3, a6 = c["abl_alpha"]
    abl_txt = (
        f"<p><b>Ablations</b> (fast loop, <code>gusty_afternoon</code>, med mismatch, fair oracle). Smoothing &alpha; = 0.1 / 0.3 / 0.6 recovers {a1['gap recovery'].split(' [')[0]} / "
        f"{a3['gap recovery'].split(' [')[0]} / {a6['gap recovery'].split(' [')[0]}; a global bias recovers {c['abl_global']['gap recovery'].split(' [')[0]} against "
        f"{c['abl_regime']['gap recovery'].split(' [')[0]} for per-regime biases, at {c['abl_global']['volume vs oracle'].split(' [')[0]} against "
        f"{c['abl_regime']['volume vs oracle'].split(' [')[0]} volume. All CIs overlap. Rendered-card verifiers (5 seeds) give {c['abl_b0']['gap recovery']} (B0) and "
        f"{c['abl_b1']['gap recovery']} (B1): too few seeds to separate them from the noisy verifier. Retreat flags were never raised, so re-treatment is untested.</p>"
        '<div class="caption"><b>Table #clabl.</b> Ablations, learned model, <code>gusty_afternoon</code>, med mismatch, reference = fair oracle.</div>'
        + md_table_html(abl[0], [r for r in abl[1] if r[1] != "verifier=noisy"], num_from=2, widths=[9, 45, 6, 12, 13, 15], cls="big"))
    bias = (f'<figure style="margin:0"><img src="{figs["bias_trace_spray"]}" alt="Example pass">'
            '<figcaption><b>Figure #bias.</b> One pass (<code>gusty_afternoon</code>, med mismatch, seed 0, learned model). Top: raw and bias-corrected prediction, '
            'card readings and true coverage under the fast loop, open loop and fast + slow loop. Bottom: bias applied to each segment.</figcaption></figure>')
    body = (design + head_tbl + verdict + figmet + met_table
            + f'<div class="twoup" style="align-items:center"><div style="flex:1">{bias}</div><div style="flex:1">{figvol}</div></div>'
            + slow + figslow + abl_txt)
    body = body.replace('<figure class="w70">', '<figure style="margin:0">')
    return body, c


# ----------------------------------------------------------------------------- extensions
def speed_boom_parts() -> tuple[str, str, dict]:
    bmd = read_md("boom_control_eval.md")
    BT = md_tables(bmd)
    ol_nom = dicts(BT[0])
    ol_spr = dicts(BT[1])
    cl = dicts(next(t for t in BT if t[0][0] == "policy/mismatch/variant"))
    smd = read_md("speed_control_eval.md")
    sp_cl = dicts(next(t for t in md_tables(smd) if t[0][0] == "policy"))
    DD, DS = "d drift vs off [95% CI]", "d success vs off [95% CI]"
    s = {
        "spr_off": pick(ol_spr, variant="off"), "spr_speed": pick(ol_spr, variant="speed"),
        "nom_off": pick(ol_nom, variant="off"), "nom_speed": pick(ol_nom, variant="speed"),
        "pilot_off": pick(sp_cl, policy="off/fast")["frac_met"], "pilot_on": pick(sp_cl, policy="on_band_0.25/fast")["frac_met"],
        "pilot_gpa_off": pick(sp_cl, policy="off/fast")["GPA"], "pilot_gpa_on": pick(sp_cl, policy="on_band_0.25/fast")["GPA"],
    }
    speed = (
        "<p><b>Design (D11, schema 1.1).</b> <code>Setting.travel_speed_kmh</code> is optional; when the selector option <code>speed_options_kmh</code> is set it "
        "chooses a grid speed within &plusmn;25% of the segment's current speed and inside the spec range. Application rate scales as GPA &prop; 1/speed. Default off; "
        "with it off every result is bit-identical.</p>"
        f"<p><b>Result: a small drift reduction, no gain in success.</b> On the 200 paired no-mismatch scenarios of the boom-height evaluation (the <code>speed</code> variant, "
        f"same seeds as <code>off</code>), the learned selector's success changes by {ci_pp(s['spr_speed'][DS])} and drift by {s['spr_speed'][DD]} (volume "
        f"{s['spr_off']['mean GPA']} to {s['spr_speed']['mean GPA']} GPA). With the mismatch-free physics as the selector model: success {ci_pp(s['nom_speed'][DS])}, drift "
        f"{s['nom_speed'][DD]}. An earlier closed-loop pilot (<code>results/speed_control_eval.md</code>: med mismatch, 4 presets, 3 seeds, no CI) saw the fast loop "
        f"meet <i>C</i><sub>req</sub> in {s['pilot_on']} of segments with speed control against {s['pilot_off']} without, at {s['pilot_gpa_on']} against {s['pilot_gpa_off']} GPA; "
        "it is too small to support a conclusion, but it points the same way as the boom-height closed loop below. Default stays off.</p>")
    Dm, Dd, Dg = "d frac_met", "d drift", "d GPA"
    b = {
        "nom_h": pick(ol_nom, variant="height"), "spr_h": pick(ol_spr, variant="height"),
        "nom_sh": pick(ol_nom, variant="speed+height"), "spr_sh": pick(ol_spr, variant="speed+height"),
    }
    for p in ("fast", "oracle"):
        for m in ("med", "high"):
            for v in ("height", "speed+height"):
                b[f"{p}_{m}_{v}"] = pick(cl, **{"policy/mismatch/variant": f"{p}/{m}/{v}"})
    boom = (
        "<p><b>Design (D12, schema 1.2).</b> <code>Setting.boom_height_m</code> is optional and combinable with speed. Candidate heights lie inside the spec range "
        "(0.30&ndash;0.75 m), within &plusmn;25% of the canopy-following set point, and above a per-family minimum from fan overlap, "
        "<i>H</i> &ge; 1.3&middot;<i>S</i>/(2 tan(&theta;/2)), because the uniform-field simulator does not model striping at low height. Default off.</p>"
        f"<p><b>Result: the drift benefit exists in the simulator, but the selector's models do not realize it.</b> In the open loop, height control alone "
        f"<i>raises</i> drift (nominal model {b['nom_h'][DD]}, learned {b['spr_h'][DD]}) at unchanged success: the selector uses the extra freedom to cut volume "
        f"({b['nom_h']['mean GPA']} against {pick(ol_nom, variant='off')['mean GPA']} GPA, nominal). In the closed loop (<code>oracle_nominal</code> model, perfect verifier, "
        f"4 presets &times; 10 seeds) the fast loop with height control meets <i>C</i><sub>req</sub> less often ({ci_pp(b['fast_med_height'][Dm])} med, {ci_pp(b['fast_high_height'][Dm])} high), and "
        f"speed + height loses more ({ci_pp(b['fast_med_speed+height'][Dm])}, {ci_pp(b['fast_high_speed+height'][Dm])}) while lowering volume ({ci_r(b['fast_med_speed+height'][Dg])}, "
        f"{ci_r(b['fast_high_speed+height'][Dg])} GPA) and drift ({b['fast_med_speed+height'][Dd]}, {b['fast_high_speed+height'][Dd]}). With the fair oracle, height alone cuts drift "
        f"({b['oracle_med_height'][Dd]} med, {b['oracle_high_height'][Dd]} high) with no detectable loss in success ({ci_pp(b['oracle_med_height'][Dm])}, {ci_pp(b['oracle_high_height'][Dm])}). "
        "So the limit is the model's prediction of the height effect, not the physics. Default stays off.</p>")
    rows = [[r["policy/mismatch/variant"], r["frac_met"], r["GPA"], r["drift"], r[Dm], r[Dd], r[Dg]] for r in cl]
    boom += ('<div class="caption"><b>Table #boom.</b> Boom height and travel speed in the closed loop (<code>oracle_nominal</code> model, perfect verifier, '
             '4 presets &times; 10 seeds; differences vs <code>off</code> with 95% paired-bootstrap CIs; <code>oracle</code> = fair oracle). '
             'Source: <code>results/boom_control_eval.md</code>.</div>'
             + md_table_html(["policy / mismatch / variant", "met", "GPA", "drift", "&Delta; met", "&Delta; drift", "&Delta; GPA"], rows, num_from=1, cls="big")
             .replace("&amp;Delta;", "&Delta;"))
    return speed, boom, {"s": s, "b": b}


def forecast_parts(figs) -> tuple[str, dict]:
    d = load_json("forecast_eval.json")
    if d is None:
        raise SystemExit("results/forecast_eval.json is required")
    T = d["table"]

    def r(loop, arm, mm="low+med+high", preset="ALL"):
        return next(x for x in T if x["loop"] == loop and x["arm"] == arm and x["mismatch"] == mm and x["preset"] == preset)

    def dmet(x):
        return f"{100 * x['d_met']:+.1f} pp [{100 * x['d_met_lo']:+.1f}, {100 * x['d_met_hi']:+.1f}]"

    def dvol(x):
        return f"{x['d_volume_pct']:+.1f}% [{x['d_volume_pct_lo']:+.1f}, {x['d_volume_pct_hi']:+.1f}]"

    def dmae(x):
        return f"{x['d_mae_ms']:+.3f} m/s [{x['d_mae_ms_lo']:+.3f}, {x['d_mae_ms_hi']:+.3f}]"

    arms = [a for a in d["arms"] if a != "off"]
    rows = []
    for loop, lab in (("fast", "fast"), ("fast_slow", "fast + slow")):
        off = r(loop, "off")
        rows.append([lab, "off (reference)", f"{off['frac_met']:.3f}", "-", "-", f"{off['fc_mae_ms']:.3f}", "-"])
        for a in arms:
            x = r(loop, a)
            rows.append([lab, f"`{a}`", f"{x['frac_met']:.3f}", dmet(x), dvol(x), f"{x['fc_mae_ms']:.3f}", dmae(x)])
        rows.append([lab, "*fair oracle (ref.)*", f"{off['oracle_frac_met']:.3f}", "", "", "", ""])
    ff, fs = r("fast", "ar1"), r("fast_slow", "ar1")
    go_f, go_fs = r("fast", "gust_oracle"), r("fast_slow", "gust_oracle")
    sens = d["sensitivity"]
    smin = min(abs(x["sens_dcov_pp"]) for x in sens)
    smax = max(abs(x["sens_dcov_pp"]) for x in sens)
    off_f = r("fast", "off")
    nseeds = d["seeds"] if isinstance(d["seeds"], int) else len(d["seeds"])
    txt = (
        "<p><b>Design (D13).</b> The selector plans each segment with the wind read at its start, but the plant applies a hidden AR(1) gust. An opt-in forecaster "
        "(<code>forecast.py</code>; statistics only) predicts the gust ratio of the next segment from an anemometer that observes the realized wind of completed "
        "segments with sd 0.3 m/s noise, never the upcoming one. Arms: <code>last</code>, <code>ewma</code>, <code>ar1</code> (online shrunk least squares), "
        "<code>ar1+gust</code> and <code>nominal+gust</code> (plan with forecast + 1 sd), and <code>gust_oracle</code>, which reads the plant's gust and bounds any "
        f"forecaster. Grid: {len(d['mismatch'])} mismatch levels &times; {len(d['presets'])} presets &times; {nseeds} seeds, paired against forecaster-off on the same passes. "
        "Defaults were fixed on dev seeds 100&ndash;103. Default off; with it off results are bit-identical.</p>"
        f"<p><b>Result: negative.</b> No forecaster raises the share of segments meeting <i>C</i><sub>req</sub>. AR(1) cuts the wind error by {dmae(ff)} in the fast loop "
        f"(off: {off_f['fc_mae_ms']:.3f} m/s) and {dmae(fs)} in fast + slow, but success changes by {dmet(ff)} and {dmet(fs)}. The 3-seed pilot had suggested "
        "an error cut of about 0.18 m/s; it did not hold. The gust-aware arms cost about half a percent of volume for no gain, and <code>nominal+gust</code>, which "
        f"widens the wind margin without any forecast, does the same. The bound explains it: <code>gust_oracle</code> changes success by {dmet(go_f)} (fast) and "
        f"{dmet(go_fs)} (fast + slow). A 30% stronger wind moves the coverage of the chosen settings by only {smin:.2f} to {smax:.2f} pp, so the selector's margin "
        "already absorbs gust risk, and the remaining shortfall comes from canopy interception, nozzle wear and card noise, which a wind forecast cannot see.</p>")
    tbl = ('<div class="caption"><b>Table #fc.</b> Wind forecaster arms, learned model, noisy verifier, low + med + high mismatch and all presets pooled. '
           '&Delta; = arm minus forecaster-off on the same passes, 95% paired-bootstrap CI. Source: <code>results/forecast_eval.json</code>.</div>'
           + md_table_html(["Loop", "Arm", "Met", "Δ met", "Δ volume", "Wind MAE (m/s)", "Δ wind MAE"], rows, num_from=2, cls="big"))
    fig = (f'<figure><img src="{figs["forecast_eval"]}" alt="Forecaster evaluation">'
           '<figcaption><b>Figure #fc.</b> Left: example pass (<code>gusty_afternoon</code>, high mismatch, seed 0): realized wind, nominal reading, AR(1) forecast and '
           'gust-aware planning wind. Middle and right: change in share meeting <i>C</i><sub>req</sub> and in volume against forecaster-off by mismatch level '
           '(filled: fast loop; open: fast + slow; 95% paired CI).</figcaption></figure>')
    info = {"ar1_f": ff, "ar1_fs": fs, "go_f": go_f, "go_fs": go_fs, "dmet": dmet, "dmae": dmae, "dvol": dvol, "off_f": off_f}
    return txt + fig + tbl, info


def hybrid_parts() -> tuple[str, dict]:
    md = read_md("hybrid_eval.md")
    sel = dicts(next(t for t in md_tables(section(md, "## Selector on no-mismatch scenarios", "## Closed loop")) if t[0][0] == "model"))
    clt = next(t for t in md_tables(section(md, "### Mismatch med+high", "### Mismatch med")) if t[0][0] == "model")
    cl = dicts(clt)
    off = dicts(next(t for t in md_tables(section(md, "### Split `random`", "### Split `U_gt_5`")) if t[0][0] == "model"))
    shares = [int(x) for x in re.findall(r"(?:coverage|drift) (\d+)%", section(md, "## Offline accuracy", "### Learning curve"))]
    oracle_line = re.search(r"Fair oracle: met ([\d.]+ ± [\d.]+), volume ([\d.]+) GPA", section(md, "### Mismatch med+high", "### Mismatch med"))
    MS, MH, PM = "SprayModel (pure ML)", "hybrid, misspecified base (optimistic)", "physics only, misspecified (optimistic)"
    E = "cov MAE vs exp."
    h = {
        "share_min": min(shares), "share_max": max(shares),
        "off_spray": pick(off, model=MS)[E], "off_hyb_o": pick(off, model=MH)[E],
        "off_hyb_p": pick(off, model="hybrid, misspecified base (pessimistic)")[E],
        "off_phys_o": pick(off, model=PM)[E], "off_phys_p": pick(off, model="physics only, misspecified (pessimistic)")[E],
        "sel_spray": pick(sel, model=MS)["success (sampled)"], "sel_hyb": pick(sel, model=MH)["success (sampled)"],
        "sel_phys": pick(sel, model=PM)["success (sampled)"],
        "oracle_met": oracle_line.group(1) if oracle_line else "?", "oracle_gpa": oracle_line.group(2) if oracle_line else "?",
    }
    for m, k in ((MS, "spray"), (MH, "hyb"), (PM, "phys"), ("hybrid (exact base)", "hybx")):
        for p in ("open", "fast", "fast_slow"):
            h[f"{k}_{p}"] = pick(cl, model=m, policy=p)
    dms = [lo_hi(h[f"hyb_{p}"]["Δ met vs spray (pp)"])[0] for p in ("open", "fast", "fast_slow")]
    dvs = [lo_hi(h[f"hyb_{p}"]["Δ volume vs spray"])[0] for p in ("open", "fast", "fast_slow")]
    h["dmet_rng"] = f"{min(abs(x) for x in dms):.1f} to {max(abs(x) for x in dms):.1f}"
    h["dvol_rng"] = f"{min(abs(x) for x in dvs):.1f} to {max(abs(x) for x in dvs):.1f}"
    met = lambda r: r["met"].split(" ±")[0]  # noqa: E731
    txt = (
        "<p><b>Design (D14).</b> <code>PhysicsResidualModel</code> predicts with a physics base (the simulator's equations without hidden mismatch, at the mean spread factor) "
        "plus a strongly regularized LightGBM residual and a conformalized interval. Because the plant is the simulator, an exact physics base is circular; the honest test "
        "uses misspecified bases (evaporation, drift threshold, spread factor and VMD each off by 10 to 30%). Default model stays <code>spray</code>.</p>"
        f"<p><b>Offline and in the selector the hybrid repairs a wrong physics base.</b> The learned residual removes {h['share_min']} to {h['share_max']}% of the misspecification's "
        f"excess coverage and drift error (four splits, both presets). Coverage MAE against the expected coverage falls from {h['off_phys_o']} / {h['off_phys_p']} pp (physics alone, "
        f"optimistic / pessimistic) to {h['off_hyb_o']} / {h['off_hyb_p']} pp, against {h['off_spray']} pp for <code>SprayModel</code>. In the selector test misspecified physics "
        f"alone meets <i>C</i><sub>req</sub> in {pct(float(h['sel_phys']))}% of scenarios, the hybrid on the same wrong base in {pct(float(h['sel_hyb']))}%, <code>SprayModel</code> in "
        f"{pct(float(h['sel_spray']))}%.</p>"
        f"<p><b>In the closed loop the hybrid is not better than pure ML.</b> With med + high mismatch (20 seeds, noisy verifier; fair oracle met {h['oracle_met']} at {h['oracle_gpa']} GPA) "
        f"the hybrid on a wrong base is {h['dmet_rng']} pp below <code>SprayModel</code> in every policy (CIs exclude zero) at {h['dvol_rng']}% less volume. Its gap recovery with "
        f"<code>fast_slow</code>, {h['hyb_fast_slow']['gap recovery']}, is statistically equal to <code>SprayModel</code>'s {h['spray_fast_slow']['gap recovery']} and meets the proposal "
        f"criterion ({h['hyb_fast_slow']['volume vs oracle']} volume). Misspecified physics alone collapses in the open loop ({met(h['phys_open'])} met against {met(h['hyb_open'])} for the hybrid) "
        f"but reaches {met(h['phys_fast_slow'])} after five passes of the slow loop, by spraying {h['phys_fast_slow']['volume vs oracle']} more than the fair oracle: the slow-loop offset absorbs "
        f"a smooth physics bias. The exact-base hybrid, the circular upper bound, is not better than <code>SprayModel</code> either ({h['hybx_fast_slow']['Δ met vs spray (pp)']} pp with "
        "<code>fast_slow</code>). Recommendation: keep <code>SprayModel</code>; use the hybrid when a trusted physics model exists but is wrong by 10 to 30%.</p>")
    keep = [r for r in clt[1] if not (r[0] in ("physics only (exact base)",) or (r[0] == "hybrid (exact base)" and r[1] != "fast_slow"))]
    cols = [0, 1, 2, 3, 6, 7, 8]
    tbl = ('<div class="caption"><b>Table #hyb.</b> Hybrid predictor in the closed loop, med + high mismatch pooled, 20 seeds, noisy verifier. Gap recovery uses each model\'s own '
           'open loop; volume against the fair oracle; &Delta; met against <code>SprayModel</code> on the same seeds (95% paired-bootstrap CI). The exact-base row is circular. '
           'Source: <code>results/hybrid_eval.md</code>.</div>'
           + md_table_html([clt[0][j] for j in cols], [[r[j] for j in cols] for r in keep], num_from=2, cls="big"))
    return txt + tbl, h


# ----------------------------------------------------------------------------- scorecard
def scorecard(mp, sel, ver, cl, sb, fc, hy, n_sim_tests) -> list[dict]:
    S = []
    add = lambda area, target, result, st, src: S.append(dict(area=area, target=target, result=result, status=st, source=src))  # noqa: E731
    e = E2E_REAL
    app = ("Streamlit app checked in a browser: all strategies and readers run, every decision is shown." if APP_CHECK
           else "Streamlit app browser check: **pending**.")
    add("End-to-end demonstration",
        "A simulated field pass runs from prediction to correction without manual steps, and every decision is logged and visualized.",
        f"`python -m coverageai.e2e --real`: {e['segments']}-segment pass with rendered cards and B0 in {e['seconds']} s, mean abs(measured - true) "
        f"{e['mean_abs_err_pp']:.2f} pp (midterm measurement); every segment logged as a SegmentLog row. {app}",
        "met" if APP_CHECK else "pending", "midterm e2e run; `app/main.py` browser check")
    add("Simulator and dataset",
        "Outputs follow the encoded trends (coverage rises with volume; drift rises with wind speed, release height, and fine-droplet share) across the M1 operating ranges.",
        f"{n_sim_tests} simulator tests pass, including a monotone sweep for each trend and the physical anchors; full suite {SUITE['tests']} tests pass.",
        "met", "`tests/test_simulator.py`")
    g = mp["_grp"]
    add("Predictive model",
        "Coverage MAE <= 3 pp on held-out conditions and lower than a linear baseline's; error reported on held-out condition regimes; sensitivity checks pass.",
        f"LightGBM {mp['lgb_rand_mae']} pp (random split) vs ridge {mp['ridge_rand_mae']} pp; {', '.join(f(x) for x in g)} pp on three withheld regimes; "
        f"{mp['sens_pass']}/{mp['sens_total']} sensitivity checks pass; noise floor {mp['noise_floor']} pp. No CI (single hold-out).",
        "met" if mp["_cov_random"] <= 3 and mp["_cov_random"] < mp["_ridge_random"] and all(x <= 3 for x in g) and mp["_sens_ok"] else "missed",
        "`results/model_v1.json`")
    c, mf = sel["real"], sel["real"]["fixed_matched_success"]
    ok = c["selector"]["success"] >= 0.9 and c.get("selector_lower_drift_than_matched_fixed") is True
    add("Parameter selection",
        "With no injected mismatch, selected settings meet C_req in >= 90% of scenarios at lower drift risk than the best fixed setting with the same success rate.",
        f"{pct(c['selector']['success'])}% of {sel['n']} scenarios; drift {f(c['selector']['mean_drift'], 4)} vs {f(mf['mean_drift'], 4)} and "
        f"{f(c['selector']['mean_gpa'], 1)} vs {f(mf['mean_gpa'], 1)} GPA for the matched fixed setting ({_setting(mf['setting'])}, {pct(mf['success'], 0)}%). No CI.",
        "met" if ok else "missed", "`results/selector_eval.json`")
    b0r, b1r, b0s, b1s = ver["b0r"], ver["b1r"], ver["b0s"], ver["b1s"]
    add("Verification: MAE, own synthetic cards", "Coverage MAE <= 2 pp on synthetic cards.",
        f"B0 {f(b0r['mae_pp'])} pp, B1 {f(b1r['mae_pp'])} pp ({b0r['n']} held-out renderer cards).",
        "met" if max(b0r["mae_pp"], b1r["mae_pp"]) <= 2 else "missed", "`results/verification_eval.json`")
    add("Verification: class agreement, own synthetic cards", "Three-class agreement >= 90% on synthetic cards.",
        f"B0 {pct(b0r['class_agreement'])}%, B1 {pct(b1r['class_agreement'])}%.",
        "met" if min(b0r["class_agreement"], b1r["class_agreement"]) >= 0.9 else "missed", "`results/verification_eval.json`")
    sb1 = ver["b1s_bands"]["30-50"]
    add("Verification: Simoes synthetic cards (out-of-generator)", "Same synthetic targets: MAE <= 2 pp, agreement >= 90%.",
        f"B0 {f(b0s['mae_pp'])} pp / {pct(b0s['class_agreement'])}%; B1 {f(b1s['mae_pp'])} pp / {pct(b1s['class_agreement'])}%. "
        f"B1 underreads every band; 30-50% band ({sb1['n']} cards) bias {sgn(sb1['bias_pp'], 2)} pp.",
        "partial", "`results/verification_eval.json`")
    t0, t1 = ver["t_b0"], ver["t_b1"]
    if B1_IDLE_RETIME:
        t = B1_IDLE_RETIME
        rt = (f"Laptop, busy CPU (upper bound): B0 p95 {f(t0['p95_s'])} s; B1 p95 {f(t1['p95_s'])} s, max {f(t1['max_s'])} s ({t1['n']} cards). "
              f"Desktop re-time, light load (not a laptop): B0 p95 {f(t['b0']['p95_s'])} s; B1 p95 {f(t['b1']['p95_s'])} s, max {f(t['b1']['max_s'])} s.")
        rst = "met" if t0["p95_s"] <= 2 and t1["p95_s"] <= 2 else "missed"
    else:
        rt = (f"B0 p95 {f(t0['p95_s'])} s; B1 p95 {f(t1['p95_s'])} s, max {f(t1['max_s'])} s ({t1['n']} cards, busy CPU). "
              "B1 idle-CPU re-timing **pending**.")
        rst = "pending"
    add("Verification: speed", "<= 2 s per card on a laptop CPU.", rt, rst, "`results/verification_eval.json` (timing)" + ("; `results/b1_idle_retime.json`" if B1_IDLE_RETIME else ""))
    add("Verification: real cards", "Coverage MAE <= 5 pp on real cards.",
        "Untested: the annotated real cards (Zenodo 14010262) are restricted; evaluation code is ready.", "blocked", "D9; `results/verification_summary.md`")
    add("Closed loop (fair oracle)",
        "Under injected mismatch, recovers >= 50% of the open-loop-to-oracle gap in requirement-met segments, using <= 10% more volume than the oracle (20 seeds, 95% CI).",
        f"fast + slow: {cl['fs_rec']} recovery, {cl['fs_vol']} volume. fast: {cl['f_rec']} (CI crosses 50%), {cl['f_vol']}.",
        "met", "`results/closed_loop_summary.md`")
    add("Closed loop (card-aware oracle, unreachable bound)", "Same criterion against `oracle_card`, which also sees the per-card spread factor.",
        f"fast + slow: {cl['c_fs_rec']} recovery, {cl['c_fs_vol']} volume. The fair oracle itself uses {cl['fair_over_card']} more volume than `oracle_card`.",
        "missed", "`results/closed_loop_summary.md`")
    s, b = sb["s"], sb["b"]
    DD, DS = "d drift vs off [95% CI]", "d success vs off [95% CI]"
    add("Extension: travel speed (D11)", "No proposal target. Adopt by default only if success holds and drift or volume falls.",
        f"200 paired scenarios, learned selector: success {ci_pp(s['spr_speed'][DS])}, drift {s['spr_speed'][DD]}; nominal: success {ci_pp(s['nom_speed'][DS])}, drift "
        f"{s['nom_speed'][DD]}. Closed-loop pilot (3 seeds, no CI): fast met {s['pilot_off']} to {s['pilot_on']}.",
        "neg", "`results/boom_control_eval.md` (`speed` variant); `results/speed_control_eval.md`")
    add("Extension: boom height (D12)", "No proposal target. Same adoption rule.",
        f"Open loop, height alone: drift {b['spr_h'][DD]} (learned). Closed loop, fast, speed+height: met {ci_pp(b['fast_med_speed+height']['d frac_met'])} (med), "
        f"{ci_pp(b['fast_high_speed+height']['d frac_met'])} (high), GPA {ci_r(b['fast_med_speed+height']['d GPA'])} (med). Fair oracle, height alone: drift "
        f"{b['oracle_high_height']['d drift']} (high), met {ci_pp(b['oracle_high_height']['d frac_met'])}.",
        "neg", "`results/boom_control_eval.md`")
    dmet, dmae = fc["dmet"], fc["dmae"]
    add("Extension: wind forecaster (D13)", "No proposal target. Adopt only if it raises the share meeting C_req.",
        f"AR(1): wind MAE {dmae(fc['ar1_f'])} (fast), success {dmet(fc['ar1_f'])}; fast + slow {dmet(fc['ar1_fs'])}. "
        f"Bound `gust_oracle`: {dmet(fc['go_f'])} (fast).",
        "neg", "`results/forecast_eval.json`, `results/forecast_eval.md`")
    add("Extension: hybrid physics + ML predictor (D14)", "No proposal target. Adopt only if better than `SprayModel` in the closed loop.",
        f"Misspecified base, fast_slow: {hy['hyb_fast_slow']['met'].split(' ±')[0]} met vs {hy['spray_fast_slow']['met'].split(' ±')[0]} "
        f"({hy['hyb_fast_slow']['Δ met vs spray (pp)']} pp), recovery {hy['hyb_fast_slow']['gap recovery']} vs {hy['spray_fast_slow']['gap recovery']}, "
        f"volume {hy['hyb_fast_slow']['Δ volume vs spray']} vs SprayModel.",
        "neg", "`results/hybrid_eval.md`")
    return S


def scorecard_html(S) -> str:
    rows = "".join(f"<tr><td>{_cell(r['area'])}</td><td>{_cell(r['target'])}</td><td>{_cell(r['result'])}</td>"
                   f"<td>{status(r['status'])}</td><td class=\"src\">{_cell(r['source'])}</td></tr>" for r in S)
    return ('<table class="big score"><tr><th style="width:14%">Criterion</th><th style="width:25%">Target (proposal Table 1, verbatim)</th>'
            '<th style="width:38%">Result (95% CI where one exists)</th><th style="width:9%">Status</th><th style="width:14%">Source</th></tr>' + rows + "</table>")


def negative_results(cl, ver, sb, fc, hy) -> list[str]:
    s, b = sb["s"], sb["b"]
    b1r, b1s = ver["b1r"], ver["b1s"]
    return [
        f"**Prediction-error variance does not shrink pass after pass.** The slow loop removes the mean residual (high mismatch: {cl['mean_p1']} pp in pass 1, "
        f"{cl['mean_p2']} pp in pass 2), but the within-pass residual variance stays at {cl['var_min']} to {cl['var_max']} pp^2 in every pass and at every mismatch "
        "level, including none. Per-card spread-factor noise and per-segment gusts are irreducible for an offset correction.",
        f"**B1 is less safe outside the generator than on our own cards.** MAE {f(b1r['mae_pp'])} pp and {pct(b1r['class_agreement'])}% class agreement on renderer cards, "
        f"but {f(b1s['mae_pp'])} pp (bias {sgn(b1s['bias_pp'], 2)} pp) and {pct(b1s['class_agreement'])}% on the Simoes synthetic cards; class agreement misses 90% for "
        "both readers. Real-card accuracy is unknown.",
        f"**The wind forecaster does not help.** AR(1) lowers the wind error ({fc['dmae'](fc['ar1_f'])}) but not coverage failures ({fc['dmet'](fc['ar1_f'])}); "
        f"even a perfect gust forecast gives {fc['dmet'](fc['go_f'])}.",
        f"**Boom height (alone or with speed) trades success for drift and volume.** In the fast loop speed + height cuts volume ({ci_r(b['fast_high_speed+height']['d GPA'])} GPA, high) "
        f"but meets C_req less often ({ci_pp(b['fast_med_speed+height']['d frac_met'])} med, {ci_pp(b['fast_high_speed+height']['d frac_met'])} high). Height alone raises drift in "
        f"the open loop ({b['spr_h']['d drift vs off [95% CI]']}, learned model). Speed alone changes success by {ci_pp(s['spr_speed']['d success vs off [95% CI]'])}.",
        f"**The hybrid predictor is not better than pure ML in the closed loop**: {hy['hyb_fast_slow']['Δ met vs spray (pp)']} pp met against `SprayModel` with "
        "`fast_slow`, and the exact-base (circular) hybrid is not better either.",
        f"**The midterm slow loop hurt.** Refitting LightGBM on the cards with record weight 30 gave {cl['abl_w30']['gap recovery']} recovery "
        f"(met {cl['abl_w30']['met'].split(' +-')[0]}); it narrowed the interval while fitting card noise. Replaced by the residual offset (D16).",
        f"**The volume target is unreachable against the card-aware oracle.** The fair oracle itself uses {cl['fair_over_card']} more volume than `oracle_card`; "
        f"fast + slow is at {cl['c_fs_vol']}.",
        f"**The fast loop alone does not clear the recovery target with confidence**: {cl['f_rec']}, weakest on `hot_dry` ({cl['hot_f']}).",
        f"**Rendered-card verifiers are inconclusive in the loop**: 5 seeds give {cl['abl_b0']['gap recovery']} (B0) and {cl['abl_b1']['gap recovery']} (B1) recovery. "
        "Re-treatment flags were never raised, so that path is untested.",
    ]


# ----------------------------------------------------------------------------- AI log and decisions
def ai_log() -> tuple[int, str]:
    p = ROOT / "Docs" / "ai_usage_log.md"
    if not p.exists():
        return 0, ""
    rows = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\| 20\d\d-", ln):
            cells = [x.strip() for x in ln.strip().strip("|").split("|")]
            rows.append(cells[:4] + [""] * (4 - len(cells)))
    body = "".join(f"<tr><td>{_cell(r[0])}</td><td>{_cell(r[1])}</td><td>{_cell(r[2])}</td><td>{_cell(r[3])}</td></tr>" for r in rows)
    return len(rows), ('<table class="ailog"><tr><th style="width:10%">Date</th><th style="width:10%">Work package</th><th style="width:12%">Model</th>'
                       '<th>Output</th></tr>' + body + "</table>")


def n_decisions() -> int:
    p = ROOT / "Docs" / "decisions.md"
    return len(re.findall(r"^\| D\d+ ", p.read_text(encoding="utf-8"), re.M))


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


# ----------------------------------------------------------------------------- summary
def summary_text(mp, sel, ver, cl, n_open) -> str:
    c = sel["real"]
    b0r, b1r, b1s = ver["b0r"], ver["b1r"], ver["b1s"]
    return (
        "We built the closed loop of the proposal end to end and evaluated it inside a physics-informed simulator: a learned coverage model with a calibrated interval, "
        "a constrained setting selector, a water-sensitive-paper (WSP) renderer, classical (B0) and U-Net (B1) card readers, a fast loop that corrects the next segment "
        "and a slow loop that corrects the model between passes. "
        f"<b>The closed-loop criterion is met against a fair oracle</b> that knows the hidden mismatch but not the per-card noise: the fast + slow loop recovers {cl['fs_rec']} "
        f"of the open-loop-to-oracle gap at {cl['fs_vol']} volume (target &ge; 50%, &le; +10%). Against the midterm's card-aware oracle the volume half is missed "
        f"({cl['c_fs_vol']}), because that oracle sees noise no controller can know. The model ({mp['lgb_rand_mae']} pp MAE, ridge {mp['ridge_rand_mae']} pp), the selector "
        f"({pct(c['selector']['success'])}% success at lower drift than the matched fixed setting) and card reading on our own cards (B0 {f(b0r['mae_pp'])} pp, B1 {f(b1r['mae_pp'])} pp MAE) "
        f"meet their targets. On out-of-generator cards B1 drops to {f(b1s['mae_pp'])} pp and {pct(b1s['class_agreement'])}% class agreement (target 90%), and the real-card "
        "criterion is blocked by restricted data. Four extensions were tested and none is adopted by default: travel speed lowers drift slightly without raising success, "
        "boom height (alone or with speed) trades success for drift and volume, "
        "a wind forecaster lowers forecast error but not coverage failures, and a physics-first hybrid predictor is not better than pure ML in the loop. "
        f"All results are simulator-internal. {n_open}")


# ----------------------------------------------------------------------------- main
def build(do_pdf: bool = True) -> None:
    global B1_IDLE_RETIME
    B1_IDLE_RETIME = idle_retime()
    figs = prep_figures()
    need = ["sim_trends", "model_parity", "wsp_examples", "b0_parity", "b1_parity_left", "selector_tradeoff",
            "closed_loop_met_spray", "closed_loop_volume_drift", "bias_trace_spray", "slow_loop_spray", "forecast_eval", "demo_live_top"]
    missing = [k for k in need if k not in figs]
    if missing:
        raise SystemExit(f"missing figures: {missing}")
    mp = model_parts()
    sp = simoes_parts()
    sel_html, sel = selector_block(figs)
    ver_html, ver = verification_block(figs)
    cl_html, cl = closed_loop_parts(figs)
    speed_html, boom_html, sb = speed_boom_parts()
    fc_html, fc = forecast_parts(figs)
    hy_html, hy = hybrid_parts()
    n_tests, sim_tests, src_lines = count_tests_and_lines()
    n_ai, ai_table = ai_log()
    S = scorecard(mp, sel, ver, cl, sb, fc, hy, sim_tests)
    neg = negative_results(cl, ver, sb, fc, hy)
    open_items = [x for x, v in (("app browser check", APP_CHECK), ("B1 idle-CPU timing", B1_IDLE_RETIME)) if not v]
    n_open = (f"Still open in this draft: {' and '.join(open_items)}." if open_items else "")

    app_html = (f"<p><b>Browser check.</b> {APP_CHECK}</p>" if APP_CHECK else
                pending("Streamlit app browser check.", "The app is being checked in a browser (all control strategies and card readers, animation controls, "
                        "end-of-run summary; any new opt-in sidebar options for speed, boom height or forecaster). Fill <code>APP_CHECK</code> in "
                        "<code>reports/final/build.py</code>."))
    counts = {k: sum(1 for r in S if r["status"] == k) for k in STATUS_LABEL}
    vals = {
        "build_date": date.today().strftime("%B %d, %Y").replace(" 0", " "),
        "summary_text": summary_text(mp, sel, ver, cl, n_open),
        "src_lines": f"{src_lines:,}", "n_tests": str(n_tests), "n_suite": str(SUITE["tests"]), "suite_machine": SUITE["machine"],
        "anchor_rows": anchor_rows(),
        "selector_block": sel_html, "verification_block": ver_html, "closed_loop_block": cl_html,
        "speed_block": speed_html, "boom_block": boom_html, "forecast_block": fc_html, "hybrid_block": hy_html,
        "app_block": app_html,
        "scorecard_table": scorecard_html(S),
        "n_met": str(counts["met"]), "n_rows": str(len(S)), "n_partial": str(counts["partial"]), "n_missed": str(counts["missed"]),
        "n_blocked": str(counts["blocked"]), "n_neg": str(counts["neg"]), "n_pending": str(counts["pending"]),
        "neg_list": "".join(f"<li>{_cell(x)}</li>" for x in neg),
        "ai_log_rows": str(n_ai), "ai_log_table": ai_table, "n_decisions": str(n_decisions()),
        "fs_rec": cl["fs_rec"], "fs_vol": cl["fs_vol"], "fair_over_card": cl["fair_over_card"],
        "b1s_mae": f(ver["b1s"]["mae_pp"]), "b1s_agree": pct(ver["b1s"]["class_agreement"]),
        "b0s_mae": f(ver["b0s"]["mae_pp"]), "b0s_agree": pct(ver["b0s"]["class_agreement"]),
        **sp, **{k: v for k, v in mp.items() if not k.startswith("_")},
        **{f"fig_{k}": v for k, v in figs.items()},
    }
    doc = (HERE / "template.html").read_text(encoding="utf-8")
    for _ in range(2):
        doc = re.sub(r"\{\{(\w+)\}\}", lambda m: str(vals[m.group(1)]) if m.group(1) in vals else m.group(0), doc)
    left = sorted(set(re.findall(r"\{\{(\w+)\}\}", doc)))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    doc = number_captions(doc)
    if "??" in doc:
        print("WARNING: unresolved cross-reference marker '??' in output", file=sys.stderr)
    out_html = HERE / "final_report.html"
    out_html.write_text(doc, encoding="utf-8")
    print(f"wrote {out_html} ({out_html.stat().st_size // 1024} KB)")
    write_scorecard_md(S, neg)
    if not do_pdf:
        return
    pdf = HERE / "final_report.pdf"
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


def write_scorecard_md(S, neg) -> None:
    esc = lambda s: s.replace("|", "/")  # noqa: E731
    lines = [
        "# Final scorecard",
        "",
        "Generated by `reports/final/build.py` from the results files named in the Source column; do not edit by hand, edit the build script or the results and rebuild "
        "(`uv run --no-sync python reports/final/build.py --no-pdf`). Targets are quoted from Table 1 of the proposal "
        "(`Docs/CoverageVisionAI_Project_Proposal (1).pdf`). All results are simulator-internal. The fair oracle (`oracle`: knows the hidden mismatch, not the per-card "
        "spread factor) is the reference; the card-aware oracle (`oracle_card`) is reported only as an unreachable bound. Brackets are 95% paired-bootstrap CIs.",
        "",
        "Status: met / missed / partial / blocked (outside dependency) / done-negative (extension evaluated, not adopted) / pending (input not yet received).",
        "",
        "| Criterion | Target | Result | Status | Source |",
        "|---|---|---|---|---|",
    ]
    for r in S:
        lines.append(f"| {esc(r['area'])} | {esc(r['target'])} | {esc(r['result'])} | {STATUS_LABEL[r['status']]} | {esc(r['source'])} |")
    lines += ["", "## Negative results", ""] + [f"{i}. {x}" for i, x in enumerate(neg, 1)] + [""]
    SCORECARD_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {SCORECARD_MD}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    build(not ap.parse_args().no_pdf)
