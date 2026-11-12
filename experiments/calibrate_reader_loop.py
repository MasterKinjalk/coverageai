"""Measure an image card reader's mean error on cards from closed-loop passes (decision D30).

The calibration table in ``results/<reader>_synthetic.json`` comes from the reader's validation
set. Used in the loop it reversed the classic reader's bias (results/feedback_model_v2_closed_loop.md,
b0 section), because the loop's cards differ from the validation cards. This script measures the
reader on the cards the loop itself produces, on development seeds only, and writes
``results/<reader>_loop_calibration.json``, which ``integration.build_modules(calibrate_reader=True)``
reads for image readers.

Protocol: scenario seeds 20000-20005 to fit, 20006-20007 held out (never 0-19, 100-119 or
21000+); 6 presets x c_req 20 / 30 / 40 x mismatch med; hidden seed s + 4036; policy fast with the
eq. 3 rule and the RAW reader; one nozzle per field; same_field=True; model "spray". Per card: true
coverage, reading, c_req, droplet class and VMD of the setting. Bands on TRUE coverage with edges
0, 15, 20, ..., 60 and an open end; a band with fewer than 60 cards is merged into its smaller
neighbour. Per band: n, bias_pp = mean(reading - true), sd_pp.

Held-out check: mean (reading - true) overall and per c_req for the raw reading, the synthetic-card
table and the new table (``integration.calibration_table`` + ``calibrated_coverage`` applied to the
stored raw readings), and the bias by droplet class after the new table. Applying a table offline to
readings from raw-reader passes is an approximation: in the loop a calibrated reading also changes
the settings chosen later in the pass.

Everything is simulation: the cards are rendered by the simulator's own renderer.

Usage:
  uv run --no-sync python experiments/calibrate_reader_loop.py --workers 26 [--readers b0,b1]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coverageai import integration, scenarios                     # noqa: E402

RESULTS = ROOT / "results"
FIT_SEEDS = tuple(range(20000, 20006))
HOLD_SEEDS = (20006, 20007)
C_REQS = (20.0, 30.0, 40.0)
EDGES = (0.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, None)
MIN_N = 60
EVAL_OFFSET = 4036
PRESETS = tuple(scenarios.list_presets())
_W: dict = {}


def _init() -> None:
    from coverageai import app_logic as al
    _W["al"] = al
    _W["mods"] = {}


def task(reader: str, seed: int, preset: str, c_req: float) -> dict:
    al = _W["al"]
    if reader not in _W["mods"]:
        _W["mods"][reader] = al.build_modules_adapter("spray", reader, calibrate_reader=False)
    conds = al.make_conditions(preset, seed)
    t0 = time.perf_counter()
    res = al.run_pass_wrapper(_W["mods"][reader], conds, "fast", seed + EVAL_OFFSET, "med", c_req,
                              fixed_nozzle="auto", same_field=True)
    if not res.used_vision:
        raise RuntimeError(f"{reader}: the pass did not read rendered cards")
    return {"reader": reader, "seed": seed, "preset": preset, "c_req": c_req, "pass_s": time.perf_counter() - t0,
            "true": [float(lg.outcome.coverage_pct) for lg in res.logs],
            "read": [float(lg.meas.coverage_pct) for lg in res.logs],
            "cls": [str(lg.derived.droplet_class) for lg in res.logs],
            "vmd": [float(lg.derived.vmd_um) for lg in res.logs]}


def fit_bands(true: np.ndarray, err: np.ndarray) -> list[dict]:
    """Bands on true coverage; merge any band under MIN_N cards into its smaller neighbour."""
    edges = [e for e in EDGES]
    bands = [[edges[i], edges[i + 1]] for i in range(len(edges) - 1)]

    def members(b):
        hi = np.inf if b[1] is None else b[1]
        return (true >= b[0]) & (true < hi)

    while len(bands) > 1:
        ns = [int(members(b).sum()) for b in bands]
        i = int(np.argmin(ns))
        if ns[i] >= MIN_N:
            break
        if i == 0:
            j = 1
        elif i == len(bands) - 1:
            j = i - 1
        else:
            j = i - 1 if ns[i - 1] <= ns[i + 1] else i + 1
        a, b = sorted((i, j))
        bands[a] = [bands[a][0], bands[b][1]]
        del bands[b]
    out = []
    for b in bands:
        m = members(b)
        e = err[m]
        out.append({"lo": float(b[0]), "hi": None if b[1] is None else float(b[1]), "n": int(m.sum()),
                    "bias_pp": float(e.mean()) if len(e) else 0.0,
                    "sd_pp": float(e.std(ddof=1)) if len(e) > 1 else 0.0})
    return out


def _bias_stats(true, read, c_req, cls, table) -> dict:
    from coverageai.integration import calibrated_coverage
    corr = read if table is None else np.array([calibrated_coverage(r, table) for r in read])
    e = corr - true
    out = {"all": {"n": int(len(e)), "bias_pp": float(e.mean()), "mae_pp": float(np.abs(e).mean())}}
    for c in C_REQS:
        m = c_req == c
        out[f"c_req {c:g}"] = {"n": int(m.sum()), "bias_pp": float(e[m].mean()), "mae_pp": float(np.abs(e[m]).mean())}
    out["by_droplet_class"] = {k: {"n": int((cls == k).sum()), "bias_pp": float(e[cls == k].mean())}
                               for k in sorted(set(cls))}
    return out


def analyse(reader: str, rows: list[dict]) -> dict:
    from coverageai.integration import calibration_table, noise_bands_from_results

    def arr(seeds, key):
        return np.array([v for r in rows if r["seed"] in seeds for v in r[key]])

    def creq(seeds):
        return np.array([r["c_req"] for r in rows if r["seed"] in seeds for _ in r["true"]])
    tf, rf = arr(FIT_SEEDS, "true"), arr(FIT_SEEDS, "read")
    bands = fit_bands(tf, rf - tf)
    new_tab = calibration_table([(b["lo"], 1e9 if b["hi"] is None else b["hi"], b["sd_pp"], b["bias_pp"])
                                 for b in bands])
    syn = noise_bands_from_results(reader)
    syn_tab = calibration_table(syn[0]) if syn else None
    th, rh, ch = arr(HOLD_SEEDS, "true"), arr(HOLD_SEEDS, "read"), creq(HOLD_SEEDS)
    clh = np.array([v for r in rows if r["seed"] in HOLD_SEEDS for v in r["cls"]])
    held = {"raw": _bias_stats(th, rh, ch, clh, None),
            "synthetic_table": _bias_stats(th, rh, ch, clh, syn_tab) if syn_tab else None,
            "loop_table": _bias_stats(th, rh, ch, clh, new_tab)}
    fit_cls = np.array([v for r in rows if r["seed"] in FIT_SEEDS for v in r["cls"]])
    by_cls_fit = {k: {"n": int((fit_cls == k).sum()), "bias_pp": float((rf - tf)[fit_cls == k].mean())}
                  for k in sorted(set(fit_cls))}
    return {"reader": reader, "bands": bands,
            "protocol": ("closed-loop cards: policy fast with the eq. 3 rule and the RAW reader, fixed_nozzle='auto', "
                         "same_field=True, model 'spray', mismatch med, hidden seed s + 4036, 6 presets x c_req "
                         "20/30/40; bands on TRUE coverage, bias = mean(reading - true), bands under "
                         f"{MIN_N} cards merged into the smaller neighbour"),
            "fit_seeds": list(FIT_SEEDS), "holdout_seeds": list(HOLD_SEEDS), "edges": [e for e in EDGES],
            "n_cards": int(len(tf)), "n_cards_holdout": int(len(th)),
            "fit_bias_overall_pp": float((rf - tf).mean()), "fit_bias_by_droplet_class": by_cls_fit,
            "synthetic_source": syn[1] if syn else None, "holdout_check": held,
            "pass_s_mean": float(np.mean([r["pass_s"] for r in rows])), "n_passes": len(rows)}


def write_md(results: dict, skipped: dict, wall: float) -> None:
    L = ["# Card reader calibration measured on closed-loop cards (D30)\n",
         "Generated by `experiments/calibrate_reader_loop.py`. Do not edit by hand. Everything is simulation: "
         "the cards are rendered by the simulator, and a real reader would need annotated real cards.\n",
         "Why: the table measured on the reader's validation cards (`results/<reader>_synthetic.json`) reversed the "
         "classic reader's bias when used in the loop (`results/feedback_model_v2_closed_loop.md`). This table is "
         "measured on the cards the loop itself produces.\n",
         f"Protocol: development seeds {FIT_SEEDS[0]}-{FIT_SEEDS[-1]} to fit and {HOLD_SEEDS[0]}-{HOLD_SEEDS[-1]} "
         "held out; 6 presets x c_req 20 / 30 / 40 x mismatch med; policy fast with the eq. 3 rule and the raw "
         "reader; one nozzle per field. Bands on true coverage; bias = mean(reading - true). The held-out "
         "check applies each table to the stored raw readings, which is an approximation of what happens "
         f"in the loop. Wall time {wall / 60:.1f} min.\n"]
    for reader, r in results.items():
        L.append(f"## {reader}\n")
        L.append(f"{r['n_cards']:,} fit cards, {r['n_cards_holdout']:,} held-out cards, {r['n_passes']} passes, "
                 f"{r['pass_s_mean']:.1f} s per pass on a loaded machine. Fit-set bias overall "
                 f"{r['fit_bias_overall_pp']:+.2f} pp.\n")
        L.append("| true coverage band % | cards | bias pp (reading - true) | sd pp |")
        L.append("|---|---|---|---|")
        for b in r["bands"]:
            hi = "+" if b["hi"] is None else f"{b['hi']:g}"
            L.append(f"| {b['lo']:g}-{hi} | {b['n']:,} | {b['bias_pp']:+.2f} | {b['sd_pp']:.2f} |")
        L.append("")
        L.append("Held-out check, mean (reading - true) in pp (mean absolute error in brackets):\n")
        keys = ["all"] + [f"c_req {c:g}" for c in C_REQS]
        L.append("| reading | " + " | ".join(keys) + " |")
        L.append("|---|" + "---|" * len(keys))
        for lab, k in (("raw", "raw"), ("synthetic-card table", "synthetic_table"), ("loop table", "loop_table")):
            h = r["holdout_check"].get(k)
            if h is None:
                L.append(f"| {lab} | " + " | ".join("-" for _ in keys) + " |")
                continue
            L.append(f"| {lab} | " + " | ".join(f"{h[x]['bias_pp']:+.2f} ({h[x]['mae_pp']:.2f})" for x in keys) + " |")
        L.append("")
        L.append("Bias by droplet class of the setting (held out, after the loop table; fit set raw for reference):\n")
        L.append("| droplet class | held-out cards | bias after loop table pp | fit-set raw bias pp |")
        L.append("|---|---|---|---|")
        for k, v in r["holdout_check"]["loop_table"]["by_droplet_class"].items():
            fb = r["fit_bias_by_droplet_class"].get(k, {}).get("bias_pp", float("nan"))
            L.append(f"| {k} | {v['n']} | {v['bias_pp']:+.2f} | {fb:+.2f} |")
        L.append("")
    for reader, why in skipped.items():
        L.append(f"## {reader}\n\nSkipped: {why}.\n")
    L.append("## What was not tested\n")
    L += ["- Mismatch levels other than med, the free nozzle choice, other policies (the table was measured under "
          "the rule; a learned policy picks somewhat different settings).",
          "- Real cards. The table corrects the reader for the simulator's renderer only.",
          "- A table per droplet class or per nozzle; this one depends on coverage only."]
    (RESULTS / "reader_loop_calibration.md").write_text("\n".join(L) + "\n", encoding="utf8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--readers", default="b0,b1")
    ap.add_argument("--b1-max-ratio", type=float, default=3.0)
    ap.add_argument("--skip-note", default=None,
                    help="reason to record for b1 when it is left out of --readers")
    args = ap.parse_args()
    readers, skipped = [], {}
    if "b1" not in args.readers.split(",") and args.skip_note:
        skipped["b1"] = args.skip_note
    for rd in args.readers.split(","):
        if not integration.image_verifier_available(rd):
            skipped[rd] = "reader not available"
        else:
            readers.append(rd)
    if "b1" in readers and "b0" in readers:
        # cost check on one development pass each (seed 20003, not part of the fit/hold-out grid's use)
        _init()
        t = {}
        for rd in ("b0", "b1"):
            t[rd] = task(rd, 20003, "gusty_afternoon", 30.0)["pass_s"]
        ratio = t["b1"] / t["b0"]
        print(f"pass time b0 {t['b0']:.1f}s, b1 {t['b1']:.1f}s, ratio {ratio:.2f}", flush=True)
        if ratio > args.b1_max_ratio:
            readers.remove("b1")
            skipped["b1"] = f"one b1 pass costs {ratio:.1f}x a b0 pass (limit {args.b1_max_ratio:g}x)"
    t0 = time.time()
    tasks = [(rd, s, p, c) for rd in readers for s in FIT_SEEDS + HOLD_SEEDS for p in PRESETS for c in C_REQS]
    rows: list[dict] = []
    with ProcessPoolExecutor(min(args.workers, 26), initializer=_init) as ex:
        futs = [ex.submit(task, *t_) for t_ in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 12 == 0 or i == len(futs):
                print(f"[calib] {i}/{len(futs)} passes, {time.time() - t0:.0f}s", flush=True)
    wall = time.time() - t0
    results = {}
    for rd in readers:
        r = analyse(rd, [x for x in rows if x["reader"] == rd])
        r["wall_s_all_readers"] = wall
        (RESULTS / f"{rd}_loop_calibration.json").write_text(json.dumps(r, indent=1), encoding="utf8")
        results[rd] = r
        txt = ", ".join(f"{b['lo']:g}-{'+' if b['hi'] is None else format(b['hi'], 'g')}: {b['bias_pp']:+.2f} "
                        f"(n {b['n']})" for b in r["bands"])
        print(f"wrote results/{rd}_loop_calibration.json: {txt}", flush=True)
    write_md(results, skipped, wall)
    print(f"wrote results/reader_loop_calibration.md ({wall / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
