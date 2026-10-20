"""Verification (O3) evaluation of the card readers B0 (classical) and B1 (U-Net) against the proposal.

    uv run python experiments/eval_verifiers.py                       # B0 + B1, all datasets
    uv run python experiments/eval_verifiers.py --verifiers b0        # B0 only (fast, ~2 min)
    uv run python experiments/eval_verifiers.py --verifiers b1 --merge  # add/replace B1, keep B0
    uv run python experiments/eval_verifiers.py --b1-weights models/unet_v1.pt --limit 20

Datasets (proposal 4.3)
  renderer  held-out test split of data/synthetic/v1 (our WSPRenderer; in-generator)
  simoes    the 300 open synthetic cards of Simoes et al. 2025 (out-of-generator).  Truth = union of the
            annotated polygons; count = number of polygons; sizes = polygon areas / spread factor.
  real      tiles of the manually annotated real cards (Zenodo 14010262, RESTRICTED).  Not downloadable
            here.  Place them under  <real-root>/real/image/*.png  and  <real-root>/real/label/*.txt
            (YOLOv8 polygon labels, same layout as the synthetic set) and re-run; optional
            <real-root>/real/px_per_mm.json  = {"default": 24.0, "<card stem>": 23.1}  gives the scale
            (without it only coverage / IoU / class are scored, size and count outputs are skipped).

Outputs per (verifier, dataset): coverage error (MAE, bias, p95), stain-mask IoU, count error (vs
annotated stains and, for the renderer, vs connected components), stain density and Dv0.1/0.5/0.9
errors, three-class agreement, runtime, saturation flag statistics; stratified by coverage band and by
true Dv0.5 stain-size class; plus pixel recall per individual stain-size bin.  B1 is reported twice for
the stain statistics: its native watershed ("b1") and B0's watershed run on B1's mask ("b1+b0ws").

Writes results/verification_eval.json and results/verification_summary.md (criterion table:
target / measured / met | missed | blocked).
"""
from __future__ import annotations

import argparse
from itertools import islice
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from coverageai.config import px_per_mm as default_ppm, spec  # noqa: E402
from coverageai.contracts import coverage_class  # noqa: E402
from coverageai.vision import real_data  # noqa: E402
from coverageai.vision.classical import ClassicalVerifier, _volume_percentiles  # noqa: E402

COV_BANDS = [("0-15", 0.0, 15.0), ("15-30", 15.0, 30.0), ("30-50", 30.0, 50.0), ("50+", 50.0, 1e9)]
# true Dv0.5 (droplet diameter, um) of the card -> stain-size class
SIZE_CLASSES = [("fine<200um", 0.0, 200.0), ("medium200-400um", 200.0, 400.0), ("coarse>=400um", 400.0, 1e9)]
# individual stains by true droplet diameter (um), for pixel recall
STAIN_BINS = [("<100", 0.0, 100.0), ("100-200", 100.0, 200.0), ("200-400", 200.0, 400.0), (">=400", 400.0, 1e9)]
TARGETS = {"cov_mae_synth_pp": 2.0, "cov_mae_real_pp": 5.0, "class_agreement": 0.90, "runtime_s": 2.0}
DV = ("dv01_um", "dv05_um", "dv09_um")


# =============================================================== datasets
# every dataset yields dicts: id, image (RGB u8), gt (bool mask, image frame), cov (true %), ppm, ppm_known,
# n (annotated stains), n_cc (connected components, optional), dens (stains/cm2), dv (3 floats or None), spread


def _spread() -> float:
    return float(np.mean(spec()["physics"]["spread_factor"]))


def dataset_renderer(data: Path, split: str, limit: int = 0) -> Iterable[dict[str, Any]]:
    from eval_classical import discover_samples, load_sample
    found, _ = discover_samples(data, split)
    if limit:
        found = found[:limit]
    for s in found:
        r = load_sample(s)
        if r["mask"] is None:
            continue
        meta = s["meta"]
        yield {"id": r["id"], "image": r["image"], "gt": r["mask"], "cov": r["cov"], "ppm": r["ppm"],
               "ppm_known": True, "n": r["n"], "n_cc": r["n_components"],
               "dens": meta.get("stains_per_cm2"), "dv": [r.get(k) for k in DV],
               "spread": meta.get("spread_factor") or _spread()}


def _polygon_areas_px(polys: list[np.ndarray], h: int, w: int) -> np.ndarray:
    a = []
    for p in polys:
        x, y = p[:, 0] * w, p[:, 1] * h
        a.append(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return np.asarray(a, float)


def _annotation_stats(polys, h, w, ppm, spread):
    """(n, density /cm2, [dv01, dv05, dv09]) of a polygon-annotated card (card = whole image)."""
    areas = _polygon_areas_px(polys, h, w)
    areas = areas[areas > 0]
    if areas.size == 0:
        return 0, 0.0, [None] * 3
    d = 2.0 * np.sqrt(areas / np.pi) / ppm * 1000.0 / spread
    cm2 = h * w / ppm ** 2 / 100.0
    return int(areas.size), float(areas.size / cm2), [float(x) for x in _volume_percentiles(d, (0.1, 0.5, 0.9))]


def _real_ppm_table(root: Path) -> dict[str, float]:
    p = root / "px_per_mm.json"
    if p.is_file():
        try:
            return {k: float(v) for k, v in json.loads(p.read_text()).items()}
        except (ValueError, TypeError):
            pass
    return {}


def _label_files(root: Path, subset: str) -> dict[str, Path]:
    out = {}
    for img_dir, lab_dir in real_data._find_pairs(root, subset):
        for lp in lab_dir.glob("*.txt"):
            out[lp.stem] = lp
    return out


def dataset_simoes(root: Path, limit: int = 0) -> Iterable[dict[str, Any]]:
    cards = real_data.load_cards(root, "synthetic", limit=limit or None)
    labels = _label_files(root, "synthetic")
    spread = _spread()
    for c in cards:
        h, w = c.mask.shape
        polys = real_data.parse_yolo_polygons(labels[c.name].read_text()) if c.name in labels else []
        n, dens, dv = _annotation_stats(polys, h, w, c.px_per_mm, spread)
        yield {"id": c.name, "image": c.image, "gt": c.mask, "cov": 100.0 * float(c.mask.mean()),
               "ppm": c.px_per_mm, "ppm_known": True, "n": n, "n_cc": None, "dens": dens, "dv": dv,
               "spread": spread}


def real_cards_available(root: Path) -> bool:
    return bool(real_data.load_cards(root, "real", limit=1)) if root.exists() else False


def dataset_real_tiles(root: Path, tile: int = 512, limit: int = 0) -> Iterable[dict[str, Any]]:
    """Tiles of the annotated real cards. Scale from <real dir>/px_per_mm.json when present."""
    cards = real_data.load_cards(root, "real")
    labels = _label_files(root, "real")
    ppm_tab: dict[str, float] = {}
    for img_dir, _ in real_data._find_pairs(root, "real"):
        ppm_tab.update(_real_ppm_table(img_dir.parent))
    spread = _spread()
    n_out = 0
    for c in cards:
        ppm = ppm_tab.get(c.name, ppm_tab.get("default"))
        known = ppm is not None
        ppm = ppm or default_ppm()
        polys = real_data.parse_yolo_polygons(labels[c.name].read_text()) if c.name in labels else []
        h, w = c.mask.shape
        # per-tile annotation stats: polygons whose centroid lies in the tile
        cent = np.array([p.mean(0) * [w, h] for p in polys]) if polys else np.zeros((0, 2))
        for r0, c0, img_t, m_t in real_data.tiles(c, tile, tile, drop_partial=True):
            th, tw = m_t.shape
            sel = [p for p, ce in zip(polys, cent) if r0 <= ce[1] < r0 + th and c0 <= ce[0] < c0 + tw]
            local = [np.stack([(p[:, 0] * w - c0) / tw, (p[:, 1] * h - r0) / th], 1) for p in sel]
            n, dens, dv = _annotation_stats(local, th, tw, ppm, spread) if known else (None, None, [None] * 3)
            yield {"id": f"{c.name}_r{r0}_c{c0}", "image": img_t, "gt": m_t, "cov": 100.0 * float(m_t.mean()),
                   "ppm": ppm, "ppm_known": known, "n": n, "n_cc": None, "dens": dens, "dv": dv,
                   "spread": spread}
            n_out += 1
            if limit and n_out >= limit:
                return


# =============================================================== verifier adapters
class Reader:
    """Uniform front end: read(image, ppm) -> dict(meas, mask, roi, n, dens, dv, runtime)."""

    def __init__(self, name: str, fn: Callable, label: str):
        self.name, self.read, self.label = name, fn, label


def _b0_reader() -> Reader:
    v = ClassicalVerifier()

    def read(image, ppm):
        meas = v.measure(image, ppm)
        info = dict(v.last_info)
        mask, roi = v.segment(image)
        return {"meas": meas, "mask": mask, "roi": roi, "n": info.get("n_stains"),
                "dens": meas.stains_per_cm2, "dv": [meas.dv01_um, meas.dv05_um, meas.dv09_um],
                "runtime": meas.runtime_s}
    return Reader("b0", read, "B0 classical")


def _b1_readers(weights: str) -> list[Reader]:
    from coverageai.vision.unet import UNetVerifier
    u = UNetVerifier(weights)
    b0 = ClassicalVerifier()
    cache: dict[str, Any] = {}

    def base(image, ppm):
        meas, mask, roi = u._measure_full(image, ppm)
        cache["x"] = (meas, mask, roi)
        return meas, mask, roi

    def native(image, ppm):
        meas, mask, roi = base(image, ppm)
        cm2 = max(int(roi.sum()), 1) / (ppm * 10.0) ** 2
        n = meas.stains_per_cm2 * cm2 if meas.stains_per_cm2 is not None else None
        return {"meas": meas, "mask": mask, "roi": roi, "n": None if n is None else int(round(n)),
                "dens": meas.stains_per_cm2, "dv": [meas.dv01_um, meas.dv05_um, meas.dv09_um],
                "runtime": meas.runtime_s}

    def with_b0_ws(image, ppm):
        meas, mask, roi = cache["x"]               # same call, reuse B1's mask (no second network pass)
        t0 = time.perf_counter()
        st = b0.stats_from_mask(mask & roi, ppm, int(roi.sum()))
        return {"meas": meas, "mask": mask, "roi": roi, "n": st["n_stains"], "dens": st["stains_per_cm2"],
                "dv": [st["dv01_um"], st["dv05_um"], st["dv09_um"]],
                "runtime": meas.runtime_s + (time.perf_counter() - t0)}

    r1 = Reader("b1", native, "B1 U-Net (native watershed)")
    r2 = Reader("b1+b0ws", with_b0_ws, "B1 mask + B0 watershed stats")
    r2.depends_on = r1                                  # type: ignore[attr-defined]
    return [r1, r2]


# =============================================================== metrics
def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = int((a | b).sum())
    return float((a & b).sum() / u) if u else 1.0


def _rel(est, tru):
    return abs(est - tru) / tru if est is not None and tru else None


def stain_recall(gt: np.ndarray, pred: np.ndarray, ppm: float, spread: float) -> dict[str, list[float]]:
    """Per stain-size bin: [n_gt_components, sum of detected fraction, gt px, detected px]."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), connectivity=8)
    out = {name: [0.0, 0.0, 0.0, 0.0] for name, _, _ in STAIN_BINS}
    if n <= 1:
        return out
    area = st[1:, cv2.CC_STAT_AREA].astype(float)
    det = np.bincount(lab[pred & gt].ravel(), minlength=n)[1:].astype(float)
    d = 2.0 * np.sqrt(area / np.pi) / ppm * 1000.0 / spread
    for name, lo, hi in STAIN_BINS:
        s = (d >= lo) & (d < hi)
        if s.any():
            out[name] = [float(s.sum()), float((det[s] / area[s]).sum()), float(area[s].sum()), float(det[s].sum())]
    return out


def evaluate(cards: Iterable[dict[str, Any]], readers: list[Reader], warmup: bool = True) -> dict[str, list[dict]]:
    """Run every reader over every card. Readers with `depends_on` reuse the preceding reader's call."""
    s = spec()["coverage_classes"]
    rows: dict[str, list[dict]] = {r.name: [] for r in readers}
    warmed = not warmup
    for i, c in enumerate(cards):
        if i and i % 25 == 0:
            print(f"  ... {i} cards", flush=True)
        if not warmed:                                   # first-call overhead (torch init) is not runtime
            for r in readers:
                if not hasattr(r, "depends_on"):
                    r.read(c["image"], c["ppm"])
            warmed = True
        for r in readers:
            o = r.read(c["image"], c["ppm"])
            meas, mask, roi = o["meas"], o["mask"], o["roi"]
            gt = c["gt"]
            if gt.shape != mask.shape:
                continue
            gt_roi = gt & roi
            tc = float(c["cov"])
            rec = stain_recall(gt_roi, mask, c["ppm"], c["spread"]) if c["ppm_known"] else None
            sized = c["ppm_known"]
            dv_err = {f"{k}_rel_err": (_rel(o["dv"][i], c["dv"][i]) if sized else None) for i, k in enumerate(DV)}
            rows[r.name].append({
                "id": c["id"], "true_cov": tc, "meas_cov": meas.coverage_pct,
                "err_pp": meas.coverage_pct - tc, "abs_err_pp": abs(meas.coverage_pct - tc),
                "iou": _iou(mask & roi, gt_roi),
                "n_true": c["n"], "n_meas": o["n"] if sized else None,
                "count_rel_err": _rel(o["n"], c["n"]) if sized else None,
                "count_rel_err_vs_cc": _rel(o["n"], c["n_cc"]) if sized else None,
                "dens_rel_err": _rel(o["dens"], c["dens"]) if sized else None,
                **dv_err, "dv05_true_um": c["dv"][1] if sized else None,
                "class_true": coverage_class(tc, s["acceptable_min"], s["marginal_min"]),
                "class_meas": meas.coverage_class, "saturated": bool(meas.saturated),
                "runtime_s": float(o["runtime"]), "recall": rec,
            })
    return rows


def _agg(rows: list[dict]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    g = lambda k: np.array([r[k] for r in rows if r.get(k) is not None], float)
    ae, err, rt = g("abs_err_pp"), g("err_pp"), g("runtime_s")
    sat = np.array([r["saturated"] for r in rows])
    out: dict[str, Any] = {
        "n": len(rows), "mae_pp": float(ae.mean()), "rmse_pp": float(np.sqrt((err ** 2).mean())),
        "median_abs_err_pp": float(np.median(ae)), "p95_abs_err_pp": float(np.percentile(ae, 95)),
        "max_abs_err_pp": float(ae.max()), "bias_pp": float(err.mean()),
        "class_agreement": float(np.mean([r["class_true"] == r["class_meas"] for r in rows])),
        "saturated_frac": float(sat.mean()),
        "mae_pp_saturated": float(ae[sat].mean()) if sat.any() else None,
        "mae_pp_not_saturated": float(ae[~sat].mean()) if (~sat).any() else None,
        "runtime_mean_s": float(rt.mean()), "runtime_median_s": float(np.median(rt)), "runtime_p95_s": float(np.percentile(rt, 95)),
        "runtime_max_s": float(rt.max()),
        "iou_mean": float(g("iou").mean()),
    }
    for k in ("count_rel_err", "count_rel_err_vs_cc", "dens_rel_err", "dv01_um_rel_err", "dv05_um_rel_err",
              "dv09_um_rel_err"):
        v = g(k)
        out[k + "_median"] = float(np.median(v)) if v.size else None
    out["n_with_size_outputs"] = int(g("count_rel_err").size)
    return out


def summarize(rows: list[dict]) -> dict[str, Any]:
    by_cov = {n: _agg([r for r in rows if lo <= r["true_cov"] < hi]) for n, lo, hi in COV_BANDS}
    sized = [r for r in rows if r.get("dv05_true_um")]
    by_size = {n: _agg([r for r in sized if lo <= r["dv05_true_um"] < hi]) for n, lo, hi in SIZE_CLASSES}
    recall = {}
    for name, _, _ in STAIN_BINS:
        tot = np.zeros(4)
        for r in rows:
            if r.get("recall"):
                tot += np.array(r["recall"][name])
        recall[name] = {"n_stains": int(tot[0]), "mean_recall": float(tot[1] / tot[0]) if tot[0] else None,
                        "pixel_recall": float(tot[3] / tot[2]) if tot[2] else None}
    return {"overall": _agg(rows), "by_coverage_band": by_cov, "by_stain_size_class": by_size,
            "stain_recall_by_size": recall}


def time_readers(cards: list[dict[str, Any]], readers: list[Reader], repeats: int = 3) -> dict[str, dict]:
    """Per-card runtime with one warm-up call and the minimum over `repeats` calls (robust to other
    processes stealing CPU; the load that remains is recorded by the caller)."""
    out = {}
    for r in readers:
        if hasattr(r, "depends_on"):
            continue
        r.read(cards[0]["image"], cards[0]["ppm"])
        per = []
        for c in cards:
            per.append(min(float(r.read(c["image"], c["ppm"])["runtime"]) for _ in range(repeats)))
        a = np.array(per)
        out[r.name] = {"n": len(a), "repeats": repeats, "median_s": float(np.median(a)), "mean_s": float(a.mean()),
                       "p95_s": float(np.percentile(a, 95)), "max_s": float(a.max())}
    return out


# =============================================================== criteria
def _status(measured, target, kind):
    if measured is None:
        return "blocked"
    return "met" if (measured <= target if kind == "le" else measured >= target) else "missed"


def criteria_table(res: dict[str, Any], timing: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Rows: criterion x verifier. `res[verifier][dataset]` = summarize() output or {"blocked": reason}."""
    rows = []

    def get(v, ds):
        x = res.get(v, {}).get(ds)
        return None if (x is None or "blocked" in x) else x["overall"]

    for v in res:
        label = {"b0": "B0", "b1": "B1", "b1+b0ws": "B1+B0ws"}.get(v, v)
        if v == "b1+b0ws":
            continue
        specs = [
            ("Coverage MAE <= 2 pp, held-out renderer cards", "renderer", "mae_pp", TARGETS["cov_mae_synth_pp"], "le", "pp"),
            ("Coverage MAE <= 2 pp, Simoes synthetic (out-of-generator)", "simoes", "mae_pp", TARGETS["cov_mae_synth_pp"], "le", "pp"),
            ("Coverage MAE <= 5 pp, real annotated cards (tiles)", "real", "mae_pp", TARGETS["cov_mae_real_pp"], "le", "pp"),
            ("Three-class agreement >= 90 %, renderer", "renderer", "class_agreement", TARGETS["class_agreement"], "ge", "frac"),
            ("Three-class agreement >= 90 %, Simoes synthetic", "simoes", "class_agreement", TARGETS["class_agreement"], "ge", "frac"),
            ("Three-class agreement >= 90 %, real tiles", "real", "class_agreement", TARGETS["class_agreement"], "ge", "frac"),
        ]
        for text, ds, key, tgt, kind, unit in specs:
            o = get(v, ds)
            m = None if o is None else o.get(key)
            blocked = res.get(v, {}).get(ds, {}).get("blocked") if m is None else None
            rows.append({"criterion": text, "verifier": label, "target": tgt, "unit": unit, "kind": kind,
                         "measured": m, "status": _status(m, tgt, kind), "note": blocked or ""})
        t = (timing or {}).get(v)
        m = t["p95_s"] if t else (get(v, "renderer") or {}).get("runtime_p95_s")
        rows.append({"criterion": "Runtime <= 2 s/card (p95, this CPU, min of repeats)" if t else
                     "Runtime <= 2 s/card (p95, this CPU, single pass, load-inflated)",
                     "verifier": label, "target": TARGETS["runtime_s"], "unit": "s", "kind": "le", "measured": m,
                     "status": _status(m, TARGETS["runtime_s"], "le"), "note": ""})
    return rows


def _fmt(x, unit=""):
    if x is None:
        return "n/a"
    return f"{100 * x:.1f} %" if unit == "frac" else f"{x:.2f} {unit}".strip()


def render_markdown(out: dict[str, Any]) -> str:
    L = ["# Verification (O3) results: B0 classical and B1 U-Net", ""]
    L.append(f"Generated by `experiments/eval_verifiers.py` on {out['meta']['machine']}; "
             f"runtimes are wall-clock per card on this CPU (no GPU).")
    L.append("")
    L.append(f"B1 weights: `{out['meta']['b1_weights']}` ({out['meta']['b1_status']}).")
    L.append("")
    L.append("## Criterion table")
    L.append("")
    L.append("| Criterion | Verifier | Target | Measured | Status |")
    L.append("|---|---|---|---|---|")
    for r in out["criteria"]:
        tgt = ("<= " if r["kind"] == "le" else ">= ") + _fmt(r["target"], r["unit"])
        status = r["status"] if r["status"] != "blocked" else "blocked: needs data access" if "real" in r["criterion"] else "blocked"
        L.append(f"| {r['criterion']} | {r['verifier']} | {tgt} | {_fmt(r['measured'], r['unit'])} | {status} |")
    L.append("")
    for v, t in (out.get("timing") or {}).items():
        L.append(f"Controlled timing {v}: {t['n']} renderer cards, min of {t['repeats']} calls after warm-up: median {t['median_s']:.2f} s, "
                 f"mean {t['mean_s']:.2f}, p95 {t['p95_s']:.2f}, max {t['max_s']:.2f}.")
    if out["meta"].get("note"):
        L.append("")
        L.append(f"Load caveat: {out['meta']['note']}")
    L.append("")
    L.append("Runtime is judged on the 95th percentile so that a few slow cards do not hide behind a mean; "
             "the mean and max are in the tables below. The proposal's 'laptop CPU' is this machine.")
    L.append("")
    for v, dss in out["results"].items():
        for ds, r in dss.items():
            L.append(f"## {v} on {ds}")
            L.append("")
            if "blocked" in r:
                L.append(f"Blocked: {r['blocked']}")
                L.append("")
                continue
            o = r["overall"]
            L.append(f"n = {o['n']}; coverage MAE {o['mae_pp']:.2f} pp (bias {o['bias_pp']:+.2f}, p95 {o['p95_abs_err_pp']:.2f}, "
                     f"max {o['max_abs_err_pp']:.2f}); IoU {o['iou_mean']:.3f}; class agreement {o['class_agreement']:.3f}; "
                     f"runtime median/mean/p95/max {o.get('runtime_median_s', float('nan')):.2f}/{o['runtime_mean_s']:.2f}/{o['runtime_p95_s']:.2f}/{o['runtime_max_s']:.2f} s; "
                     f"flagged saturated {100 * o['saturated_frac']:.1f} %"
                     + (f" (MAE flagged {o['mae_pp_saturated']:.2f} vs unflagged {o['mae_pp_not_saturated']:.2f} pp)"
                        if o['mae_pp_saturated'] is not None and o['mae_pp_not_saturated'] is not None else "") + ".")
            L.append("")
            L.append("Secondary outputs (median relative error): "
                     + ", ".join(f"{k.replace('_rel_err_median', '')} {100 * o[k]:.0f} %"
                                 for k in ("count_rel_err_median", "count_rel_err_vs_cc_median", "dens_rel_err_median",
                                           "dv01_um_rel_err_median", "dv05_um_rel_err_median", "dv09_um_rel_err_median")
                                 if o.get(k) is not None) or "not scored (scale unknown)")
            L.append("")
            for title, key in (("coverage band (true %)", "by_coverage_band"), ("true Dv0.5 stain-size class", "by_stain_size_class")):
                L.append(f"| {title} | n | MAE pp | bias pp | IoU | class agr. | count rel. err | Dv0.5 rel. err | saturated |")
                L.append("|---|---|---|---|---|---|---|---|---|")
                for name, a in r[key].items():
                    if not a.get("n"):
                        L.append(f"| {name} | 0 |  |  |  |  |  |  |  |")
                        continue
                    f = lambda k: "n/a" if a.get(k) is None else f"{a[k]:.2f}"
                    L.append(f"| {name} | {a['n']} | {a['mae_pp']:.2f} | {a['bias_pp']:+.2f} | {a['iou_mean']:.3f} | "
                             f"{a['class_agreement']:.2f} | {f('count_rel_err_median')} | {f('dv05_um_rel_err_median')} | "
                             f"{100 * a['saturated_frac']:.0f} % |")
                L.append("")
            rec = r["stain_recall_by_size"]
            L.append("Pixel recall of annotated stains by droplet size (um): "
                     + ", ".join(f"{k}: {v['mean_recall']:.2f} (n={v['n_stains']})" for k, v in rec.items() if v["mean_recall"] is not None)
                     if any(v["mean_recall"] is not None for v in rec.values()) else "")
            L.append("")
    L.append("## Not covered / blocked")
    L.append("")
    for line in out["blockers"]:
        L.append(f"* {line}")
    L.append("")
    L.append("## What to run to update these numbers")
    L.append("")
    L.extend(out["user_actions"])
    L.append("")
    return "\n".join(L)


# =============================================================== main
def run(args) -> dict[str, Any]:
    data = Path(args.data) if Path(args.data).is_absolute() else ROOT / args.data
    simoes = Path(args.simoes_root) if Path(args.simoes_root).is_absolute() else ROOT / args.simoes_root
    want = [v.strip() for v in args.verifiers.split(",") if v.strip()]
    readers: list[Reader] = []
    notes: dict[str, str] = {}
    if "b0" in want:
        readers.append(_b0_reader())
    if "b1" in want:
        w = Path(args.b1_weights) if Path(args.b1_weights).is_absolute() else ROOT / args.b1_weights
        try:
            readers += _b1_readers(str(w))
        except Exception as e:                       # torch missing / no weights
            notes["b1"] = f"B1 not evaluated: {type(e).__name__}: {e}"
            print(notes["b1"])
    results: dict[str, dict[str, Any]] = {r.name: {} for r in readers}

    plans: list[tuple[str, Callable[[], Iterable[dict]]]] = []
    if "renderer" in args.datasets:
        plans.append(("renderer", lambda: islice(dataset_renderer(data, args.split, args.limit), 0, None, args.stride)))
    if "simoes" in args.datasets:
        plans.append(("simoes", lambda: islice(dataset_simoes(simoes, args.limit), 0, None, args.stride)))
    real_here = "real" in args.datasets and real_cards_available(simoes)
    if real_here:
        plans.append(("real", lambda: dataset_real_tiles(simoes, args.tile, args.limit)))
    elif "real" in args.datasets:
        for r in readers:
            results[r.name]["real"] = {"blocked": "annotated real cards (Zenodo 14010262, restricted) are not on disk; "
                                                  f"place them under {simoes}/real/image + real/label"}

    wall = {}
    timing = {}
    if args.timing:
        tcards = list(islice(dataset_renderer(data, args.split, 0), 0, None, max(1, 180 // args.timing)))[: args.timing]
        timing = time_readers(tcards, readers, args.timing_repeats)
        for k, v in timing.items():
            print(f"timing {k}: median {v['median_s']:.2f} mean {v['mean_s']:.2f} p95 {v['p95_s']:.2f} max {v['max_s']:.2f} s", flush=True)
    for ds, make in plans:
        t0 = time.perf_counter()
        rows = evaluate(make(), readers)
        wall[ds] = time.perf_counter() - t0
        for r in readers:
            results[r.name][ds] = {**summarize(rows[r.name]), "eval_wall_s": wall[ds],
                                   "cards": [{k: v for k, v in x.items() if k != "recall"} for x in rows[r.name]]}
            o = results[r.name][ds]["overall"]
            print(f"{r.name:8s} {ds:9s} n={o['n']:4d} MAE={o['mae_pp']:.2f} IoU={o['iou_mean']:.3f} "
                  f"cls={o['class_agreement']:.3f} rt(mean/p95)={o['runtime_mean_s']:.2f}/{o['runtime_p95_s']:.2f}", flush=True)
    return {"results": results, "notes": notes, "wall": wall, "timing": timing}


def build_output(run_out: dict[str, Any], args, previous: dict[str, Any] | None = None) -> dict[str, Any]:
    import platform
    results = dict(previous["results"]) if previous else {}
    timing = dict(previous.get("timing") or {}) if previous else {}
    timing.update(run_out.get("timing") or {})
    for k, v in run_out["results"].items():       # merge per dataset so partial runs add up
        results[k] = {**results.get(k, {}), **{ds: x for ds, x in v.items()
                                               if not ("blocked" in x and ds in results.get(k, {})
                                                       and "blocked" not in results[k][ds])}}
    w = args.b1_weights
    status = "checkpoint not readable"
    try:
        import torch
        wp = Path(w) if Path(w).is_absolute() else ROOT / w
        ck = torch.load(wp, map_location="cpu", weights_only=False)
        ep = int(ck.get("epoch", -1))
        status = (f"checkpoint epoch {ep}, val Dice {float(ck.get('val_dice', float('nan'))):.3f}, "
                  f"amp={ck.get('amp')}, {ck.get('n_train_cards')} training cards"
                  + ("; PARTIAL training, not the final GPU model" if ep < 20 else ""))
    except Exception as e:                                          # noqa: BLE001
        status = f"checkpoint not readable ({type(e).__name__})"
    blockers = [
        "Real annotated cards: the Simoes et al. real cards (Zenodo 14010262) are a restricted record; the open "
        "record 13995950 holds only the 300 synthetic cards. The real-card tile evaluation is implemented "
        "(`dataset_real_tiles`) and runs automatically when the files are present. Criteria on real cards stay "
        "'blocked: needs data access' until then.",
        "Stain counts: the renderer annotates every droplet while touching stains merge visibly, so count error "
        "against annotated droplets is bounded below by overlap; the count against connected components "
        "(`count_rel_err_vs_cc`) isolates the watershed quality. Simoes polygons are per droplet.",
        "Droplet-size outputs depend on the spread factor (1.6 to 2.2 in the renderer, assumed 1.9 by the readers) "
        "and, for the Simoes cards, on an inferred scale (76 mm card width / image width); neither is published.",
    ]
    actions = [
        "1. B1 full GPU training is DONE (27 epochs, bf16, best val Dice 0.969 / IoU 0.940, see "
        "`results/unet_train_log.csv`; GPU-host test eval in `results/b1_synthetic.json`, runtime there is NOT this laptop). "
        "The old 6-epoch CPU weights are `models/unet_v1_cpu_partial.pt`. Nothing more to run on Colab unless you retrain.",
        "2. After any new weights: `uv run python experiments/eval_verifiers.py --verifiers b1 --merge` (about 15 to 30 "
        "min on CPU) refreshes the B1 rows here and writes the local re-evaluation `results/b1_synthetic_local.json`. "
        "Re-run it on an idle machine for clean runtimes (runtime is wall-clock and sensitive to other CPU load).",
        "3. Real cards: request access to Zenodo record 14010262, unzip so that the files are "
        "`data/external/simoes2025/real/image/*.png` and `real/label/*.txt` (optional `real/px_per_mm.json` = "
        "{\"default\": px_per_mm}), then `uv run python experiments/eval_verifiers.py --merge` fills the real-tile rows "
        "and flips the three 'blocked' criteria to met/missed.",
    ]
    out = {"meta": {"machine": f"{platform.processor() or platform.machine()} / {platform.system()} CPU-only",
                    "b1_weights": str(w), "b1_status": status, "split": args.split, "limit": args.limit, "stride": args.stride, "note": args.note,
                    "tile": args.tile, "targets": TARGETS, "size_classes": SIZE_CLASSES, "coverage_bands": COV_BANDS,
                    "notes": run_out["notes"]},
           "results": results, "blockers": blockers, "user_actions": actions}
    out["timing"] = timing
    out["criteria"] = criteria_table(results, timing)
    return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/synthetic/v1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--simoes-root", default="data/external/simoes2025")
    ap.add_argument("--verifiers", default="b0,b1")
    ap.add_argument("--datasets", default="renderer,simoes,real")
    ap.add_argument("--b1-weights", default="models/unet_v1.pt")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="cards (tiles for real) per dataset; 0 = all")
    ap.add_argument("--stride", type=int, default=1, help="evaluate every k-th card (B1 on CPU is slow)")
    ap.add_argument("--timing", type=int, default=0, help="also time N renderer cards (min of repeats)")
    ap.add_argument("--timing-repeats", type=int, default=3)
    ap.add_argument("--note", default="", help="free text stored in the JSON (e.g. CPU contention during the run)")
    ap.add_argument("--out-json", default="results/verification_eval.json")
    ap.add_argument("--out-md", default="results/verification_summary.md")
    ap.add_argument("--merge", action="store_true", help="keep other verifiers' entries from the existing JSON")
    args = ap.parse_args(argv)
    args.datasets = [d.strip() for d in args.datasets.split(",")]
    oj = Path(args.out_json) if Path(args.out_json).is_absolute() else ROOT / args.out_json
    om = Path(args.out_md) if Path(args.out_md).is_absolute() else ROOT / args.out_md
    prev = json.loads(oj.read_text()) if args.merge and oj.is_file() else None
    run_out = run(args)
    out = build_output(run_out, args, prev)
    oj.parent.mkdir(parents=True, exist_ok=True)
    default = lambda o: o.item() if hasattr(o, "item") else str(o)
    oj.write_text(json.dumps(out, indent=1, default=default))
    b1r = out["results"].get("b1", {}).get("renderer")
    if b1r and "overall" in b1r:       # local CPU re-evaluation, kept apart from the GPU-host results/b1_synthetic.json
        (oj.parent / "b1_synthetic_local.json").write_text(json.dumps(
            {"verifier": "B1_unet (local CPU re-evaluation, eval_verifiers.py)", "weights": args.b1_weights,
             "split": args.split, "note": args.note, "overall": b1r["overall"],
             "by_coverage_band": b1r["by_coverage_band"], "by_stain_size_class": b1r["by_stain_size_class"]},
            indent=1, default=default))
    om.write_text(render_markdown(out), encoding="utf8")
    print(f"-> {oj}\n-> {om}")
    return out


if __name__ == "__main__":
    main()
