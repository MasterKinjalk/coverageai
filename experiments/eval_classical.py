"""Evaluate the classical verifier (B0) on a synthetic WSP dataset.

    uv run python experiments/eval_classical.py --data data/synthetic/v1 --split test

Reports per-card coverage abs error (pp), stain-mask IoU, relative stain-count error,
3-class agreement and runtime, overall and stratified by true-coverage band, and writes
``results/b0_synthetic.json`` plus ``experiments/figures/b0_parity.png``.

Dataset discovery is defensive (the WP-03 layout was not final when this was written):
  * samples = JSON sidecar files holding a coverage key; PNG image and PNG mask are found
    by names stored in the JSON, or by stem (<id>.png / <id>_image.png, <id>_mask.png, ...);
  * split = a ``<split>/`` directory, or a splits/manifest file (json, csv, txt), else all;
  * keys: coverage_pct | coverage | ..., px_per_mm | ppm, n_stains | count | ...
If the data directory does not exist, cards from ``coverageai.vision.simple_cards`` are
generated instead and ``data_source`` in the JSON says so.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coverageai.config import px_per_mm as default_ppm, spec  # noqa: E402
from coverageai.contracts import coverage_class  # noqa: E402
from coverageai.vision.classical import ClassicalVerifier  # noqa: E402

BANDS = [("0-15", 0.0, 15.0), ("15-30", 15.0, 30.0), ("30-50", 30.0, 50.0), ("50-80+", 50.0, 1e9)]
COV_KEYS = ["coverage_pct", "true_coverage_pct", "gt_coverage_pct", "coverage_percent", "cov_pct", "coverage"]
N_KEYS = ["stain_count", "n_stains", "n_stain", "num_stains", "n_droplets", "n_stains_gt", "count"]
DV_KEYS = {"dv01_um": ["dv01_um", "dv0.1_um", "dv10_um"], "dv05_um": ["dv05_um", "dv0.5_um", "dv50_um"],
           "dv09_um": ["dv09_um", "dv0.9_um", "dv90_um"]}
PPM_KEYS = ["px_per_mm", "ppm", "pixels_per_mm"]
MANIFEST_STEMS = {"splits", "split", "manifest", "index", "metadata", "labels", "dataset", "config"}


# ------------------------------------------------------------------ loading
def imread(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    buf = np.fromfile(str(path), np.uint8)
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = d
    return out


def _find_key(flat: dict[str, Any], names: list[str]) -> Any:
    last = {k.split(".")[-1].lower(): v for k, v in flat.items()}
    for n in names:                       # priority order
        if n in last and isinstance(last[n], (int, float)) and not isinstance(last[n], bool):
            return float(last[n])
    return None


def _norm(x: Any) -> str:
    """Id key that ignores zero padding (splits.json may hold ints, files '00003')."""
    t = Path(str(x)).stem
    return t.lstrip("0") or "0" if t.isdigit() else t


def _split_ids(root: Path, split: str) -> set[str] | None:
    """Ids listed for ``split`` by a splits/manifest file, or None if there is none."""
    cands = [root / "splits.json", root / "split.json", root / "splits" / f"{split}.json",
             root / "splits" / f"{split}.txt", root / "splits" / f"{split}.csv",
             root / f"{split}.txt", root / f"{split}.csv", root / f"{split}.json"]
    cands += [root / n for n in ("splits.csv", "split.csv", "manifest.csv", "index.csv",
                                 "metadata.csv", "labels.csv")]
    for p in cands:
        if not p.exists():
            continue
        try:
            if p.suffix == ".json":
                data = json.loads(p.read_text())
                if isinstance(data, dict):
                    if split in data:
                        data = data[split]
                    else:
                        continue
                return {_norm(x["id"] if isinstance(x, dict) and "id" in x else x) for x in data}
            if p.suffix == ".txt":
                return {_norm(l.strip()) for l in p.read_text().splitlines() if l.strip()}
            if p.suffix == ".csv":
                rows = list(csv.DictReader(p.open(encoding="utf8")))
                if not rows:
                    continue
                cols = {c.lower(): c for c in rows[0]}
                idc = next((cols[c] for c in ("id", "card_id", "name", "file", "image", "filename", "stem") if c in cols), None)
                if idc is None:
                    continue
                if "split" in cols:
                    return {_norm(r[idc]) for r in rows if r[cols["split"]] == split}
                if p.stem == split:
                    return {_norm(r[idc]) for r in rows}
        except (ValueError, KeyError, OSError):
            continue
    return None


def discover_samples(data_dir: Path, split: str) -> tuple[list[dict[str, Any]], str]:
    """Return ([{id, image, mask, meta}], how the split was resolved)."""
    root = data_dir
    how = "all samples (no split information found)"
    if (data_dir / split).is_dir():
        root, how = data_dir / split, f"directory {split}/"
        ids_filter = None
    else:
        ids_filter = _split_ids(data_dir, split)
        if ids_filter is not None:
            how = f"split file ({len(ids_filter)} ids)"
    pngs: dict[str, list[Path]] = {}
    for p in root.rglob("*.png"):
        pngs.setdefault(p.stem.lower(), []).append(p)
    samples = []
    for jp in sorted(root.rglob("*.json")):
        if jp.stem.lower() in MANIFEST_STEMS:
            continue
        try:
            meta = json.loads(jp.read_text())
        except ValueError:
            continue
        if not isinstance(meta, dict):
            continue
        flat = _flatten(meta)
        if _find_key(flat, COV_KEYS) is None:
            continue
        sid = jp.stem
        if ids_filter is not None and _norm(sid) not in ids_filter:
            continue
        img_p = mask_p = None
        for k, v in flat.items():            # explicit file names inside the JSON
            if isinstance(v, str) and v.lower().endswith(".png"):
                for base in (jp.parent, root, data_dir):
                    if (base / v).exists():
                        if "mask" in k.lower():
                            mask_p = base / v
                        elif any(t in k.lower() for t in ("image", "img", "file", "path")):
                            img_p = base / v
                        break
        if img_p is None or mask_p is None:
            def pick(names: list[str], want_mask: bool) -> Path | None:
                hits = [p for n in names for p in pngs.get(n.lower(), [])]
                hits = [p for p in hits if (("mask" in p.stem.lower() or "mask" in p.parent.name.lower()
                                             or "gt" in p.stem.lower().split("_")) == want_mask)]
                hits.sort(key=lambda p: (p.parent != jp.parent, len(str(p))))
                return hits[0] if hits else None
            img_p = img_p or pick([sid, sid + "_image", sid + "_img", sid + "_rgb", "image_" + sid, "img_" + sid], False)
            mask_p = mask_p or pick([sid + "_mask", sid + "_gt", sid + "_label", "mask_" + sid, sid + "_stain_mask",
                                     sid, sid + "_seg"], True)
        if img_p is None:
            continue
        samples.append({"id": sid, "image": img_p, "mask": mask_p, "meta": meta})
    return samples, how


def load_sample(s: dict[str, Any]) -> dict[str, Any]:
    flat = _flatten(s["meta"])
    img = cv2.cvtColor(imread(s["image"]), cv2.COLOR_BGR2RGB)
    mask = None
    if s["mask"] is not None:
        m = imread(s["mask"], cv2.IMREAD_GRAYSCALE)
        mask = m > 0
    ppm = _find_key(flat, PPM_KEYS)
    return {"id": s["id"], "image": img, "mask": mask,
            "cov": _find_key(flat, COV_KEYS), "n": _find_key(flat, N_KEYS),
            "ppm": ppm if ppm else default_ppm(), "ppm_from_meta": bool(ppm), "count_reliable": True,
            "n_components": _find_key(flat, ["stain_components"]),
            **{k: _find_key(flat, v) for k, v in DV_KEYS.items()}}


def generated_samples(n: int, seed: int = 0):
    """Fallback data: simple test cards stratified over the coverage bands."""
    from coverageai.vision.simple_cards import make_card
    rng = np.random.default_rng(seed)
    for i in range(n):
        lo, hi = BANDS[i % 4][1], min(BANDS[i % 4][2], 80.0)
        cov = float(rng.uniform(max(lo, 0.3), hi))
        kind = ["clean", "degraded", "phone"][(i // 4) % 3]
        kw: dict[str, Any] = dict(overlap=cov > 25.0, ellipse=True)
        if kind != "clean":
            kw.update(blur_px=float(rng.uniform(0.5, 1.5)), noise_sigma=float(rng.uniform(2, 10)),
                      gradient=float(rng.uniform(0.0, 0.25)))
        if kind == "phone":
            kw.update(perspective=True)
        c = make_card(rng, cov, **kw)
        yield {"id": f"sim_{i:04d}_{kind}", "image": c.image, "mask": c.mask, "cov": c.coverage_pct,
               "n": float(c.n_stains), "ppm": c.px_per_mm, "ppm_from_meta": True,
               "count_reliable": not kw["overlap"]}


# --------------------------------------------------------------- evaluation
def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = int((a | b).sum())
    return float((a & b).sum() / u) if u else 1.0


def evaluate(samples, verifier: ClassicalVerifier) -> list[dict[str, Any]]:
    spec_cls = spec()["coverage_classes"]
    rows = []
    for s in samples:
        if not isinstance(s, dict) or "image" not in s or isinstance(s["image"], Path):
            s = load_sample(s)
        meas = verifier.measure(s["image"], s["ppm"])
        mask, _ = verifier.segment(s["image"])
        gt_mask = s["mask"]
        if gt_mask is not None and gt_mask.shape != mask.shape:
            gt_mask = None
        true_cov = s["cov"] if s["cov"] is not None else (100.0 * gt_mask.mean() if gt_mask is not None else None)
        if true_cov is None:
            continue
        n_true = s["n"]
        dv_err = {}
        for k in DV_KEYS:
            est, tru = getattr(meas, k), s.get(k)
            dv_err[k + "_rel_err"] = abs(est - tru) / tru if est is not None and tru else None
        rows.append({
            "id": s["id"], "true_cov": float(true_cov), "meas_cov": meas.coverage_pct,
            "abs_err_pp": abs(meas.coverage_pct - true_cov), "err_pp": meas.coverage_pct - true_cov,
            "iou": iou(mask, gt_mask) if gt_mask is not None else None,
            "n_true": n_true, "n_meas": verifier.last_info.get("n_stains"),
            "count_rel_err": (abs(verifier.last_info.get("n_stains", 0) - n_true) / max(n_true, 1.0)
                              if n_true is not None and s["count_reliable"] else None),
            "count_rel_err_vs_components": (abs(verifier.last_info.get("n_stains", 0) - s["n_components"])
                                            / max(s["n_components"], 1.0) if s.get("n_components") else None),
            **dv_err,
            "class_true": coverage_class(true_cov, spec_cls["acceptable_min"], spec_cls["marginal_min"]),
            "class_meas": meas.coverage_class,
            "saturated": meas.saturated, "runtime_s": meas.runtime_s,
            "threshold_method": verifier.last_info.get("threshold_method"),
            "geometry": verifier.last_info.get("geometry"),
        })
    return rows


def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    f = lambda k: [r[k] for r in rows if r[k] is not None]
    ae = np.array(f("abs_err_pp"))
    out: dict[str, Any] = {
        "n": len(rows), "mae_pp": float(ae.mean()), "median_abs_err_pp": float(np.median(ae)),
        "p95_abs_err_pp": float(np.percentile(ae, 95)), "max_abs_err_pp": float(ae.max()),
        "bias_pp": float(np.mean(f("err_pp"))),
        "class_agreement": float(np.mean([r["class_true"] == r["class_meas"] for r in rows])),
        "saturated_frac": float(np.mean([r["saturated"] for r in rows])),
        "runtime_mean_s": float(np.mean(f("runtime_s"))), "runtime_max_s": float(np.max(f("runtime_s"))),
    }
    iou_v, ce = f("iou"), f("count_rel_err")
    out["iou_mean"] = float(np.mean(iou_v)) if iou_v else None
    out["count_rel_err_median"] = float(np.median(ce)) if ce else None
    out["count_rel_err_mean"] = float(np.mean(ce)) if ce else None
    out["n_with_count"] = len(ce)
    for k in ("count_rel_err_vs_components", "dv01_um_rel_err", "dv05_um_rel_err", "dv09_um_rel_err"):
        v = f(k)
        out[k + "_median"] = float(np.median(v)) if v else None
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strata = {}
    for name, lo, hi in BANDS:
        strata[name] = _agg([r for r in rows if lo <= r["true_cov"] < hi])
    overall = _agg(rows)
    overall["targets"] = {
        "coverage_mae_pp<=2": bool(overall.get("mae_pp", 99) <= 2.0),
        "class_agreement>=0.90": bool(overall.get("class_agreement", 0) >= 0.90),
        "runtime_max_s<=2": bool(overall.get("runtime_max_s", 99) <= 2.0),
    }
    return {"overall": overall, "by_coverage_band": strata}


def parity_plot(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"0-15": "#2a9d8f", "15-30": "#e9c46a", "30-50": "#f4a261", "50-80+": "#9b2226"}
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=150)
    top = max([r["true_cov"] for r in rows] + [r["meas_cov"] for r in rows] + [10.0]) * 1.05
    xs = np.array([0, top])
    ax.plot(xs, xs, color="0.25", lw=1, label="y = x")
    ax.fill_between(xs, xs - 2, xs + 2, color="0.85", alpha=0.6, label="+/- 2 pp", lw=0)
    for name, lo, hi in BANDS:
        sub = [r for r in rows if lo <= r["true_cov"] < hi]
        ax.scatter([r["true_cov"] for r in sub], [r["meas_cov"] for r in sub], s=14, c=colors[name],
                   alpha=0.85, edgecolors="none", label=f"{name} % (n={len(sub)})")
    for t in (15, 30):
        ax.axvline(t, color="0.6", lw=0.6, ls=":")
        ax.axhline(t, color="0.6", lw=0.6, ls=":")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("true coverage (%)")
    ax.set_ylabel("B0 measured coverage (%)")
    ax.set_title("Classical verifier B0: coverage parity")
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/synthetic/v1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results/b0_synthetic.json")
    ap.add_argument("--fig", default="experiments/figures/b0_parity.png")
    ap.add_argument("--n-fallback", type=int, default=120, help="cards to generate if --data is missing")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    verifier = ClassicalVerifier()
    t0 = time.perf_counter()
    if data_dir.is_dir():
        found, how = discover_samples(data_dir, args.split)
        if args.limit:
            found = found[: args.limit]
        if not found:
            raise SystemExit(f"no samples found under {data_dir} (split={args.split})")
        source = {"data_source": f"dataset:{data_dir}", "split": args.split, "split_resolution": how}
        rows = evaluate(found, verifier)
    else:
        n = args.limit or args.n_fallback
        source = {"data_source": f"generated by coverageai.vision.simple_cards ({n} cards, seed 0); "
                                 f"dataset {data_dir} not found", "split": "n/a", "split_resolution": "n/a"}
        rows = evaluate(generated_samples(n), verifier)
    result = {**source, "verifier": "ClassicalVerifier (B0)", "spread_factor": verifier.spread_factor,
              "eval_wall_s": time.perf_counter() - t0, **summarize(rows), "cards": rows}
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    fig = Path(args.fig)
    parity_plot(rows, fig if fig.is_absolute() else ROOT / fig)
    o = result["overall"]
    print(f"{result['data_source']}\nn={o['n']}  MAE={o['mae_pp']:.2f} pp  bias={o['bias_pp']:+.2f}  "
          f"IoU={o['iou_mean']}  class agreement={o['class_agreement']:.3f}  "
          f"runtime mean/max={o['runtime_mean_s']:.2f}/{o['runtime_max_s']:.2f} s\n-> {out}")
    return result


if __name__ == "__main__":
    main()
