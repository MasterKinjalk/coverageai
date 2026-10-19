"""WP-05: loader for the Simoes et al. (2025) water-sensitive-paper (WSP) dataset.

Source: Simoes, I., Sousa, A.J., Baltazar, A., Santos, F. (2025) "Spray quality
assessment on water-sensitive paper comparing AI and classical computer vision
methods", Agriculture 15(3):261.  Data: Zenodo concept record 13995949.

* v1  (record 13995950, open, CC BY 4.0): 300 synthetic WSP images (PNG, RGB) with one
  YOLOv8 *polygon* label file per image (``class x1 y1 x2 y2 ...`` normalised to [0, 1]).
* v2  (record 14010262, latest): adds 2 manually annotated real cards and 127 pre-annotated
  real squares, but its files are **restricted** (access request needed) and cannot be
  downloaded anonymously.  ``load_cards(root, "real")`` therefore returns ``[]`` unless the
  user has placed the real files under ``root`` (any ``image/`` + ``label/`` folder pair
  whose path contains ``real``).

The archives carry no dpi metadata.  Synthetic cards are 76 x 26 mm WSP renders (aspect
ratio 2.92 = 76/26), so ``px_per_mm = width_px / 76`` is *inferred*, not published.

CLI::

    uv run python -m coverageai.vision.real_data --download --summary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np
import requests

log = logging.getLogger(__name__)

ZENODO_API = "https://zenodo.org/api/records"
CONCEPT_RECORD = "13995949"
SYNTHETIC_RECORD = "13995950"   # v1: open, synthetic only
LATEST_RECORD = "14010262"      # v2: synthetic + real, files restricted
DEFAULT_DEST = "data/external/simoes2025"
LARGE_FILE_BYTES = 2 * 1024**3
CARD_WIDTH_MM = 76.0            # standard WSP card, long side (Section 3 of PLAN.md)
CARD_HEIGHT_MM = 26.0
_IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# --------------------------------------------------------------------------- data type
@dataclass
class CardSample:
    image: np.ndarray            # (H, W, 3) RGB uint8
    mask: np.ndarray             # (H, W) bool, union of droplet polygons
    n_droplets: int              # number of valid polygons
    px_per_mm: float | None      # None when unknown
    name: str
    subset: str = "synthetic"


# --------------------------------------------------------------------------- labels
def parse_yolo_polygons(text: str) -> list[np.ndarray]:
    """Parse YOLOv8 segmentation label text -> list of (N, 2) float arrays (normalised xy).

    Lines with fewer than 3 points (e.g. plain bbox rows with 5 numbers) are skipped.
    """
    polys: list[np.ndarray] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 7:          # class + at least 3 xy pairs
            continue
        vals = np.asarray(parts[1:], dtype=np.float64)
        if vals.size % 2:
            vals = vals[:-1]
        polys.append(vals.reshape(-1, 2))
    return polys


def rasterize_polygons(polys: Sequence[np.ndarray], height: int, width: int) -> np.ndarray:
    """Rasterise normalised polygons into a bool mask with ``cv2.fillPoly``."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for p in polys:
        pts = np.round(p * np.array([width, height], dtype=np.float64)).astype(np.int32)
        np.clip(pts[:, 0], 0, width - 1, out=pts[:, 0])
        np.clip(pts[:, 1], 0, height - 1, out=pts[:, 1])
        cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


# --------------------------------------------------------------------------- loading
def _find_pairs(root: Path, subset: str) -> list[tuple[Path, Path]]:
    """Find (image_dir, label_dir) pairs under ``root`` belonging to ``subset``."""
    pairs = []
    for img_dir in sorted(p for p in root.rglob("*") if p.is_dir() and p.name.lower() in ("image", "images")):
        label_dir = img_dir.with_name("label")
        if not label_dir.is_dir():
            label_dir = img_dir.with_name("labels")
        if not label_dir.is_dir():
            continue
        is_real = "real" in str(img_dir.relative_to(root)).lower()
        if (subset == "real") == is_real:
            pairs.append((img_dir, label_dir))
    return pairs


def _natural_key(p: Path):
    stem = p.stem
    return (0, int(stem), "") if stem.isdigit() else (1, 0, stem)


def load_cards(root: str | Path = DEFAULT_DEST, subset: str = "synthetic",
               limit: int | None = None) -> list[CardSample]:
    """Load annotated cards. Returns [] (with a warning) if the subset is not on disk."""
    if subset not in ("synthetic", "real"):
        raise ValueError("subset must be 'synthetic' or 'real'")
    root = Path(root)
    pairs = _find_pairs(root, subset) if root.exists() else []
    if not pairs:
        log.warning("no %s cards found under %s", subset, root)
        return []
    out: list[CardSample] = []
    for img_dir, label_dir in pairs:
        for img_path in sorted((p for p in img_dir.iterdir() if p.suffix.lower() in _IMG_EXT), key=_natural_key):
            if limit is not None and len(out) >= limit:
                return out
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                log.warning("unreadable image %s", img_path)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            lab = label_dir / (img_path.stem + ".txt")
            polys = parse_yolo_polygons(lab.read_text()) if lab.exists() else []
            # synthetic cards are 76 x 26 mm renders; real cards: unknown scale
            ppm = (w / CARD_WIDTH_MM) if subset == "synthetic" else None
            out.append(CardSample(rgb, rasterize_polygons(polys, h, w), len(polys), ppm,
                                  img_path.stem, subset))
    return out


def tiles(sample: CardSample, size: int = 512, stride: int = 512,
          drop_partial: bool = False) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield ``(row0, col0, image_tile, mask_tile)`` windows over a large card.

    Edge tiles are cropped (smaller than ``size``) unless ``drop_partial`` is set.
    The last window along each axis is always emitted so every pixel is covered.
    """
    h, w = sample.mask.shape

    def starts(n: int) -> list[int]:
        s = list(range(0, max(n - size, 0) + 1, stride))
        if not drop_partial and s[-1] + size < n:
            s.append(s[-1] + stride)
        return s

    for r in starts(h):
        for c in starts(w):
            img_t = sample.image[r:r + size, c:c + size]
            if drop_partial and img_t.shape[:2] != (size, size):
                continue
            yield r, c, img_t, sample.mask[r:r + size, c:c + size]


# --------------------------------------------------------------------------- download
def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def list_remote_files(record_id: str = SYNTHETIC_RECORD) -> list[dict]:
    """Zenodo file entries ``{key,size,md5,url,record}`` across all versions of the concept.

    The newest version (with real cards) has restricted files; those yield no entries and
    are reported by :func:`describe_versions`.
    """
    entries = []
    r = requests.get(f"{ZENODO_API}/{record_id}/versions", timeout=30)
    r.raise_for_status()
    for hit in r.json()["hits"]["hits"]:
        for f in hit.get("files", []):
            entries.append({
                "key": f["key"], "size": f["size"],
                "md5": f["checksum"].split(":", 1)[-1],
                "url": f["links"]["self"], "record": str(hit["id"]),
            })
    return entries


def describe_versions(record_id: str = SYNTHETIC_RECORD) -> list[dict]:
    r = requests.get(f"{ZENODO_API}/{record_id}/versions", timeout=30)
    r.raise_for_status()
    return [{"id": h["id"], "title": h["metadata"]["title"],
             "date": h["metadata"]["publication_date"],
             "access": h["metadata"].get("access_right"),
             "license": (h["metadata"].get("license") or {}).get("id"),
             "n_files": len(h.get("files", []))} for h in r.json()["hits"]["hits"]]


def _download_one(entry: dict, dest: Path) -> Path:
    """Stream with HTTP Range resume, verify md5."""
    target = dest / entry["key"]
    if target.exists() and target.stat().st_size == entry["size"] and _md5(target) == entry["md5"]:
        print(f"  [ok] {entry['key']} already present, md5 verified")
        return target
    part = target.with_name(target.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(entry["url"], headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 200:      # server ignored Range -> restart
            have = 0
        elif r.status_code != 206:
            r.raise_for_status()
        with open(part, "ab" if have else "wb") as f:
            done = have
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                print(f"\r  {entry['key']}: {done / 1e6:8.1f} / {entry['size'] / 1e6:.1f} MB", end="", flush=True)
    print()
    got = _md5(part)
    if got != entry["md5"]:
        part.unlink()
        raise IOError(f"md5 mismatch for {entry['key']}: {got} != {entry['md5']}")
    part.replace(target)
    return target


def download(dest: str | Path = DEFAULT_DEST, files: Sequence[str] | None = None,
             allow_large: bool = False, extract: bool = True) -> list[Path]:
    """Download (and extract) the open Zenodo files. Files > 2 GB skipped unless ``allow_large``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    versions = describe_versions()
    for v in versions:
        print(f"  record {v['id']}: '{v['title']}' ({v['date']}) access={v['access']} "
              f"license={v['license']} downloadable_files={v['n_files']}")
        if v["access"] == "restricted":
            print("    -> restricted: real annotated cards cannot be fetched anonymously "
                  "(request access on Zenodo or contact the authors).")
    entries = list_remote_files()
    paths: list[Path] = []
    for e in entries:
        print(f"  file {e['key']}: {e['size'] / 1e6:.1f} MB (record {e['record']}, md5 {e['md5']})")
        if files is not None and e["key"] not in files:
            continue
        if e["size"] > LARGE_FILE_BYTES and not allow_large:
            print(f"  [skip] {e['key']} > 2 GB (pass allow_large=True)")
            continue
        p = _download_one(e, dest)
        paths.append(p)
        marker = dest / f".extracted_{p.name}"
        if extract and p.suffix == ".zip" and not marker.exists():
            with zipfile.ZipFile(p) as z:
                z.extractall(dest)
            marker.write_text("ok")
    return paths


# --------------------------------------------------------------------------- summary
def summarize(root: str | Path = DEFAULT_DEST, subsets: Sequence[str] = ("synthetic", "real")) -> dict:
    out: dict = {"root": str(root), "subsets": {}}
    for s in subsets:
        cards = load_cards(root, s)
        if not cards:
            out["subsets"][s] = {"n_images": 0, "available": False}
            continue
        cov = np.array([100.0 * c.mask.mean() for c in cards])
        sizes = sorted({f"{c.image.shape[1]}x{c.image.shape[0]}" for c in cards})
        nd = np.array([c.n_droplets for c in cards])
        out["subsets"][s] = {
            "available": True, "n_images": len(cards), "image_sizes_wxh": sizes,
            "n_droplets_total": int(nd.sum()), "n_droplets_mean": float(nd.mean()),
            "n_droplets_min": int(nd.min()), "n_droplets_max": int(nd.max()),
            "coverage_pct_mean": float(cov.mean()), "coverage_pct_min": float(cov.min()),
            "coverage_pct_max": float(cov.max()),
            "px_per_mm_inferred": sorted({round(c.px_per_mm, 2) for c in cards if c.px_per_mm}),
        }
    return out


def print_table(summary: dict) -> None:
    hdr = f"{'subset':<10}{'n_img':>6}  {'image sizes (WxH)':<34}{'droplets':>10}{'cov% mean':>10}{'min':>7}{'max':>7}"
    print(hdr)
    print("-" * len(hdr))
    for s, d in summary["subsets"].items():
        if not d["available"]:
            print(f"{s:<10}{0:>6}  {'(not available - files restricted on Zenodo)':<34}")
            continue
        print(f"{s:<10}{d['n_images']:>6}  {', '.join(d['image_sizes_wxh']):<34}{d['n_droplets_total']:>10}"
              f"{d['coverage_pct_mean']:>10.2f}{d['coverage_pct_min']:>7.2f}{d['coverage_pct_max']:>7.2f}")


def save_example_figure(root: str | Path, path: str | Path, names: Sequence[str] = ("0", "100", "200")) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cards = {c.name: c for c in load_cards(root, "synthetic")}
    picks = [cards[n] for n in names if n in cards] or list(cards.values())[:3]
    fig, axes = plt.subplots(len(picks), 2, figsize=(11, 2.2 * len(picks) + 0.6))
    axes = np.atleast_2d(axes)
    for ax_row, c in zip(axes, picks):
        ov = c.image.copy()
        ov[c.mask] = (0.45 * ov[c.mask] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
        ax_row[0].imshow(c.image)
        ax_row[0].set_title(f"synthetic card {c.name}: {c.image.shape[1]}x{c.image.shape[0]} px, "
                            f"{c.n_droplets} droplets", fontsize=8)
        ax_row[1].imshow(ov)
        ax_row[1].set_title(f"polygon mask overlay (coverage {100 * c.mask.mean():.1f}%)", fontsize=8)
        for a in ax_row:
            a.axis("off")
    fig.suptitle("Simoes et al. 2025 synthetic WSP (CC BY 4.0, Zenodo 13995950)", fontsize=9)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--allow-large", action="store_true")
    ap.add_argument("--out", default="results/real_data_summary.json")
    ap.add_argument("--figure", default=None, help="write example overlay figure to this path")
    a = ap.parse_args(argv)
    if a.download:
        download(a.dest, allow_large=a.allow_large)
    if a.summary:
        s = summarize(a.dest)
        s["zenodo"] = {"concept_record": CONCEPT_RECORD, "versions": describe_versions(),
                       "license": "CC BY 4.0", "label_format": "YOLOv8 polygon (normalised xy)"}
        print_table(s)
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(s, indent=2))
        print(f"wrote {out}")
    if a.figure:
        print(f"wrote {save_example_figure(a.dest, a.figure)}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
