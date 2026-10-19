"""Quick random-ellipse water-sensitive-paper generator for developing and testing
the learned verifier before the realistic renderer (WP-03) lands.

Not a physical model: stains are random ellipses on a yellow card with illumination
gradient, blur, noise and JPEG artefacts. Never use for reported results.

Layout written by `make_toy_dataset` (one flat folder per split):
    <root>/<split>/card_000001_img.png, card_000001_mask.png, card_000001.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from coverageai.config import px_per_mm as _spec_ppm, spec


def generate_card(rng: np.random.Generator, target_cov_pct: float, h: int | None = None,
                  w: int | None = None, ppm: float | None = None,
                  degrade: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (RGB uint8 image, bool mask, meta). Mask is exact before degradation."""
    s = spec()
    ppm = float(ppm or _spec_ppm())
    h = h or int(round(s["card"]["height_mm"] * ppm))
    w = w or int(round(s["card"]["width_mm"] * ppm))
    mask = np.zeros((h, w), np.uint8)
    shade = np.zeros((h, w), np.uint8)
    target = target_cov_pct / 100.0
    median_px = rng.uniform(0.25, 0.6) * ppm          # median stain diameter
    sigma = rng.uniform(0.35, 0.7)
    n_stains, n_batch = 0, 100
    for _ in range(400):
        if target <= 0 or cv2.countNonZero(mask) / (h * w) >= target:
            break
        for _k in range(n_batch):
            d = float(np.clip(median_px * rng.lognormal(0.0, sigma), 2.0, 3.0 * ppm))
            ecc = rng.uniform(1.0, 1.35)
            ax = (max(int(round(d * ecc / 2)), 1), max(int(round(d / ecc / 2)), 1))
            c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            ang = float(rng.uniform(0, 180))
            cv2.ellipse(mask, c, ax, ang, 0, 360, 255, -1)
            cv2.ellipse(shade, c, ax, ang, 0, 360, int(rng.integers(120, 255)), -1)
            n_stains += 1
    hard = mask > 0
    # compose: soft alpha, stain colour with per-stain shade, rim darkening
    alpha = cv2.GaussianBlur(mask, (0, 0), 0.8).astype(np.float32)[..., None] / 255.0
    bg = np.array([235, 210, 40], np.float32) + rng.normal(0, 6, 3).astype(np.float32)
    dark = np.array([25, 35, 110], np.float32) + rng.normal(0, 6, 3).astype(np.float32)
    light = np.array([70, 90, 170], np.float32)
    sh = (shade.astype(np.float32) / 255.0)[..., None]
    stain_col = dark * sh + light * (1 - sh) * 0.6 + dark * (1 - sh) * 0.4
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)[..., None]
    stain_col = stain_col * (0.8 + 0.2 * np.clip(dist / 3.0, 0, 1))
    img = bg * (1 - alpha) + stain_col * alpha
    if degrade:
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        gx, gy = rng.uniform(-0.15, 0.15, 2)
        img = img * (1 + gx * (xx / w - 0.5) + gy * (yy / h - 0.5))[..., None]
        img = img * rng.uniform(0.85, 1.1, 3).astype(np.float32)
        sig = rng.uniform(0.0, 1.2)
        if sig > 0.2:
            img = cv2.GaussianBlur(img, (0, 0), sig)
        img = img + rng.normal(0, rng.uniform(0, 4), img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if degrade and rng.random() < 0.5:
        ok, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 96))])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
    meta = {"coverage_pct": 100.0 * float(hard.mean()), "n_stains": n_stains, "px_per_mm": ppm,
            "target_coverage_pct": float(target_cov_pct)}
    return np.ascontiguousarray(img), hard, meta


def make_toy_dataset(root: str | Path, n_train: int = 200, n_val: int = 30, n_test: int = 30,
                     seed: int = 0, h: int | None = None, w: int | None = None,
                     max_cov: float = 70.0) -> Path:
    """Write train/val/test folders of *_img.png, *_mask.png, *.json; coverage uniform in [0, max_cov]."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        d = root / split
        d.mkdir(parents=True, exist_ok=True)
        covs = rng.uniform(0.0, max_cov, n)
        for i, cov in enumerate(covs):
            img, mask, meta = generate_card(rng, float(cov), h=h, w=w)
            stem = f"card_{i:06d}"
            cv2.imwrite(str(d / f"{stem}_img.png"), img[..., ::-1])
            cv2.imwrite(str(d / f"{stem}_mask.png"), mask.astype(np.uint8) * 255)
            (d / f"{stem}.json").write_text(json.dumps(meta))
    return root


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--train", type=int, default=200)
    ap.add_argument("--val", type=int, default=30)
    ap.add_argument("--test", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    make_toy_dataset(a.out, a.train, a.val, a.test, a.seed)
    print("wrote", a.out)
