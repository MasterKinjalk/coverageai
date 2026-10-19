"""Tiny test-card generator (yellow card, blue disks/ellipses) with known ground truth.

Independent of the realistic renderer (WP-03); used by the classical-verifier tests and
as the fallback data source of ``experiments/eval_classical.py``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

YELLOW = np.array([235.0, 210.0, 40.0])
BLUE = np.array([30.0, 40.0, 120.0])


@dataclass
class SimCard:
    image: np.ndarray        # RGB uint8
    mask: np.ndarray         # bool GT stain mask, image frame
    card_mask: np.ndarray    # bool, card region in the image frame
    coverage_pct: float      # mask pixels / card pixels * 100
    n_stains: int
    px_per_mm: float
    diameters_px: np.ndarray


def make_card(rng: np.random.Generator, coverage_pct: float, ppm: float = 600 / 25.4,
              size_mm: tuple[float, float] = (76.0, 26.0),
              radius_mm: tuple[float, float] = (0.10, 0.35), overlap: bool = False,
              ellipse: bool = False, gap_px: int = 3, blur_px: float = 0.0,
              noise_sigma: float = 0.0, gradient: float = 0.0, perspective: bool = False,
              canvas: tuple[int, int] | None = None, max_tries: int = 200000) -> SimCard:
    """Draw stains until ``coverage_pct`` is reached (or placement saturates).

    overlap=False places stains with at least ``gap_px`` between them (count is exact);
    overlap=True lets them merge (GT mask is the union).
    blur_px: Gaussian sigma of the optical blur; noise_sigma: sensor noise in 0-255 levels;
    gradient: multiplicative illumination tilt (0.2 -> +-20 % across the card);
    perspective: warp onto a gray background canvas (default 3000x1100 for a 76x26 mm card).
    """
    W, H = int(round(size_mm[0] * ppm)), int(round(size_mm[1] * ppm))
    mask = np.zeros((H, W), np.uint8)
    target = coverage_pct / 100.0 * H * W
    area = 0
    n = 0
    diam = []
    rlo, rhi = radius_mm[0] * ppm, radius_mm[1] * ppm
    for _ in range(max_tries):
        if area >= target:
            break
        r = float(np.clip(np.exp(rng.uniform(np.log(rlo), np.log(rhi))), 1.5, None))
        ecc = rng.uniform(0.75, 1.0) if ellipse else 1.0
        a, b = r, r * ecc
        ang = rng.uniform(0, 180)
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        pad = int(np.ceil(a)) + gap_px + 1
        y0, y1 = max(int(cy) - pad, 0), min(int(cy) + pad + 1, H)
        x0, x1 = max(int(cx) - pad, 0), min(int(cx) + pad + 1, W)
        if y1 <= y0 or x1 <= x0:
            continue
        sub = mask[y0:y1, x0:x1]
        tmp = np.zeros_like(sub)
        cv2.ellipse(tmp, (int(round(cx - x0)), int(round(cy - y0))), (max(int(round(a)), 1), max(int(round(b)), 1)),
                    ang, 0, 360, 1, -1)
        if not overlap:
            grown = cv2.dilate(tmp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap_px + 1,) * 2))
            if (grown & sub).any():
                continue
            # stay inside the card so that no stain is cut by the border
            if cx - pad < 0 or cy - pad < 0 or cx + pad >= W or cy + pad >= H:
                continue
        before = int(sub.sum())
        sub |= tmp
        area += int(sub.sum()) - before
        n += 1
        diam.append(2 * np.sqrt(a * b))
    gt = mask.astype(bool)

    alpha = gt.astype(np.float32)
    if blur_px > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), blur_px)
    img = YELLOW[None, None] * (1 - alpha[..., None]) + BLUE[None, None] * alpha[..., None]
    if gradient:
        gx = np.linspace(-1, 1, W)[None, :] * gradient
        gy = np.linspace(-1, 1, H)[:, None] * gradient * 0.5
        img = img * (1 + gx + gy)[..., None]
    card_mask = np.ones((H, W), bool)

    if perspective:
        ch, cw = canvas if canvas is not None else (1100, 3000)
        sx = 0.78 * cw / W
        # quad: card corners placed with random jitter on the canvas
        cx0, cy0 = cw / 2, ch / 2
        w2, h2 = W * sx / 2, H * sx / 2
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        dst = np.float32([[cx0 - w2, cy0 - h2], [cx0 + w2, cy0 - h2], [cx0 + w2, cy0 + h2], [cx0 - w2, cy0 + h2]])
        dst += rng.uniform(-1, 1, dst.shape).astype(np.float32) * np.float32([cw * 0.015, ch * 0.08])
        Mx = cv2.getPerspectiveTransform(src, dst)
        bg = np.full((ch, cw, 3), 125.0, np.float32)
        if gradient:
            bg = bg * (1 + np.linspace(-1, 1, cw)[None, :, None] * gradient)
        warped = cv2.warpPerspective(img.astype(np.float32), Mx, (cw, ch), flags=cv2.INTER_LINEAR)
        cm = cv2.warpPerspective(np.ones((H, W), np.uint8), Mx, (cw, ch), flags=cv2.INTER_NEAREST).astype(bool)
        gt = cv2.warpPerspective(mask, Mx, (cw, ch), flags=cv2.INTER_NEAREST).astype(bool)
        img = np.where(cm[..., None], warped, bg)
        card_mask = cm
        # the scale on the canvas differs from the flat card
        ppm = ppm * sx
    if noise_sigma > 0:
        img = img + rng.normal(0, noise_sigma, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    cov = 100.0 * gt.sum() / card_mask.sum()
    return SimCard(img, gt, card_mask, float(cov), n, float(ppm), np.asarray(diam))


def save_dataset(root: Path, cards: list[SimCard], split: str = "test") -> None:
    """Write cards in a WP-03-like layout: <root>/<split>/<id>.png, <id>_mask.png, <id>.json."""
    d = Path(root) / split
    d.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(cards):
        cv2.imwrite(str(d / f"card_{i:04d}.png"), cv2.cvtColor(c.image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(d / f"card_{i:04d}_mask.png"), c.mask.astype(np.uint8) * 255)
        (d / f"card_{i:04d}.json").write_text(json.dumps(
            {"coverage_pct": c.coverage_pct, "n_stains": c.n_stains, "px_per_mm": c.px_per_mm}))
