"""Synthetic water-sensitive-paper (WSP) card renderer and dataset generator (WP-03).

Turns the droplets deposited on a card (``Outcome.droplets``) into an RGB image
plus an exact boolean ground-truth stain mask.

Geometry
    Every detectable droplet (diameter >= ``physics.min_detectable_droplet_um``)
    leaves an elliptical stain of equivalent diameter ``s = spread_factor * d``.
    The ellipse is area preserving (a * b = s^2 / 4 for any axis ratio), so the
    union of stains follows the Boolean model, ``C = 1 - exp(-sum(pi s^2 / 4) / A)``,
    that the simulator uses analytically.
Rasterization
    A small vectorized rasterizer replaces ``cv2.ellipse``, whose polygon
    approximation misses 10 to 15 percent of the area for stains 2 to 3 px in
    radius. The ground-truth mask is the set of pixels whose center lies inside
    at least one ellipse (unbiased in expectation for any stain size); the image
    uses anti-aliased edges and a darker rim from the same geometry.
Degradations
    ``degrade=True`` places the card on a neutral background, applies a
    perspective warp (same matrix applied to the mask), illumination gradient,
    vignetting, blur, sensor noise and JPEG compression. Two profiles:
    ``scanner`` (flat, clean) and ``phone`` (warp, strong shading).

Dataset CLI::

    uv run python -m coverageai.vision.render --out data/synthetic/v1 --n 1200 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from coverageai.config import ROOT, spec
from coverageai.contracts import DropletSample, Outcome

PROFILES = ("scanner", "phone")
BANDS = ((0.0, 15.0), (15.0, 30.0), (30.0, 50.0), (50.0, 80.0))
DATASET_CLASSES = ("F", "M", "C", "VC", "XC")
# Droplets in synthetic samples are placed on the card plus this margin so that
# stains straddling the card edge are represented (removes the edge effect from
# the Boolean-model comparison).
SYNTH_MARGIN_MM = 1.5

_PALETTE_DARK = np.array([25.0, 45.0, 130.0], np.float32)      # dark blue
_PALETTE_TEAL = np.array([15.0, 115.0, 150.0], np.float32)     # blue-green
_PALETTE_LIGHT = np.array([95.0, 175.0, 165.0], np.float32)    # light greenish (small stains)
_PAPER_YELLOW = np.array([244.0, 216.0, 64.0], np.float32)


# ------------------------------------------------------------------ droplets

def _lognormal_sigma(relative_span: float) -> float:
    """sigma of ln d for a log-normal volume distribution with the given span.

    span = (Dv0.9 - Dv0.1) / Dv0.5 = 2 sinh(z90 sigma), z90 = 1.2816.
    """
    return math.asinh(relative_span / 2.0) / 1.2815515655446004


def _quadrature_moments(vmd_um: float, relative_span: float, spread_factor: float,
                        min_d_um: float) -> tuple[float, float]:
    """(E[droplet volume] in uL, E[detectable stain area] in mm^2) per droplet."""
    sig = _lognormal_sigma(relative_span)
    mu = math.log(vmd_um) - 3.0 * sig * sig       # number-weighted median (Hatch-Choate)
    z = np.linspace(-8.0, 8.0, 3201)
    w = np.exp(-0.5 * z * z)
    w /= w.sum()
    d = np.exp(mu + sig * z)                      # um
    vol_ul = float((w * (math.pi / 6.0) * d ** 3).sum()) * 1e-9
    area_mm2 = float((w * (d >= min_d_um) * (math.pi / 4.0) * (spread_factor * d * 1e-3) ** 2).sum())
    return vol_ul, area_mm2


def deposition_for_coverage(coverage_pct: float, vmd_um: float, relative_span: float = 1.2,
                            spread_factor: float = 1.9, min_d_um: float | None = None) -> float:
    """Deposition (uL/cm^2) whose expected Boolean-model coverage equals coverage_pct."""
    min_d = spec()["physics"]["min_detectable_droplet_um"] if min_d_um is None else min_d_um
    rho = -math.log(1.0 - min(coverage_pct, 99.9) / 100.0)
    vol_ul, area_mm2 = _quadrature_moments(vmd_um, relative_span, spread_factor, min_d)
    n_per_mm2 = rho / area_mm2
    return n_per_mm2 * vol_ul * 100.0


def synthetic_droplets(vmd_um: float, deposition_ul_cm2: float, rng: np.random.Generator,
                       relative_span: float = 1.2, spread_factor: float | None = None,
                       margin_mm: float = SYNTH_MARGIN_MM) -> DropletSample:
    """Standalone droplet sample: log-normal volume distribution, Poisson count,
    uniform positions on the card (plus ``margin_mm`` around it).

    ``vmd_um`` is the volume median diameter, ``relative_span`` the
    (Dv0.9 - Dv0.1) / Dv0.5 span. Count is Poisson with mean deposited volume
    divided by the mean droplet volume. All droplets are returned, including
    those below the detection limit; the renderer ignores those.
    """
    s = spec()["card"]
    cw, ch = float(s["width_mm"]), float(s["height_mm"])
    if spread_factor is None:
        lo, hi = spec()["physics"]["spread_factor"]
        spread_factor = float(rng.uniform(lo, hi))
    sig = _lognormal_sigma(relative_span)
    mu = math.log(vmd_um) - 3.0 * sig * sig
    vol_ul, _ = _quadrature_moments(vmd_um, relative_span, spread_factor, 0.0)
    area_cm2 = (cw + 2 * margin_mm) * (ch + 2 * margin_mm) / 100.0
    n = int(rng.poisson(max(deposition_ul_cm2, 0.0) * area_cm2 / vol_ul))
    d = np.exp(mu + sig * rng.standard_normal(n))
    x = rng.uniform(-margin_mm, cw + margin_mm, n)
    y = rng.uniform(-margin_mm, ch + margin_mm, n)
    return DropletSample(diameters_um=d, x_mm=x, y_mm=y, spread_factor=float(spread_factor))


def analytic_coverage_pct(sample: DropletSample, margin_mm: float = 0.0,
                          min_d_um: float | None = None) -> float:
    """Boolean-model coverage 1 - exp(-sum(pi s^2 / 4) / A) in percent.

    Uses droplets whose center lies within ``margin_mm`` of the card; A is the
    matching area. Use margin_mm=SYNTH_MARGIN_MM for synthetic_droplets samples
    and 0 for samples placed on the card only.
    """
    s = spec()
    cw, ch = float(s["card"]["width_mm"]), float(s["card"]["height_mm"])
    min_d = s["physics"]["min_detectable_droplet_um"] if min_d_um is None else min_d_um
    d = np.asarray(sample.diameters_um, float)
    x, y = np.asarray(sample.x_mm, float), np.asarray(sample.y_mm, float)
    inside = ((x >= -margin_mm) & (x <= cw + margin_mm) & (y >= -margin_mm) & (y <= ch + margin_mm)
              & (d >= min_d))
    area_stain = (math.pi / 4.0) * (sample.spread_factor * d[inside] * 1e-3) ** 2   # mm^2
    area = (cw + 2 * margin_mm) * (ch + 2 * margin_mm)
    return 100.0 * (1.0 - math.exp(-area_stain.sum() / area))


def dv_quantiles(diameters_um: np.ndarray, qs=(0.1, 0.5, 0.9)) -> list[float | None]:
    """Volume-weighted diameter quantiles (Dv0.1, Dv0.5, Dv0.9) of droplets."""
    d = np.sort(np.asarray(diameters_um, float))
    if d.size == 0:
        return [None for _ in qs]
    cum = np.cumsum(d ** 3)
    cum /= cum[-1]
    return [float(np.interp(q, cum, d)) for q in qs]


def class_of_vmd(vmd_um: float) -> str:
    """ASABE class whose bounds contain the VMD (spec droplets.vmd_bounds_um)."""
    for name, (lo, hi) in spec()["droplets"]["vmd_bounds_um"].items():
        if lo <= vmd_um < hi:
            return name
    return "UC" if vmd_um >= 650 else "VF"


# ------------------------------------------------------------------ rasterizer

def _rasterize(height: int, width: int, cx, cy, a, b, theta, col, rim_f, rim_w,
               chunk_entries: int = 150_000):
    """Rasterize ellipses (pixel-center coordinates, semi-axes in px).

    Returns (alpha, color, hit): union opacity (H, W) in [0, 1], alpha-weighted mean
    stain color (H, W, 3), and the exact mask of pixels whose center is inside at
    least one ellipse.
    """
    n = len(cx)
    if n == 0:
        return (np.zeros((height, width), np.float32), np.zeros((height, width, 3), np.float32),
                np.zeros((height, width), bool))
    h = np.ceil(np.maximum(a, b) + 1.0).astype(np.int64)
    keep = (cx > -h) & (cx < width + h) & (cy > -h) & (cy < height + h)
    cx, cy, a, b, theta, col, rim_f, rim_w, h = (v[keep] for v in
                                                 (cx, cy, a, b, theta, col, rim_f, rim_w, h))
    if len(cx) == 0:
        return (np.zeros((height, width), np.float32), np.zeros((height, width, 3), np.float32),
                np.zeros((height, width), bool))
    pad = 2 * int(h.max())
    wp, hp = width + 2 * pad, height + 2 * pad
    idx_all, a_all, w_all, hit_all = [], [], [], []
    cos_t, sin_t = np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)
    rcx, rcy = np.rint(cx), np.rint(cy)
    col = np.ascontiguousarray(col, np.float32)
    f32 = np.float32
    # Stains with b < 2.5 px and a < 3 px (no rim) use an area-equivalent circle: the
    # orientation of such a stain is below the pixel grid, and this path is ~4x cheaper.
    small = (b < 2.5) & (a < 3.0)
    base_pix = (rcy + pad).astype(np.int64) * wp + (rcx + pad).astype(np.int64)
    for is_small in (True, False):
        for hv in np.unique(h[small == is_small]):
            members = np.nonzero((h == hv) & (small == is_small))[0]
            k = 2 * int(hv) + 1
            step = max(1, chunk_entries // (k * k))
            ox = np.arange(-hv, hv + 1, dtype=f32)
            off = ((np.arange(k) - hv)[:, None] * wp + (np.arange(k) - hv)[None, :]).ravel()
            for lo in range(0, len(members), step):
                m = members[lo:lo + step]
                u = (rcx[m] - cx[m]).astype(f32)[:, None, None] + ox[None, None, :]
                v = (rcy[m] - cy[m]).astype(f32)[:, None, None] + ox[None, :, None]
                if is_small:
                    rr = np.sqrt(a[m] * b[m]).astype(f32)[:, None, None]
                    sd = np.sqrt(u * u + v * v)
                    sd -= rr
                else:
                    c, s = cos_t[m][:, None, None], sin_t[m][:, None, None]
                    ur = u * c + v * s
                    vr = v * c - u * s
                    aa = a[m].astype(f32)[:, None, None]
                    bb = b[m].astype(f32)[:, None, None]
                    gx, gy = ur / aa, vr / bb
                    f = gx * gx + gy * gy - 1.0
                    g = np.sqrt((gx / aa) ** 2 + (gy / bb) ** 2)
                    sd = f / (2.0 * np.maximum(g, 1e-3))          # signed distance (px), > 0 outside
                alpha = np.clip(0.5 - sd, 0.0, 1.0)
                flat = np.flatnonzero(alpha > 0.0)
                if flat.size == 0:
                    continue
                kk = k * k
                ni = flat // kk
                sidx = m[ni]
                pix = base_pix[sidx] + off[flat - ni * kk]
                al, sdv = alpha.ravel()[flat], sd.ravel()[flat]
                if is_small:
                    wgt = al
                else:
                    t = np.clip(1.0 + sdv / rim_w[sidx], 0.0, 1.0)     # rim weight (1 at edge)
                    wgt = al * (1.0 - t * (1.0 - rim_f[sidx]))
                idx_all.append(pix)
                a_all.append(al)
                w_all.append(wgt[:, None] * col[sidx])
                hit_all.append(pix[sdv <= 0.0])
    size = hp * wp
    if not idx_all:
        return (np.zeros((height, width), np.float32), np.zeros((height, width, 3), np.float32),
                np.zeros((height, width), bool))
    idx = np.concatenate(idx_all)
    al = np.concatenate(a_all)
    wc = np.concatenate(w_all)
    s_op = np.bincount(idx, weights=-np.log1p(-np.minimum(al, 0.9999)), minlength=size)
    w_sum = np.bincount(idx, weights=al, minlength=size)
    col_sum = np.stack([np.bincount(idx, weights=wc[:, k], minlength=size) for k in range(3)], -1)
    hits = np.bincount(np.concatenate(hit_all), minlength=size) if hit_all else np.zeros(size)
    sl = (slice(pad, pad + height), slice(pad, pad + width))
    s_op = s_op.reshape(hp, wp)[sl]
    w_sum = w_sum.reshape(hp, wp)[sl]
    col_sum = col_sum.reshape(hp, wp, 3)[sl]
    hit = hits.reshape(hp, wp)[sl] > 0
    alpha_u = (1.0 - np.exp(-s_op)).astype(np.float32)
    color = (col_sum / np.maximum(w_sum, 1e-9)[..., None]).astype(np.float32)
    return alpha_u, color, hit


# ------------------------------------------------------------------ renderer

@dataclass
class RenderResult:
    image: np.ndarray                 # RGB uint8, image frame
    mask: np.ndarray                  # bool, image frame (warped identically to the image)
    card_image: np.ndarray            # RGB uint8, card only, no degradation
    card_mask: np.ndarray             # bool, card only
    profile: str | None
    px_per_mm: float                  # effective scale in the returned image
    nominal_px_per_mm: float
    params: dict[str, Any] = field(default_factory=dict)


class WSPRenderer:
    """Satisfies ``contracts.Renderer``. ``px_per_mm`` and ``last`` expose the scale
    and parameters of the most recent render."""

    def __init__(self) -> None:
        s = spec()
        self.nominal_px_per_mm = float(s["card"]["dpi"]) / 25.4
        self.card_w_mm = float(s["card"]["width_mm"])
        self.card_h_mm = float(s["card"]["height_mm"])
        self.width = int(round(self.card_w_mm * self.nominal_px_per_mm))
        self.height = int(round(self.card_h_mm * self.nominal_px_per_mm))
        self.min_d_um = float(s["physics"]["min_detectable_droplet_um"])
        self.px_per_mm = self.nominal_px_per_mm
        self.last: RenderResult | None = None

    # -- public API ---------------------------------------------------------
    def render(self, outcome: Outcome, rng: np.random.Generator, degrade: bool = True,
               profile: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        res = self.render_full(outcome, rng, degrade=degrade, profile=profile)
        return res.image, res.mask

    def render_full(self, outcome: Outcome, rng: np.random.Generator, degrade: bool = True,
                    profile: str | None = None) -> RenderResult:
        if outcome.droplets is None:
            raise ValueError("Outcome.droplets is None; simulate with sample_droplets=True")
        card_f, card_mask = self._render_card(outcome.droplets, rng)
        card_u8 = np.clip(card_f + 0.5, 0, 255).astype(np.uint8)
        if not degrade:
            res = RenderResult(card_u8, card_mask, card_u8, card_mask, None,
                               self.nominal_px_per_mm, self.nominal_px_per_mm, {})
        else:
            if profile is None:
                profile = PROFILES[int(rng.integers(len(PROFILES)))]
            if profile not in PROFILES:
                raise ValueError(f"unknown profile {profile!r}; choose from {PROFILES}")
            img, mask, scale, params = self._degrade(card_f, card_mask, rng, profile)
            res = RenderResult(img, mask, card_u8, card_mask, profile,
                               self.nominal_px_per_mm * scale, self.nominal_px_per_mm, params)
        self.px_per_mm = res.px_per_mm
        self.last = res
        return res

    # -- card ---------------------------------------------------------------
    def _render_card(self, dr: DropletSample, rng: np.random.Generator):
        ppm = self.nominal_px_per_mm
        d = np.asarray(dr.diameters_um, float)
        det = d >= self.min_d_um
        d = d[det]
        x_px = np.asarray(dr.x_mm, float)[det] * ppm - 0.5
        y_px = np.asarray(dr.y_mm, float)[det] * ppm - 0.5
        n = len(d)
        s_px = dr.spread_factor * d * 1e-3 * ppm                  # equivalent stain diameter
        r = 0.5 * s_px
        q = rng.uniform(0.85, 1.0, n)                             # axis ratio
        a, b = r / np.sqrt(q), r * np.sqrt(q)                     # area preserving: a*b = r^2
        theta = rng.uniform(0.0, np.pi, n)
        # colors: dark blue -> blue-green, small stains lighter and greener
        tcol = rng.beta(2.0, 3.0, n).astype(np.float32)
        col = _PALETTE_DARK[None] * (1 - tcol[:, None]) + _PALETTE_TEAL[None] * tcol[:, None]
        w_small = np.clip((9.0 - s_px) / 9.0, 0.0, 0.7).astype(np.float32)
        col = col * (1 - w_small[:, None]) + _PALETTE_LIGHT[None] * w_small[:, None]
        col *= rng.uniform(0.9, 1.1, (n, 1)).astype(np.float32)
        rimmed = b >= 2.5
        rim_f = np.where(rimmed, rng.uniform(0.55, 0.72, n), 1.0).astype(np.float32)
        rim_w = np.clip(0.14 * b, 0.6, 3.0).astype(np.float32)
        alpha, color, hit = _rasterize(self.height, self.width, x_px, y_px, a, b, theta,
                                       col, rim_f, rim_w)
        paper = self._paper(rng)
        img = (_jittered_paper(rng)[None, None] * (1 - alpha)[..., None]
               + alpha[..., None] * color) * paper[..., None]
        return img.astype(np.float32), hit

    def _paper(self, rng: np.random.Generator) -> np.ndarray:
        h, w = self.height, self.width
        low = rng.standard_normal((h // 40 + 2, w // 40 + 2), dtype=np.float32)
        low = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC) * 0.018
        grain = rng.standard_normal((h, w), dtype=np.float32) * 0.011
        return 1.0 + low + grain

    # -- degradations ---------------------------------------------------------
    def _degrade(self, card_f: np.ndarray, card_mask: np.ndarray, rng: np.random.Generator,
                 profile: str):
        h, w = self.height, self.width
        phone = profile == "phone"
        if phone:
            margin = int(rng.integers(130, 191))
            rot = rng.uniform(-3.0, 3.0)
            scale = rng.uniform(0.85, 1.05)
            jit = 0.008
            grad_amp, vig = rng.uniform(0.05, 0.25), rng.uniform(0.05, 0.30)
            blur = rng.uniform(0.3, 1.5)
            noise = rng.uniform(1.5, 6.0)
            jpeg_q = int(rng.integers(60, 91))
            bg_gray = rng.uniform(70, 190)
            cast = rng.normal(1.0, 0.03, 3).astype(np.float32)
        else:
            margin = int(rng.integers(24, 49))
            rot = rng.uniform(-0.4, 0.4)
            scale = rng.uniform(0.99, 1.01)
            jit = 0.0
            grad_amp, vig = rng.uniform(0.0, 0.06), rng.uniform(0.0, 0.05)
            blur = rng.uniform(0.0, 0.6)
            noise = rng.uniform(0.5, 2.5)
            jpeg_q = int(rng.integers(85, 96))
            bg_gray = rng.uniform(215, 245)
            cast = np.ones(3, np.float32)
        fw, fh = w + 2 * margin, h + 2 * margin
        canvas = np.empty((fh, fw, 3), np.float32)
        canvas[:] = bg_gray * cast
        canvas[margin:margin + h, margin:margin + w] = card_f
        mcanvas = np.zeros((fh, fw), np.uint8)
        mcanvas[margin:margin + h, margin:margin + w] = card_mask.astype(np.uint8) * 255

        src = np.array([[margin, margin], [margin + w, margin], [margin + w, margin + h],
                        [margin, margin + h]], np.float32)
        ctr = np.array([fw / 2.0, fh / 2.0], np.float32)
        ang = math.radians(rot)
        rmat = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]], np.float32)
        for _ in range(8):
            dxy = np.clip(rng.normal(0.0, jit * w, (4, 2)), -2.5 * jit * w, 2.5 * jit * w).astype(np.float32)
            dst = ((src - ctr) * scale) @ rmat.T + ctr + dxy
            if (dst >= 2).all() and (dst[:, 0] <= fw - 2).all() and (dst[:, 1] <= fh - 2).all():
                break
        else:
            dst = src.copy()
            scale = 1.0
            dxy = np.zeros((4, 2), np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        if phone or rot != 0.0:
            canvas = cv2.warpPerspective(canvas, M, (fw, fh), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE)
            mwarp = cv2.warpPerspective(mcanvas, M, (fw, fh), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask = mwarp > 127
        else:
            mask = mcanvas > 127
        # effective scale at the card center from the local Jacobian of the warp
        c0 = np.array([[[fw / 2.0, fh / 2.0]]], np.float32)
        p0 = cv2.perspectiveTransform(c0, M)[0, 0]
        px = cv2.perspectiveTransform(c0 + np.array([[[1.0, 0.0]]], np.float32), M)[0, 0] - p0
        py = cv2.perspectiveTransform(c0 + np.array([[[0.0, 1.0]]], np.float32), M)[0, 0] - p0
        eff_scale = float(math.sqrt(abs(px[0] * py[1] - px[1] * py[0])))

        # illumination gradient + vignetting
        xs = np.linspace(-0.5, 0.5, fw, dtype=np.float32)[None, :]
        ys = np.linspace(-0.5, 0.5, fh, dtype=np.float32)[:, None]
        phi = rng.uniform(0, 2 * np.pi)
        gain = 1.0 + grad_amp * (math.cos(phi) * xs + math.sin(phi) * ys)
        gain = gain * (1.0 - vig * ((xs * 2) ** 2 + (ys * 2) ** 2) / 2.0)
        canvas *= gain[..., None]
        if blur > 0.05:
            canvas = cv2.GaussianBlur(canvas, (0, 0), blur)
        canvas += rng.standard_normal(canvas.shape, dtype=np.float32) * np.float32(noise)
        bgr = cv2.cvtColor(np.clip(canvas + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
        img = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        params = {"profile": profile, "margin_px": margin, "rotation_deg": float(rot),
                  "scale": float(eff_scale), "gradient_amp": float(grad_amp),
                  "vignette": float(vig), "blur_sigma_px": float(blur),
                  "noise_sigma": float(noise), "jpeg_quality": jpeg_q,
                  "corner_jitter_px": float(np.abs(dxy).max())}
        return img, mask, eff_scale, params


def _jittered_paper(rng: np.random.Generator) -> np.ndarray:
    return _PAPER_YELLOW + rng.normal(0.0, [5.0, 5.0, 6.0]).astype(np.float32)


# ------------------------------------------------------------------ dataset generation

def _cell_list(classes=DATASET_CLASSES, bands=BANDS) -> list[tuple[int, str]]:
    return [(bi, c) for bi in range(len(bands)) for c in classes]


def _plan_synthetic(seed: int, i: int, cell: tuple[int, str]) -> dict[str, Any]:
    rng = np.random.default_rng([seed, i, 1])
    bi, dc = cell
    lo, hi = BANDS[bi]
    target = float(rng.uniform(max(lo, 0.3), hi))
    blo, bhi = spec()["droplets"]["vmd_bounds_um"][dc]
    vmd = float(rng.uniform(blo + 0.05 * (bhi - blo), bhi - 0.05 * (bhi - blo)))
    span = float(rng.uniform(0.9, 1.5))
    sf = float(rng.uniform(*spec()["physics"]["spread_factor"]))
    dep = deposition_for_coverage(target, vmd, span, sf)
    return dict(target_pct=target, vmd_um=vmd, span=span, spread_factor=sf, deposition_ul_cm2=dep,
                droplet_class=dc, band=bi)


def _make_card(job: dict[str, Any]) -> dict[str, Any]:
    """Render and save one card; returns its metadata record."""
    out = Path(job["out"])
    i, seed = job["i"], job["seed"]
    rng = np.random.default_rng([seed, i, 2])
    renderer = WSPRenderer()
    cs = spec()["card"]
    margin = SYNTH_MARGIN_MM
    if job.get("outcome") is not None:
        outcome: Outcome = job["outcome"]
        margin = 0.0
        plan = job["plan"]
    else:
        plan = job["plan"]
        ds = synthetic_droplets(plan["vmd_um"], plan["deposition_ul_cm2"], rng, plan["span"],
                                plan["spread_factor"])
        area_cm2 = cs["width_mm"] * cs["height_mm"] / 100.0
        xin = (ds.x_mm >= 0) & (ds.x_mm <= cs["width_mm"]) & (ds.y_mm >= 0) & (ds.y_mm <= cs["height_mm"])
        det = ds.diameters_um >= spec()["physics"]["min_detectable_droplet_um"]
        outcome = Outcome(coverage_pct=analytic_coverage_pct(ds, margin), deposition_ul_cm2=plan["deposition_ul_cm2"],
                          drift_frac=0.0, displacement_mm=0.0,
                          stains_per_cm2=float((xin & det).sum() / area_cm2), droplets=ds)
    ds = outcome.droplets
    t0 = time.perf_counter()
    res = renderer.render_full(outcome, rng, degrade=True, profile=job.get("profile"))
    render_s = time.perf_counter() - t0
    cid = f"{i:05d}"
    for sub in ("images", "masks", "meta") + (("cards",) if job["crop_card"] else ()):
        (out / sub).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "images" / f"{cid}.png"), cv2.cvtColor(res.image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "masks" / f"{cid}.png"), res.mask.astype(np.uint8) * 255)
    if job["crop_card"]:
        cv2.imwrite(str(out / "cards" / f"{cid}_card.png"), cv2.cvtColor(res.card_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "cards" / f"{cid}_card_mask.png"), res.card_mask.astype(np.uint8) * 255)

    det = np.asarray(ds.diameters_um) >= spec()["physics"]["min_detectable_droplet_um"]
    inside = ((ds.x_mm >= 0) & (ds.x_mm <= cs["width_mm"]) & (ds.y_mm >= 0) & (ds.y_mm <= cs["height_mm"]))
    dv = dv_quantiles(np.asarray(ds.diameters_um)[det & inside])
    n_comp = int(cv2.connectedComponents(res.card_mask.astype(np.uint8), connectivity=8)[0] - 1)
    meta = {
        "id": cid, "index": i, "seed": seed, "profile": res.profile,
        "px_per_mm": res.px_per_mm, "nominal_px_per_mm": res.nominal_px_per_mm,
        "image_shape": list(res.image.shape[:2]), "card_shape": list(res.card_image.shape[:2]),
        "true_coverage_pct": 100.0 * float(res.card_mask.mean()),
        "analytic_coverage_pct": float(outcome.coverage_pct),
        "stain_count": int((det & inside).sum()), "stain_components": n_comp,
        "stains_per_cm2": float(outcome.stains_per_cm2),
        "dv01_um": dv[0], "dv05_um": dv[1], "dv09_um": dv[2],
        "deposition_ul_cm2": float(outcome.deposition_ul_cm2),
        "vmd_um": plan.get("vmd_um"), "droplet_class": plan.get("droplet_class"),
        "coverage_band": plan.get("band"), "target_coverage_pct": plan.get("target_pct"),
        "spread_factor": float(ds.spread_factor), "source": job.get("source", "synthetic"),
        "render_s": render_s, "degradation": res.params,
    }
    (out / "meta" / f"{cid}.json").write_text(json.dumps(meta), encoding="utf8")
    return meta


def _simulator_source():
    """Return (simulate_fn, name) using SpraySimulator if importable, else None."""
    try:
        from coverageai.simulator import SpraySimulator  # type: ignore
    except Exception:
        return None
    try:
        sim = SpraySimulator()
    except Exception:
        return None
    from coverageai.contracts import Conditions
    from coverageai.mocks import mock_candidates
    try:
        from coverageai.catalog import candidates as cat_candidates  # type: ignore
        cands = list(cat_candidates())
    except Exception:
        cands = mock_candidates()
    rg = spec()["ranges"]

    def draw(rng: np.random.Generator) -> Outcome | None:
        x = Conditions(*(float(rng.uniform(*rg[k])) for k in (
            "wind_speed_ms", "wind_dir_deg", "heading_deg", "temp_c", "rh_pct",
            "travel_speed_kmh", "boom_height_m")))
        u = cands[int(rng.integers(len(cands)))]
        try:
            out = sim.simulate(x, u, rng, "none", sample_droplets=True)
        except Exception:
            return None
        return out if out.droplets is not None else None

    return draw


def _stratified_splits(metas_cells: dict[int, tuple[int, str]], seed: int,
                       fractions=(0.70, 0.15, 0.15)) -> dict[str, list[int]]:
    """70/15/15 split by index, stratified within (band, class) cells, fixed seed."""
    rng = np.random.default_rng([seed, 99])
    by_cell: dict[tuple[int, str], list[int]] = {}
    for i, c in metas_cells.items():
        by_cell.setdefault(c, []).append(i)
    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for cell in sorted(by_cell):
        idx = np.array(by_cell[cell])
        rng.shuffle(idx)
        n_val = int(round(fractions[1] * len(idx)))
        n_test = int(round(fractions[2] * len(idx)))
        out["val"] += idx[:n_val].tolist()
        out["test"] += idx[n_val:n_val + n_test].tolist()
        out["train"] += idx[n_val + n_test:].tolist()
    return {k: sorted(v) for k, v in out.items()}


def generate_dataset(out: Path, n: int, seed: int = 0, crop_card: bool = False,
                     workers: int | None = None, source: str = "auto") -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cells = _cell_list()
    jobs = []
    sim_draw = None if source == "synthetic" else _simulator_source()
    if source == "simulator" and sim_draw is None:
        raise RuntimeError("source=simulator requested but coverageai.simulator.SpraySimulator is not importable")
    cell_of = {i: cells[i % len(cells)] for i in range(n)}
    sim_outcomes: dict[int, tuple[Outcome, dict]] = {}
    if sim_draw is not None:
        # rejection-sample simulator outcomes into the (band, class) cells; fill gaps synthetically
        rng = np.random.default_rng([seed, 7])
        quota = {c: [i for i in range(n) if cell_of[i] == c] for c in cells}
        budget = 30 * n
        while budget > 0 and any(quota.values()):
            budget -= 1
            oc = sim_draw(rng)
            if oc is None:
                continue
            d = np.asarray(oc.droplets.diameters_um)
            d = d[d >= spec()["physics"]["min_detectable_droplet_um"]]
            if len(d) == 0:
                continue
            cov = oc.coverage_pct
            bi = next((j for j, (lo, hi) in enumerate(BANDS) if lo <= cov < hi), None)
            vmd = dv_quantiles(d)[1]
            cell = (bi, class_of_vmd(vmd)) if bi is not None else None
            if cell in quota and quota[cell]:
                i = quota[cell].pop()
                sim_outcomes[i] = (oc, dict(vmd_um=vmd, droplet_class=cell[1], band=bi, target_pct=cov))
    for i in range(n):
        job = dict(out=str(out), i=i, seed=seed, crop_card=crop_card)
        if i in sim_outcomes:
            job.update(outcome=sim_outcomes[i][0], plan=sim_outcomes[i][1], source="simulator")
        else:
            job.update(plan=_plan_synthetic(seed, i, cell_of[i]), source="synthetic")
        jobs.append(job)
    t0 = time.perf_counter()
    if workers is None:
        import os
        workers = max(1, (os.cpu_count() or 2) - 1)
    if workers <= 1 or n < 4:
        metas = [_make_card(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            metas = list(ex.map(_make_card, jobs, chunksize=4))
    elapsed = time.perf_counter() - t0
    splits = _stratified_splits({i: cell_of[i] for i in range(n)}, seed)
    split_of = {i: s for s, ids in splits.items() for i in ids}
    for m in metas:
        m["split"] = split_of[m["index"]]
    (out / "splits.json").write_text(json.dumps(splits), encoding="utf8")
    with open(out / "manifest.jsonl", "w", encoding="utf8") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")
    cols = ["id", "split", "profile", "droplet_class", "coverage_band", "true_coverage_pct",
            "analytic_coverage_pct", "stain_count", "dv01_um", "dv05_um", "dv09_um", "px_per_mm"]
    with open(out / "manifest.csv", "w", newline="", encoding="utf8") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for m in metas:
            wr.writerow([m[c] for c in cols])
    cov = np.array([m["true_coverage_pct"] for m in metas])
    ana = np.array([m["analytic_coverage_pct"] for m in metas])
    summary = {
        "n": n, "seed": seed, "seconds": elapsed, "workers": workers,
        "splits": {k: len(v) for k, v in splits.items()},
        "mean_abs_diff_true_vs_analytic_pp": float(np.abs(cov - ana).mean()),
        "mean_signed_diff_pp": float((cov - ana).mean()),
        "coverage_by_band_counts": [int(((cov >= lo) & (cov < hi)).sum()) for lo, hi in BANDS],
        "profiles": {p: sum(m["profile"] == p for m in metas) for p in PROFILES},
        "sources": {s: sum(m["source"] == s for m in metas) for s in ("synthetic", "simulator")},
        "mean_render_s": float(np.mean([m["render_s"] for m in metas])),
        "max_render_s": float(np.max([m["render_s"] for m in metas])),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf8")
    return summary


# ------------------------------------------------------------------ example figure

def make_examples_figure(path: Path, seed: int = 3) -> list[dict[str, Any]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan = [(5, "F", "scanner"), (22, "M", "phone"), (38, "C", "scanner"),
            (47, "VC", "phone"), (63, "XC", "scanner"), (76, "M", "phone")]
    rnd = WSPRenderer()
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.2))
    info = []
    for k, ((tgt, dc, prof), ax) in enumerate(zip(plan, axes.ravel())):
        rng = np.random.default_rng([seed, k])
        vmd = float(np.mean(spec()["droplets"]["vmd_bounds_um"][dc]))
        sf = float(rng.uniform(*spec()["physics"]["spread_factor"]))
        dep = deposition_for_coverage(tgt, vmd, 1.2, sf)
        ds = synthetic_droplets(vmd, dep, rng, 1.2, sf)
        oc = Outcome(analytic_coverage_pct(ds, SYNTH_MARGIN_MM), dep, 0.0, 0.0, 0.0, ds)
        res = rnd.render_full(oc, rng, degrade=True, profile=prof)
        cov = float(100.0 * res.card_mask.mean())
        small = cv2.resize(res.image, (900, int(900 * res.image.shape[0] / res.image.shape[1])),
                           interpolation=cv2.INTER_AREA)
        ax.imshow(small)
        ax.set_title(f"true coverage {cov:.1f}%  |  class {dc} (VMD {vmd:.0f} um)  |  {prof}", fontsize=9)
        ax.axis("off")
        info.append(dict(coverage=cov, droplet_class=dc, profile=prof))
    fig.suptitle("Synthetic water-sensitive-paper cards (renderer v1)", fontsize=11)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return info


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate the synthetic WSP dataset.")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "synthetic" / "v1")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop-card", action="store_true",
                    help="also save the card-only (unwarped, undegraded) image and mask")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--source", choices=["auto", "synthetic", "simulator"], default="auto",
                    help="droplet source; auto uses SpraySimulator when importable")
    ap.add_argument("--examples", type=Path, default=None,
                    help="only write the example figure (6 cards) to this path and exit")
    args = ap.parse_args(argv)
    if args.examples is not None:
        for row in make_examples_figure(args.examples, args.seed):
            print(row)
        return
    summary = generate_dataset(args.out, args.n, args.seed, args.crop_card, args.workers, args.source)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
