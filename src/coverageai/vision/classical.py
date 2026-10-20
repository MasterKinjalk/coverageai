"""Classical (non-learned) water-sensitive-paper verifier, baseline B0.

Pipeline (all CPU, OpenCV + scikit-image):

1. Card ROI. Card-coloured pixels (saturated yellow or blue) are found on a
   downscaled copy; the largest blob's convex hull is reduced to a quadrilateral
   (approxPolyDP) and perspective-corrected to a rectangle. If the card fills the
   image the full image is used; if no quadrilateral is found the filled hull (or
   the full image) is used without warping. The coverage denominator is the
   number of ROI pixels only.
2. Illumination flattening. A coarse per-block 90th percentile of the brightest
   channel gives a smooth gain field (stains cannot dominate a block unless it is
   almost entirely stained); the image is divided by it.
3. Stain channel f = (R - B) / 255 on the flattened image: about +0.76 on yellow and
   -0.35 on blue, and linear in the yellow/blue mixing fraction, so an Otsu
   threshold at the class midpoint keeps blurred stain edges at half height
   (a ratio such as (R-B)/(R+B) is nonlinear and biased the area low). Otsu on the ROI histogram; rejected (and
   a card-referenced fixed threshold used instead) when the two classes are not
   well separated (empty or fully stained card, unimodal histogram) or when the
   threshold leaves the plausible band.
4. Cleanup: 3x3 cross opening, then components below ``min_stain_px`` dropped.
5. coverage = stained px / ROI px * 100.
6. Stains: distance transform + watershed (markers from h-maxima of log distance,
   so the split criterion is scale free: peaks must exceed the saddle by 1/0.7).
   Equivalent diameter -> um -> / spread factor -> droplet diameter; Dv0.1/0.5/0.9
   are volume weighted (weights d^3). stains_per_cm2 uses the ROI area.
7. ``saturated`` when coverage > 60 %, or when coverage > 30 % and one connected
   component holds more than 70 % of the stained area (the stain network has
   percolated, so counts and sizes are unreliable). Coverage itself stays valid.
8. Class from ``contracts.coverage_class`` with the spec thresholds.

Input images are RGB uint8 (the Renderer contract).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from coverageai.config import spec
from coverageai.contracts import CardMeasurement, coverage_class

_NBINS = 512


@dataclass
class _Geometry:
    """Maps the original image to the analysed (possibly warped/cropped) card."""
    mode: str                                   # "full" | "crop" | "warp"
    shape: tuple[int, int]                      # original (h, w)
    matrix: np.ndarray | None = None            # 3x3, original -> warped frame (warp)
    out_shape: tuple[int, int] | None = None    # warped frame (h, w) before margin crop
    sl: tuple[slice, slice] | None = None       # ROI slice inside warped/original frame
    found_quad: bool = False

    def to_original(self, roi_img: np.ndarray) -> np.ndarray:
        """Place an ROI-frame uint8/bool image back into the original image frame."""
        if self.mode == "full":
            return roi_img.astype(bool)
        h, w = self.shape
        if self.mode == "crop":
            out = np.zeros((h, w), bool)
            out[self.sl] = roi_img
            return out
        oh, ow = self.out_shape  # type: ignore[misc]
        padded = np.zeros((oh, ow), np.uint8)
        padded[self.sl] = roi_img.astype(np.uint8)
        back = cv2.warpPerspective(padded, self.matrix, (w, h),
                                   flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST,
                                   borderValue=0)
        return back.astype(bool)


@dataclass
class ClassicalVerifier:
    """Verifier protocol implementation (``measure``) plus ``segment``."""
    spread_factor: float = field(default_factory=lambda: float(np.mean(spec()["physics"]["spread_factor"])))
    acceptable_min: float = field(default_factory=lambda: float(spec()["coverage_classes"]["acceptable_min"]))
    marginal_min: float = field(default_factory=lambda: float(spec()["coverage_classes"]["marginal_min"]))
    saturation_cov_pct: float = 60.0
    percolation_cov_pct: float = 30.0
    percolation_frac: float = 0.70
    min_stain_px: int = 3
    split_h: float = 0.36              # h of the h-maxima on log(distance); ~ saddle/peak < 0.70
    fixed_threshold: float = 0.20      # fallback on f (midpoint of yellow and blue) when unimodal
    min_separation: float = 0.2        # min distance between Otsu class means (f units)
    blur_sigma: float = 0.8
    last_info: dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ API
    def measure(self, image: np.ndarray, px_per_mm: float) -> CardMeasurement:
        t0 = time.perf_counter()
        if px_per_mm <= 0:
            raise ValueError("px_per_mm must be positive")
        img = self._as_rgb_u8(image)
        geo, crop = self._card_roi(img)
        mask, info = self._stain_mask(crop)
        card_px = mask.size
        stained = int(mask.sum())
        cov = 100.0 * stained / card_px
        stats = self._stain_stats(mask, px_per_mm, card_px)
        sat = bool(cov > self.saturation_cov_pct or
                   (cov > self.percolation_cov_pct and stats["largest_cc_frac"] > self.percolation_frac))
        self.last_info = {**info, **stats, "geometry": geo.mode, "found_quad": geo.found_quad,
                          "card_px": card_px, "coverage_pct": cov}
        return CardMeasurement(
            coverage_pct=float(cov),
            stains_per_cm2=stats["stains_per_cm2"],
            dv01_um=stats["dv01_um"], dv05_um=stats["dv05_um"], dv09_um=stats["dv09_um"],
            coverage_class=coverage_class(cov, self.acceptable_min, self.marginal_min),
            saturated=sat,
            runtime_s=time.perf_counter() - t0,
        )

    def segment(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (stain_mask, card_roi_mask), both bool and in the input image frame."""
        img = self._as_rgb_u8(image)
        geo, crop = self._card_roi(img)
        mask, _ = self._stain_mask(crop)
        roi = geo.to_original(np.ones(mask.shape, np.uint8))
        return geo.to_original(mask.astype(np.uint8)), roi

    # ----------------------------------------------------------- card ROI
    @staticmethod
    def _as_rgb_u8(image: np.ndarray) -> np.ndarray:
        img = np.asarray(image)
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        if img.shape[2] == 4:
            img = img[..., :3]
        if img.dtype != np.uint8:
            f = img.astype(np.float32)
            if f.max() <= 1.0:
                f = f * 255.0
            img = np.clip(f, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(img)

    def _card_roi(self, img: np.ndarray) -> tuple[_Geometry, np.ndarray]:
        h, w = img.shape[:2]
        full = _Geometry("full", (h, w))
        scale = min(1.0, 640.0 / max(h, w))
        small = cv2.resize(img, (max(8, round(w * scale)), max(8, round(h * scale))),
                           interpolation=cv2.INTER_AREA) if scale < 1 else img
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        colored = (sat >= 60) & (val >= 40) & (((hue >= 12) & (hue <= 50)) | ((hue >= 85) & (hue <= 140)))
        colored = colored.astype(np.uint8)
        if colored.mean() < 0.01:
            return full, img
        k = max(3, int(round(0.012 * max(small.shape[:2]))) | 1)
        colored = cv2.morphologyEx(colored, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        cnts, _ = cv2.findContours(colored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return full, img
        cnt = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        harea = cv2.contourArea(hull)
        sh, sw = small.shape[:2]
        if harea < 0.02 * sh * sw:
            return full, img
        if harea > 0.985 * sh * sw:
            return full, img

        quad = self._hull_to_quad(hull, harea)
        inv = 1.0 / scale
        if quad is None:
            # no clean quadrilateral: keep the original frame, ROI = filled hull
            roi = np.zeros((sh, sw), np.uint8)
            cv2.fillConvexPoly(roi, hull, 1)
            roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.nonzero(roi)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            geo = _Geometry("crop", (h, w), sl=(slice(y0, y1), slice(x0, x1)))
            return geo, img[geo.sl]

        quad = self._order_quad(quad * inv)
        margin = int(max(4, np.ceil(1.5 * inv)))
        # axis-aligned rectangle (within 0.4 %): plain crop, no resampling
        xs_, ys_ = quad[:, 0], quad[:, 1]
        tol = max(3.0, 0.004 * max(h, w))
        if (abs(xs_[0] - xs_[3]) < tol and abs(xs_[1] - xs_[2]) < tol and
                abs(ys_[0] - ys_[1]) < tol and abs(ys_[2] - ys_[3]) < tol):
            x0 = int(np.ceil(max(xs_[0], xs_[3]))) + margin
            x1 = int(np.floor(min(xs_[1], xs_[2]))) - margin
            y0 = int(np.ceil(max(ys_[0], ys_[1]))) + margin
            y1 = int(np.floor(min(ys_[2], ys_[3]))) - margin
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, w), min(y1, h)
            if x1 - x0 > 8 and y1 - y0 > 8:
                geo = _Geometry("crop", (h, w), sl=(slice(y0, y1), slice(x0, x1)), found_quad=True)
                return geo, img[geo.sl]
        e = lambda a, b: float(np.hypot(*(quad[a] - quad[b])))
        ow = int(round(max(e(0, 1), e(3, 2))))
        oh = int(round(max(e(0, 3), e(1, 2))))
        dst = np.array([[0, 0], [ow, 0], [ow, oh], [0, oh]], np.float32)
        M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        warped = cv2.warpPerspective(img, M, (ow, oh), flags=cv2.INTER_LINEAR)
        sl = (slice(margin, oh - margin), slice(margin, ow - margin))
        geo = _Geometry("warp", (h, w), matrix=M, out_shape=(oh, ow), sl=sl, found_quad=True)
        return geo, warped[sl]

    @staticmethod
    def _hull_to_quad(hull: np.ndarray, harea: float) -> np.ndarray | None:
        peri = cv2.arcLength(hull, True)
        for eps in (0.01, 0.015, 0.02, 0.03, 0.045, 0.06, 0.08):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                if cv2.contourArea(ap) > 0.9 * harea:
                    return ap.reshape(4, 2).astype(np.float64)
                return None
            if len(ap) < 4:
                break
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        if cv2.contourArea(box.astype(np.float32)) > 0 and harea / cv2.contourArea(box.astype(np.float32)) > 0.93:
            return box.astype(np.float64)
        return None

    @staticmethod
    def _order_quad(q: np.ndarray) -> np.ndarray:
        """Order as tl, tr, br, bl, then rotate so the longer side pair is horizontal."""
        s, d = q.sum(1), q[:, 0] - q[:, 1]
        o = np.array([q[np.argmin(s)], q[np.argmax(d)], q[np.argmax(s)], q[np.argmin(d)]])
        top = 0.5 * (np.hypot(*(o[0] - o[1])) + np.hypot(*(o[3] - o[2])))
        side = 0.5 * (np.hypot(*(o[0] - o[3])) + np.hypot(*(o[1] - o[2])))
        if top < side:
            o = np.roll(o, -1, axis=0)       # tr, br, bl, tl -> longer sides become horizontal
        return o

    # -------------------------------------------------------- stain mask
    def _flatten(self, crop: np.ndarray, ppm_hint: float | None = None) -> np.ndarray:
        """Divide by a smooth gain field estimated from block 90th percentiles of V."""
        f = crop.astype(np.float32)
        hc, wc = crop.shape[:2]
        b = int(max(16, min(hc, wc) // 6))
        hb, wb = hc // b, wc // b
        if hb < 2 or wb < 2:
            return f
        v = f[: hb * b, : wb * b].max(axis=2)
        blocks = v.reshape(hb, b, wb, b).transpose(0, 2, 1, 3).reshape(hb, wb, b * b)
        g = np.percentile(blocks, 90, axis=2).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), 1.0, borderType=cv2.BORDER_REPLICATE)
        ref = float(np.percentile(g, 75))
        g = np.clip(g / max(ref, 1e-3), 0.7, 1.4)
        gain = cv2.resize(g, (wb * b, hb * b), interpolation=cv2.INTER_LINEAR)
        gain = cv2.copyMakeBorder(gain, 0, hc - hb * b, 0, wc - wb * b, cv2.BORDER_REPLICATE)
        # resize maps block centres poorly at the border; the clip keeps this benign
        return f / gain[..., None]

    def _stain_channel(self, crop: np.ndarray) -> np.ndarray:
        flat = self._flatten(crop)
        r = cv2.GaussianBlur(flat[..., 0], (0, 0), self.blur_sigma)
        bl = cv2.GaussianBlur(flat[..., 2], (0, 0), self.blur_sigma)
        return (r - bl) * (1.0 / 255.0)

    def _threshold(self, f: np.ndarray) -> tuple[float, str, dict[str, float]]:
        q = np.clip((f + 1.0) * 0.5 * (_NBINS - 1), 0, _NBINS - 1).astype(np.int32)
        hist = np.bincount(q.ravel(), minlength=_NBINS).astype(np.float64)
        centers = np.linspace(-1.0, 1.0, _NBINS)
        p = hist / hist.sum()
        w0 = np.cumsum(p)
        m = np.cumsum(p * centers)
        mt = m[-1]
        w1 = 1.0 - w0
        with np.errstate(divide="ignore", invalid="ignore"):
            sb = (mt * w0 - m) ** 2 / (w0 * w1)
        sb[~np.isfinite(sb)] = 0.0
        i = int(np.argmax(sb))
        t_otsu = float(0.5 * (centers[i] + centers[min(i + 1, _NBINS - 1)]))
        m0 = m[i] / max(w0[i], 1e-12)
        m1 = (mt - m[i]) / max(w1[i], 1e-12)
        sep = float(m1 - m0)
        eta = float(sb[i] / max(np.sum(p * (centers - mt) ** 2), 1e-12))
        ok = sep >= self.min_separation and abs(t_otsu - self.fixed_threshold) <= 0.4
        med = float(np.median(f[:: max(1, f.shape[0] // 64), :: max(1, f.shape[1] // 64)]))
        diag = {"otsu_threshold": t_otsu, "class_separation": sep, "between_class_ratio": eta,
                "median_f": med}
        if ok:
            return t_otsu, "otsu", diag
        # card-referenced fallback: yellow reference minus half the yellow-blue distance
        thr = med - 0.40 if med > 0.45 else self.fixed_threshold
        return float(np.clip(thr, 0.0, 0.4)), "fixed", diag

    def _stain_mask(self, crop: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        f = self._stain_channel(crop)
        thr, method, diag = self._threshold(f)
        mask = (f < thr).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
        if self.min_stain_px > 1:
            n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            keep = stats[:, cv2.CC_STAT_AREA] >= self.min_stain_px
            keep[0] = False
            mask = keep[lab].astype(np.uint8)
        return mask.astype(bool), {"threshold": thr, "threshold_method": method, **diag}

    # -------------------------------------------------- stains and sizes
    def stats_from_mask(self, mask: np.ndarray, px_per_mm: float, card_px: int) -> dict[str, Any]:
        """Stain count / density / Dv0.1-0.9 / largest-component share of ANY boolean stain mask
        (e.g. the U-Net's) with this verifier's distance-transform watershed. ``card_px`` is the
        number of card pixels (the density denominator). Same keys as the stats in ``last_info``."""
        return self._stain_stats(np.asarray(mask, bool), px_per_mm, int(card_px))

    def _stain_areas(self, mask: np.ndarray) -> tuple[np.ndarray, float]:
        """Pixel area of every stain after splitting touching ones, and the largest
        connected component's share of the stained area.

        Only connected components holding >= 2 distance-transform markers go through the
        watershed; very large masks (> 1e6 px, i.e. saturated cards) are split at half
        resolution, where tiny stains no longer matter.
        """
        ds = 2 if int(mask.sum()) > 1_000_000 else 1
        m8 = mask.astype(np.uint8)
        if ds == 2:
            m8 = (cv2.resize(m8 * 255, (mask.shape[1] // 2, mask.shape[0] // 2),
                             interpolation=cv2.INTER_AREA) > 127).astype(np.uint8)
        m = m8.astype(bool)
        ncc, cc, st, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
        cc_area = st[:, cv2.CC_STAT_AREA].astype(np.float64)
        cc_area[0] = 0.0
        lcc = float(cc_area.max() / max(cc_area.sum(), 1.0))

        dist = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
        q = 40.0    # log-scale quantisation: h-maxima threshold becomes a peak/saddle ratio
        ld = np.minimum(np.log(np.maximum(dist, 1.0)) * q, 255.0).astype(np.uint8)
        peaks = _h_maxima_u8(ld, int(round(self.split_h * q))) & m
        nmk, mk = cv2.connectedComponents(peaks.astype(np.uint8), connectivity=8)
        ys, xs = np.nonzero(peaks)
        pairs = np.unique(np.stack([mk[ys, xs], cc[ys, xs]]), axis=1)
        n_markers = np.bincount(pairs[1], minlength=ncc)
        multi = n_markers >= 2
        multi[0] = False

        areas = cc_area[(~multi) & (np.arange(ncc) > 0)]
        if multi.any():
            sel = multi[cc]
            # cv2.watershed floods an 8-bit 3-channel image from the markers; it leaves
            # 1-px (-1) lines between regions, which are handed back to a neighbour
            # background = 255 vs stain <= 200: a step the flood only crosses last, so
            # labels cannot leak from one connected component into another
            gray = np.where(sel, 200 - (ld * 0.75), 255).astype(np.uint8)
            lab = np.where(sel, mk, 0).astype(np.int32)
            cv2.watershed(cv2.merge([gray, gray, gray]), lab)
            lab[~sel] = 0
            line = sel & (lab <= 0)
            if line.any():
                lab[line] = 0
                lab = np.where(line, cv2.dilate(lab.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32), lab)
            a2 = np.bincount(lab[lab > 0].ravel()).astype(np.float64)
            areas = np.concatenate([areas, a2[a2 > 0]])
        return areas * ds * ds, lcc

    def _stain_stats(self, mask: np.ndarray, ppm: float, card_px: int) -> dict[str, Any]:
        out: dict[str, Any] = {"n_stains": 0, "stains_per_cm2": 0.0, "dv01_um": None,
                               "dv05_um": None, "dv09_um": None, "largest_cc_frac": 0.0}
        if not mask.any():
            return out
        areas, out["largest_cc_frac"] = self._stain_areas(mask)
        areas = areas[areas >= self.min_stain_px]
        if areas.size == 0:
            return out
        d_stain_um = 2.0 * np.sqrt(areas / np.pi) / ppm * 1000.0
        d_drop = d_stain_um / self.spread_factor
        card_cm2 = card_px / ppm ** 2 / 100.0
        out["n_stains"] = int(areas.size)
        out["stains_per_cm2"] = float(areas.size / card_cm2)
        dv = _volume_percentiles(d_drop, (0.1, 0.5, 0.9))
        out["dv01_um"], out["dv05_um"], out["dv09_um"] = (float(x) for x in dv)
        return out


def _volume_percentiles(d: np.ndarray, qs: tuple[float, ...]) -> np.ndarray:
    """Diameters below which the given fractions of the total volume lie (interpolated)."""
    order = np.argsort(d)
    ds = d[order]
    vol = ds ** 3
    cum = (np.cumsum(vol) - 0.5 * vol) / vol.sum()      # mid-point cumulative fraction
    return np.interp(qs, cum, ds)


def _h_maxima_u8(f: np.ndarray, h: int, max_iter: int = 2000) -> np.ndarray:
    """h-maxima of a uint8 image by geodesic reconstruction with OpenCV (same markers as
    skimage.morphology.h_maxima, ~10x faster here). Apex pixels of every peak whose
    dynamic is >= h are True."""
    k = np.ones((3, 3), np.uint8)
    rec = cv2.subtract(f, np.full_like(f, h))
    it = 0
    while it < max_iter:
        prev = rec
        for _ in range(8):
            rec = cv2.min(cv2.dilate(rec, k), f)
        it += 8
        if np.array_equal(prev, rec):
            break
    return cv2.subtract(f, rec) >= h
