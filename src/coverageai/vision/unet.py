"""Learned verifier B1: U-Net stain segmentation on water-sensitive-paper cards (WP-13).

    uv run python -m coverageai.vision.unet train --data data/synthetic/v1 --epochs 30 --max-minutes 90
    uv run python -m coverageai.vision.unet eval  --data data/synthetic/v1 --split test
    uv run python -m coverageai.vision.unet bench --data data/synthetic/v1 --split test

Pieces: file pairing (`find_pairs`), numpy/cv2 augmentation (`CardDataset`), model builder
(`build_model`), `train`, `evaluate` and `UNetVerifier` (satisfies `contracts.Verifier`).
Requires the `vision` extra (torch, segmentation-models-pytorch). Everything that does not
need torch (pairing, augmentation, stain statistics) imports without it.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from coverageai.config import ROOT, px_per_mm as _spec_ppm, spec
from coverageai.contracts import CardMeasurement, coverage_class

try:  # optional heavy deps
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset as _TorchDataset
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore
    _TorchDataset = object  # type: ignore
    HAS_TORCH = False

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
DEFAULT_ENCODER = "mobilenet_v2"
DEFAULT_DECODER = (128, 64, 32, 16, 16)       # light decoder: keeps CPU inference well under 2 s/card
SATURATED_COVERAGE_PCT = 60.0                  # same rule as PLAN WP-04: coverage > 60% => saturated
CKPT_VERSION = 1


def _require_torch() -> None:
    if not HAS_TORCH:
        raise ImportError("torch / segmentation-models-pytorch missing; run `uv sync --extra vision`")


# ====================================================================== data pairing

IMG_TOKENS = ("_img", "_image", "_rgb", "_card")
MASK_TOKENS = ("_mask", "_gt", "_label", "_seg", "_labels")
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


@dataclass
class PairSpec:
    """How to find (image, mask) pairs. Defaults auto-detect the usual layouts:

    * flat folder:    <stem>_img.png + <stem>_mask.png (+ <stem>.json)
    * split folders:  images/<stem>.png + masks/<stem>.png (+ json/<stem>.json)

    Set `img_suffix` / `mask_suffix` (e.g. "_img", "_mask", or "") to force a flat pattern;
    `img_dir` / `mask_dir` to rename the folders.
    """
    img_suffix: str | None = None
    mask_suffix: str | None = None
    img_dir: str = "images"
    mask_dir: str = "masks"
    meta_dirs: tuple[str, ...] = ("json", "meta", "annotations", "labels_json")
    exts: tuple[str, ...] = IMG_EXTS


@dataclass(frozen=True)
class Pair:
    stem: str
    image: Path
    mask: Path
    meta: Path | None = None


def _strip(stem: str, tokens: Sequence[str]) -> tuple[str, bool]:
    for t in tokens:
        if t and stem.endswith(t):
            return stem[: -len(t)], True
    return stem, False


def _find_meta(root: Path, stem: str, ps: PairSpec, extra_stems: Sequence[str] = ()) -> Path | None:
    names = [stem, f"{stem}_meta", f"{stem}_info", f"{stem}_img", *extra_stems]
    for d in [root, *[root / m for m in ps.meta_dirs]]:
        for n in names:
            p = d / f"{n}.json"
            if p.is_file():
                return p
    return None


def find_pairs(root: str | Path, ps: PairSpec | None = None) -> list[Pair]:
    """Pair images and masks under `root` by stem. Raises FileNotFoundError when none found."""
    ps = ps or PairSpec()
    root = Path(root)
    pairs: list[Pair] = []
    img_dir, mask_dir = root / ps.img_dir, root / ps.mask_dir
    if img_dir.is_dir() and mask_dir.is_dir():
        masks = {}
        for p in sorted(mask_dir.rglob("*")):
            if p.suffix.lower() in ps.exts:
                base, _ = _strip(p.stem, [ps.mask_suffix] if ps.mask_suffix is not None else MASK_TOKENS)
                masks[base] = p
        for p in sorted(img_dir.rglob("*")):
            if p.suffix.lower() in ps.exts:
                base, _ = _strip(p.stem, [ps.img_suffix] if ps.img_suffix else IMG_TOKENS)
                if base in masks:
                    pairs.append(Pair(base, p, masks[base], _find_meta(root, base, ps, [p.stem])))
    else:
        imgs: dict[str, Path] = {}
        masks = {}
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() not in ps.exts or not p.is_file():
                continue
            if ps.img_suffix is not None or ps.mask_suffix is not None:
                m_tok = [ps.mask_suffix] if ps.mask_suffix is not None else MASK_TOKENS
                i_tok = [ps.img_suffix] if ps.img_suffix is not None else IMG_TOKENS
                base, is_mask = _strip(p.stem, [t for t in m_tok if t])
                if is_mask:
                    masks[(p.parent, base)] = p
                    continue
                base, is_img = _strip(p.stem, [t for t in i_tok if t])
                if is_img or ps.img_suffix == "":
                    imgs[(p.parent, base)] = p
                continue
            base, is_mask = _strip(p.stem, MASK_TOKENS)
            if is_mask:
                masks[(p.parent, base)] = p
            else:
                base, _ = _strip(p.stem, IMG_TOKENS)
                imgs[(p.parent, base)] = p
        for key, ip in sorted(imgs.items(), key=lambda kv: str(kv[1])):
            if key in masks:
                parent, base = key
                pairs.append(Pair(base, ip, masks[key], _find_meta(parent, base, ps, [ip.stem])))
    if not pairs:
        raise FileNotFoundError(f"no image/mask pairs found under {root} (pair spec: {ps})")
    return pairs


def read_meta(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def meta_get(meta: dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    """Look up the first of `keys` anywhere in a (possibly nested) metadata dict."""
    for k in keys:
        if k in meta and meta[k] is not None:
            return meta[k]
    for v in meta.values():
        if isinstance(v, dict):
            r = meta_get(v, keys, None)
            if r is not None:
                return r
    return default


def split_pairs(data_dir: str | Path, split: str, ps: PairSpec | None = None,
                fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> list[Pair]:
    """Pairs of one split. Uses <data_dir>/<split>/ when it exists, otherwise a deterministic
    80/10/10 split of everything under data_dir (with a warning)."""
    data_dir = Path(data_dir)
    if (data_dir / split).is_dir():
        return find_pairs(data_dir / split, ps)
    allp = find_pairs(data_dir, ps)
    sj = data_dir / "splits.json"                  # WP-03 layout: {"train": [ids], "val": [...], "test": [...]}
    if sj.is_file():
        try:
            ids = json.loads(sj.read_text()).get(split)
        except ValueError:
            ids = None
        if ids is not None:
            def _key(x: Any) -> str:
                return str(int(x)) if str(x).isdigit() else str(x)
            want = {_key(i) for i in ids}
            return [p for p in allp if _key(p.stem) in want]
    warnings.warn(f"{data_dir}/{split} not found; using a deterministic split of {len(allp)} pairs")
    n = len(allp)
    n_tr, n_va = int(round(fractions[0] * n)), int(round(fractions[1] * n))
    if n >= 3:
        n_tr = min(max(n_tr, 1), n - 2)
        n_va = min(max(n_va, 1), n - n_tr - 1)
    parts = {"train": allp[:n_tr], "val": allp[n_tr:n_tr + n_va], "test": allp[n_tr + n_va:]}
    if split not in parts:
        raise FileNotFoundError(f"split {split!r} not found under {data_dir}")
    return parts[split]


def load_pair(p: Pair) -> tuple[np.ndarray, np.ndarray]:
    """RGB uint8 image, uint8 {0,1} mask."""
    bgr = cv2.imread(str(p.image), cv2.IMREAD_COLOR)
    m = cv2.imread(str(p.mask), cv2.IMREAD_UNCHANGED)
    if bgr is None or m is None:
        raise OSError(f"cannot read {p.image} / {p.mask}")
    if m.ndim == 3:
        m = m[..., :3].max(axis=2)
    if m.shape != bgr.shape[:2]:
        m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(bgr[..., ::-1]), (m > 0).astype(np.uint8)


# ====================================================================== augmentation

@dataclass
class AugConfig:
    rot_deg: float = 12.0
    scale: tuple[float, float] = (0.85, 1.2)
    p_flip: float = 0.5
    brightness: float = 25.0          # additive, 0-255 scale
    contrast: tuple[float, float] = (0.75, 1.25)
    channel_gain: float = 0.07
    hue_deg: float = 6.0              # opencv hue units (0-180), so +-12 degrees
    sat_scale: tuple[float, float] = (0.8, 1.2)
    p_hue: float = 0.5
    p_gradient: float = 0.5
    p_blur: float = 0.4
    blur_sigma: tuple[float, float] = (0.3, 1.6)
    p_noise: float = 0.5
    noise_sigma: tuple[float, float] = (1.0, 7.0)
    p_jpeg: float = 0.4
    jpeg_q: tuple[int, int] = (50, 95)


def sample_crop(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator, crop: int,
                aug: bool, cfg: AugConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One crop x crop patch. With `aug`: random similarity warp (rotation, scale) + flips, sampled
    straight from the source (cheap even for 1800x600 cards); without: plain random crop."""
    cfg = cfg or AugConfig()
    h, w = mask.shape
    if not aug:
        if h < crop or w < crop:
            ph, pw = max(crop - h, 0), max(crop - w, 0)
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect" if min(h, w) > 1 else "edge")
            mask = np.pad(mask, ((0, ph), (0, pw)), mode="reflect" if min(h, w) > 1 else "edge")
            h, w = mask.shape
        y, x = int(rng.integers(0, h - crop + 1)), int(rng.integers(0, w - crop + 1))
        return img[y:y + crop, x:x + crop].copy(), mask[y:y + crop, x:x + crop].copy()
    cx, cy = float(rng.uniform(0, w)), float(rng.uniform(0, h))
    angle = float(rng.uniform(-cfg.rot_deg, cfg.rot_deg))
    scale = float(rng.uniform(*cfg.scale))
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += crop / 2 - cx
    M[1, 2] += crop / 2 - cy
    out_i = cv2.warpAffine(img, M, (crop, crop), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    out_m = cv2.warpAffine(mask * 255, M, (crop, crop), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    out_m = (out_m > 127).astype(np.uint8)
    if rng.random() < cfg.p_flip:
        out_i, out_m = out_i[:, ::-1], out_m[:, ::-1]
    if rng.random() < cfg.p_flip:
        out_i, out_m = out_i[::-1], out_m[::-1]
    return np.ascontiguousarray(out_i), np.ascontiguousarray(out_m)


def photometric(img: np.ndarray, rng: np.random.Generator, cfg: AugConfig | None = None) -> np.ndarray:
    """Blur, illumination gradient, brightness/contrast/gain, hue/saturation, noise, JPEG (RGB uint8)."""
    cfg = cfg or AugConfig()
    x = img.astype(np.float32)
    if rng.random() < cfg.p_blur:
        x = cv2.GaussianBlur(x, (0, 0), float(rng.uniform(*cfg.blur_sigma)))
    if rng.random() < cfg.p_gradient:
        h, w = x.shape[:2]
        gx, gy = rng.uniform(-0.2, 0.2, 2)
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        x *= (1.0 + gx * (xx / w - 0.5) + gy * (yy / h - 0.5))[..., None]
    x = x * float(rng.uniform(*cfg.contrast)) * (1 + rng.uniform(-cfg.channel_gain, cfg.channel_gain, 3).astype(np.float32))
    x += float(rng.uniform(-cfg.brightness, cfg.brightness))
    x = np.clip(x, 0, 255)
    if rng.random() < cfg.p_hue:
        hsv = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-cfg.hue_deg, cfg.hue_deg)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(*cfg.sat_scale), 0, 255)
        x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    if rng.random() < cfg.p_noise:
        x = x + rng.normal(0, rng.uniform(*cfg.noise_sigma), x.shape).astype(np.float32)
    out = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < cfg.p_jpeg:
        ok, buf = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(cfg.jpeg_q[0], cfg.jpeg_q[1] + 1))])
        if ok:
            out = np.ascontiguousarray(cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1])
    return out


def to_tensor_array(img: np.ndarray) -> np.ndarray:
    """RGB uint8 HxWx3 -> normalised float32 3xHxW."""
    x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


class CardDataset(_TorchDataset):
    """Random crops from (image, mask) cards. `crops_per_card` crops per card make one epoch.

    train=True: random warp + photometric augmentation, fresh randomness every epoch.
    train=False: deterministic plain crops (seeded by index), no augmentation.
    Images are cached in RAM when the whole split fits in `cache_mb`.
    """

    def __init__(self, pairs: Sequence[Pair], crop: int = 256, crops_per_card: int = 4,
                 train: bool = True, seed: int = 0, aug: AugConfig | None = None,
                 cache_mb: float = 1500.0):
        self.pairs = list(pairs)
        if not self.pairs:
            raise ValueError("CardDataset needs at least one pair")
        self.crop, self.cpc, self.train, self.seed = crop, crops_per_card, train, seed
        self.aug = aug or AugConfig()
        self.cache: dict[int, tuple[np.ndarray, np.ndarray]] | None = None
        probe = load_pair(self.pairs[0])[0]
        est_mb = probe.nbytes * 1.34 * len(self.pairs) / 1e6
        if est_mb <= cache_mb:
            self.cache = {}

    def __len__(self) -> int:
        return len(self.pairs) * self.cpc

    def _card(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.cache is not None:
            if k not in self.cache:
                self.cache[k] = load_pair(self.pairs[k])
            return self.cache[k]
        return load_pair(self.pairs[k])

    def __getitem__(self, i: int):
        k = i // self.cpc
        img, mask = self._card(k)
        if self.train:
            seed = int(torch.empty((), dtype=torch.int64).random_().item() % (2**63)) if HAS_TORCH else None
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng([self.seed, i])
        im, m = sample_crop(img, mask, rng, self.crop, self.train, self.aug)
        if self.train:
            im = photometric(im, rng, self.aug)
        return torch.from_numpy(to_tensor_array(im)), torch.from_numpy(m[None].astype(np.float32))


# ====================================================================== model

def build_model(encoder: str = DEFAULT_ENCODER, pretrained: bool = True,
                decoder_channels: Sequence[int] = DEFAULT_DECODER):
    """smp.Unet with one output channel (logits). Falls back to random init, with a warning,
    when ImageNet weights cannot be downloaded."""
    _require_torch()
    import segmentation_models_pytorch as smp
    kw = dict(encoder_name=encoder, in_channels=3, classes=1, decoder_channels=tuple(decoder_channels))
    if pretrained:
        try:
            return smp.Unet(encoder_weights="imagenet", **kw)
        except Exception as e:  # network/cache problems
            warnings.warn(f"ImageNet weights unavailable ({type(e).__name__}); training from random init")
    return smp.Unet(encoder_weights=None, **kw)


def bce_dice_loss(logits, target, eps: float = 1.0):
    """BCE + soft Dice (Dice pooled over the batch so empty crops do not blow up)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    dice = (2 * inter + eps) / (p.sum() + target.sum() + eps)
    return bce + (1 - dice)


def _cosine_lr(step: int, total: int, lr0: float, warmup: int = 30, lr_min_frac: float = 0.01) -> float:
    if step < warmup:
        return lr0 * (step + 1) / warmup
    t = min((step - warmup) / max(total - warmup, 1), 1.0)
    return lr0 * (lr_min_frac + (1 - lr_min_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def _resolve_amp(amp: str, device: str) -> str:
    if not device.startswith("cuda") or amp in ("none", "off", "fp32"):
        return "none"
    if amp == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if amp not in ("bf16", "fp16"):
        raise ValueError(f"amp must be auto|bf16|fp16|none, got {amp!r}")
    return amp


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
    return device


def save_checkpoint(path: str | Path, model, meta: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": CKPT_VERSION, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                **meta}, path)


def _validate(model, loader, device: str) -> dict[str, float]:
    model.eval()
    inter = pred_sum = true_sum = 0.0
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += float(bce_dice_loss(logits, y)) * len(x)
            n += len(x)
            p = (logits > 0).float()
            inter += float((p * y).sum())
            pred_sum += float(p.sum())
            true_sum += float(y.sum())
    dice = (2 * inter + 1) / (pred_sum + true_sum + 1)
    iou = (inter + 1) / (pred_sum + true_sum - inter + 1)
    return {"val_loss": loss_sum / max(n, 1), "val_dice": dice, "val_iou": iou}


def train(data_dir: str | Path, out: str | Path = "models/unet_v1.pt", epochs: int = 30, batch: int = 8,
          crop: int = 256, max_minutes: float = 90.0, device: str = "cpu", seed: int = 0, *,
          encoder: str = DEFAULT_ENCODER, decoder_channels: Sequence[int] = DEFAULT_DECODER,
          pretrained: bool = True, lr: float = 1e-3, weight_decay: float = 1e-4, patience: int = 8,
          crops_per_card: int = 4, val_crops_per_card: int = 2, max_steps_per_epoch: int | None = None,
          num_workers: int = 0, log_path: str | Path = "results/unet_train_log.csv",
          pair_spec: PairSpec | None = None, train_split: str = "train", val_split: str = "val",
          aug: AugConfig | None = None, amp: str = "auto", cache_mb: float = 6000.0,
          verbose: bool = True) -> dict[str, Any]:
    """Train and save the best-val-Dice weights to `out`. Stops at `epochs`, when val Dice has not
    improved for `patience` epochs, or at the `max_minutes` wall-clock cap. The cosine schedule is
    stretched/compressed to the epochs that fit in the time cap (estimated after epochs 1 and 2).
    `amp`: "auto" (bf16 on a bf16-capable GPU, fp16 on other GPUs, off on CPU), "bf16", "fp16" or "none".
    Also writes `<out stem>.last.pt` after every epoch. Images are cached in RAM only when
    `num_workers == 0` and the split fits in `cache_mb`."""
    _require_torch()
    t_start = time.perf_counter()
    device = _resolve_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(1)
    tr_pairs = split_pairs(data_dir, train_split, pair_spec)
    va_pairs = split_pairs(data_dir, val_split, pair_spec)
    cache = cache_mb if num_workers == 0 else 0.0
    ds_tr = CardDataset(tr_pairs, crop, crops_per_card, True, seed, aug, cache)
    ds_va = CardDataset(va_pairs, crop, val_crops_per_card, False, seed, aug, cache)
    g = torch.Generator().manual_seed(seed)
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, drop_last=len(ds_tr) >= batch, generator=g,
                       num_workers=num_workers, persistent_workers=num_workers > 0,
                       pin_memory=device.startswith("cuda"))
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=min(num_workers, 2),
                       pin_memory=device.startswith("cuda"))
    model = build_model(encoder, pretrained, decoder_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = min(len(dl_tr), max_steps_per_epoch or len(dl_tr))
    total_steps = epochs * steps_per_epoch
    budget_s = max_minutes * 60.0
    amp_mode = _resolve_amp(amp, device)
    use_amp = amp_mode != "none"
    amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_mode == "fp16")
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["epoch", "step", "lr", "train_loss", "val_loss", "val_dice", "val_iou", "epoch_s", "elapsed_s"]
    meta_base = {"encoder": encoder, "decoder_channels": list(decoder_channels), "crop": crop,
                 "train_px_per_mm": _spec_ppm(), "seed": seed, "n_train_cards": len(tr_pairs),
                 "n_val_cards": len(va_pairs), "torch": torch.__version__, "pretrained": pretrained}
    best, bad, step, eff_epochs, history = -1.0, 0, 0, epochs, []
    stop_reason = "epochs"
    with open(log_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, cols)
        wr.writeheader()
        for ep in range(epochs):
            t_ep = time.perf_counter()
            model.train()
            run, nb = 0.0, 0
            for bi, (x, y) in enumerate(dl_tr):
                if bi >= steps_per_epoch:
                    break
                cur_lr = _cosine_lr(step, total_steps, lr)
                for gp in opt.param_groups:
                    gp["lr"] = cur_lr
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    loss = bce_dice_loss(model(x).float(), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                run += loss.item()
                nb += 1
                step += 1
                if time.perf_counter() - t_start > budget_s:
                    stop_reason = "max_minutes"
                    break
            val = _validate(model, dl_va, device)
            ep_s = time.perf_counter() - t_ep
            elapsed = time.perf_counter() - t_start
            row = {"epoch": ep + 1, "step": step, "lr": cur_lr, "train_loss": run / max(nb, 1), **val,
                   "epoch_s": ep_s, "elapsed_s": elapsed}
            wr.writerow(row)
            fh.flush()
            history.append(row)
            if verbose:
                print(f"epoch {ep + 1}/{eff_epochs} train {row['train_loss']:.4f} val_dice {val['val_dice']:.4f} "
                      f"val_iou {val['val_iou']:.4f} lr {cur_lr:.2e} {ep_s:.0f}s (total {elapsed / 60:.1f} min)", flush=True)
            ck_meta = {**meta_base, "epoch": ep + 1, "val_dice": val["val_dice"], "val_iou": val["val_iou"],
                       "amp": amp_mode}
            save_checkpoint(Path(out).with_suffix(".last.pt"), model, ck_meta)
            if val["val_dice"] > best:
                best, bad = val["val_dice"], 0
                save_checkpoint(out, model, {**ck_meta, "val_dice": best})
            else:
                bad += 1
            if stop_reason == "max_minutes":
                break
            if ep < 2:  # fit the cosine schedule into the wall-clock cap
                fit = ep + 1 + int(max(budget_s - elapsed, 0.0) // max(ep_s, 1e-3))
                eff_epochs = max(ep + 1, min(epochs, fit))
                total_steps = eff_epochs * steps_per_epoch
            if ep + 1 >= eff_epochs:
                stop_reason = "epochs" if eff_epochs == epochs else "max_minutes"
                break
            if bad >= patience:
                stop_reason = "early_stop"
                break
            if elapsed + ep_s > budget_s:
                stop_reason = "max_minutes"
                break
    wall = time.perf_counter() - t_start
    result = {"best_val_dice": best, "epochs_run": len(history), "wall_s": wall, "stop_reason": stop_reason,
              "out": str(out), "log": str(log_path), "device": device, "steps": step}
    if verbose:
        print(json.dumps(result))
    return result


# ====================================================================== inference

def fuse_conv_bn(module) -> None:
    """In place: fold each eval-mode BatchNorm2d into the Conv2d right before it (inference only).
    Saves about 10% of CPU inference time; outputs are unchanged up to float rounding."""
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    for child in module.children():
        fuse_conv_bn(child)
    if isinstance(module, torch.nn.Sequential):
        keys = list(module._modules.keys())
        for k1, k2 in zip(keys, keys[1:]):
            a, b = module._modules[k1], module._modules[k2]
            if isinstance(a, torch.nn.Conv2d) and isinstance(b, torch.nn.BatchNorm2d):
                module._modules[k1] = fuse_conv_bn_eval(a, b)
                module._modules[k2] = torch.nn.Identity()


def load_checkpoint(path: str | Path, device: str = "cpu", fuse: bool = True):
    """Return (model in eval mode, metadata dict). `fuse` folds BatchNorm into convs (inference only)."""
    _require_torch()
    p = Path(path)
    if not p.is_absolute() and not p.exists() and (ROOT / p).exists():
        p = ROOT / p
    ck = torch.load(p, map_location=device, weights_only=False)
    model = build_model(ck.get("encoder", DEFAULT_ENCODER), pretrained=False,
                        decoder_channels=ck.get("decoder_channels", DEFAULT_DECODER))
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    if fuse:
        fuse_conv_bn(model)
    return model, {k: v for k, v in ck.items() if k != "state_dict"}


def _positions(n: int, tile: int, stride: int) -> list[int]:
    if n <= tile:
        return [0]
    pos = list(range(0, n - tile + 1, stride))
    if pos[-1] != n - tile:
        pos.append(n - tile)
    return pos


def _ramp(n: int, overlap: int) -> np.ndarray:
    """1 in the interior, linear down to ~0.05 over `overlap` px at both ends."""
    r = np.ones(n, np.float32)
    k = max(min(overlap, n // 2), 1)
    edge = np.linspace(0.05, 1.0, k, dtype=np.float32)
    r[:k], r[-k:] = edge, edge[::-1]
    return r


def _ceil32(n: int) -> int:
    return -(-n // 32) * 32


def predict_proba(model, image: np.ndarray, tile: int | tuple[int, int] = (832, 640), overlap: int = 32,
                  batch: int = 1, device: str = "cpu") -> np.ndarray:
    """Sliding-window sigmoid probabilities for an RGB uint8 image (float32 HxW).

    `tile` is (height, width) or an int; each side is clipped to the image rounded up to a multiple of 32,
    so a 614 px tall card is covered by a single row of tiles. Tiles overlap by `overlap` px and are blended
    with a ramp window; the image is mirror-padded to the tile size when smaller."""
    _require_torch()
    th, tw = (tile, tile) if isinstance(tile, int) else tile
    h, w = image.shape[:2]
    th, tw = min(th, _ceil32(h)), min(tw, _ceil32(w))
    ph, pw = max(th - h, 0), max(tw - w, 0)
    img = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="symmetric") if (ph or pw) else image
    H, W = img.shape[:2]
    ys, xs = _positions(H, th, max(th - overlap, 1)), _positions(W, tw, max(tw - overlap, 1))
    coords = [(y, x) for y in ys for x in xs]
    win = np.outer(_ramp(th, overlap), _ramp(tw, overlap))
    acc = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    norm = to_tensor_array(img)
    with torch.inference_mode():
        for i in range(0, len(coords), batch):
            chunk = coords[i:i + batch]
            xb = torch.from_numpy(np.stack([norm[:, y:y + th, x:x + tw] for y, x in chunk])).to(device)
            pb = torch.sigmoid(model(xb))[:, 0].float().cpu().numpy()
            for (y, x), p in zip(chunk, pb):
                acc[y:y + th, x:x + tw] += p * win
                wsum[y:y + th, x:x + tw] += win
    return (acc / np.maximum(wsum, 1e-6))[:h, :w]


# ---------------------------------------------------------------- stain statistics

def watershed_labels(mask: np.ndarray, min_dist: int = 3) -> np.ndarray:
    """Distance-transform watershed split of a boolean mask into stains (int32 labels, 0 = background)."""
    from skimage.segmentation import watershed
    m8 = mask.astype(np.uint8)
    dist = cv2.distanceTransform(m8, cv2.DIST_L2, 3)
    sm = cv2.GaussianBlur(dist, (0, 0), 1.0)
    k = 2 * min_dist + 1
    peaks = (sm >= cv2.dilate(sm, np.ones((k, k), np.uint8))) & mask & (sm > 0)
    _, markers = cv2.connectedComponents(peaks.astype(np.uint8), connectivity=8)
    return watershed(-sm, markers, mask=mask)


def stain_stats(mask: np.ndarray, px_per_mm: float, spread_factor: float | None = None,
                min_area_px: int = 4) -> tuple[int, np.ndarray]:
    """(stain count, droplet diameters in um) from a boolean mask: watershed split, equivalent
    circle diameter of each stain, divided by the spread factor midpoint (spec.yaml)."""
    sf = spread_factor or float(np.mean(spec()["physics"]["spread_factor"]))
    if not mask.any():
        return 0, np.zeros(0)
    labels = watershed_labels(mask)
    areas = np.bincount(labels.ravel())[1:].astype(np.float64)
    areas = areas[areas >= min_area_px]
    stain_um = 2.0 * np.sqrt(areas / math.pi) / px_per_mm * 1000.0
    return int(len(areas)), stain_um / sf


def volume_percentiles(d_um: np.ndarray, qs: Sequence[float] = (0.1, 0.5, 0.9)) -> list[float | None]:
    """Volume-weighted diameter percentiles (Dv0.1, Dv0.5, Dv0.9); None for too few stains."""
    if len(d_um) < 5:
        return [None] * len(qs)
    d = np.sort(d_um)
    w = d ** 3
    cum = (np.cumsum(w) - 0.5 * w) / w.sum()
    return [float(np.interp(q, cum, d)) for q in qs]


def hull_card_roi(image: np.ndarray, sat_thresh: int = 55) -> np.ndarray | None:
    """Card mask for a yellow WSP card on a neutral background: convex hull of the largest saturated
    region (paper and stains are both saturated; the background is gray). None when nothing is found."""
    hsv = cv2.cvtColor(np.ascontiguousarray(image[..., :3]), cv2.COLOR_RGB2HSV)
    sat = (hsv[..., 1] > sat_thresh).astype(np.uint8)
    sat = cv2.morphologyEx(sat, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(sat, connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    cnts, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    hull = cv2.convexHull(np.vstack(cnts))
    roi = np.zeros(image.shape[:2], np.uint8)
    cv2.fillConvexPoly(roi, hull, 1)
    return roi.astype(bool)


# ---------------------------------------------------------------- verifier

class UNetVerifier:
    """B1 learned verifier: tiled U-Net inference + watershed stain statistics. Satisfies the
    `contracts.Verifier` protocol.

    Card ROI: reuses `coverageai.vision.classical.ClassicalVerifier` card detection when that
    module is importable, otherwise a saturation convex-hull detector, otherwise the whole image.
    """

    def __init__(self, weights: str | Path = "models/unet_v1.pt", device: str = "cpu",
                 tile: int | tuple[int, int] = (832, 640), overlap: int = 32, batch: int = 1,
                 threshold: float = 0.5, num_threads: int | None = None,
                 roi_mode: str = "auto", min_area_px: int = 4, resize_tol: float = 0.25):
        _require_torch()
        self.device, self.tile, self.overlap, self.batch = device, tile, overlap, batch
        self.threshold, self.min_area_px, self.resize_tol = threshold, min_area_px, resize_tol
        if device == "cpu":
            # half the logical CPUs: hyper-threads / efficiency cores slow conv inference down
            torch.set_num_threads(num_threads or max(1, min((os.cpu_count() or 2) // 2, 8)))
        self.model, self.meta = load_checkpoint(weights, device)
        self.train_ppm = float(self.meta.get("train_px_per_mm", _spec_ppm()))
        s = spec()
        self.acceptable_min = s["coverage_classes"]["acceptable_min"]
        self.marginal_min = s["coverage_classes"]["marginal_min"]
        self.spread = float(np.mean(s["physics"]["spread_factor"]))
        if roi_mode not in ("auto", "classical", "hull", "full"):
            raise ValueError("roi_mode must be auto|classical|hull|full")
        self.roi_mode = roi_mode
        self._classical = None
        if roi_mode in ("auto", "classical"):
            try:
                from coverageai.vision.classical import ClassicalVerifier  # type: ignore
                self._classical = ClassicalVerifier()
            except Exception:
                self._classical = None

    # -- card ROI
    def _classical_roi(self, image: np.ndarray, px_per_mm: float) -> np.ndarray | None:
        if self._classical is None or not hasattr(self._classical, "segment"):
            return None
        try:
            try:
                res = self._classical.segment(image)
            except TypeError:
                res = self._classical.segment(image, px_per_mm)
            return np.asarray(res[1]).astype(bool)
        except Exception:
            return None

    def card_roi(self, image: np.ndarray, px_per_mm: float) -> np.ndarray:
        """Boolean card mask. auto: ClassicalVerifier's ROI when importable, else the saturation hull
        (`hull_card_roi`), else the whole image. Implausible ROIs (< 20% of the image) are rejected."""
        full = np.ones(image.shape[:2], bool)
        if self.roi_mode == "full":
            return full
        cands = []
        if self.roi_mode in ("auto", "classical"):
            cands.append(self._classical_roi(image, px_per_mm))
        if self.roi_mode in ("auto", "hull"):
            cands.append(hull_card_roi(image))
        for roi in cands:
            if roi is not None and roi.shape == full.shape and roi.mean() >= 0.2:
                return roi
        return full

    # -- segmentation
    def probability(self, image: np.ndarray, px_per_mm: float) -> np.ndarray:
        img = np.ascontiguousarray(image[..., :3]).astype(np.uint8, copy=False)
        s = self.train_ppm / px_per_mm if px_per_mm else 1.0
        if abs(s - 1.0) > self.resize_tol:      # match training resolution
            small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            p = predict_proba(self.model, small, self.tile, self.overlap, self.batch, self.device)
            return cv2.resize(p, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        return predict_proba(self.model, img, self.tile, self.overlap, self.batch, self.device)

    def segment(self, image: np.ndarray, px_per_mm: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(stain mask, card ROI mask), both boolean, same shape as `image`."""
        ppm = px_per_mm or self.train_ppm
        roi = self.card_roi(image, ppm)
        h, w = roi.shape
        ys, xs = np.flatnonzero(roi.any(1)), np.flatnonzero(roi.any(0))
        if len(ys) == 0:
            return np.zeros_like(roi), roi
        pad = 16                                   # run the network on the card's bounding box only
        y0, y1 = max(ys[0] - pad, 0), min(ys[-1] + 1 + pad, h)
        x0, x1 = max(xs[0] - pad, 0), min(xs[-1] + 1 + pad, w)
        prob = self.probability(image[y0:y1, x0:x1], ppm)
        mask = np.zeros((h, w), bool)
        mask[y0:y1, x0:x1] = (prob > self.threshold) & roi[y0:y1, x0:x1]
        return mask, roi

    # -- Verifier protocol
    def measure(self, image: np.ndarray, px_per_mm: float) -> CardMeasurement:
        return self._measure_full(image, px_per_mm)[0]

    def _measure_full(self, image: np.ndarray, px_per_mm: float) -> tuple[CardMeasurement, np.ndarray, np.ndarray]:
        t0 = time.perf_counter()
        mask, roi = self.segment(image, px_per_mm)
        card_px = max(int(roi.sum()), 1)
        stained = int(mask.sum())
        cov = 100.0 * stained / card_px
        n, d_um = stain_stats(mask, px_per_mm, self.spread, self.min_area_px)
        dv01, dv05, dv09 = volume_percentiles(d_um)
        card_cm2 = card_px / (px_per_mm * 10.0) ** 2
        n_cc, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        largest = float(st[1:, cv2.CC_STAT_AREA].max()) / stained if stained and n_cc > 1 else 0.0
        saturated = bool(cov > SATURATED_COVERAGE_PCT or (cov > 45.0 and largest > 0.5))
        meas = CardMeasurement(
            coverage_pct=float(cov), stains_per_cm2=float(n / card_cm2), dv01_um=dv01, dv05_um=dv05,
            dv09_um=dv09, coverage_class=coverage_class(cov, self.acceptable_min, self.marginal_min),
            saturated=saturated, runtime_s=time.perf_counter() - t0)
        return meas, mask, roi


# ====================================================================== evaluation

BANDS = [(0, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 100.01)]


def _band_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    err = np.array([r["pred_cov"] - r["true_cov"] for r in rows])
    return {"n": len(rows), "coverage_mae_pp": float(np.abs(err).mean()), "coverage_bias_pp": float(err.mean()),
            "iou": float(np.mean([r["iou"] for r in rows])),
            "class_agreement": float(np.mean([r["true_class"] == r["pred_class"] for r in rows])),
            "runtime_s_mean": float(np.mean([r["runtime_s"] for r in rows]))}


def evaluate(data_dir: str | Path, split: str = "test", weights: str | Path = "models/unet_v1.pt",
             out_json: str | Path = "results/b1_synthetic.json",
             fig_path: str | Path | None = "experiments/figures/b1_parity.png", device: str = "cpu",
             pair_spec: PairSpec | None = None, max_cards: int | None = None, verifier: Any = None,
             verbose: bool = True) -> dict[str, Any]:
    """Run the verifier over a split. Truth = the card's true coverage from the per-card JSON
    (`true_coverage_pct`, defined on the undistorted card); falls back to the GT mask inside the
    detected ROI when no JSON is present. Metrics: coverage MAE (pp), IoU of the stain mask, 3-class
    agreement, runtime, overall and stratified by coverage band. The mask-based MAE is also reported."""
    pairs = split_pairs(data_dir, split, pair_spec)[:max_cards]
    ver = verifier or UNetVerifier(weights, device)
    rows = []
    for i, p in enumerate(pairs):
        img, gt = load_pair(p)
        meta = read_meta(p.meta)
        ppm = float(meta_get(meta, ["px_per_mm", "pixels_per_mm", "ppm"], _spec_ppm()))
        meas, mask, roi = ver._measure_full(img, ppm)
        gt_b = gt.astype(bool) & roi
        mask_cov = 100.0 * gt_b.sum() / max(int(roi.sum()), 1)
        union = (mask | gt_b).sum()
        iou = float((mask & gt_b).sum() / union) if union else 1.0
        meta_cov = meta_get(meta, ["true_coverage_pct", "coverage_pct", "coverage"], None)
        true_cov = mask_cov if meta_cov is None else float(meta_cov)
        rows.append({"stem": p.stem, "true_cov": float(true_cov), "mask_cov": float(mask_cov),
                     "pred_cov": meas.coverage_pct, "iou": iou,
                     "true_class": coverage_class(true_cov), "pred_class": meas.coverage_class,
                     "runtime_s": meas.runtime_s, "meta_cov": None if meta_cov is None else float(meta_cov)})
        if verbose and (i + 1) % 10 == 0:
            print(f"  evaluated {i + 1}/{len(pairs)}", flush=True)
    result: dict[str, Any] = {"verifier": "B1_unet", "split": split, "data": str(data_dir),
                              "weights": str(weights), "overall": _band_stats(rows), "by_band": {}}
    for lo, hi in BANDS:
        sel = [r for r in rows if lo <= r["true_cov"] < hi]
        result["by_band"][f"{lo:g}-{min(hi, 100):g}"] = _band_stats(sel)
    result["truth"] = "json" if all(r["meta_cov"] is not None for r in rows) else "mask"
    if rows:
        result["overall"]["coverage_mae_vs_mask_pp"] = float(np.mean([abs(r["pred_cov"] - r["mask_cov"]) for r in rows]))
        result["overall"]["runtime_s_max"] = float(max(r["runtime_s"] for r in rows))
    result["targets"] = {"coverage_mae_pp": 2.0, "class_agreement": 0.90, "runtime_s": 2.0}
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))
    if fig_path:
        parity_plot(rows, fig_path, result["overall"])
    if verbose:
        print(json.dumps(result["overall"], indent=2))
    return result


def parity_plot(rows: list[dict[str, Any]], path: str | Path, overall: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([r["true_cov"] for r in rows])
    p = np.array([r["pred_cov"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    ax[0].scatter(t, p, s=14, alpha=0.7)
    lim = [0, max(80.0, float(max(t.max(initial=0), p.max(initial=0))) + 2)]
    ax[0].plot(lim, lim, "k--", lw=1)
    for v in (15, 30):
        ax[0].axvline(v, color="gray", lw=0.5)
        ax[0].axhline(v, color="gray", lw=0.5)
    ax[0].set(xlabel="true coverage (%)", ylabel="U-Net coverage (%)", xlim=lim, ylim=lim,
              title=f"B1 parity, n={len(rows)}, MAE {overall.get('coverage_mae_pp', float('nan')):.2f} pp")
    ax[1].scatter(t, p - t, s=14, alpha=0.7)
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set(xlabel="true coverage (%)", ylabel="error (pp)", title="residual vs coverage")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bench(data_dir: str | Path, split: str, weights: str | Path, n: int = 5, device: str = "cpu",
          pair_spec: PairSpec | None = None) -> dict[str, float]:
    """Time `measure` on full cards (first call excluded as warm-up)."""
    ver = UNetVerifier(weights, device)
    pairs = split_pairs(data_dir, split, pair_spec)[:n + 1]
    ts = []
    for k, p in enumerate(pairs):
        img, _ = load_pair(p)
        m = ver.measure(img, float(meta_get(read_meta(p.meta), ["px_per_mm"], _spec_ppm())))
        if k:
            ts.append(m.runtime_s)
    res = {"shape": list(img.shape[:2]), "threads": torch.get_num_threads(), "mean_s": float(np.mean(ts)),
           "max_s": float(np.max(ts)), "n": len(ts)}
    print(json.dumps(res))
    return res


# ====================================================================== CLI

def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m coverageai.vision.unet", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data", required=True)
        p.add_argument("--img-suffix", default=None, help='flat layout image suffix, e.g. "_img"')
        p.add_argument("--mask-suffix", default=None, help='flat layout mask suffix, e.g. "_mask"')
        p.add_argument("--device", default="cpu")

    t = sub.add_parser("train")
    common(t)
    t.add_argument("--out", default="models/unet_v1.pt")
    t.add_argument("--log", default="results/unet_train_log.csv")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--crop", type=int, default=256)
    t.add_argument("--max-minutes", type=float, default=90.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--encoder", default=DEFAULT_ENCODER)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--patience", type=int, default=8)
    t.add_argument("--crops-per-card", type=int, default=4)
    t.add_argument("--max-steps-per-epoch", type=int, default=None)
    t.add_argument("--workers", type=int, default=0)
    t.add_argument("--no-pretrained", action="store_true")
    t.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "none"])
    t.add_argument("--cache-mb", type=float, default=6000.0, help="RAM image cache (only with --workers 0)")

    e = sub.add_parser("eval")
    common(e)
    e.add_argument("--split", default="test")
    e.add_argument("--weights", default="models/unet_v1.pt")
    e.add_argument("--out", default="results/b1_synthetic.json")
    e.add_argument("--fig", default="experiments/figures/b1_parity.png")
    e.add_argument("--max-cards", type=int, default=None)

    b = sub.add_parser("bench")
    common(b)
    b.add_argument("--split", default="test")
    b.add_argument("--weights", default="models/unet_v1.pt")
    b.add_argument("-n", type=int, default=5)

    a = ap.parse_args(argv)
    ps = PairSpec(img_suffix=a.img_suffix, mask_suffix=a.mask_suffix)
    if a.cmd == "train":
        train(a.data, a.out, a.epochs, a.batch, a.crop, a.max_minutes, a.device, a.seed, encoder=a.encoder,
              lr=a.lr, patience=a.patience, crops_per_card=a.crops_per_card,
              max_steps_per_epoch=a.max_steps_per_epoch, num_workers=a.workers, log_path=a.log,
              pair_spec=ps, pretrained=not a.no_pretrained, amp=a.amp, cache_mb=a.cache_mb)
    elif a.cmd == "eval":
        evaluate(a.data, a.split, a.weights, a.out, a.fig, a.device, ps, a.max_cards)
    else:
        bench(a.data, a.split, a.weights, a.n, a.device, ps)


if __name__ == "__main__":
    main()
