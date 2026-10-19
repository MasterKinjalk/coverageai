import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from coverageai.contracts import CardMeasurement, Verifier
from coverageai.vision.classical import ClassicalVerifier
from coverageai.vision.simple_cards import make_card, save_dataset

ROOT = Path(__file__).resolve().parents[1]
V = ClassicalVerifier()


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def test_protocol_and_types():
    assert isinstance(V, Verifier)
    c = make_card(np.random.default_rng(0), 10)
    m = V.measure(c.image, c.px_per_mm)
    assert isinstance(m, CardMeasurement)
    mask, roi = V.segment(c.image)
    assert mask.dtype == bool and roi.dtype == bool
    assert mask.shape == roi.shape == c.image.shape[:2]
    assert m.coverage_class in ("acceptable", "marginal", "critical")


@pytest.mark.parametrize("cov", [2, 8, 15, 25, 35])
def test_clean_coverage_error(cov):
    c = make_card(np.random.default_rng(cov), cov)
    m = V.measure(c.image, c.px_per_mm)
    assert abs(m.coverage_pct - c.coverage_pct) <= 1.5
    mask, _ = V.segment(c.image)
    assert iou(mask, c.mask) > 0.9


@pytest.mark.parametrize("cov", [3, 12, 22, 33])
def test_degraded_coverage_error(cov):
    c = make_card(np.random.default_rng(100 + cov), cov, ellipse=True, blur_px=1.5, noise_sigma=10, gradient=0.25)
    m = V.measure(c.image, c.px_per_mm)
    assert abs(m.coverage_pct - c.coverage_pct) <= 3.0
    mask, _ = V.segment(c.image)
    assert iou(mask, c.mask) > 0.8


@pytest.mark.parametrize("cov", [3, 10, 20])
def test_count_non_touching(cov):
    c = make_card(np.random.default_rng(200 + cov), cov, ellipse=True, blur_px=0.8, noise_sigma=4)
    V.measure(c.image, c.px_per_mm)
    n = V.last_info["n_stains"]
    assert abs(n - c.n_stains) <= 0.10 * c.n_stains


def test_count_split_touching_pair():
    # two equal disks overlapping ~25 % of the diameter must be split; a single disk must not
    img = np.zeros((200, 400, 3), np.uint8)
    img[:] = (235, 210, 40)
    import cv2
    for cx in (100, 130):
        cv2.circle(img, (cx, 100), 25, (30, 40, 120), -1)
    cv2.circle(img, (300, 100), 25, (30, 40, 120), -1)
    V.measure(img, 23.6)
    assert V.last_info["n_stains"] == 3


def test_empty_card():
    rng = np.random.default_rng(1)
    for kw in ({}, dict(noise_sigma=8, gradient=0.2, blur_px=1.0)):
        c = make_card(rng, 0.0, **kw)
        m = V.measure(c.image, c.px_per_mm)
        assert m.coverage_pct < 0.5
        assert m.coverage_class == "critical" and not m.saturated
        assert V.last_info["n_stains"] == 0 and m.dv05_um is None


def test_perspective_on_gray_background():
    for seed, cov in ((3, 8.0), (4, 20.0), (5, 32.0)):
        c = make_card(np.random.default_rng(seed), cov, ellipse=True, perspective=True,
                      blur_px=1.0, noise_sigma=6, gradient=0.2)
        assert c.image.shape[:2] == (1100, 3000)
        m = V.measure(c.image, c.px_per_mm)
        assert V.last_info["geometry"] == "warp" and V.last_info["found_quad"]
        assert abs(m.coverage_pct - c.coverage_pct) <= 3.0
        mask, roi = V.segment(c.image)
        assert iou(roi, c.card_mask) > 0.9          # denominator is the card, not the background
        assert iou(mask, c.mask) > 0.75


def test_saturated_flag_and_classes():
    c = make_card(np.random.default_rng(7), 70, overlap=True, radius_mm=(0.15, 0.5))
    m = V.measure(c.image, c.px_per_mm)
    assert m.saturated and abs(m.coverage_pct - c.coverage_pct) <= 3.0
    c = make_card(np.random.default_rng(8), 12)
    assert not V.measure(c.image, c.px_per_mm).saturated
    # class edges follow spec thresholds
    for cov, cls in ((8, "critical"), (22, "marginal"), (36, "acceptable")):
        c = make_card(np.random.default_rng(9 + cov), cov)
        assert V.measure(c.image, c.px_per_mm).coverage_class == cls


def test_droplet_size_distribution():
    c = make_card(np.random.default_rng(11), 10, radius_mm=(0.08, 0.45))
    m = V.measure(c.image, c.px_per_mm)
    d = c.diameters_px / c.px_per_mm * 1000.0 / V.spread_factor
    order = np.argsort(d)
    cum = np.cumsum(d[order] ** 3) / np.sum(d ** 3)
    for est, q in ((m.dv01_um, 0.1), (m.dv05_um, 0.5), (m.dv09_um, 0.9)):
        true = d[order][np.searchsorted(cum, q)]
        assert abs(est - true) / true < 0.15
    assert m.dv01_um < m.dv05_um < m.dv09_um
    assert m.stains_per_cm2 == pytest.approx(c.n_stains / (c.image.shape[0] * c.image.shape[1] / c.px_per_mm ** 2 / 100), rel=0.1)


@pytest.mark.parametrize("persp", [False, True])
def test_runtime_full_res(persp):
    c = make_card(np.random.default_rng(21), 30, overlap=True, ellipse=True, perspective=persp,
                  blur_px=1.0, noise_sigma=6, gradient=0.15)
    V.measure(c.image, c.px_per_mm)   # warm-up
    m = V.measure(c.image, c.px_per_mm)
    assert m.runtime_s <= 2.0
    assert c.image.shape[:2] == ((1100, 3000) if persp else (614, 1795))


def _load_eval():
    spec = importlib.util.spec_from_file_location("eval_classical", ROOT / "experiments" / "eval_classical.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eval_script_reads_dataset_and_falls_back(tmp_path):
    ev = _load_eval()
    rng = np.random.default_rng(5)
    cards = [make_card(rng, c, blur_px=0.8) for c in (5, 10, 22)]
    save_dataset(tmp_path / "ds", cards, split="test")
    res = ev.main(["--data", str(tmp_path / "ds"), "--split", "test", "--out", str(tmp_path / "r.json"),
                   "--fig", str(tmp_path / "p.png")])
    assert res["overall"]["n"] == 3 and res["overall"]["mae_pp"] < 1.5
    assert res["overall"]["class_agreement"] == 1.0
    assert (tmp_path / "p.png").exists() and json.loads((tmp_path / "r.json").read_text())["by_coverage_band"]
    res = ev.main(["--data", str(tmp_path / "missing"), "--n-fallback", "4", "--out", str(tmp_path / "f.json"),
                   "--fig", str(tmp_path / "f.png")])
    assert "not found" in res["data_source"] and res["overall"]["n"] == 4
