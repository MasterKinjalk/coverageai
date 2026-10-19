"""WP-03 tests: synthetic WSP renderer and dataset generator."""
from __future__ import annotations

import json
import sys
import time
import types

import cv2
import numpy as np
import pytest

from coverageai.contracts import DropletSample, Outcome, Renderer
from coverageai.vision.render import (BANDS, DATASET_CLASSES, PROFILES, SYNTH_MARGIN_MM,
                                      WSPRenderer, analytic_coverage_pct,
                                      deposition_for_coverage, dv_quantiles,
                                      generate_dataset, synthetic_droplets)
from coverageai.config import spec

VMD = {"F": 138.0, "M": 212.0, "C": 312.0, "VC": 412.0, "XC": 550.0}


def make_outcome(target_pct: float, dc: str, seed: int) -> Outcome:
    rng = np.random.default_rng(seed)
    vmd = VMD[dc]
    sf = float(rng.uniform(*spec()["physics"]["spread_factor"]))
    dep = deposition_for_coverage(target_pct, vmd, 1.2, sf)
    ds = synthetic_droplets(vmd, dep, rng, 1.2, sf)
    return Outcome(analytic_coverage_pct(ds, SYNTH_MARGIN_MM), dep, 0.0, 0.0, 0.0, ds)


def blueish(img: np.ndarray) -> np.ndarray:
    """Pixels clearly less yellow than paper: (R+G)/2 - B well below the paper value (~165)."""
    r, g, b = (img[..., i].astype(np.int32) for i in range(3))
    return (r + g) // 2 - b < 60


def test_satisfies_renderer_protocol():
    assert isinstance(WSPRenderer(), Renderer)


def test_card_geometry_matches_spec():
    r = WSPRenderer()
    assert (r.width, r.height) == (1795, 614)
    assert abs(r.nominal_px_per_mm - 600 / 25.4) < 1e-9


def test_mask_coverage_matches_analytic_boolean_model():
    rnd = WSPRenderer()
    rng = np.random.default_rng(123)
    diffs = []
    for k in range(100):
        dc = DATASET_CLASSES[k % len(DATASET_CLASSES)]
        target = float(rng.uniform(1.0, 80.0))
        oc = make_outcome(target, dc, 1000 + k)
        _, mask = rnd.render(oc, np.random.default_rng(k), degrade=False)
        diffs.append(100.0 * mask.mean() - oc.coverage_pct)
    diffs = np.array(diffs)
    assert np.abs(diffs).mean() <= 2.0, f"mean |mask - analytic| = {np.abs(diffs).mean():.2f} pp"
    assert abs(diffs.mean()) <= 1.0, f"bias = {diffs.mean():.2f} pp"


def test_deterministic_per_seed():
    oc = make_outcome(30.0, "C", 5)
    rnd = WSPRenderer()
    for degrade in (False, True):
        i1, m1 = rnd.render(oc, np.random.default_rng(7), degrade=degrade, profile="phone" if degrade else None)
        i2, m2 = WSPRenderer().render(oc, np.random.default_rng(7), degrade=degrade, profile="phone" if degrade else None)
        assert np.array_equal(i1, i2) and np.array_equal(m1, m2)
    a, _ = rnd.render(oc, np.random.default_rng(7), degrade=False)
    b, _ = rnd.render(oc, np.random.default_rng(8), degrade=False)
    assert not np.array_equal(a, b)


def test_render_time_under_one_second_worst_case():
    rnd = WSPRenderer()
    oc = make_outcome(78.0, "F", 11)            # fine droplets at high coverage: most stains
    rnd.render(oc, np.random.default_rng(0), degrade=True, profile="phone")   # warm-up
    times = []
    for k in range(3):
        t0 = time.perf_counter()
        rnd.render(oc, np.random.default_rng(k), degrade=True, profile="phone")
        times.append(time.perf_counter() - t0)
    assert float(np.median(times)) <= 1.5, times  # 1 s target when idle; slack for parallel load


@pytest.mark.parametrize("profile", PROFILES)
def test_mask_shape_matches_image_and_dtypes(profile):
    oc = make_outcome(25.0, "M", 3)
    rnd = WSPRenderer()
    img, mask = rnd.render(oc, np.random.default_rng(1), degrade=True, profile=profile)
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
    assert mask.dtype == bool and mask.shape == img.shape[:2]
    assert img.shape[0] > rnd.height and img.shape[1] > rnd.width     # card placed with margin
    assert 0.5 * rnd.nominal_px_per_mm < rnd.px_per_mm < 1.5 * rnd.nominal_px_per_mm
    img0, mask0 = rnd.render(oc, np.random.default_rng(1), degrade=False)
    assert img0.shape[:2] == mask0.shape == (rnd.height, rnd.width)


def test_stains_align_with_mask_without_degradation():
    rnd = WSPRenderer()
    for dc, tgt in (("F", 20.0), ("C", 45.0), ("XC", 60.0)):
        oc = make_outcome(tgt, dc, 21)
        img, mask = rnd.render(oc, np.random.default_rng(2), degrade=False)
        frac = blueish(img)[mask].mean()
        assert frac > 0.98, (dc, frac)
        # and the paper outside the mask is yellow
        assert (~blueish(img)[~mask]).mean() > 0.98


@pytest.mark.parametrize("profile", PROFILES)
def test_warped_mask_aligns_with_degraded_image(profile):
    rnd = WSPRenderer()
    oc = make_outcome(45.0, "C", 22)
    img, mask = rnd.render(oc, np.random.default_rng(4), degrade=True, profile=profile)
    core = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    assert blueish(img)[core].mean() > 0.9


def test_undetectable_droplets_leave_no_stain():
    n = 5000
    rng = np.random.default_rng(0)
    ds = DropletSample(np.full(n, 40.0), rng.uniform(0, 76, n), rng.uniform(0, 26, n), 1.9)
    oc = Outcome(0.0, 0.1, 0.0, 0.0, 0.0, ds)
    img, mask = WSPRenderer().render(oc, np.random.default_rng(0), degrade=False)
    assert not mask.any()
    assert (~blueish(img)).all()


def test_missing_droplets_raises():
    with pytest.raises(ValueError):
        WSPRenderer().render(Outcome(10.0, 0.1, 0.0, 0.0, 1.0, None), np.random.default_rng(0))


def test_dv_quantiles_monotone():
    d = np.random.default_rng(0).lognormal(5.0, 0.4, 2000)
    q = dv_quantiles(d)
    assert q[0] < q[1] < q[2]
    assert dv_quantiles(np.array([])) == [None, None, None]


def test_dataset_generator_small(tmp_path):
    out = tmp_path / "v1"
    summary = generate_dataset(out, n=10, seed=0, crop_card=True, workers=1, source="synthetic")
    assert summary["n"] == 10
    splits = json.loads((out / "splits.json").read_text())
    assert sorted(sum(splits.values(), [])) == list(range(10))
    meta = json.loads((out / "meta" / "00003.json").read_text())
    for key in ("true_coverage_pct", "analytic_coverage_pct", "stain_count", "dv01_um", "dv05_um",
                "dv09_um", "profile", "px_per_mm", "vmd_um", "deposition_ul_cm2"):
        assert key in meta
    img = cv2.imread(str(out / "images" / "00003.png"))
    msk = cv2.imread(str(out / "masks" / "00003.png"), cv2.IMREAD_GRAYSCALE)
    assert img.shape[:2] == msk.shape
    card = cv2.imread(str(out / "cards" / "00003_card.png"))
    assert card.shape[:2] == (614, 1795)
    # coverage recomputed from the saved card mask equals the JSON value
    cm = cv2.imread(str(out / "cards" / "00003_card_mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    assert abs(100 * cm.mean() - meta["true_coverage_pct"]) < 1e-6
    # the same seed reproduces the same card
    out2 = tmp_path / "v1b"
    generate_dataset(out2, n=10, seed=0, workers=1, source="synthetic")
    assert (out / "images" / "00003.png").read_bytes() == (out2 / "images" / "00003.png").read_bytes()


def test_dataset_stratification_cells_cover_bands_and_classes(tmp_path):
    n = len(BANDS) * len(DATASET_CLASSES)           # one card per (band, class) cell
    summary = generate_dataset(tmp_path / "s", n=n, seed=1, workers=1, source="synthetic")
    metas = [json.loads(l) for l in (tmp_path / "s" / "manifest.jsonl").read_text().splitlines()]
    cells = {(m["coverage_band"], m["droplet_class"]) for m in metas}
    assert len(cells) == n
    # realized coverage lands in (or next to) the requested band; Poisson noise only
    for m in metas:
        lo, hi = BANDS[m["coverage_band"]]
        assert lo - 6 <= m["true_coverage_pct"] <= hi + 6
    assert summary["mean_abs_diff_true_vs_analytic_pp"] < 3.0


def test_dataset_uses_simulator_when_importable(tmp_path, monkeypatch):
    class FakeSim:
        def simulate(self, x, u, rng, mismatch="none", sample_droplets=False):
            dc = list(VMD)[int(rng.integers(len(VMD)))]
            tgt = float(rng.uniform(1, 79))
            sf = 1.9
            dep = deposition_for_coverage(tgt, VMD[dc], 1.2, sf, None)
            ds = synthetic_droplets(VMD[dc], dep, rng, 1.2, sf, margin_mm=0.0)
            return Outcome(analytic_coverage_pct(ds, 0.0), dep, 0.0, 0.0, 0.0, ds)

    mod = types.ModuleType("coverageai.simulator")
    mod.SpraySimulator = FakeSim
    monkeypatch.setitem(sys.modules, "coverageai.simulator", mod)
    summary = generate_dataset(tmp_path / "sim", n=6, seed=2, workers=1, source="simulator")
    assert summary["sources"]["simulator"] >= 1
    assert sum(summary["sources"].values()) == 6
