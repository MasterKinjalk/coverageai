from pathlib import Path

import cv2
import numpy as np
import pytest

from coverageai.vision.real_data import (
    CardSample,
    load_cards,
    parse_yolo_polygons,
    rasterize_polygons,
    tiles,
)

REAL_ROOT = Path(__file__).resolve().parents[1] / "data" / "external" / "simoes2025"


def test_parse_skips_bbox_rows():
    polys = parse_yolo_polygons("0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n0 0.5 0.5 0.2 0.2\n\n")
    assert len(polys) == 1 and polys[0].shape == (4, 2)


def test_rasterize_square_and_union():
    # 10x20 image; one square covering x in [0.2, 0.6], y in [0.2, 0.6] of (W=20, H=10)
    sq = np.array([[0.2, 0.2], [0.6, 0.2], [0.6, 0.6], [0.2, 0.6]])
    m = rasterize_polygons([sq], height=10, width=20)
    assert m.shape == (10, 20) and m.dtype == bool
    # fillPoly includes the boundary: x 4..12 (9 px), y 2..6 (5 px)
    assert m.sum() == 9 * 5
    assert m[4, 8] and not m[0, 0] and not m[9, 19]
    # overlapping duplicate does not change the union
    assert rasterize_polygons([sq, sq], 10, 20).sum() == m.sum()


def _make_fixture(root: Path, subset_dir: str = "") -> None:
    img_dir = root / subset_dir / "image"
    lab_dir = root / subset_dir / "label"
    img_dir.mkdir(parents=True)
    lab_dir.mkdir(parents=True)
    img = np.full((26, 76, 3), (255, 230, 0), np.uint8)  # BGR irrelevant for fixture
    cv2.imwrite(str(img_dir / "0.png"), img)
    cv2.imwrite(str(img_dir / "1.png"), img)
    (lab_dir / "0.txt").write_text(
        "0 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4\n0 0.6 0.5 0.8 0.5 0.7 0.9\n")
    # image 1 has no label file -> zero droplets


def test_loader_on_fixture(tmp_path):
    _make_fixture(tmp_path / "Dataset WSP")
    cards = load_cards(tmp_path, "synthetic")
    assert [c.name for c in cards] == ["0", "1"]
    c0, c1 = cards
    assert isinstance(c0, CardSample)
    assert c0.image.shape == (26, 76, 3) and c0.image.dtype == np.uint8
    assert c0.mask.shape == (26, 76) and c0.mask.dtype == bool
    assert c0.n_droplets == 2 and c0.mask.sum() > 0
    assert c0.px_per_mm == pytest.approx(1.0)  # 76 px / 76 mm
    assert c1.n_droplets == 0 and not c1.mask.any()
    # RGB conversion: fixture BGR (255,230,0) -> RGB (0,230,255)
    assert tuple(c0.image[0, 0]) == (0, 230, 255)
    assert load_cards(tmp_path, "real") == []
    assert len(load_cards(tmp_path, "synthetic", limit=1)) == 1


def test_real_subset_detected_by_path(tmp_path):
    _make_fixture(tmp_path / "real_cards")
    real = load_cards(tmp_path, "real")
    assert len(real) == 2 and real[0].px_per_mm is None
    assert load_cards(tmp_path, "synthetic") == []


def test_missing_root_returns_empty(tmp_path):
    assert load_cards(tmp_path / "nope", "synthetic") == []


def test_tiles_cover_every_pixel():
    h, w = 600, 1100
    s = CardSample(np.zeros((h, w, 3), np.uint8), np.zeros((h, w), bool), 0, None, "t")
    cover = np.zeros((h, w), int)
    for r, c, img, m in tiles(s, size=512, stride=512):
        assert img.shape[:2] == m.shape
        cover[r:r + m.shape[0], c:c + m.shape[1]] += 1
    assert (cover == 1).all()
    full = list(tiles(s, 512, 512, drop_partial=True))
    assert len(full) == 2 and all(t[2].shape[:2] == (512, 512) for t in full)


@pytest.mark.skipif(not REAL_ROOT.exists(), reason="Zenodo data not downloaded")
def test_downloaded_synthetic_cards():
    cards = load_cards(REAL_ROOT, "synthetic", limit=3)
    if not cards:
        pytest.skip("synthetic subset not extracted")
    for c in cards:
        assert c.mask.any() and c.n_droplets > 100
        assert abs(c.image.shape[1] / c.image.shape[0] - 76 / 26) < 0.02
