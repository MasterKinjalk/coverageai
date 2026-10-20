"""Tests for experiments/eval_verifiers.py (O3 verification evaluation) on tiny generated cards."""
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

from coverageai.vision.classical import ClassicalVerifier
from coverageai.vision.simple_cards import make_card

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
_spec = importlib.util.spec_from_file_location("eval_verifiers", ROOT / "experiments" / "eval_verifiers.py")
EV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EV)


def _cards(n=4):
    rng = np.random.default_rng(3)
    for i, cov in enumerate([6, 12, 25, 40][:n]):
        c = make_card(rng, cov, overlap=False, ellipse=True)
        yield {"id": f"c{i}", "image": c.image, "gt": c.mask, "cov": c.coverage_pct, "ppm": c.px_per_mm,
               "ppm_known": True, "n": float(c.n_stains), "n_cc": float(c.n_stains), "dens": None,
               "dv": [None, None, None], "spread": 1.9}


def test_stats_from_mask_matches_measure():
    c = make_card(np.random.default_rng(1), 15, overlap=False, ellipse=True)
    v = ClassicalVerifier()
    m = v.measure(c.image, c.px_per_mm)
    mask, roi = v.segment(c.image)
    st = v.stats_from_mask(mask & roi, c.px_per_mm, int(roi.sum()))
    assert st["n_stains"] == v.last_info["n_stains"] > 0
    assert abs(st["stains_per_cm2"] - m.stains_per_cm2) / m.stains_per_cm2 < 0.02


def test_evaluate_b0_and_summarize():
    rows = EV.evaluate(_cards(), [EV._b0_reader()], warmup=False)["b0"]
    assert len(rows) == 4 and all(r["runtime_s"] > 0 for r in rows)
    s = EV.summarize(rows)
    assert s["overall"]["n"] == 4 and s["overall"]["mae_pp"] < 3.0
    assert set(s["by_coverage_band"]) == {"0-15", "15-30", "30-50", "50+"}
    assert sum(a.get("n", 0) for a in s["by_coverage_band"].values()) == 4
    assert s["overall"]["runtime_p95_s"] >= s["overall"]["runtime_mean_s"] * 0.5
    # exact-count cards: B0 count error is small
    assert s["overall"]["count_rel_err_median"] < 0.3
    assert s["stain_recall_by_size"]["100-200"]["n_stains"] >= 0


def test_annotation_stats_and_recall():
    sq = np.array([[0.1, 0.1], [0.3, 0.1], [0.3, 0.5], [0.1, 0.5]])
    n, dens, dv = EV._annotation_stats([sq], 100, 200, ppm=10.0, spread=2.0)
    assert n == 1 and dens > 0
    d_expect = 2 * np.sqrt(40 * 40 / np.pi) / 10 * 1000 / 2.0     # area (0.2*200)*(0.4*100) = 1600 px
    assert abs(dv[1] - d_expect) < 1e-6
    gt = np.zeros((50, 50), bool)
    gt[10:20, 10:20] = True
    pred = np.zeros_like(gt)
    pred[10:20, 10:15] = True
    rec = EV.stain_recall(gt, pred, ppm=50.0, spread=1.0)         # 11 px diam -> 220 um
    assert rec["200-400"][0] == 1 and abs(rec["200-400"][1] - 0.5) < 1e-9


def test_real_tiles_pipeline_and_blocked(tmp_path):
    # no real data -> unavailable (the criterion must report blocked, not crash)
    assert not EV.real_cards_available(tmp_path)
    out = EV.criteria_table({"b0": {"renderer": {"blocked": "x"}, "real": {"blocked": "no data"}}})
    assert {r["status"] for r in out if "real" in r["criterion"]} == {"blocked"}
    # fixture: one "real" card with YOLO polygon labels derived from its mask
    c = make_card(np.random.default_rng(5), 15, overlap=False, ellipse=True)
    img_dir, lab_dir = tmp_path / "real" / "image", tmp_path / "real" / "label"
    img_dir.mkdir(parents=True)
    lab_dir.mkdir(parents=True)
    cv2.imwrite(str(img_dir / "cardA.png"), cv2.cvtColor(c.image, cv2.COLOR_RGB2BGR))
    h, w = c.mask.shape
    cnts, _ = cv2.findContours(c.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for ct in cnts:
        if len(ct) >= 3:
            pts = ct.reshape(-1, 2) / np.array([w, h])
            lines.append("0 " + " ".join(f"{v:.5f}" for v in pts.ravel()))
    (lab_dir / "cardA.txt").write_text("\n".join(lines))
    (tmp_path / "real" / "px_per_mm.json").write_text('{"default": %f}' % c.px_per_mm)
    assert EV.real_cards_available(tmp_path)
    tiles = list(EV.dataset_real_tiles(tmp_path, tile=256))
    assert len(tiles) >= 4 and tiles[0]["ppm_known"]
    rows = EV.evaluate(tiles[:3], [EV._b0_reader()], warmup=False)["b0"]
    s = EV.summarize(rows)
    assert s["overall"]["n"] == 3 and s["overall"]["iou_mean"] > 0.3
