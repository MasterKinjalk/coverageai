"""WP-13 tests: U-Net verifier. Skipped cleanly when the `vision` extra is not installed."""
import json

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

import torch  # noqa: E402

from coverageai.contracts import CardMeasurement, Verifier  # noqa: E402
from coverageai.mocks import MockRenderer  # noqa: E402
from coverageai.contracts import Outcome  # noqa: E402
from coverageai.vision import unet  # noqa: E402
from coverageai.vision.toy_data import generate_card, make_toy_dataset  # noqa: E402


@pytest.fixture(scope="module")
def toy_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("toy")
    # small cards (192x384 px at full 600 dpi) keep the CPU tests fast
    return make_toy_dataset(root, n_train=20, n_val=6, n_test=4, seed=1, h=192, w=384, max_cov=40.0)


@pytest.fixture(scope="module")
def trained(toy_dir, tmp_path_factory):
    """A 2-minute-capped CPU overfit run shared by several tests."""
    d = tmp_path_factory.mktemp("run")
    res = unet.train(toy_dir, d / "toy.pt", epochs=15, batch=8, crop=128, max_minutes=2.0, device="cpu",
                     seed=0, lr=2e-3, patience=100, crops_per_card=8, log_path=d / "log.csv", verbose=False)
    res["dir"] = d
    return res


def test_model_forward_shape():
    model = unet.build_model(pretrained=False).eval()
    with torch.inference_mode():
        out = model(torch.randn(2, 3, 64, 96))
    assert out.shape == (2, 1, 64, 96)


def test_conv_bn_fusion_preserves_outputs():
    model = unet.build_model(pretrained=False)
    for m in model.modules():                         # non-trivial BN statistics
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.normal_(0, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.normal_(0, 0.3)
    model.eval()
    x = torch.randn(1, 3, 96, 128)
    with torch.inference_mode():
        before = model(x)
    unet.fuse_conv_bn(model)
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    with torch.inference_mode():
        after = model(x)
    assert torch.allclose(before, after, atol=1e-4)


def test_predict_proba_tiling_matches_single_pass():
    model = unet.build_model(pretrained=False).eval()
    unet.fuse_conv_bn(model)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (96, 224, 3), dtype=np.uint8)
    whole = unet.predict_proba(model, img, tile=(96, 224), overlap=32)
    tiled = unet.predict_proba(model, img, tile=(96, 128), overlap=32)
    assert whole.shape == tiled.shape == (96, 224)
    assert np.abs(whole - tiled).max() < 0.15           # only tile-border effects


def test_pairing_flat_and_folder_layouts(tmp_path):
    img = np.full((40, 50, 3), 200, np.uint8)
    msk = np.zeros((40, 50), np.uint8)
    msk[5:10, 5:10] = 255
    flat = tmp_path / "flat"
    flat.mkdir()
    for s in ("a", "b"):
        cv2.imwrite(str(flat / f"{s}_img.png"), img)
        cv2.imwrite(str(flat / f"{s}_mask.png"), msk)
    (flat / "a.json").write_text(json.dumps({"coverage_pct": 1.0, "px_per_mm": 20.0}))
    cv2.imwrite(str(flat / "orphan_img.png"), img)          # no mask: ignored
    pairs = unet.find_pairs(flat)
    assert [p.stem for p in pairs] == ["a", "b"]
    assert pairs[0].meta is not None and pairs[1].meta is None
    assert unet.meta_get(unet.read_meta(pairs[0].meta), ["px_per_mm"]) == 20.0
    im, m = unet.load_pair(pairs[0])
    assert im.shape == (40, 50, 3) and set(np.unique(m)) == {0, 1}

    nested = tmp_path / "nested"
    (nested / "images").mkdir(parents=True)
    (nested / "masks").mkdir()
    cv2.imwrite(str(nested / "images" / "c1.png"), img)
    cv2.imwrite(str(nested / "masks" / "c1.png"), msk)
    assert [p.stem for p in unet.find_pairs(nested)] == ["c1"]

    custom = tmp_path / "custom"
    custom.mkdir()
    cv2.imwrite(str(custom / "x.rgb.png"), img)
    cv2.imwrite(str(custom / "x.gt.png"), msk)
    ps = unet.PairSpec(img_suffix=".rgb", mask_suffix=".gt")
    assert [p.stem for p in unet.find_pairs(custom, ps)] == ["x"]

    with pytest.raises(FileNotFoundError):
        unet.find_pairs(tmp_path / "nested" / "images")


def test_toy_pairing_and_dataset(toy_dir):
    pairs = unet.split_pairs(toy_dir, "train")
    assert len(pairs) == 20
    ds = unet.CardDataset(pairs, crop=64, crops_per_card=2, train=True)
    assert len(ds) == 40
    x, y = ds[3]
    assert x.shape == (3, 64, 64) and y.shape == (1, 64, 64)
    assert set(np.unique(y.numpy())) <= {0.0, 1.0}
    dv = unet.CardDataset(unet.split_pairs(toy_dir, "val"), crop=64, crops_per_card=1, train=False)
    assert torch.equal(dv[2][0], dv[2][0])                   # validation crops are deterministic


def test_sample_crop_pads_small_images():
    rng = np.random.default_rng(0)
    img = np.zeros((20, 30, 3), np.uint8)
    msk = np.zeros((20, 30), np.uint8)
    for aug in (False, True):
        i, m = unet.sample_crop(img, msk, rng, 64, aug)
        assert i.shape == (64, 64, 3) and m.shape == (64, 64)


def test_stain_stats_two_disks():
    mask = np.zeros((200, 300), bool)
    cv2.circle(mask.view(np.uint8), (60, 100), 20, 1, -1)
    cv2.circle(mask.view(np.uint8), (200, 100), 10, 1, -1)
    ppm = 20.0
    n, d = unet.stain_stats(mask, ppm, spread_factor=2.0)
    assert n == 2
    # stain diameters 41 and 21 px -> 2.05 and 1.05 mm -> droplets 1025 and 525 um
    assert sorted(np.round(d, -1)) == pytest.approx([530, 1030], abs=60)
    assert unet.volume_percentiles(d) == [None] * 3          # too few stains for percentiles
    big = np.linspace(100, 500, 50)
    dv = unet.volume_percentiles(big)
    assert dv[0] < dv[1] < dv[2]


def test_cpu_overfit_reaches_dice(trained):
    assert trained["best_val_dice"] > 0.8, trained
    assert trained["wall_s"] < 150
    log = (trained["dir"] / "log.csv").read_text().splitlines()
    assert log[0].startswith("epoch,step,lr,train_loss,val_loss,val_dice")
    assert len(log) == trained["epochs_run"] + 1


def test_verifier_measure_valid(trained, toy_dir):
    ver = unet.UNetVerifier(trained["out"], tile=128, overlap=32)
    assert isinstance(ver, Verifier)
    p = unet.split_pairs(toy_dir, "test")[0]
    img, gt = unet.load_pair(p)
    meas = ver.measure(img, 23.62)
    assert isinstance(meas, CardMeasurement)
    assert 0.0 <= meas.coverage_pct <= 100.0
    assert meas.coverage_class in ("acceptable", "marginal", "critical")
    assert meas.runtime_s > 0 and isinstance(meas.saturated, bool)
    assert meas.stains_per_cm2 is not None and meas.stains_per_cm2 >= 0
    for v in (meas.dv01_um, meas.dv05_um, meas.dv09_um):
        assert v is None or v > 0
    if None not in (meas.dv01_um, meas.dv09_um):
        assert meas.dv01_um <= meas.dv05_um <= meas.dv09_um
    # the trained toy model should be within a few pp of the truth on a held-out card
    assert abs(meas.coverage_pct - 100 * gt.mean()) < 8.0


def test_verifier_on_mock_renderer_card(trained):
    ver = unet.UNetVerifier(trained["out"], tile=256, overlap=32)
    out = Outcome(coverage_pct=20.0, deposition_ul_cm2=1.0, drift_frac=0.1, displacement_mm=0.0, stains_per_cm2=10.0)
    img, mask = MockRenderer().render(out, np.random.default_rng(0))
    meas = ver.measure(img, 600 / 25.4)
    assert 0.0 <= meas.coverage_pct <= 100.0 and meas.runtime_s > 0


def test_evaluate_writes_outputs(trained, toy_dir, tmp_path):
    res = unet.evaluate(toy_dir, "test", trained["out"], tmp_path / "b1.json", tmp_path / "parity.png",
                        max_cards=2, verbose=False)
    assert (tmp_path / "parity.png").stat().st_size > 0
    saved = json.loads((tmp_path / "b1.json").read_text())
    assert saved["overall"]["n"] == 2 and "coverage_mae_pp" in saved["overall"]
    assert set(saved["by_band"]) == {"0-10", "10-20", "20-30", "30-45", "45-60", "60-100"}
    assert res["overall"]["iou"] >= 0.0


def test_splits_json_layout(tmp_path):
    """WP-03 layout: images/, masks/, meta/ plus splits.json with integer ids."""
    for d in ("images", "masks", "meta"):
        (tmp_path / d).mkdir()
    img = np.full((40, 50, 3), 200, np.uint8)
    msk = np.zeros((40, 50), np.uint8)
    for i in range(6):
        cv2.imwrite(str(tmp_path / "images" / f"{i:05d}.png"), img)
        cv2.imwrite(str(tmp_path / "masks" / f"{i:05d}.png"), msk)
        (tmp_path / "meta" / f"{i:05d}.json").write_text(json.dumps({"px_per_mm": 22.0, "true_coverage_pct": 1.0}))
    (tmp_path / "splits.json").write_text(json.dumps({"train": [0, 2, 3], "val": [1], "test": [4, 5]}))
    assert [p.stem for p in unet.split_pairs(tmp_path, "train")] == ["00000", "00002", "00003"]
    test = unet.split_pairs(tmp_path, "test")
    assert [p.stem for p in test] == ["00004", "00005"]
    assert unet.meta_get(unet.read_meta(test[0].meta), ["px_per_mm"]) == 22.0


def test_hull_card_roi_finds_card_on_gray_background():
    img = np.full((200, 400, 3), 128, np.uint8)
    img[30:170, 50:350] = (244, 216, 64)             # yellow card
    img[60:90, 100:140] = (25, 45, 130)              # a stain on the card
    roi = unet.hull_card_roi(img)
    assert roi is not None
    assert abs(int(roi.sum()) - 140 * 300) / (140 * 300) < 0.02
    assert roi[75, 120] and not roi[5, 5]
