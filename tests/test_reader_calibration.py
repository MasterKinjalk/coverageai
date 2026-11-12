"""Card-reader calibration (opt-in, D30): remove a reader's known mean error per coverage band."""
from __future__ import annotations

import copy
import dataclasses
import json

import numpy as np
import pytest

from coverageai import app_logic as al
from coverageai import integration as it
from coverageai.contracts import CardMeasurement

# the shape of the classic reader's measured error: reads sparse cards a little high, dense cards low
BANDS = ((0.0, 15.0, 0.5, 0.7), (15.0, 30.0, 0.9, 0.9), (30.0, 50.0, 0.8, -0.5), (50.0, 1e9, 2.5, -3.3))


def _card(cov: float) -> CardMeasurement:
    return CardMeasurement(cov, 100.0, None, None, None, "acceptable", False, 0.0)


def test_table_edges_sit_between_the_bands_on_the_measured_scale():
    table = it.calibration_table(BANDS)
    assert [b for _e, b in table] == [0.7, 0.9, -0.5, -3.3]
    assert table[0][0] == pytest.approx(15.0 + 0.8) and table[1][0] == pytest.approx(30.0 + 0.2)
    assert table[2][0] == pytest.approx(50.0 - 1.9) and table[3][0] == float("inf")


@pytest.mark.parametrize("true", [5.0, 22.0, 40.0, 58.0, 75.0])
def test_calibration_gives_back_the_true_coverage_inside_a_band(true):
    table = it.calibration_table(BANDS)
    bias = next(b for lo, hi, _sd, b in BANDS if lo <= true < hi)
    assert it.calibrated_coverage(true + bias, table) == pytest.approx(true)
    assert 0.0 <= it.calibrated_coverage(-5.0, table) and it.calibrated_coverage(150.0, table) <= 100.0


def test_wrapper_recomputes_the_class_and_keeps_the_readers_own_methods():
    class OnlyOutcome:                       # like the image-free noisy reader
        colour = "blue"

        def measure_outcome(self, outcome, rng):
            return _card(outcome)

    class OnlyImages:                        # like the image readers
        def measure(self, image, ppm):
            return _card(image)

        def measure_full(self, image, ppm):
            return _card(image), "mask"

        def segment(self, image):
            return "mask", "labels"

    a = it.CalibratedVerifier(OnlyOutcome(), BANDS, "test")
    assert hasattr(a, "measure_outcome") and not hasattr(a, "measure_full") and not hasattr(a, "measure")
    assert a.colour == "blue" and a.calibration_source == "test"
    m = a.measure_outcome(29.5, None)        # reads 29.5: band 15 to 30 reads 0.9 high, so 28.6: Borderline
    assert m.coverage_pct == pytest.approx(28.6) and m.coverage_class == "marginal"
    assert a.measure_outcome(52.0, None).coverage_pct == pytest.approx(55.3)
    b = it.CalibratedVerifier(OnlyImages(), BANDS)
    assert hasattr(b, "measure_full") and not hasattr(b, "measure_outcome")
    full = b.measure_full(52.0, 1.0)
    assert full[0].coverage_pct == pytest.approx(55.3) and full[1] == "mask"
    assert b.measure(10.0, 1.0).coverage_pct == pytest.approx(9.3) and b.segment(0) == ("mask", "labels")
    c = copy.deepcopy(a)
    assert c.table == a.table and c.measure_outcome(52.0, None).coverage_pct == pytest.approx(55.3)


def test_default_stack_is_not_calibrated_and_the_opt_in_one_is():
    plain = al.build_modules_adapter("oracle_nominal", "noisy")
    assert type(plain.verifier).__name__ == "NoisyVerifier"            # default: nothing changes
    cal = al.build_modules_adapter("oracle_nominal", "noisy", calibrate_reader=True)
    assert type(cal.verifier).__name__ == "CalibratedVerifier" and hasattr(cal.verifier, "measure_outcome")
    assert al.build_modules_adapter("oracle_nominal", "perfect", calibrate_reader=True).verifier is None
    # the wrapped reader's mean error is gone, band by band (away from the band edges)
    conds = al.make_conditions("gusty_afternoon", 20005, n_segments=4)
    o = al.run_pass_wrapper(al.build_modules_adapter("oracle_nominal", "perfect"), conds, "open", 20005,
                            "none", 30.0).logs[0].outcome
    for true in (22.0, 40.0, 60.0):
        oo = dataclasses.replace(o, coverage_pct=true)
        raw = np.mean([plain.verifier.measure_outcome(oo, np.random.default_rng(i)).coverage_pct for i in range(3000)])
        fix = np.mean([cal.verifier.measure_outcome(oo, np.random.default_rng(i)).coverage_pct for i in range(3000)])
        bias = plain.verifier.error_params(true)[1]
        assert raw - true == pytest.approx(bias, abs=0.15)
        assert abs(fix - true) < 0.15, (true, fix)


def test_a_calibrated_run_reads_cards_closer_to_the_truth_and_replays_consistently():
    conds = al.make_conditions("gusty_afternoon", 20005)
    kw = dict(fixed_nozzle="auto")
    err = {}
    for name, flag in (("raw", False), ("calibrated", True)):
        mods = al.build_modules_adapter("oracle_nominal", "noisy", calibrate_reader=flag)
        res = al.run_pass_wrapper(mods, conds, "fast", 20005, "med", 45.0, 0.6, **kw)   # dense cards
        err[name] = float(np.mean([lg.meas.coverage_pct - lg.outcome.coverage_pct for lg in res.logs]))
        assert all(lg.meas.coverage_class in ("acceptable", "marginal", "critical") for lg in res.logs)
        assert al.replay_card(res, 3).consistent
    assert err["raw"] < -1.0 and abs(err["calibrated"]) < abs(err["raw"]) / 2


def test_an_image_reader_is_calibrated_only_from_a_table_measured_in_the_loop(tmp_path):
    """D30: the table from the reader's validation cards over-corrects in the loop, so an image
    reader is calibrated only from ``<reader>_loop_calibration.json``; without it nothing changes."""
    assert it.loop_calibration_bands("b0", tmp_path) is None
    assert not it.reader_calibration_available("b0", tmp_path)
    assert it.reader_calibration_available("noisy", tmp_path) and not it.reader_calibration_available("perfect", tmp_path)
    plain = it.build_modules(model="oracle_nominal", verifier="b0", calibrate_reader=True, results_dir=tmp_path)
    assert type(plain.verifier).__name__ == "ClassicalVerifier"

    (tmp_path / "b0_loop_calibration.json").write_text(json.dumps({"reader": "b0", "bands": [
        {"lo": 30.0, "hi": None, "n": 100, "bias_pp": -1.0, "sd_pp": 2.0},
        {"lo": 0.0, "hi": 30.0, "n": 100, "bias_pp": 0.4, "sd_pp": 1.0}]}), encoding="utf8")
    bands, source = it.loop_calibration_bands("b0", tmp_path)
    assert source == "b0_loop_calibration.json" and bands == ((0.0, 30.0, 1.0, 0.4), (30.0, 1e9, 2.0, -1.0))
    assert it.reader_calibration_available("b0", tmp_path)
    cal = it.build_modules(model="oracle_nominal", verifier="b0", calibrate_reader=True, results_dir=tmp_path)
    assert type(cal.verifier).__name__ == "CalibratedVerifier" and hasattr(cal.verifier, "measure_full")
    assert cal.verifier.calibration_source == "b0_loop_calibration.json"
    assert cal.verifier.table == it.calibration_table(bands)
    # still off unless asked for
    off = it.build_modules(model="oracle_nominal", verifier="b0", results_dir=tmp_path)
    assert type(off.verifier).__name__ == "ClassicalVerifier"

    (tmp_path / "b1_loop_calibration.json").write_text("{}", encoding="utf8")      # unreadable = no table
    assert it.loop_calibration_bands("b1", tmp_path) is None


def test_the_validation_card_table_is_kept_only_to_reproduce_the_negative_result():
    if it.noise_bands_from_results("b0") is None:
        pytest.skip("results/b0_synthetic.json has no error bands")
    syn = it.build_modules(model="oracle_nominal", verifier="b0", calibrate_reader="synthetic")
    assert type(syn.verifier).__name__ == "CalibratedVerifier"
    assert syn.verifier.calibration_source == "b0_synthetic.json"
    # for the noisy reader the two modes are the same thing: its own bands
    a = it.build_modules(model="oracle_nominal", verifier="noisy", calibrate_reader=True)
    b = it.build_modules(model="oracle_nominal", verifier="noisy", calibrate_reader="synthetic")
    assert a.verifier.table == b.verifier.table
