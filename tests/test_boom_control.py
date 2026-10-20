import json

import numpy as np
import pytest

from coverageai import catalog
from coverageai.config import spec
from coverageai.contracts import (CardMeasurement, Conditions, Outcome, Prediction, SegmentLog,
                                  Setting, effective_boom_height_m, segment_log_to_json)
from coverageai.mocks import MockModel, mock_candidates, mock_derive
from coverageai.model import features as F
from coverageai.selector import ConstrainedSelector
from coverageai.simulator import SpraySimulator

X = Conditions(wind_speed_ms=3.0, wind_dir_deg=90, heading_deg=0, temp_c=25, rh_pct=50,
               travel_speed_kmh=16, boom_height_m=0.6)
OPTS = [0.30, 0.40, 0.50, 0.60, 0.75]


def test_none_is_identical_to_v1():
    u = Setting("XR", "04", 40.0)
    assert u.boom_height_m is None
    assert u == Setting("XR", "04", 40.0, None, None) and hash(u) == hash(Setting("XR", "04", 40.0))
    assert Setting("XR", "04", 40.0, None, 0.5) != u
    assert effective_boom_height_m(X, u) == 0.6
    assert effective_boom_height_m(X, Setting("XR", "04", 40.0, None, 0.4)) == 0.4
    assert catalog.derive(X, u) == catalog.derive(X, Setting("XR", "04", 40.0, None, 0.4))  # derive ignores height
    sim = SpraySimulator(derive_fn=catalog.derive)
    a = sim.simulate(X, u, np.random.default_rng(1))
    b = sim.simulate(X, Setting("XR", "04", 40.0, None, X.boom_height_m), np.random.default_rng(1))
    assert a.coverage_pct == b.coverage_pct and a.drift_frac == b.drift_frac


def test_lower_height_lower_drift_simulator():
    sim = SpraySimulator(derive_fn=catalog.derive)
    u = Setting("XR", "04", 40.0)
    hi = sim.simulate(X, Setting("XR", "04", 40.0, None, 0.75), np.random.default_rng(2))
    lo = sim.simulate(X, Setting("XR", "04", 40.0, None, 0.30), np.random.default_rng(2))
    assert lo.drift_frac < hi.drift_frac
    assert abs(lo.displacement_mm) < abs(hi.displacement_mm)
    arr = sim.nominal_arrays([X, X], [Setting("XR", "04", 40.0, None, 0.30), Setting("XR", "04", 40.0, None, 0.75)])
    assert arr["drift_frac"][0] < arr["drift_frac"][1]
    # batch path equals condition-height path
    x30 = Conditions(**{**X.__dict__, "boom_height_m": 0.30})
    a = sim.nominal_arrays([X], [Setting("XR", "04", 40.0, None, 0.30)])
    b = sim.nominal_arrays([x30], [u])
    assert np.allclose(a["coverage_pct"], b["coverage_pct"]) and np.allclose(a["drift_frac"], b["drift_frac"])


def test_features_and_mock_use_effective_height():
    u0, u1 = Setting("XR", "04", 40.0), Setting("XR", "04", 40.0, None, 0.4)
    x4 = Conditions(**{**X.__dict__, "boom_height_m": 0.4})
    assert np.allclose(F.featurize([X], [u1]), F.featurize([x4], [u0]))
    assert not np.allclose(F.featurize([X], [u1]), F.featurize([X], [u0]))
    m = MockModel()
    assert m.predict(X, u1).drift_frac < m.predict(X, u0).drift_frac


def test_min_overlap_height_rule():
    spacing = spec()["boom"]["nozzle_spacing_in"] * 0.0254
    for fam, ang in spec()["spray_angle_deg"].items():
        h = catalog.min_boom_height_m(fam)
        assert h == pytest.approx(1.3 * spacing / (2 * np.tan(np.radians(ang) / 2)))
        # the fan at that height spans 1.3x the nozzle spacing
        assert 2 * h * np.tan(np.radians(ang) / 2) == pytest.approx(1.3 * spacing)
    assert catalog.min_boom_height_m("TP65") > catalog.min_boom_height_m("XR")


def test_grid_bounds_and_default_off():
    assert catalog.boom_height_grid(0.6, "XR") == []
    assert catalog.boom_height_grid(0.6, "XR", OPTS, 0.25) == [0.5, 0.75]          # current excluded, band 0.45-0.75
    assert catalog.boom_height_grid(0.6, "TP65", OPTS, 0.25) == [0.75]            # TP65 min 0.518
    assert catalog.boom_height_grid(0.6, "DGX40", OPTS, 0.25) == []               # needs 0.91 m > spec max
    lo, hi = spec()["ranges"]["boom_height_m"]
    assert all(lo <= h <= hi for h in catalog.boom_height_grid(0.5, "ULD", [0.1, 0.35, 0.9], 0.5))


def test_selector_off_matches_v1_and_on_respects_constraints():
    def mk(**kw):
        return ConstrainedSelector(candidates_fn=mock_candidates, derive_fn=mock_derive, **kw)
    model = MockModel()
    off = mk()
    on = mk(boom_height_options_m=OPTS, boom_height_band_frac=0.4)
    both = mk(boom_height_options_m=OPTS, boom_height_band_frac=0.4, speed_options_kmh=[12.0, 20.0, 24.0])
    lo, hi = spec()["ranges"]["boom_height_m"]
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = Conditions(rng.uniform(0, 5), rng.uniform(0, 360), rng.uniform(0, 360), 25, 50,
                       float(rng.uniform(10, 25)), float(rng.uniform(lo, hi)))
        u0, i0 = off.select(x, model, 0.0, 30.0)
        assert u0.boom_height_m is None and i0["n_candidates"] == len(off.candidates)
        for sel in (on, both):
            u1, i1 = sel.select(x, model, 0.0, 30.0)
            assert i1["n_candidates"] >= i0["n_candidates"]
            if not i0["infeasible"]:
                assert not i1["infeasible"]
            for u in sel.candidates_for(x):
                if u.boom_height_m is not None:
                    assert lo <= u.boom_height_m <= hi
                    assert u.boom_height_m >= catalog.min_boom_height_m(u.family) - 1e-9
                    assert abs(u.boom_height_m - x.boom_height_m) <= 0.4 * x.boom_height_m + 1e-9
    n_on, n_both = len(on.candidates_for(X)), len(both.candidates_for(X))
    assert n_both > n_on > len(off.candidates_for(X))
    assert any(u.boom_height_m is not None and u.travel_speed_kmh is not None for u in both.candidates_for(X))


def test_segment_log_json_roundtrip_with_height():
    for u, expect in ((Setting("XR", "04", 40.0, 12.5, 0.4), 0.4), (Setting("XR", "04", 40.0), None)):
        d = mock_derive(X, u)
        log = SegmentLog(
            pass_id="p", seed=0, k=0, policy="selector", x=X, u=u, derived=d,
            pred=Prediction(30, 26, 34, 1.0, 0.1, 5.0), corrected_pred_cov=30.0,
            outcome=Outcome(31, 1.0, 0.1, 5.0, 10.0),
            meas=CardMeasurement(31, None, None, None, None, "acceptable", False, 0.01),
            residual=0.0, bias=0.0, flagged_retreat=False, selector_infeasible=False)
        j = json.loads(segment_log_to_json(log))
        assert j["schema_version"] == "1.2"
        assert j["u"]["boom_height_m"] == expect
        assert Setting(**j["u"]) == u
        # old (1.1) records without the field still load
        old = {k: v for k, v in j["u"].items() if k != "boom_height_m"}
        assert Setting(**old).boom_height_m is None
