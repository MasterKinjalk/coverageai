import json

import numpy as np
import pytest

from coverageai import catalog
from coverageai.config import spec
from coverageai.contracts import (CardMeasurement, Conditions, Outcome, Prediction, SegmentLog,
                                  Setting, effective_speed_kmh, segment_log_to_json)
from coverageai.mocks import MockModel, mock_candidates, mock_derive
from coverageai.model import features as F
from coverageai.selector import ConstrainedSelector

X = Conditions(wind_speed_ms=2.0, wind_dir_deg=90, heading_deg=0, temp_c=25, rh_pct=50,
               travel_speed_kmh=16, boom_height_m=0.5)
OPTS = [8.0, 12.0, 14.0, 16.0, 18.0, 20.0, 28.0]


def test_none_is_identical_to_v1():
    u = Setting("XR", "04", 40.0)
    assert u.travel_speed_kmh is None
    assert u == Setting("XR", "04", 40.0, None) and hash(u) == hash(Setting("XR", "04", 40.0))
    assert {(X, u): 1}[(X, Setting("XR", "04", 40.0))] == 1
    assert Setting("XR", "04", 40.0, 16.0) != u
    assert effective_speed_kmh(X, u) == 16
    assert catalog.derive(X, u) == catalog.derive(X, Setting("XR", "04", 40.0, X.travel_speed_kmh))


def test_derive_gpa_inverse_in_speed():
    base = catalog.derive(X, Setting("XR", "04", 40.0))
    for v in (8.0, 12.0, 24.0):
        d = catalog.derive(X, Setting("XR", "04", 40.0, v))
        assert d.app_rate_gpa == pytest.approx(base.app_rate_gpa * X.travel_speed_kmh / v)
        assert d.flow_gpm == base.flow_gpm and d.vmd_um == base.vmd_um
    assert mock_derive(X, Setting("XR", "04", 40.0, 8.0)).app_rate_gpa == pytest.approx(
        2 * mock_derive(X, Setting("XR", "04", 40.0)).app_rate_gpa)


def test_features_use_effective_speed():
    u0, u1 = Setting("XR", "04", 40.0), Setting("XR", "04", 40.0, 8.0)
    x8 = Conditions(**{**X.__dict__, "travel_speed_kmh": 8.0})
    assert np.allclose(F.featurize([X], [u1]), F.featurize([x8], [u0]))
    assert not np.allclose(F.featurize([X], [u1]), F.featurize([X], [u0]))


def test_speed_grid_bounds_and_default_off():
    assert catalog.speed_grid(16.0) == catalog.speed_grid(16.0, spec()["selector"]["speed_options_kmh"])
    assert catalog.speed_grid(16.0, OPTS, 0.25) == [12.0, 14.0, 18.0, 20.0]   # current excluded
    lo, hi = spec()["ranges"]["travel_speed_kmh"]
    assert catalog.speed_grid(7.0, [1.0, 6.0, 7.5], 5.0) == [7.5]             # clipped to spec range
    assert all(lo <= v <= hi for v in catalog.speed_grid(30.0, [30.0, 33.0, 40.0, 25.0], 1.0))


def test_selector_off_matches_v1_and_on_respects_ranges():
    def mk(**kw):
        return ConstrainedSelector(candidates_fn=mock_candidates, derive_fn=mock_derive, **kw)
    model = MockModel()
    off, on = mk(speed_options_kmh=[]), mk(speed_options_kmh=OPTS, speed_band_frac=0.25)
    rng = np.random.default_rng(0)
    lo, hi = spec()["ranges"]["travel_speed_kmh"]
    for _ in range(25):
        x = Conditions(rng.uniform(0, 5), rng.uniform(0, 360), rng.uniform(0, 360), 25, 50,
                       float(rng.uniform(lo, hi)), 0.5)
        u0, i0 = off.select(x, model, 0.0, 30.0)
        assert u0.travel_speed_kmh is None and i0["n_candidates"] == len(off.candidates)
        u1, i1 = on.select(x, model, 0.0, 30.0)
        if u1.travel_speed_kmh is not None:
            assert lo <= u1.travel_speed_kmh <= hi
            assert abs(u1.travel_speed_kmh - x.travel_speed_kmh) <= 0.25 * x.travel_speed_kmh + 1e-9
        assert i1["n_candidates"] >= i0["n_candidates"]
        if not i0["infeasible"]:   # superset of candidates: still feasible
            assert not i1["infeasible"]


def _log(u):
    d = mock_derive(X, u)
    return SegmentLog(
        pass_id="p", seed=0, k=0, policy="selector", x=X, u=u, derived=d,
        pred=Prediction(30, 26, 34, 1.0, 0.1, 5.0), corrected_pred_cov=30.0,
        outcome=Outcome(31, 1.0, 0.1, 5.0, 10.0),
        meas=CardMeasurement(31, None, None, None, None, "acceptable", False, 0.01),
        residual=0.0, bias=0.0, flagged_retreat=False, selector_infeasible=False)


def test_segment_log_json_roundtrip_with_speed():
    for u, expect in ((Setting("XR", "04", 40.0, 12.5), 12.5), (Setting("XR", "04", 40.0), None)):
        d = json.loads(segment_log_to_json(_log(u)))
        assert d["schema_version"] == "1.2"
        assert d["u"]["travel_speed_kmh"] == expect
        assert Setting(**d["u"]) == u
