import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from coverageai.contracts import Conditions, Prediction, SegmentLog, Setting
from coverageai.e2e import make_conditions, run_all_policies
from coverageai.feedback import BiasFeedback
from coverageai.loop import (default_mock_modules, logs_to_frame, run_pass, run_passes_slow,
                             save_logs, summarize)

C_REQ = 30.0
K = 12


class OffsetModel:
    """Mock model that underpredicts coverage by a constant `offset` pp."""

    def __init__(self, base, offset):
        self.base, self.offset, self.fits = base, offset, []

    def predict(self, x, u):
        p = self.base.predict(x, u)
        return dataclasses.replace(p, coverage_pct=p.coverage_pct - self.offset,
                                   coverage_lo=p.coverage_lo - self.offset,
                                   coverage_hi=p.coverage_hi - self.offset)

    def predict_batch(self, xs, us):
        return [self.predict(x, u) for x, u in zip(xs, us)]

    def fit(self, records):
        self.fits.append(list(records))


def test_run_pass_fields_complete():
    m = default_mock_modules()
    conds = make_conditions(K, 1)
    logs = run_pass(conds, "fast", m, 1, "med", C_REQ, "t")
    assert len(logs) == K
    for k, l in enumerate(logs):
        assert isinstance(l, SegmentLog) and l.k == k and l.policy == "fast" and l.pass_id == "t"
        assert l.x is conds[k]
        assert l.corrected_pred_cov == pytest.approx(l.pred.coverage_pct + l.bias)
        assert l.residual == pytest.approx(l.meas.coverage_pct - l.corrected_pred_cov)
        assert l.meas.runtime_s >= 0
        for key in ("regime", "raw_residual", "bias_after", "t_select_s", "t_sim_s",
                    "t_render_s", "t_measure_s", "sel_n_candidates"):
            assert key in l.extra
    assert logs[0].bias == 0.0           # nothing measured yet


def test_all_policies_run():
    res = run_all_policies(6, seed=3)
    assert set(res) == {"open", "fast", "oracle", "best_fixed", "fast_slow"}
    assert len(res["fast_slow"]) == 12
    assert {l.extra["slow_pass"] for l in res["fast_slow"]} == {0, 1}


def test_open_policy_bias_zero_and_unknown_policy():
    m = default_mock_modules()
    logs = run_pass(make_conditions(5, 0), "open", m, 0, "med", C_REQ, "o")
    assert all(l.bias == 0.0 for l in logs)
    with pytest.raises(ValueError):
        run_pass(make_conditions(2, 0), "nope", m, 0, "med", C_REQ, "o")
    with pytest.raises(ValueError):
        run_pass(make_conditions(2, 0), "best_fixed", m, 0, "med", C_REQ, "o")


def test_best_fixed_uses_setting():
    m = default_mock_modules()
    u = Setting("XR", "04", 40.0)
    logs = run_pass(make_conditions(4, 0), "best_fixed", m, 0, "none", C_REQ, "b", fixed_setting=u)
    assert all(l.u == u for l in logs)


def test_oracle_at_least_open_under_med_mismatch():
    m = default_mock_modules()
    conds = make_conditions(30, 5)
    open_ = run_pass(conds, "open", m, 5, "med", C_REQ, "a", render=False)
    orc = run_pass(conds, "oracle", m, 5, "med", C_REQ, "b", render=False)
    f_open, f_orc = summarize(open_, C_REQ)["frac_met"], summarize(orc, C_REQ)["frac_met"]
    assert f_orc >= f_open
    assert f_orc > f_open or f_orc == 1.0


def test_fast_loop_learns_constant_offset():
    m = default_mock_modules()
    m.model = OffsetModel(m.model, 8.0)         # truth is 8 pp above the model, no noise
    fb = BiasFeedback(per_regime=False)
    logs = run_pass(make_conditions(20, 2), "fast", m, 2, "none", C_REQ, "f", feedback=fb,
                    render=False)
    assert logs[-1].extra["bias_after"] == pytest.approx(8.0, abs=1.0)


def test_perfect_measurement_when_no_verifier():
    m = default_mock_modules()
    m.verifier = None
    logs = run_pass(make_conditions(4, 0), "fast", m, 0, "med", C_REQ, "p")
    assert all(l.meas.coverage_pct == l.outcome.coverage_pct for l in logs)


def test_determinism():
    def run():
        return run_pass(make_conditions(K, 9), "fast", default_mock_modules(), 9, "med", C_REQ, "d")
    a, b = logs_to_frame(run()), logs_to_frame(run())
    pd.testing.assert_frame_equal(a.drop(columns=[c for c in a if c.endswith("_s")]),
                                  b.drop(columns=[c for c in b if c.endswith("_s")]))
    c = logs_to_frame(run_pass(make_conditions(K, 9), "fast", default_mock_modules(), 10, "med",
                               C_REQ, "d"))
    assert not np.allclose(a["out_coverage_pct"], c["out_coverage_pct"])


def test_slow_loop_fits_incrementally_with_measured_coverage():
    """Legacy (midterm) slow loop: model.fit receives only the new records of each pass."""
    m = default_mock_modules()
    m.model = OffsetModel(m.model, 0.0)
    logs = run_passes_slow(make_conditions(5, 0), m, 0, "med", C_REQ, n_passes=3,
                           slow_mode="refit")
    assert len(logs) == 15
    assert len(m.model.fits) == 2 and all(len(f) == 5 for f in m.model.fits)
    x, u, o = m.model.fits[0][0]
    assert o.coverage_pct == logs[0].meas.coverage_pct
    assert o.drift_frac == logs[0].outcome.drift_frac


def test_logs_frame_and_save(tmp_path):
    logs = run_pass(make_conditions(4, 0), "fast", default_mock_modules(), 0, "med", C_REQ, "s")
    df = logs_to_frame(logs)
    assert len(df) == 4 and {"x_wind_speed_ms", "u_family", "out_coverage_pct", "extra_regime"} <= set(df)
    pq, jl = save_logs(logs, tmp_path / "run")
    assert len(pd.read_parquet(pq)) == 4
    assert len(jl.read_text().splitlines()) == 4
    assert json.loads(jl.read_text().splitlines()[0])["schema_version"] == "1.2"


def test_summarize():
    logs = run_pass(make_conditions(K, 0), "open", default_mock_modules(), 0, "med", C_REQ, "s")
    s = summarize(logs, C_REQ)
    cov = np.array([l.outcome.coverage_pct for l in logs])
    assert s["frac_met"] == pytest.approx(np.mean(cov >= C_REQ))
    assert s["mean_shortfall_pp"] == pytest.approx(np.mean(np.maximum(C_REQ - cov, 0)))
    assert s["total_volume_gpa"] == pytest.approx(s["mean_volume_gpa"] * K)
    assert s["n"] == K
