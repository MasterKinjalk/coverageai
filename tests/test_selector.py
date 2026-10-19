import time

import numpy as np
import pytest

from coverageai.config import spec
from coverageai.contracts import Conditions, Prediction, Setting
from coverageai.mocks import STUB_CATALOG, MockModel, mock_candidates, mock_derive
from coverageai.selector import ConstrainedSelector, best_fixed_setting

X = Conditions(wind_speed_ms=2.0, wind_dir_deg=90, heading_deg=0, temp_c=25, rh_pct=50,
               travel_speed_kmh=16, boom_height_m=0.5)


def make(**kw):
    return ConstrainedSelector(candidates_fn=mock_candidates, derive_fn=mock_derive, **kw)


def dense_candidates(n_p=20):
    out = []
    for fam, sizes in STUB_CATALOG.items():
        lo, hi = spec()["pressure_range_psi"][fam]
        for size in sizes:
            out += [Setting(fam, size, float(p)) for p in np.linspace(lo, hi, n_p)]
    return out


def feasible_set(sel, model, x, bias, c_req):
    cands = sel.candidates
    preds = model.predict_batch([x] * len(cands), cands)
    cov = np.array([p.coverage_pct for p in preds])
    lo = np.array([p.coverage_lo for p in preds])
    ok = cov + bias - sel.margin(cov, lo) >= c_req
    return cands, preds, ok


def test_feasibility_respected():
    sel, model = make(), MockModel()
    for c_req in (10.0, 20.0, 30.0):
        u, info = sel.select(X, model, 0.0, c_req)
        assert not info["infeasible"]
        p = model.predict(X, u)
        assert p.coverage_pct - max(0.5 * sel.margin_pp, p.coverage_pct - p.coverage_lo) >= c_req - 1e-9
        assert info["n_feasible"] >= 1 and info["n_candidates"] == len(mock_candidates())
        assert len(info["top"]) <= 5 and info["top"][0]["score"] is not None


def test_info_fields_and_wind_gate():
    sel = make()
    _, info = sel.select(X, MockModel(), 0.0, 20.0)
    for k in ("infeasible", "n_candidates", "n_feasible", "top", "wind_limit_exceeded"):
        assert k in info
    assert info["wind_limit_exceeded"] is False
    assert {"family", "size", "pressure_psi", "pred_cov", "corrected_cov", "drift", "gpa", "score"} <= set(info["top"][0])
    windy = Conditions(**{**X.__dict__, "wind_speed_ms": 7.0})
    _, info = sel.select(windy, MockModel(), 0.0, 5.0)
    assert info["wind_limit_exceeded"] is True


def test_raising_creq_never_lowers_chosen_coverage():
    sel, model = make(), MockModel()
    prev = -1.0
    for c_req in np.arange(5, 60, 2.5):
        u, _ = sel.select(X, model, 0.0, float(c_req))
        cov = model.predict(X, u).coverage_pct
        assert cov >= prev - 1e-9
        prev = cov


def test_lambda_extremes():
    model = MockModel()
    c_req = 20.0
    s1, s0 = make(lam=1.0), make(lam=0.0)
    u1, _ = s1.select(X, model, 0.0, c_req)
    u0, _ = s0.select(X, model, 0.0, c_req)
    cands, preds, ok = feasible_set(s1, model, X, 0.0, c_req)
    drift = {c: p.drift_frac for c, p, o in zip(cands, preds, ok) if o}
    vol = {c: mock_derive(X, c).app_rate_gpa for c in drift}
    assert drift[u1] == pytest.approx(min(drift.values()))
    assert vol[u0] == pytest.approx(min(vol.values()))


def test_infeasible_returns_max_coverage_and_flag():
    sel, model = make(), MockModel()
    u, info = sel.select(X, model, 0.0, 1e6)
    assert info["infeasible"] is True and info["n_feasible"] == 0
    cands = sel.candidates
    best = max(p.coverage_pct for p in model.predict_batch([X] * len(cands), cands))
    assert model.predict(X, u).coverage_pct == pytest.approx(best)
    assert info["top"][0]["score"] is None


def test_bias_shifts_feasibility():
    sel, model = make(), MockModel()
    _, _, ok0 = feasible_set(sel, model, X, 0.0, 30.0)
    _, _, okp = feasible_set(sel, model, X, 10.0, 30.0)
    _, _, okn = feasible_set(sel, model, X, -10.0, 30.0)
    assert okp.sum() >= ok0.sum() >= okn.sum() and okp.sum() > okn.sum()
    _, i_neg = sel.select(X, model, -1000.0, 30.0)
    assert i_neg["infeasible"]
    _, i_pos = sel.select(X, model, 1000.0, 30.0)
    assert not i_pos["infeasible"] and i_pos["n_feasible"] == len(sel.candidates)


def test_interval_vs_fixed_margin():
    class Wide(MockModel):
        def predict(self, x, u):
            p = super().predict(x, u)
            return Prediction(p.coverage_pct, p.coverage_pct - 20, p.coverage_pct + 20,
                              p.deposition_ul_cm2, p.drift_frac, p.displacement_mm)
    n_int = make(use_interval_margin=True).select(X, Wide(), 0.0, 20.0)[1]["n_feasible"]
    n_fix = make(use_interval_margin=False, margin_pp=3.0).select(X, Wide(), 0.0, 20.0)[1]["n_feasible"]
    assert n_int < n_fix


def test_zero_range_normalization():
    # single feasible candidate / constant volume must not divide by zero
    sel = ConstrainedSelector(candidates_fn=lambda: [Setting("XR", "04", 40.0)], derive_fn=mock_derive)
    u, info = sel.select(X, MockModel(), 0.0, 0.0)
    assert u.family == "XR" and info["top"][0]["score"] == 0.0


def test_runtime_300_candidates():
    cands = dense_candidates()
    assert len(cands) >= 300
    sel = ConstrainedSelector(candidates_fn=lambda: cands, derive_fn=mock_derive)
    model = MockModel()
    sel.select(X, model, 0.0, 20.0)  # warm up
    t_tot, t_sel = [], []
    for _ in range(10):
        t0 = time.perf_counter()
        _, info = sel.select(X, model, 0.0, 20.0)
        t_tot.append(time.perf_counter() - t0)
        t_sel.append(info["selector_time_s"])
    assert np.median(t_sel) < 0.050, f"selector-only {np.median(t_sel)*1e3:.1f} ms"


def test_best_fixed_setting():
    model = MockModel()
    rng = np.random.default_rng(0)
    scen = [Conditions(float(rng.uniform(0, 4)), 90, 0, 25, 50, float(rng.uniform(8, 20)),
                       float(rng.uniform(0.3, 0.7))) for _ in range(30)]
    cands = mock_candidates()
    u, st = best_fixed_setting(scen, model, 20.0, cands, mock_derive, return_stats=True)
    assert st["success"][st["index"]] == st["success"].max()
    tied = st["success"] == st["success"].max()
    assert st["mean_drift"][st["index"]] == pytest.approx(st["mean_drift"][tied].min())
    assert u == best_fixed_setting(scen, model, 20.0, cands, mock_derive)


def test_best_fixed_accepts_simulator():
    from coverageai.mocks import MockSimulator
    scen = [X, X]
    u = best_fixed_setting(scen, MockSimulator(), 20.0, mock_candidates(), mock_derive)
    assert isinstance(u, Setting)
