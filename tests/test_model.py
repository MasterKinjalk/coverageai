"""WP-10 tests: features, SprayModel protocol, append semantics, timing, sensitivity signs."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from coverageai import catalog
from coverageai.contracts import Conditions, Model, Outcome, Prediction, Setting
from coverageai.model import features as F
from coverageai.model.predictor import SprayModel
from coverageai.scenarios import random_conditions
from coverageai.simulator import SpraySimulator

N_TRAIN = 4000
X0 = Conditions(3.0, 90.0, 0.0, 25.0, 50.0, 19.0, 0.5)


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    xs = random_conditions(n, seed)
    cands = catalog.candidates()
    us = [cands[i] for i in rng.integers(0, len(cands), n)]
    out = SpraySimulator().simulate_arrays(xs, us, np.random.default_rng(seed + 1))
    df = pd.DataFrame([{**x.__dict__, "family": u.family, "size": u.size,
                        "pressure_psi": u.pressure_psi} for x, u in zip(xs, us)])
    for k, v in out.items():
        df[k] = v
    return df


@pytest.fixture(scope="module")
def trained() -> SprayModel:
    return SprayModel(seed=0).fit_frame(_frame(N_TRAIN, 0))


def test_features_match_catalog_derive():
    cands = catalog.candidates()[::17]
    xs = random_conditions(len(cands), 3)
    A = F.featurize(xs, cands)
    assert A.shape == (len(cands), F.N_FEATURES) and np.isfinite(A).all()
    for i, (x, u) in enumerate(zip(xs, cands)):
        d = catalog.derive(x, u)
        assert A[i, F.feature_index("gpa")] == pytest.approx(d.app_rate_gpa, rel=1e-9)
        assert A[i, F.feature_index("u_cross")] == pytest.approx(d.u_cross_ms, abs=1e-9)
        assert A[i, F.feature_index("u_along")] == pytest.approx(d.u_along_ms, abs=1e-9)
        assert A[i, F.feature_index("vmd_um")] == pytest.approx(d.vmd_um)
        assert A[i, F.feature_index("flow_gpm")] == pytest.approx(d.flow_gpm)


def test_features_have_no_raw_angles_and_are_circular():
    assert not any(n in F.FEATURE_NAMES for n in ("wind_dir_deg", "heading_deg", "rel_wind_deg"))
    u = Setting("XR", "04", 40.0)
    a = F.featurize([Conditions(3, 10, 0, 25, 50, 19, 0.5)], [u])
    b = F.featurize([Conditions(3, 370, 360, 25, 50, 19, 0.5)], [u])
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_feature_batch_is_fast():
    cands = catalog.candidates()
    xs = [X0] * len(cands)
    F.featurize(xs, cands)
    t = time.perf_counter()
    for _ in range(20):
        F.featurize(xs, cands)
    assert (time.perf_counter() - t) / 20 < 0.010


def test_protocol_conformance_and_ranges(trained):
    assert isinstance(trained, Model)
    p = trained.predict(X0, Setting("XR", "04", 40.0))
    assert isinstance(p, Prediction)
    assert p.coverage_lo <= p.coverage_pct <= p.coverage_hi
    assert 0.0 <= p.drift_frac <= 1.0 and p.deposition_ul_cm2 >= 0.0


def test_predict_batch_shape_matches_predict_and_timing(trained):
    cands = catalog.candidates()
    xs = [X0] * len(cands)
    batch = trained.predict_batch(xs, cands)
    assert len(batch) == len(cands) and all(isinstance(b, Prediction) for b in batch)
    single = trained.predict(X0, cands[5])
    assert single.coverage_pct == pytest.approx(batch[5].coverage_pct, abs=1e-9)
    ts = []
    for _ in range(15):
        t = time.perf_counter()
        trained.predict_batch(xs, cands)
        ts.append(time.perf_counter() - t)
    # target 20 ms; the small test model is smaller, generous bound for loaded CI machines
    assert np.median(ts) < 0.040
    assert trained.predict_batch([], []) == []


def test_unfitted_raises():
    with pytest.raises(RuntimeError):
        SprayModel().predict(X0, Setting("XR", "04", 40.0))


def test_fit_appends_and_refits():
    m = SprayModel(seed=1, n_estimators=20, quantile_n_estimators=20).fit_frame(_frame(1500, 5))
    n0 = m.n_train
    u = Setting("XR", "04", 40.0)
    before = m.predict(X0, u).coverage_pct
    # new records say coverage is far higher than the model thinks
    recs = [(X0, u, Outcome(before + 40.0, 1.0, 0.1, 10.0, 100.0)) for _ in range(300)]
    m.fit(recs)
    assert m.n_train == n0 + 300
    m.fit(recs[:7])
    assert m.n_train == n0 + 307                      # appended, not replaced
    assert m.predict(X0, u).coverage_pct > before + 10.0   # refit moved the prediction
    m.fit([])
    assert m.n_train == n0 + 307


def test_save_load_roundtrip(trained, tmp_path):
    path = trained.save(tmp_path / "m.joblib")
    m2 = SprayModel.load(path)
    u = Setting("ULD", "04", 40.0)
    assert m2.predict(X0, u) == trained.predict(X0, u)
    assert m2.n_train == trained.n_train


def test_interval_coverage_reasonable(trained):
    df = _frame(2000, 77)
    a = trained.predict_features(F.featurize_frame(df))
    y = df["coverage_pct"].to_numpy()
    cov = ((y >= a["coverage_lo"]) & (y <= a["coverage_hi"])).mean()
    assert 0.65 < cov < 0.95


def _pred(m, xs, us, key):
    return np.array([getattr(p, key) for p in m.predict_batch(xs, us)])


def test_sensitivity_signs(trained):
    u = Setting("XR", "04", 40.0)
    # dC/dGPA > 0: lower travel speed raises GPA at fixed spectrum
    speeds = np.linspace(6.4, 32.2, 15)
    c = _pred(trained, [Conditions(3, 90, 0, 25, 50, float(v), 0.5) for v in speeds], [u] * 15, "coverage_pct")
    assert c[0] > c[-1] + 5.0 and (np.diff(c) <= 1.0).all()
    # ddrift/dU > 0 under crosswind
    ws = np.linspace(0.0, 7.0, 15)
    d = _pred(trained, [Conditions(float(v), 90, 0, 25, 50, 19, 0.5) for v in ws], [u] * 15, "drift_frac")
    assert d[-1] > d[0] + 0.02 and (np.diff(d) >= -0.01).all()
    # ddrift/dH > 0
    hs = np.linspace(0.3, 0.75, 15)
    d = _pred(trained, [Conditions(3, 90, 0, 25, 50, 19, float(h)) for h in hs], [u] * 15, "drift_frac")
    assert d[-1] > d[0] + 0.02 and (np.diff(d) >= -0.01).all()
    # finer class -> more drift (VMD vs drift anti-correlated at one condition)
    cands = catalog.candidates()
    vmd = np.array([catalog.derive(X0, v).vmd_um for v in cands])
    d = _pred(trained, [Conditions(4, 90, 0, 25, 50, 19, 0.5)] * len(cands), cands, "drift_frac")
    from scipy.stats import spearmanr
    assert spearmanr(vmd, d).statistic < -0.5
