"""Physics-first hybrid (model/hybrid.py): protocol, calibration, misspecification recovery,
persistence, integration branch."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coverageai import catalog
from coverageai.contracts import Conditions, Model, Outcome, Prediction, Setting
from coverageai.model import features as F
from coverageai.model.hybrid import (PhysicsBase, PhysicsOnlyModel, PhysicsResidualModel,
                                     resolve_misspec)
from coverageai.scenarios import random_conditions
from coverageai.simulator import SpraySimulator

N_TRAIN, N_TEST = 5000, 3000
X0 = Conditions(3.0, 90.0, 0.0, 25.0, 50.0, 19.0, 0.5)
TARGETS = ["coverage_pct", "deposition_ul_cm2", "drift_frac", "displacement_mm"]


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
def data():
    return _frame(N_TRAIN, 11), _frame(N_TEST, 12)


@pytest.fixture(scope="module")
def exact(data):
    return PhysicsResidualModel(seed=0, n_jobs=2).fit_frame(data[0])


@pytest.fixture(scope="module")
def misspec(data):
    return PhysicsResidualModel(misspec="optimistic", seed=0, n_jobs=2).fit_frame(data[0])


def _cov(model, df):
    Fm = F.featurize_frame(df)
    return model.predict_from(Fm, model.base.matrix_frame(df))


def test_protocol_conformance(exact):
    assert isinstance(exact, Model)
    assert isinstance(PhysicsOnlyModel(), Model)
    cands = catalog.candidates()[:40]
    preds = exact.predict_batch([X0] * len(cands), cands)
    assert len(preds) == len(cands) and all(isinstance(p, Prediction) for p in preds)
    for p in preds:
        assert 0.0 <= p.coverage_lo <= p.coverage_pct <= p.coverage_hi <= 100.0
        assert 0.0 <= p.drift_frac <= 1.0 and p.deposition_ul_cm2 >= 0 and p.displacement_mm >= 0
    one = exact.predict(X0, cands[3])
    assert one.coverage_pct == pytest.approx(preds[3].coverage_pct)
    assert exact.predict_batch([], []) == []


def test_exact_base_equals_nominal_physics():
    from coverageai.integration import NominalPhysicsModel
    cands = catalog.candidates()[::25]
    a = PhysicsOnlyModel(derive_fn=catalog.derive).predict_batch([X0] * len(cands), cands)
    b = NominalPhysicsModel().predict_batch([X0] * len(cands), cands)
    for p, q in zip(a, b):
        assert p.coverage_pct == pytest.approx(q.coverage_pct)
        assert p.coverage_lo == pytest.approx(q.coverage_lo)
        assert p.drift_frac == pytest.approx(q.drift_frac)


def test_misspec_presets_and_validation():
    assert resolve_misspec(None) == {} and resolve_misspec("none") == {}
    assert resolve_misspec("optimistic")["drift_threshold_scale"] == 0.75
    with pytest.raises(ValueError):
        resolve_misspec("nope")
    with pytest.raises(ValueError):
        resolve_misspec({"bogus": 1.0})
    assert PhysicsBase().exact and not PhysicsBase("optimistic").exact


def test_interval_holds_target_share_on_held_out(data, exact, misspec):
    test = data[1]
    y = test["coverage_pct"].to_numpy()
    for m in (exact, misspec):
        a = _cov(m, test)
        below = float((y < a["coverage_lo"]).mean())
        inside = float(((y >= a["coverage_lo"]) & (y <= a["coverage_hi"])).mean())
        assert 0.05 <= below <= 0.15, below          # one-sided target 10 %
        assert 0.72 <= inside <= 0.88, inside        # nominal 80 % band


def test_residual_repairs_misspecified_physics(data, misspec):
    test = data[1]
    Y = test[TARGETS].to_numpy(float)
    P = misspec.base.matrix_frame(test)
    a = _cov(misspec, test)
    phys_mae = np.abs(P[:, 0] - Y[:, 0]).mean()
    hyb_mae = np.abs(a["coverage_pct"] - Y[:, 0]).mean()
    assert hyb_mae < phys_mae - 0.5, (hyb_mae, phys_mae)
    assert np.abs(a["drift_frac"] - Y[:, 2]).mean() < 0.5 * np.abs(P[:, 4] - Y[:, 2]).mean()
    # misspecified physics alone is badly calibrated in the dangerous (optimistic) direction
    assert (Y[:, 0] < P[:, 1]).mean() > 0.25
    assert (Y[:, 0] < a["coverage_lo"]).mean() < 0.15


def test_save_load_round_trip(tmp_path, misspec):
    p = misspec.save(tmp_path / "h.joblib")
    m2 = PhysicsResidualModel.load(p)
    assert m2.misspec == misspec.misspec
    cands = catalog.candidates()[::30]
    a = misspec.predict_batch([X0] * len(cands), cands)
    b = m2.predict_batch([X0] * len(cands), cands)
    for p1, p2 in zip(a, b):
        assert p1 == pytest.approx(p2)


def test_fit_appends_records(exact):
    import copy
    m = copy.deepcopy(exact)
    n0 = m.n_train
    xs = random_conditions(30, 5)
    us = catalog.candidates()[:30]
    recs = [(x, u, Outcome(25.0, 1.0, 0.05, 20.0, 100.0, None)) for x, u in zip(xs, us)]
    m.fit(recs)
    assert m.n_train == n0 + 30
    m.fit([])
    assert m.n_train == n0 + 30
    assert m.predict(X0, us[0]).coverage_pct >= 0


def test_unfitted_raises():
    with pytest.raises(RuntimeError):
        PhysicsResidualModel().predict(X0, catalog.candidates()[0])


def test_build_modules_hybrid_branch(tmp_path, exact):
    from coverageai import integration
    assert "hybrid" in integration.MODEL_CHOICES and "hybrid_misspec" in integration.MODEL_CHOICES
    p = exact.save(tmp_path / "hybrid_v1.joblib")
    mods = integration.build_modules("hybrid", "perfect", model_path=p)
    assert isinstance(mods.model, PhysicsResidualModel)
    assert mods.model.predict(X0, catalog.candidates()[0]).coverage_pct >= 0
