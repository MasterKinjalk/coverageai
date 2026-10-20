"""Redesigned slow loop: residual-correction model on a frozen base, refit between passes."""
import dataclasses

import numpy as np
import pytest

from coverageai.contracts import Outcome
from coverageai.e2e import make_conditions
from coverageai.feedback import GLOBAL_KEY, BiasFeedback, ResidualCorrectedModel
from coverageai.loop import default_mock_modules, run_pass, run_passes_slow
from coverageai.mocks import MockModel, mock_candidates, mock_derive

C_REQ = 30.0


class OffsetModel:
    """Mock model that underpredicts coverage by a constant `offset` pp; records fit calls."""

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


def _records(model, n, offset_fn, noise_sd=0.0, seed=0):
    """(x, u, Outcome) with measured coverage = base prediction + offset_fn(x, u) + noise."""
    rng = np.random.default_rng(seed)
    xs = make_conditions(n, seed)
    cands = mock_candidates()
    out = []
    for i, x in enumerate(xs):
        u = cands[i % len(cands)]
        c = model.predict(x, u).coverage_pct + offset_fn(x, u) + noise_sd * rng.standard_normal()
        out.append((x, u, Outcome(c, 1.0, 0.01, 1.0, 10.0, None)))
    return out


def test_learns_constant_offset_and_keeps_interval_width():
    base = MockModel()
    rc = ResidualCorrectedModel(base, mock_derive, per_regime=False)
    x = make_conditions(1, 0)[0]
    u = max(mock_candidates(), key=lambda c: base.predict(x, c).coverage_pct)  # no clipping at 0
    assert base.predict(x, u).coverage_lo > 15.0
    assert rc.predict(x, u) == base.predict(x, u)            # no records: identical to base
    rc.fit(_records(base, 60, lambda x, u: -6.0, noise_sd=1.0))
    assert rc.global_corr == pytest.approx(-6.0, abs=0.5)
    p, b = rc.predict(x, u), base.predict(x, u)
    d = rc.global_corr
    assert p.coverage_pct == pytest.approx(b.coverage_pct + d)
    assert p.coverage_pct - p.coverage_lo == pytest.approx(b.coverage_pct - b.coverage_lo)
    assert p.coverage_hi - p.coverage_pct == pytest.approx(b.coverage_hi - b.coverage_pct)
    assert p.drift_frac == b.drift_frac
    assert rc.regime(x, u) == GLOBAL_KEY


def test_shrinks_toward_zero_with_few_noisy_records():
    base = MockModel()
    rc = ResidualCorrectedModel(base, mock_derive, per_regime=False, prior_sd_pp=2.0)
    recs = _records(base, 3, lambda x, u: 10.0, noise_sd=5.0, seed=1)
    rc.fit(recs)
    raw_mean = np.mean([o.coverage_pct - base.predict(x, u).coverage_pct for x, u, o in recs])
    assert 0.0 < rc.global_corr < raw_mean


def test_refits_do_not_compound():
    """Residuals are taken against the frozen base, so fitting A then B equals fitting A+B."""
    base = MockModel()
    a = _records(base, 30, lambda x, u: -4.0, noise_sd=2.0, seed=2)
    b = _records(base, 30, lambda x, u: -4.0, noise_sd=2.0, seed=3)
    r1 = ResidualCorrectedModel(base, mock_derive)
    r1.fit(a)
    r1.fit(b)
    r2 = ResidualCorrectedModel(base, mock_derive)
    r2.fit(a + b)
    assert r1.global_corr == pytest.approx(r2.global_corr)
    assert r1.regime_corr == pytest.approx(r2.regime_corr)
    assert len(r1.history) == 2 and r1.history[-1]["n_records"] == 60


def test_per_regime_offsets_are_pooled_when_regimes_agree_and_separated_when_not():
    base = MockModel()
    fb_regime = BiasFeedback().regime
    # same offset everywhere, noise only: regime deviations shrink fully to the global offset
    rc = ResidualCorrectedModel(base, mock_derive)
    rc.fit(_records(base, 200, lambda x, u: -3.0, noise_sd=3.0, seed=4))
    if len(rc.regime_corr) >= 2:
        spread = max(rc.regime_corr.values()) - min(rc.regime_corr.values())
        assert spread < 1.5
    # strongly regime-dependent offset: per-regime corrections follow it
    def off(x, u):
        return -8.0 if fb_regime(x, mock_derive(x, u)).endswith("fine") else 2.0
    rc2 = ResidualCorrectedModel(base, mock_derive)
    rc2.fit(_records(base, 300, off, noise_sd=1.0, seed=5))
    fine = [v for k, v in rc2.regime_corr.items() if k.endswith("fine")]
    coarse = [v for k, v in rc2.regime_corr.items() if k.endswith("coarse")]
    assert fine and coarse
    assert max(fine) < -5.0 and min(coarse) > 0.0


def test_run_passes_slow_residual_mode_learns_offset_and_restores_model():
    m = default_mock_modules()
    m.verifier = None                         # perfect measurement
    base = OffsetModel(m.model, 8.0)          # truth ~ 8 pp above the model (minus mismatch)
    m.model = base
    conds = make_conditions(20, 2)
    logs = run_passes_slow(conds, m, 2, "none", C_REQ, n_passes=3, render=False)
    assert m.model is base and base.fits == []        # base frozen, wrapper removed
    n = len(conds)
    p1, p3 = logs[:n], logs[2 * n:]
    assert all(l.extra["slow_corr"] == 0.0 for l in p1)
    assert all(l.extra["slow_corr"] > 4.0 for l in p3)
    # first segment of every pass: fast bias reset (bias 0), but pass 3 already corrected
    assert p3[0].bias == 0.0
    raw1 = np.mean([abs(l.extra["raw_residual"]) for l in p1])
    raw3 = np.mean([abs(l.extra["raw_residual"]) for l in p3])
    assert raw3 < raw1
    assert {l.extra["slow_pass"] for l in logs} == {0, 1, 2}
    assert logs[-1].extra["slow_n_records"] == 2 * n


def test_run_passes_slow_first_pass_equals_fast_loop():
    """Before any record the slow loop is exactly the fast loop."""
    m = default_mock_modules()
    m.verifier = None
    conds = make_conditions(8, 6)
    fast = run_pass(conds, "fast", m, 6, "med", C_REQ, "f", render=False)
    slow = run_passes_slow(conds, m, 6, "med", C_REQ, n_passes=2, render=False)
    assert [l.u for l in fast] == [l.u for l in slow[:8]]
    assert [l.outcome.coverage_pct for l in fast] == [l.outcome.coverage_pct for l in slow[:8]]


def test_run_passes_slow_rejects_unknown_mode():
    with pytest.raises(ValueError):
        run_passes_slow(make_conditions(2, 0), default_mock_modules(), 0, "med", C_REQ,
                        n_passes=1, slow_mode="nope")
