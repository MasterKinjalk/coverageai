"""Fair oracle (knows the hidden mismatch, not the per-card spread factor) vs card-aware oracle."""
import numpy as np
import pytest

from coverageai import catalog, scenarios
from coverageai.e2e import make_conditions
from coverageai.integration import NominalPhysicsModel, build_modules
from coverageai.loop import (POLICIES, FairOracleModel, OracleModel, default_mock_modules,
                             make_oracle, run_pass, segment_rng, summarize)
from coverageai.simulator import SpraySimulator

C_REQ = 30.0
SEED = 11


@pytest.fixture(scope="module")
def conds():
    return scenarios.generate(scenarios.get("gusty_afternoon"), 2)


@pytest.fixture(scope="module")
def sim():
    return SpraySimulator(derive_fn=catalog.derive)


def test_policies_and_factory(sim):
    assert "oracle" in POLICIES and "oracle_card" in POLICIES
    assert isinstance(make_oracle("oracle", sim, "med", 1), FairOracleModel)
    assert isinstance(make_oracle("oracle_card", sim, "med", 1), OracleModel)
    with pytest.raises(ValueError):
        make_oracle("open", sim, "med", 1)


def test_no_mismatch_equals_nominal_physics_for_any_seed(sim, conds):
    """With no mismatch the fair oracle knows nothing a nominal model does not: its answer
    is the mean-spread-factor physics with the 10/90 % spread-factor interval, whatever the
    seed (it does NOT see the card's spread factor)."""
    us = catalog.candidates()[::40]
    xs = [conds[0]] * len(us)
    nominal = NominalPhysicsModel(sim).predict_batch(xs, us)
    for seed in (0, 1, 2):
        fo = FairOracleModel(sim, "none", seed)
        fo.k = 3
        p = fo.predict_batch(xs, us)
        for a, b in zip(p, nominal):
            assert a.coverage_pct == pytest.approx(b.coverage_pct, abs=1e-9)
            assert a.coverage_lo == pytest.approx(b.coverage_lo, abs=1e-9)
            assert a.coverage_hi == pytest.approx(b.coverage_hi, abs=1e-9)
    # ... while the card-aware oracle changes with the seed (it sees the spread factor)
    c = [OracleModel(sim, "none", s).predict_batch(xs[:1], us[:1])[0].coverage_pct for s in (0, 1, 2)]
    assert len(set(np.round(c, 9))) == 3


def test_sees_hidden_mismatch(sim, conds):
    """Under mismatch the fair oracle's prediction differs from nominal physics (it knows
    gust, canopy and wear), and the realized outcome lies inside the spread-factor range
    evaluated at that hidden state."""
    u = catalog.candidates()[200]
    nominal = NominalPhysicsModel(sim).predict(conds[0], u).coverage_pct
    sim.begin_pass(SEED, "high")
    try:
        fo = FairOracleModel(sim, "high", SEED)
        diffs = []
        for k in range(10):
            fo.k = k
            p = fo.predict(conds[k], u)
            out = sim.simulate(conds[k], u, segment_rng(SEED, k), "high")
            assert p.coverage_lo <= p.coverage_pct <= p.coverage_hi
            # full spread-factor range [a, b] brackets the outcome
            fo.interval_q = 0.0
            p0 = fo.predict(conds[k], u)
            fo.interval_q = 0.1
            assert p0.coverage_lo - 1e-9 <= out.coverage_pct <= p0.coverage_hi + 1e-9
            if k == 0:
                diffs.append(abs(p.coverage_pct - nominal))
        assert diffs[0] > 0.5
    finally:
        sim.end_pass()


def test_lower_bound_is_the_10pct_quantile_of_card_noise(sim):
    """Coverage is increasing in the uniform spread factor, so P(true < lo) = 0.1 exactly."""
    xs = make_conditions(300, 4)
    u = catalog.candidates()[150]
    sim.begin_pass(SEED, "med")
    below = []
    try:
        fo = FairOracleModel(sim, "med", SEED)
        for k, x in enumerate(xs):
            fo.k = k
            p = fo.predict(x, u)
            out = sim.simulate(x, u, segment_rng(SEED, k), "med")
            if p.coverage_hi - p.coverage_lo > 1e-6:      # skip saturated (0 or 100 %) cases
                below.append(out.coverage_pct < p.coverage_lo)
    finally:
        sim.end_pass()
    assert len(below) > 200
    assert 0.05 <= np.mean(below) <= 0.16


def test_policies_run_and_card_oracle_dominates(conds):
    m = build_modules("oracle_nominal", "perfect")
    fair = run_pass(conds[:12], "oracle", m, SEED, "high", C_REQ, "f", render=False)
    card = run_pass(conds[:12], "oracle_card", m, SEED, "high", C_REQ, "c", render=False)
    assert {l.extra["regime"] for l in fair} == {"oracle"}
    assert {l.extra["regime"] for l in card} == {"oracle_card"}
    sf, sc = summarize(fair, C_REQ), summarize(card, C_REQ)
    assert sc["frac_met"] == 1.0
    assert sf["frac_met"] >= 0.75
    # needing a margin for the card noise costs volume
    assert sf["mean_volume_gpa"] >= sc["mean_volume_gpa"]


def test_risk_level_override(sim, conds):
    """A more conservative fair oracle (lower q) selects at least as much volume."""
    assert make_oracle("oracle", sim, "med", 1, 0.05).interval_q == 0.05
    assert make_oracle("oracle", sim, "med", 1).interval_q == 0.1
    m = build_modules("oracle_nominal", "perfect")
    v = {q: summarize(run_pass(conds[:8], "oracle", m, SEED, "med", C_REQ, "q", render=False,
                               oracle_interval_q=q), C_REQ)["mean_volume_gpa"] for q in (0.02, 0.3)}
    assert v[0.02] >= v[0.3]


def test_fallback_for_simulators_without_hidden_state():
    m = default_mock_modules()
    fo = FairOracleModel(m.simulator, "med", 0)
    assert fo.fallback
    logs = run_pass(make_conditions(4, 0), "oracle", m, 0, "med", C_REQ, "m", render=False)
    assert len(logs) == 4
