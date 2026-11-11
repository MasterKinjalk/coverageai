"""Operator recommendations (coverageai.recommend, decision D20)."""
from __future__ import annotations

import dataclasses

import pytest

from coverageai import app_logic as al
from coverageai import catalog, integration, loop
from coverageai.config import spec
from coverageai.contracts import effective_boom_height_m, effective_speed_kmh
from coverageai.recommend import (KNOBS, current_value, knob_values, recommend, recommend_for_logs,
                                  with_value)


@pytest.fixture(scope="module")
def mods():
    return integration.build_modules("oracle_nominal", "perfect")


@pytest.fixture(scope="module")
def conds():
    return al.make_conditions("gusty_afternoon", 1, n_segments=12)


@pytest.fixture(scope="module")
def case(mods, conds):
    """A (conditions, setting) pair at an interior setting so every control can move both ways."""
    x = conds[0]
    u = next(s for s in mods.selector.candidates if (s.family, s.size) == ("ULD", "06")
             and s.pressure_psi == 75.0)
    return x, u


def _legal(knob, x, u):
    s = spec()
    if knob == "pressure":
        lo, hi = catalog.pressure_range(u.family, u.size)
        r = s["pressure_range_psi"][u.family]
        return max(lo, r[0]), min(hi, r[1])
    if knob == "speed":
        return tuple(s["ranges"]["travel_speed_kmh"])
    lo, hi = s["ranges"]["boom_height_m"]
    return max(lo, catalog.min_boom_height_m(u.family)), hi


def test_observed_above_target_is_ok(mods, case):
    x, u = case
    r = recommend(mods.model, x, u, observed_cov=55.0, c_req=30.0)
    assert r.status == "ok" and r.options == () and r.combined == () and r.setting is None
    assert r.shortfall_pp == 0.0 and r.best is None
    assert "no change" in r.headline().lower()


def test_below_target_adjustments_are_legal_minimal_and_physical(mods, case):
    x, u = case
    pred = mods.model.predict(x, u).coverage_pct
    target = pred + 4.0
    r = recommend(mods.model, x, u, observed_cov=pred, c_req=target)
    assert r.status == "adjust" and r.options and r.best is not None
    assert r.shortfall_pp == pytest.approx(4.0)
    for a in r.options:
        lo, hi = _legal(a.knob, x, u)
        assert lo - 1e-6 <= a.proposed <= hi + 1e-6
        assert a.current == pytest.approx(current_value(a.knob, x, u))
        assert a.gain_pp > 0 and a.expected_cov == pytest.approx(pred + a.gain_pp)
        assert a.reaches_target == (a.expected_cov >= target)
        # the claimed gain is what the model predicts at the proposed value
        p_new = mods.model.predict(x, with_value(a.knob, u, a.proposed)).coverage_pct
        assert p_new - pred == pytest.approx(a.gain_pp)
        if a.reaches_target:
            # smallest move for this control: brute force over every alternative value
            vals = knob_values(a.knob, x, u)
            preds = mods.model.predict_batch([x] * len(vals), [with_value(a.knob, u, v) for v in vals])
            closer = [(v, p.coverage_pct - pred) for v, p in zip(vals, preds)
                      if abs(v - a.current) < abs(a.proposed - a.current) - 1e-9]
            assert all(g < 4.0 - 1e-9 for _, g in closer), (a.knob, closer)
    ok = [a for a in r.options if a.reaches_target]
    assert r.options[0] is ok[0]
    assert ok[0].move_frac == min(a.move_frac for a in ok)
    # the recommended setting differs from the current one in exactly the best control
    assert r.setting != u
    assert current_value(ok[0].knob, x, r.setting) == pytest.approx(ok[0].proposed)


def test_speed_option_slows_down_and_adds_volume(mods, case):
    x, u = case
    pred = mods.model.predict(x, u).coverage_pct
    r = recommend(mods.model, x, u, observed_cov=pred, c_req=pred + 3.0)
    sp = [a for a in r.options if a.knob == "speed"]
    assert sp, "speed should help coverage in the physics model"
    assert sp[0].delta < 0 and sp[0].volume_change_gpa > 0       # GPA ~ 1/speed
    assert sp[0].describe().startswith("Slow down")


def test_describe_and_headline_contain_numbers(mods, case):
    x, u = case
    pred = mods.model.predict(x, u).coverage_pct
    r = recommend(mods.model, x, u, observed_cov=pred, c_req=pred + 3.0)
    a = r.best
    nd = {"pressure": 0, "speed": 1, "boom_height": 2}[a.knob]
    d = a.describe()
    assert f"{a.proposed:.{nd}f}" in d and f"{a.current:.{nd}f}" in d
    assert f"{a.expected_cov:.0f}%" in d and f"{a.gain_pp:+.1f} pp" in d
    assert r.headline() == d and r.lines()[0] == d


def _check_joint(r, x, u):
    assert r.combined and r.combined_cov is not None and r.setting is not None
    assert len(r.combined) >= 2                       # no single control was enough
    for a in r.combined:
        lo, hi = _legal(a.knob, x, u)
        assert lo - 1e-6 <= a.proposed <= hi + 1e-6
        assert current_value(a.knob, x, r.setting) == pytest.approx(a.proposed)
    assert f"{r.combined_cov:.0f}%" in r.headline()


def test_single_control_enough_for_moderate_shortfall(mods, case):
    x, u = case
    r = recommend(mods.model, x, u, observed_cov=1.0, c_req=30.0)      # speed alone gets there
    assert r.status == "adjust" and r.combined == ()


def test_large_shortfall_needs_combined_move(mods, case):
    x, u = case
    r = recommend(mods.model, x, u, observed_cov=1.0, c_req=50.0)
    assert r.status == "combine" and r.best is None
    _check_joint(r, x, u)
    assert r.combined_cov >= 50.0
    assert all(not o.reaches_target for o in r.options)


def test_unreachable_target_reports_best_achievable(mods, case):
    x, u = case
    r = recommend(mods.model, x, u, observed_cov=1.0, c_req=99.0)
    assert r.status == "unreachable"
    _check_joint(r, x, u)
    assert 1.0 < r.combined_cov < 99.0
    assert "not reachable" in r.headline()
    # the unreachable-best is at least as good as every single control
    assert all(r.combined_cov >= o.expected_cov - 1e-9 for o in r.options)


def test_nozzle_not_in_catalog_is_unavailable_not_an_exception(mods, case):
    x, u = case
    m = loop.default_mock_modules()
    u_fake = dataclasses.replace(u, family="MOCK", size="99")
    r = recommend(m.model, x, u_fake, observed_cov=5.0, c_req=30.0, derive_fn=m.derive_fn)
    assert r.status == "unavailable" and r.options == () and r.combined == ()
    assert r.setting is None and "catalog" in r.headline()


class _Counting:
    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    def predict_batch(self, xs, us):
        self.calls += 1
        return self.inner.predict_batch(xs, us)


def test_recommend_for_logs_batches_model_calls(mods, conds):
    logs = loop.run_pass(list(conds), "fast", mods, 1, "med", 30.0, "r", render=False)
    hi = max(lg.meas.coverage_pct for lg in logs) + 5.0     # every section short
    for n in (3, 12):
        cm = _Counting(mods.model)
        recs = recommend_for_logs(cm, logs[:n], hi)
        assert len(recs) == n and all(r.status != "ok" for r in recs)
        assert 1 <= cm.calls <= 2
    cm = _Counting(mods.model)
    recs = recommend_for_logs(cm, logs, 0.0)                # nothing short: no model call at all
    assert cm.calls == 0 and all(r.status == "ok" for r in recs)


def test_recommend_for_logs_margin_and_observed(mods, conds):
    logs = loop.run_pass(list(conds), "fast", mods, 1, "med", 30.0, "r", render=False)
    lg = logs[0]
    cov = lg.meas.coverage_pct
    assert recommend_for_logs(mods.model, [lg], cov - 0.5)[0].status == "ok"
    r = recommend_for_logs(mods.model, [lg], cov - 0.5, margin_pp=2.0)[0]
    assert r.status != "ok" and r.target_cov == pytest.approx(cov + 1.5)
    rp = recommend_for_logs(mods.model, [lg], lg.corrected_pred_cov + 1.0, observed="predicted")[0]
    assert rp.observed_cov == pytest.approx(lg.corrected_pred_cov) and rp.status != "ok"


def test_knob_values_inside_legal_range_and_exclude_current(case):
    x, u = case
    for kn in KNOBS:
        vals = knob_values(kn, x, u)
        lo, hi = _legal(kn, x, u)
        cur = current_value(kn, x, u)
        assert vals and all(lo - 1e-6 <= v <= hi + 1e-6 for v in vals)
        assert all(abs(v - cur) > 1e-6 for v in vals)
    assert current_value("speed", x, u) == effective_speed_kmh(x, u)
    assert current_value("boom_height", x, u) == effective_boom_height_m(x, u)
