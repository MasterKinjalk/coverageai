"""Fixed application rate (coverageai.rate_control, selector.rate_lock, recommend rate_lock)."""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from coverageai import catalog, rate_control as rc
from coverageai import scenarios
from coverageai.app_logic import NominalSimModel
from coverageai.config import spec
from coverageai.contracts import Conditions, Setting
from coverageai.recommend import RATE_KNOBS, recommend, recommend_for_logs
from coverageai.selector import ConstrainedSelector

NOZ = ("ULD", "04")
GPA = 15.0
TOL = 0.05


@pytest.fixture(scope="module")
def model():
    return NominalSimModel()


@pytest.fixture(scope="module")
def conds():
    return scenarios.random_conditions(6, seed=3)


def _sel(**kw):
    return ConstrainedSelector(nozzle_lock=NOZ, rate_lock=GPA, **kw)


def test_speed_pressure_round_trip_and_derive():
    lo, hi = rc.pressure_bounds(*NOZ)
    x = Conditions(2.0, 90.0, 0.0, 25.0, 50.0, 16.0, 0.5)
    for p in (lo, 0.5 * (lo + hi), hi):
        v = rc.speed_for_rate(*NOZ, p, GPA)

        u = Setting(*NOZ, p, travel_speed_kmh=v)
        assert catalog.derive(x, u).app_rate_gpa == pytest.approx(GPA, rel=1e-6)
        p2 = rc.pressure_for_rate(*NOZ, GPA, v)
        assert p2 == pytest.approx(p, abs=1e-5)
    assert rc.pressure_for_rate(*NOZ, GPA, 0.1) is None
    assert rc.pressure_for_rate(*NOZ, GPA, 1000.0) is None


def test_iso_rate_candidates_within_band():
    c = rc.iso_rate_candidates(*NOZ, GPA, TOL)
    assert c
    x = Conditions(2.0, 90.0, 0.0, 25.0, 50.0, 16.0, 0.5)
    lo_v, hi_v = spec()["ranges"]["travel_speed_kmh"]
    seen_inexact = False
    for u in c:
        dev = abs(rc.rate_deviation(x, u, GPA))
        assert dev <= TOL + 1e-9
        assert lo_v - 1e-9 <= u.travel_speed_kmh <= hi_v + 1e-9
        if dev < 1e-6:
            assert not seen_inexact          # exact ones first
        else:
            seen_inexact = True
    assert any(abs(rc.rate_deviation(x, u, GPA)) < 1e-6 for u in c)
    assert c == rc.iso_rate_candidates(*NOZ, GPA, TOL)      # deterministic
    ps = {u.pressure_psi for u in c}
    assert min(ps) >= rc.pressure_bounds(*NOZ)[0] - 1e-9 and max(ps) <= rc.pressure_bounds(*NOZ)[1] + 1e-9


def test_selector_rate_mode_stays_on_nozzle_and_in_band(model, conds):
    sel = _sel()
    for x in conds:
        u, info = sel.select(x, model, 0.0, 30.0)
        assert (u.family, u.size) == NOZ
        assert abs(rc.rate_deviation(x, u, GPA)) <= TOL + 1e-9
        assert info["rate_mode"] and info["rate_target_gpa"] == GPA
        assert info["rate_tier"] in ("exact", "band", "unreachable")
        assert info["target_unreachable_at_rate"] == (info["rate_tier"] == "unreachable")
        if info["rate_tier"] == "exact":
            assert abs(rc.rate_deviation(x, u, GPA)) < 1e-6
        assert 1 <= len(info["top"]) <= 5


def test_unreachable_stays_in_band_and_max_coverage(model):
    x = Conditions(3.0, 90.0, 0.0, 30.0, 40.0, 16.0, 0.5)
    sel = ConstrainedSelector(nozzle_lock=NOZ, rate_lock=8.0)
    u, info = sel.select(x, model, 0.0, 95.0)
    assert info["rate_tier"] == "unreachable" and info["infeasible"]
    assert info["target_unreachable_at_rate"] is True
    assert abs(rc.rate_deviation(x, u, 8.0)) <= TOL + 1e-9
    cands = sel.candidates_for(x)
    best = max(p.coverage_pct for p in model.predict_batch([x] * len(cands), cands))
    assert model.predict(x, u).coverage_pct == pytest.approx(best)


def test_rate_lock_without_nozzle_lock_raises(model, conds):
    sel = ConstrainedSelector(rate_lock=GPA)
    with pytest.raises(ValueError):
        sel.candidates_for(conds[0])
    with pytest.raises(ValueError):
        sel.select(conds[0], model, 0.0, 30.0)


def test_default_selector_unchanged(model, conds):
    a, b = ConstrainedSelector(), ConstrainedSelector(rate_lock=None, rate_tol_frac=None)
    al, bl = ConstrainedSelector(nozzle_lock=NOZ), ConstrainedSelector(nozzle_lock=NOZ, rate_lock=None)
    for x in conds[:3]:
        assert a.candidates_for(x) == b.candidates_for(x)
        assert al.candidates_for(x) == bl.candidates_for(x)
        ua, ia = a.select(x, model, 0.0, 30.0)
        ub, ib = b.select(x, model, 0.0, 30.0)
        assert ua == ub
        drop = ("model_time_s", "selector_time_s")
        assert {k: v for k, v in ia.items() if k not in drop} == {k: v for k, v in ib.items() if k not in drop}
        assert not {"rate_mode", "rate_tier", "rate_target_gpa", "target_unreachable_at_rate"} & set(ia)


def test_boom_height_variants_in_rate_mode(model, conds):
    sel = _sel(boom_height_options_m=[0.4, 0.5, 0.6, 0.75], speed_options_kmh=[10.0, 20.0])
    x = conds[0]
    cands = sel.candidates_for(x)
    assert len({u.boom_height_m for u in cands}) > 1
    assert all(abs(rc.rate_deviation(x, u, GPA)) <= TOL + 1e-9 for u in cands)


def test_alternative_nozzles(model, conds):
    out = rc.alternative_nozzles(conds[:3], model, 30.0, GPA, exclude=NOZ, top=5)
    assert 0 < len(out) <= 5
    assert all((r["family"], r["size"]) != NOZ for r in out)
    assert all(0.0 <= r["share_meeting"] <= 1.0 for r in out)
    keys = [(-r["share_meeting"], r["mean_drift"]) for r in out]
    assert keys == sorted(keys)
    r = out[0]
    assert r["pressure_range_psi"][0] <= r["pressure_range_psi"][1]
    assert r["speed_range_kmh"][0] <= r["speed_range_kmh"][1]


def test_recommend_rate_mode_holds_rate(model):
    x = Conditions(4.0, 90.0, 0.0, 30.0, 40.0, 16.0, 0.6)
    u = rc.iso_rate_candidates(*NOZ, GPA, TOL, levels=(1.0,))[3]
    rec = recommend(model, x, u, observed_cov=5.0, c_req=30.0, rate_lock=GPA)
    assert rec.status in ("adjust", "combine", "unreachable")
    moves = list(rec.options) + list(rec.combined)
    assert moves
    assert {a.knob for a in moves} <= set(RATE_KNOBS)
    if rec.setting is not None:
        assert abs(rc.rate_deviation(x, rec.setting, GPA)) <= TOL + 1e-9
    for a in moves:
        assert abs(a.volume_change_gpa) <= TOL * GPA + 1e-9
        if a.knob == "pressure_at_rate":
            assert a.speed_proposed is not None and a.rate_gpa == GPA
            assert "rate held at" in a.describe()
    # logs path, joint moves too
    lg = SimpleNamespace(x=x, u=u, meas=SimpleNamespace(coverage_pct=5.0), corrected_pred_cov=5.0)
    r2 = recommend_for_logs(model, [lg], 30.0, rate_lock=GPA)[0]
    if r2.setting is not None:
        assert abs(rc.rate_deviation(x, r2.setting, GPA)) <= TOL + 1e-9
    assert dataclasses.is_dataclass(r2)


def test_unreachable_text_in_rate_mode_does_not_say_larger_nozzle(model):
    """At a fixed rate a larger tip is not necessarily better, so the advice must not say so."""
    x = Conditions(4.0, 90.0, 0.0, 30.0, 40.0, 16.0, 0.6)
    u = rc.iso_rate_candidates(*NOZ, GPA, TOL, levels=(1.0,))[3]
    rec = recommend(model, x, u, observed_cov=1.0, c_req=60.0, rate_lock=GPA)
    assert rec.status == "unreachable" and rec.rate_gpa == GPA
    assert "larger nozzle" not in rec.headline() and "switch nozzle" in rec.headline()
    free = recommend(model, x, u, observed_cov=1.0, c_req=60.0)
    assert free.rate_gpa is None and "larger nozzle" in free.headline()
