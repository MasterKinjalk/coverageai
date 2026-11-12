"""Operator console logic (coverageai.operator_console) and the opt-in ``rate_lock`` of
``app_logic.run_pass_wrapper``. Fast stack: physics model, perfect reader, short fields."""
from __future__ import annotations

import pytest

from coverageai import app_logic as al
from coverageai import catalog
from coverageai import operator_console as oc
from coverageai import rate_control as rc
from coverageai.contracts import Conditions

NOZ = ("XR", "06")
GPA, SPEED = 20.0, 16.0


@pytest.fixture(scope="module")
def mods():
    return al.build_modules_adapter("oracle_nominal", "perfect")


@pytest.fixture(scope="module")
def conds():
    return al.make_conditions("gusty_afternoon", 1, n_segments=8)


@pytest.fixture(scope="module")
def run(mods, conds):
    return oc.run_console(mods, conds, 1, "med", *NOZ, GPA, SPEED, boom_control=True)


def _inner(mods):
    return getattr(mods.selector, "inner", mods.selector)


def _untouched(mods):
    inner = _inner(mods)
    assert inner.rate_lock is None and inner.nozzle_lock is None
    assert inner.boom_height_options_m == [] and inner.speed_options_kmh == []


# ------------------------------------------------------------------ setup

def test_setup_implies_the_starting_pressure():
    s = oc.setup_for(*NOZ, GPA, SPEED)
    assert s.ok and s.problem() is None
    assert s.pressure_psi == pytest.approx(rc.pressure_for_rate(*NOZ, GPA, SPEED))
    assert s.droplet_class == catalog.flow_and_class(*NOZ, s.pressure_psi)[1]
    x = Conditions(2.0, 90.0, 0.0, 25.0, 50.0, 9.0, 0.5)          # field speed is overridden
    assert catalog.derive(x, s.setting).app_rate_gpa == pytest.approx(GPA, rel=1e-6)
    assert s.setting.travel_speed_kmh == SPEED and s.setting.boom_height_m is None
    lo, hi = s.rate_band
    assert lo == pytest.approx(GPA * (1 - s.tol_frac)) and hi == pytest.approx(GPA * (1 + s.tol_frac))
    a, b = s.speed_range_kmh
    assert a <= SPEED <= b
    assert s.needed_psi == pytest.approx(s.pressure_psi, rel=0.05)     # sqrt law vs the catalog


def test_setup_out_of_the_pressure_range_says_which_speeds_work():
    s = oc.setup_for("XR", "04", 15.0, 16.0)        # would need about 66 psi, XR stops at 60
    assert not s.ok and s.pressure_psi is None and s.setting is None
    assert s.needed_psi > rc.pressure_bounds("XR", "04")[1]
    a, b = s.speed_range_kmh
    assert rc.pressure_for_rate("XR", "04", 15.0, 0.5 * (a + b)) is not None
    txt = s.problem()
    assert "psi" in txt and "km/h" in txt and "15 gal/acre" in txt


def test_setup_with_no_speed_that_holds_the_rate():
    s = oc.setup_for("XR", "01", 40.0, 30.0)
    assert not s.ok and s.speed_range_kmh is None and s.n_candidates == 0
    assert "cannot apply" in s.problem()


# ------------------------------------------------------------------ the two runs

def test_two_paired_runs(run, conds, mods):
    s = run.setup
    held, con = run.held, run.console
    assert held.conds == con.conds == list(conds) and held.seed == con.seed == 1
    assert held.policy == "best_fixed" and con.policy == "fast"
    assert all(lg.u == s.setting for lg in held.logs)                 # the starting setting, held
    assert {(lg.u.family, lg.u.size) for lg in con.logs} == {NOZ}     # nozzle stays on the boom
    for lg in con.logs:                                               # rate held inside the band
        assert abs(rc.rate_deviation(lg.x, lg.u, GPA)) <= s.tol_frac + 1e-9
    assert con.meta["rate"] == {"target_gpa": GPA, "tol_frac": pytest.approx(s.tol_frac)}
    assert con.meta["nozzle"] == {"mode": "fixed", "family": NOZ[0], "size": NOZ[1]}
    assert "rate" not in held.meta
    assert run.c_min == oc.MIN_COVERAGE == 30.0 and held.c_req == con.c_req == 30.0
    _untouched(mods)                                                  # rule 4: restored


def test_console_advice_is_rate_mode_and_feeds_the_replay(run):
    from coverageai.field_anim import field_animation_payload
    assert run.console.meta["recommendations"] is run.recs
    assert al.recommendations(run.console) is run.recs                # the replay reads these
    for rec in run.recs:
        for a in list(rec.options) + list(rec.combined):
            assert a.knob in ("pressure_at_rate", "boom_height")
    pay = field_animation_payload(run.console, run.c_min, cards=False)
    assert pay["n"] == run.n and pay["target"] == 30.0
    adv = " ".join(" ".join(s_["advice"]) for s_ in pay["sections"])
    assert "Speed up" not in adv and "Slow down" not in adv          # speed alone would change the rate


def test_summary_uses_compute_kpis_and_counts_tiers(run):
    sm = oc.summary(run)
    for key, res in ((oc.HELD, run.held), (oc.CONSOLE, run.console)):
        kp = al.compute_kpis(res.logs, 30.0)
        assert {k: sm[key][k] for k in kp} == kp
        r = sm[key]["rate"]
        assert r["min"] <= r["mean"] <= r["max"] and r["in_band"] and r["n_out_of_band"] == 0
    assert sm[oc.HELD]["rate"]["max_dev_frac"] == pytest.approx(0.0, abs=1e-9)
    t = oc.tiers(run.console)
    assert all(v in oc.TIERS for v in t)
    assert sm[oc.CONSOLE]["tiers"] == {k: t.count(k) for k in oc.TIERS}
    assert sm[oc.CONSOLE]["n_unreachable"] == t.count("unreachable")
    assert sum(sm[oc.CONSOLE]["tiers"].values()) == run.n


def test_tables(run):
    df = oc.section_frame(run)
    assert len(df) == run.n and list(df["section"]) == list(range(1, run.n + 1))
    assert set(df["tier"]) <= set(oc.TIERS)
    assert df["rate_gpa"].between(GPA * (1 - run.setup.tol_frac) - 1e-9, GPA * (1 + run.setup.tol_frac) + 1e-9).all()
    adv = oc.advice_frame(run)
    assert len(adv) == sum(r.status != "ok" for r in run.recs)
    assert (adv["card_cov"] < 30.0).all() and (adv["advice"] != "").all()
    k = oc.advice_counts(run)
    assert k["n"] == run.n and k["n_short"] == len(adv) == k["n_fixable"] + k["n_unfixable"]


def test_nozzle_ranking(run):
    assert len(run.ranking) == sum(1 for f in catalog.families() for z in catalog.sizes(f)
                                   if rc.iso_rate_candidates(f, z, GPA, levels=(1.0,)))
    keys = [oc.rank_key(r) for r in run.ranking]
    assert keys == sorted(keys)
    own = run.own_rank()
    assert own is not None and (own["family"], own["size"]) == NOZ
    alts = run.alternatives(5)
    assert len(alts) == 5 and all((r["family"], r["size"]) != NOZ for r in alts)
    df = oc.alternatives_frame(run, 5)
    assert list(df["better"]) == [oc.rank_key(r) < oc.rank_key(own) for r in alts]


def test_rank_key_reaches_first_then_drift_or_coverage():
    a = {"family": "A", "size": "1", "share_meeting": 0.5, "mean_drift": 0.2, "mean_cov": 31.0}
    b = {"family": "B", "size": "1", "share_meeting": 0.5, "mean_drift": 0.1, "mean_cov": 30.0}
    c = {"family": "C", "size": "1", "share_meeting": 0.0, "mean_drift": 0.01, "mean_cov": 20.0}
    d = {"family": "D", "size": "1", "share_meeting": 0.0, "mean_drift": 0.10, "mean_cov": 28.0}
    assert sorted([c, a, d, b], key=oc.rank_key) == [b, a, d, c]


def test_unreachable_nozzle_at_this_rate_and_a_better_one(mods):
    """ULD 04 at 15 gal/acre in a steady 3 m/s crosswind: very coarse droplets cannot reach 30 %
    at this rate; the console says so and the ranking has nozzles predicted to do better."""
    xs = [Conditions(3.0, 90.0, 0.0, 25.0, 50.0, 16.0, 0.5)] * 6
    run = oc.run_console(mods, xs, 3, "none", "ULD", "04", 15.0, 16.0, boom_control=True)
    sm = oc.summary(run)
    assert sm[oc.CONSOLE]["n_unreachable"] > 0
    assert any(r.status != "ok" for r in run.recs)
    better = oc.alternatives_frame(run, 5)
    assert better["better"].any()
    _untouched(mods)


def test_invalid_setup_raises_before_any_run(mods, conds):
    with pytest.raises(ValueError, match="No pressure"):
        oc.run_console(mods, conds, 1, "med", "XR", "04", 15.0, 16.0)
    _untouched(mods)


# ------------------------------------------------------------------ run_pass_wrapper(rate_lock=)

@pytest.mark.parametrize("policy,nozzle", [("fast", None), ("fast", "free"), ("fast", "auto"),
                                           ("best_fixed", "XR-06")])
def test_rate_lock_needs_a_named_nozzle(mods, conds, policy, nozzle):
    with pytest.raises(ValueError, match="named nozzle"):
        al.run_pass_wrapper(mods, conds, policy, 1, "med", 30.0, fixed_nozzle=nozzle, rate_lock=GPA)
    _untouched(mods)


def test_rate_lock_rejects_a_rate_the_nozzle_cannot_hold(mods, conds):
    with pytest.raises(ValueError, match="travel-speed range"):
        al.run_pass_wrapper(mods, conds, "fast", 1, "med", 30.0, fixed_nozzle="XR-01", rate_lock=40.0)
    with pytest.raises(ValueError, match="positive"):
        al.run_pass_wrapper(mods, conds, "fast", 1, "med", 30.0, fixed_nozzle="XR-06", rate_lock=0.0)
    _untouched(mods)


def test_rate_lock_is_restored_when_the_run_fails(mods, conds):
    def stop(done, total):
        if done >= 2:
            raise RuntimeError("stop")
    with pytest.raises(RuntimeError):
        al.run_pass_wrapper(mods, conds, "fast", 1, "med", 30.0, fixed_nozzle=NOZ, rate_lock=GPA,
                            boom_control=True, progress=stop)
    _untouched(mods)


def test_rate_lock_also_runs_with_the_learning_strategy(mods, conds):
    res = al.run_pass_wrapper(mods, conds[:4], "fast_slow", 1, "med", 30.0, fixed_nozzle="XR-06",
                              rate_lock=GPA, n_learn=1, same_field=True)
    assert all(abs(rc.rate_deviation(lg.x, lg.u, GPA)) <= rc.tolerance_frac() + 1e-9 for lg in res.logs)
    _untouched(mods)


def test_without_rate_lock_nothing_is_recorded(mods, conds):
    res = al.run_pass_wrapper(mods, conds, "fast", 1, "med", 30.0, fixed_nozzle=NOZ)
    assert "rate" not in res.meta
    assert all(lg.u.travel_speed_kmh is None for lg in res.logs)
