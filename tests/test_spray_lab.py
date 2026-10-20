"""Spray lab: fate bookkeeping vs the simulator, particle payload, and a UI smoke test."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from coverageai import spray_lab as sl
from coverageai.contracts import Setting
from coverageai.physics import drift as drift_mod
from coverageai.simulator import SpraySimulator

FINE = Setting("XR", "04", 40.0)
COARSE = Setting("ULD", "04", 40.0)


def cond(wind=3.0, rel=90.0, temp=25.0, rh=50.0, speed=12.0, h=0.5):
    return sl.make_conditions(wind, rel, temp, rh, speed, h)


def lab(u=FINE, x=None, **kw):
    kw.setdefault("render_card", False)
    return sl.run_lab(u, x or cond(), seed=3, **kw)


def test_fate_fractions_sum_to_one():
    for u in (FINE, COARSE):
        for x in (cond(0.0), cond(5.0, temp=35.0, rh=20.0)):
            r = lab(u, x)
            assert sum(r.fate.values()) == pytest.approx(1.0, abs=1e-9)
            assert sum(r.particle_fate.values()) == pytest.approx(1.0, abs=1e-9)
            assert all(v >= 0 for v in r.fate.values())
            cols = r.bins[["landed", "blown", "evaporated"]].sum(axis=1)
            assert np.allclose(cols, r.bins["volume_frac"])


@pytest.mark.parametrize("wind,temp,rh", [(0.0, 25.0, 50.0), (3.0, 25.0, 50.0),
                                          (6.0, 35.0, 20.0), (2.0, 15.0, 90.0)])
def test_drift_matches_simulator(wind, temp, rh):
    x = cond(wind, temp=temp, rh=rh)
    r = lab(FINE, x)
    o = SpraySimulator().simulate(x, FINE, np.random.default_rng(0), "none")
    lost = r.fate["blown"] + r.fate["evaporated"]
    assert lost == pytest.approx(o.drift_frac, abs=1e-6)
    assert 100 * r.outcome.drift_frac == pytest.approx(100 * o.drift_frac, abs=1e-9)
    # the animated particle sample tells the same story
    p_lost = r.particle_fate["blown"] + r.particle_fate["evaporated"]
    assert p_lost == pytest.approx(o.drift_frac, abs=0.02)
    assert r.particle_fate["blown"] == pytest.approx(r.fate["blown"], abs=0.02)


def test_more_crosswind_more_drift():
    d = [lab(FINE, cond(w)).outcome.drift_frac for w in (0.0, 2.0, 4.0, 6.0)]
    assert all(b > a for a, b in zip(d, d[1:])), d
    blown = [lab(FINE, cond(w)).fate["blown"] for w in (0.0, 3.0, 6.0)]
    assert blown[0] == pytest.approx(0.0, abs=1e-9) and blown[1] < blown[2]


def test_headwind_does_not_blow_off_target():
    r = lab(FINE, cond(5.0, rel=180.0))
    assert r.fate["blown"] == pytest.approx(0.0, abs=1e-9)
    assert r.particles["U"] == pytest.approx(0.0, abs=1e-9)


def test_finer_nozzle_more_drift():
    x = cond(4.0)
    assert lab(FINE, x).outcome.drift_frac > lab(COARSE, x).outcome.drift_frac
    assert lab(FINE, x).fate["blown"] > lab(COARSE, x).fate["blown"]


def test_hot_dry_more_evaporation():
    wet = lab(FINE, cond(1.0, temp=15.0, rh=90.0)).fate["evaporated"]
    dry = lab(FINE, cond(1.0, temp=38.0, rh=15.0)).fate["evaporated"]
    assert dry > wet


def _all_finite(v):
    if isinstance(v, dict):
        return all(_all_finite(x) for x in v.values())
    if isinstance(v, list):
        return all(_all_finite(x) for x in v)
    if isinstance(v, float):
        return math.isfinite(v)
    return True


def test_payload_json_serializable_and_finite():
    for u, x in ((FINE, cond(6.0, temp=38.0, rh=12.0)), (COARSE, cond(0.0, rh=100.0))):
        P = lab(u, x).particles
        s = json.dumps(P, allow_nan=False)
        assert len(s) < 400_000
        assert _all_finite(P)
        n = P["n"]
        assert 600 <= n <= 1500
        for k in ("t0", "noz", "fan", "d0", "vt", "tau", "mode", "tS", "zS", "beta", "tEnd",
                  "fate", "r3", "w", "xLand"):
            assert len(P[k]) == n, k
        assert min(P["t0"]) >= 0 and max(P["t0"]) <= P["emit"]
        assert set(P["fate"]) <= {0, 1, 2}
        assert sum(P["w"]) == pytest.approx(1.0, abs=0.01)


def test_particles_include_large_droplets():
    P = lab(FINE).particles
    d = np.array(P["d0"])
    assert (d > 250).sum() >= 50          # big droplets are visible
    assert (d < 120).sum() >= 50          # and so are drift-prone small ones


def test_kinematics_match_simulator_physics():
    """Per-particle kinematics reproduce physics.drift.fall_with_exit_velocity."""
    p = sl._phys()
    d0 = np.geomspace(20, 900, 60)
    for temp, rh, U in ((25.0, 50.0, 3.0), (38.0, 12.0, 6.0), (15.0, 95.0, 0.0)):
        lam = float(sl.evaporation_lambda(temp, rh, p["evaporation_lambda_coeff"]))
        v0 = float(drift_mod.exit_velocity(40.0, p["exit_velocity_coeff"], p["water_density_kg_m3"]))
        k = sl.particle_kinematics(d0, 0.5, v0, lam, U)
        d_f, t_f, alive, x_c, _ = drift_mod.fall_with_exit_velocity(
            d0[None], np.array([[0.5]]), np.array([[v0]]), np.array([[lam]]),
            np.array([[U]]), np.array([[0.0]]))
        assert np.array_equal(k["alive"], alive[0])
        a = alive[0]
        assert np.allclose(k["t_end"][a], t_f[0][a], rtol=1e-9)
        assert np.allclose(k["x_cross_m"][a], x_c[0][a], rtol=1e-9, atol=1e-12)
        assert np.allclose(k["d_final_um"][a], d_f[0][a], rtol=1e-9)


def test_trajectories_land_where_predicted():
    r = lab(FINE, cond(4.0, temp=32.0, rh=25.0))
    P = r.particles
    for i in range(0, P["n"], 7):
        x, h, d = sl.particle_state(P, i, P["tEnd"][i])
        if P["fate"][i] == 2:
            assert d < 0.05 * P["d0"][i] + 1.5      # fully evaporated
            assert h >= -1e-6
        else:
            assert h == pytest.approx(0.0, abs=2e-3 * P["H"]), i
            assert x == pytest.approx(P["xLand"][i], abs=2e-3 + 1e-3 * abs(P["xLand"][i]))
            assert d == pytest.approx(P["d0"][i] * P["r3"][i] ** (1 / 3), rel=0.02, abs=0.5)
        # starts at the nozzle
        x0, h0, d00 = sl.particle_state(P, i, 0.0)
        assert h0 == pytest.approx(P["H"]) and x0 == pytest.approx(P["nozX"][P["noz"][i]])
        # falls monotonically
        ts = np.linspace(0, P["tEnd"][i], 25)
        hs = [sl.particle_state(P, i, t)[1] for t in ts]
        assert all(b <= a + 1e-9 for a, b in zip(hs, hs[1:]))


def test_zero_wind_symmetric_landing():
    P = lab(FINE, cond(0.0, rh=95.0, temp=15.0)).particles
    assert P["U"] == 0.0
    noz_x = np.array(P["nozX"])
    assert noz_x.sum() == pytest.approx(0.0, abs=1e-9)
    xl = np.array(P["xLand"])
    fan = np.array(P["fan"])
    off = xl - noz_x[np.array(P["noz"])]
    alive = np.array(P["fate"]) != 2
    # without wind every droplet lands exactly on its fan ray: no sideways push
    assert np.allclose(off[alive], fan[alive] * P["H"], atol=1e-3)
    assert 1 not in P["fate"]
    # the fan is symmetric about each nozzle
    assert abs(off[alive].mean()) < 0.03
    assert abs(np.mean(np.sign(off[alive]))) < 0.1
    assert P["xmin"] == pytest.approx(-P["xmax"])


def test_wind_direction_mapping():
    from coverageai.catalog import derive
    lr = derive(cond(3.0, rel=sl.WIND_DIRECTIONS["Crosswind, blowing left → right"]), FINE)
    rl = derive(cond(3.0, rel=sl.WIND_DIRECTIONS["Crosswind, blowing right → left"]), FINE)
    assert lr.u_cross_ms == pytest.approx(3.0) and rl.u_cross_ms == pytest.approx(-3.0)
    P = lab(FINE, cond(3.0, rel=270.0)).particles
    assert P["U"] < 0 and P["xmax"] < -P["xmin"]       # downwind side (left) gets more room


def test_coverage_rating_and_labels():
    assert sl.coverage_rating(35, 30) == ("acceptable", "Good")
    assert sl.coverage_rating(20, 30) == ("marginal", "Borderline")
    assert sl.coverage_rating(10, 30) == ("critical", "Poor")
    assert sl.coverage_rating(45, 50)[1] == "Borderline"
    assert sl.droplet_class_name("XC") == "Extremely coarse"
    from coverageai import catalog
    assert set(catalog.families()) <= set(sl.FAMILY_INFO)


def test_run_lab_renders_card_and_chart_frame():
    r = sl.run_lab(FINE, cond(), seed=1)
    assert r.card_image is not None and r.card_image.ndim == 3 and r.card_image.dtype == np.uint8
    df = sl.fate_chart_frame(r.bins)
    assert set(df["fate"]) == set(sl.FATE_LABELS.values())
    assert df["pct"].sum() == pytest.approx(100.0, abs=0.5)
    assert df["group"].nunique() <= 14
    s = r.particles["summary"]
    assert s["coverage"] == pytest.approx(r.outcome.coverage_pct)
    assert s["drift"] == pytest.approx(s["blown"] + s["evaporated"], abs=1e-6)


# ------------------------------------------------------------------ UI smoke test

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP_DIR = Path(__file__).resolve().parents[1] / "app"

SCRIPT = f"""
import sys
sys.path.insert(0, {str(APP_DIR)!r})
import streamlit as st
from spray_lab_view import render_spray_lab
render_spray_lab()
"""


def test_html_template_has_payload_slot():
    html = (APP_DIR / "static_spray_lab.html").read_text(encoding="utf8")
    assert html.count("/*__PAYLOAD__*/null") == 1
    assert "requestAnimationFrame" in html and "Initial diameter" in html
    assert "id=\"scrub\"" in html and "Sprayer" not in html
    assert "http://" not in html and "https://" not in html      # no network dependencies


def test_ui_start_simulation_smoke():
    at = AppTest.from_string(SCRIPT, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    for w in [*at.selectbox, *at.slider, *at.button]:
        assert w.key is None or str(w.key).startswith("lab_"), w.key
    btn = next(b for b in at.button if b.label == "Start simulation")
    btn.click().run()
    assert not at.exception, at.exception
    res = at.session_state["lab_result"]
    assert isinstance(res, sl.LabResult)
    # changing a control flags stale results without crashing
    at.slider(key="lab_wind").set_value(6.0).run()
    assert not at.exception, at.exception
    assert any("changed the settings" in w.value for w in at.warning)
    # switch nozzle family (size / pressure widgets are re-keyed per nozzle)
    at.selectbox(key="lab_family").set_value("ULD").run()
    assert not at.exception, at.exception
    next(b for b in at.button if b.label == "Start simulation").click().run()
    assert not at.exception, at.exception
    assert at.session_state["lab_result"].setting.family == "ULD"


# ------------------------------------------------------------------ companion figure data

def _finite_frame(df):
    num = df.select_dtypes("number")
    return bool(np.isfinite(num.to_numpy(float)[~np.isnan(num.to_numpy(float))]).all())


def test_figures_present_and_finite():
    r = lab(FINE, cond(3.0))
    F = r.figures
    for key in ("trajectories", "trajectory_ends", "potential", "spectrum", "cumulative", "profile",
                "sensitivity"):
        assert len(F[key]) > 0, key
        assert _finite_frame(F[key]), key
    assert not F["potential"]["t_s"].isna().any()
    for meta in ("potential_meta", "spectrum_meta", "profile_meta", "sensitivity_meta"):
        for v in F[meta].values():
            if isinstance(v, float):
                assert math.isfinite(v), (meta, v)


def test_drift_vs_wind_monotonic():
    F = lab(FINE, cond(3.0)).figures
    m = F["sensitivity_meta"]
    for name, g in F["sensitivity"].groupby("nozzle"):
        d = g.sort_values("wind_ms")["drift_pct"].to_numpy()
        assert np.all(np.diff(d) >= -1e-9)
        if name == m["nozzle"]:                     # the fine XR clearly drifts more with wind
            assert d[-1] > d[0] + 5
    cur = F["sensitivity"][F["sensitivity"]["nozzle"] == m["nozzle"]]
    assert np.interp(m["wind_ms"], cur["wind_ms"], cur["drift_pct"]) == pytest.approx(m["drift_pct"], abs=0.5)
    # the air-induction alternative drifts less at the top wind speed
    top = F["sensitivity"][F["sensitivity"]["wind_ms"] == F["sensitivity"]["wind_ms"].max()]
    assert m["alt"].startswith("ULD")
    assert float(top[top["nozzle"] == m["alt"]]["drift_pct"].iloc[0]) < \
        float(top[top["nozzle"] == m["nozzle"]]["drift_pct"].iloc[0])


def test_alternative_nozzle_similar_flow():
    from coverageai.catalog import derive
    x = cond(3.0)
    alt = sl.alternative_nozzle(FINE, x)
    assert alt.family == "ULD"
    ratio = derive(x, alt).flow_gpm / derive(x, FINE).flow_gpm
    assert 0.6 < ratio < 1.6
    assert sl.alternative_nozzle(COARSE, x).family == "XR"


def test_displacement_grows_as_diameter_shrinks():
    for wind in (2.0, 5.0):
        df, m = sl.drift_potential_frame(cond(wind), FINE, _derive(FINE, cond(wind)), lab(FINE, cond(wind)).bins)
        alive = df.dropna(subset=["x_land_m"]).sort_values("d_um")
        x = alive["x_land_m"].to_numpy()
        assert np.all(np.diff(x) <= 1e-12), "displacement must fall as diameter grows"
        t = df.sort_values("d_um")["t_s"].to_numpy()
        assert t[0] > t[-1]
        assert m["d_crit_um"] is not None and 50 < m["d_crit_um"] < 300
        lost = df[df["d_um"] < m["d_crit_um"] * 0.95]
        kept = df[df["d_um"] > m["d_crit_um"] * 1.05]
        assert (lost["fate"] != sl.FATE_SHORT[0]).all() and (kept["fate"] == sl.FATE_SHORT[0]).all()
    df0, m0 = sl.drift_potential_frame(cond(0.0, rh=95.0, temp=15.0), COARSE,
                                       _derive(COARSE, cond(0.0)), lab(COARSE, cond(0.0)).bins)
    assert np.nanmax(df0["x_land_m"]) == pytest.approx(0.0, abs=1e-12)


def _derive(u, x):
    from coverageai.catalog import derive
    return derive(x, u)


def test_trajectories_start_at_nozzle_and_end_on_target_or_evaporate():
    x = cond(4.0, temp=35.0, rh=20.0)
    lines, ends = sl.trajectory_frame(x, FINE, _derive(FINE, x))
    first = lines[lines["step"] == 0]
    assert np.allclose(first["h_m"], x.boom_height_m) and np.allclose(first["x_m"], 0.0)
    for _, e in ends.iterrows():
        if e["outcome"] == "lands":
            assert e["h_m"] == pytest.approx(0.0, abs=1e-3)
        else:
            assert 0.0 < e["h_m"] < x.boom_height_m
    xe = ends.sort_values("d_um")["x_m"].to_numpy()
    assert np.all(np.diff(xe) <= 1e-12)           # smaller droplets are carried farther
    assert set(ends["d_um"]) == set(sl.TRAJ_DIAMETERS_UM)


def test_spectrum_bookkeeping():
    r = lab(FINE, cond(4.0))
    per, cum, m = sl.spectrum_frames(r.bins)
    piv = per.pivot(index="d_um", columns="series", values="pct")
    assert piv["Emitted"].sum() == pytest.approx(100.0, abs=1e-6)
    parts = piv["Deposited on target"] + piv["Drifted (displaced > threshold)"] + piv["Evaporated"]
    assert np.allclose(parts, piv["Emitted"])
    assert cum["cum_pct"].iloc[-1] == pytest.approx(100.0) and np.all(np.diff(cum["cum_pct"]) >= 0)
    assert m["dv10_um"] < m["dv50_um"] < m["dv90_um"]
    assert m["dv50_um"] == pytest.approx(r.derived.vmd_um, rel=0.02)


@pytest.mark.parametrize("u,wind", [(FINE, 0.0), (FINE, 4.0), (COARSE, 3.0), (Setting("TP65", "01", 40.0), 6.0)])
def test_profile_integrates_to_landed_fraction(u, wind):
    r = lab(u, cond(wind))
    df, m = r.figures["profile"], r.figures["profile_meta"]
    on = df[df["component"] == "Deposited on target"]["ul_cm2"].sum() * m["dx_m"]
    expect = m["applied_ul_cm2"] * 5 * m["spacing_m"] * r.fate["landed"]
    assert on == pytest.approx(expect, rel=0.01)
    assert (df["ul_cm2"] >= 0).all()
    # with no wind and nothing lost the mean deposit over the swath is the applied rate
    if wind == 0.0:
        sw = df[(df["x_m"].abs() <= m["swath_half_m"])].groupby("x_m")["ul_cm2"].sum()
        assert sw.mean() == pytest.approx(m["applied_ul_cm2"] * r.fate["landed"], rel=0.15)
        assert m["cv_pct"] is not None and m["cv_pct"] < 30


def test_fan_pattern_matches_cos2():
    s = sl.fan_pattern_samples(200_000, np.random.default_rng(0))
    assert s.min() >= -0.5 and s.max() <= 0.5 and abs(s.mean()) < 0.005
    # variance of the density 2 cos^2(pi s) on [-1/2, 1/2]: 1/12 - 1/(2 pi^2)
    assert s.var() == pytest.approx(1 / 12 - 1 / (2 * math.pi ** 2), rel=0.02)
