"""WP-02 tests: physics checkpoints, monotone trends, analytic vs Monte-Carlo coverage."""
import math
import time

import numpy as np
import pytest

from coverageai import units
from coverageai.config import spec
from coverageai.contracts import Conditions, Derived, Setting, Simulator
from coverageai.physics.droplets import (lognormal_volume_bins, sigma_from_span,
                                         terminal_velocity)
from coverageai.physics.evaporation import (evaporation_lambda, fall_with_evaporation,
                                            wet_bulb_c)
from coverageai.simulator import SpraySimulator

CARD_W, CARD_H = spec()["card"]["width_mm"], spec()["card"]["height_mm"]
VMD_CENTER = spec()["droplets"]["vmd_center_um"]


# ------------------------------------------------------------------ helpers
def cond(u=0.0, wd=90.0, hd=0.0, t=25.0, rh=50.0, v=12.7, h=0.5) -> Conditions:
    return Conditions(u, wd, hd, t, rh, v, h)


def make_sim(vmd: float = 215.0, flow_gpm: float = 0.4, **kw) -> SpraySimulator:
    """Simulator with a physics-only stub catalog: Derived built from fixed VMD and flow,
    so trend tests do not depend on catalog contents."""
    def derive(x: Conditions, u: Setting) -> Derived:
        mph = units.kmh_to_mph(x.travel_speed_kmh)
        g = units.gpa(flow_gpm, mph, 20.0)
        rel = units.wrap_deg(x.wind_dir_deg - x.heading_deg)
        r = math.radians(rel)
        return Derived(flow_gpm, g, units.gpa_to_lha(g), "M", vmd, 110.0, rel,
                       x.wind_speed_ms * math.sin(r), x.wind_speed_ms * math.cos(r))
    return SpraySimulator(derive_fn=derive, **kw)


U0 = Setting("XR", "04", 40.0)


def run(sim, x, seed=1, mismatch="none", sample=False):
    return sim.simulate(x, U0, np.random.default_rng(seed), mismatch, sample)


def mean_outcome(sim, x, field, n=12):
    return float(np.mean([getattr(run(sim, x, seed=s), field) for s in range(n)]))


# ------------------------------------------------------------------ droplets
@pytest.mark.parametrize("d_um,expected", [(20, 0.012), (100, 0.265), (200, 0.70), (400, 1.80)])
def test_terminal_velocity_checkpoints(d_um, expected):
    assert terminal_velocity(d_um) == pytest.approx(expected, rel=0.25)


def test_terminal_velocity_monotone():
    d = np.linspace(5, 1500, 300)
    assert np.all(np.diff(terminal_velocity(d)) > 0)


def test_spectrum_percentiles_match_span():
    s = sigma_from_span(1.2)
    d, w, _ = lognormal_volume_bins([300.0], s, n_bins=400)
    cdf = np.cumsum(w)
    q = lambda p: np.interp(p, cdf, d[0])   # noqa: E731
    assert q(0.5) == pytest.approx(300.0, rel=0.02)
    assert (q(0.9) - q(0.1)) / q(0.5) == pytest.approx(1.2, rel=0.05)
    assert w.sum() == pytest.approx(1.0)


# ------------------------------------------------------------------ evaporation
def test_wet_bulb_reasonable():
    assert wet_bulb_c(25.0, 50.0) == pytest.approx(18.0, abs=0.6)
    assert wet_bulb_c(30.0, 100.0) == pytest.approx(30.0, abs=1.0)


def test_evaporation_monotone_in_T_and_RH():
    d0 = np.array([90.0, 120.0, 200.0])
    def lost(t, rh):
        lam = evaporation_lambda(t, rh)
        d_f, _, alive = fall_with_evaporation(d0, 0.5, lam)
        return 1.0 - np.where(alive, (d_f / d0) ** 3, 0.0)
    temps = [10, 20, 30, 40]
    losses = np.array([lost(t, 40.0) for t in temps])
    assert np.all(np.diff(losses, axis=0) > 0)
    rhs = [20, 40, 60, 80, 95]
    losses = np.array([lost(30.0, rh) for rh in rhs])
    assert np.all(np.diff(losses, axis=0) < 0)


def test_evaporation_zero_when_saturated_and_small_droplets_die():
    d_f, t, alive = fall_with_evaporation(np.array([200.0]), 0.5, np.array([0.0]))
    assert alive[0] and d_f[0] == 200.0 and t[0] == pytest.approx(0.5 / terminal_velocity(200.0))
    # 20 um droplet falling 0.75 m in hot dry air cannot survive
    lam = evaporation_lambda(40.0, 10.0)
    assert not fall_with_evaporation(np.array([20.0]), 0.75, lam)[2][0]


def test_simulator_drift_from_evaporation_grows_with_T_down_RH():
    sim = make_sim(vmd=90.0)                      # very fine, no wind: loss = evaporation only
    d_hot = run(sim, cond(0.0, t=38.0, rh=15.0)).drift_frac
    d_mild = run(sim, cond(0.0, t=20.0, rh=60.0)).drift_frac
    d_humid = run(sim, cond(0.0, t=38.0, rh=90.0)).drift_frac
    assert d_hot > d_mild > 0.0
    assert d_hot > d_humid


# ------------------------------------------------------------------ trends
def test_protocol_and_ranges():
    sim = make_sim()
    assert isinstance(sim, Simulator)
    o = run(sim, cond(2.0))
    assert 0 <= o.coverage_pct <= 100 and 0 <= o.drift_frac <= 1
    assert o.deposition_ul_cm2 >= 0 and o.stains_per_cm2 >= 0 and o.droplets is None


def test_coverage_increases_with_gpa():
    covs = [mean_outcome(make_sim(flow_gpm=q), cond(0.5), "coverage_pct") for q in (0.1, 0.2, 0.4, 0.8)]
    assert all(b > a for a, b in zip(covs, covs[1:]))
    # lower travel speed -> higher GPA -> more coverage
    sim = make_sim()
    slow = mean_outcome(sim, cond(0.5, v=8.0), "coverage_pct")
    fast = mean_outcome(sim, cond(0.5, v=24.0), "coverage_pct")
    assert slow > fast


def test_coverage_decreases_with_coarser_spray_at_equal_volume():
    covs = [mean_outcome(make_sim(vmd=VMD_CENTER[c]), cond(0.0), "coverage_pct")
            for c in ("F", "M", "C", "VC")]
    assert all(b < a for a, b in zip(covs, covs[1:]))


def test_deposition_equals_rate_without_loss():
    sim = make_sim(vmd=400.0)
    o = run(sim, cond(0.0, t=15.0, rh=90.0))
    expect = units.gpa_to_ul_cm2(units.gpa(0.4, units.kmh_to_mph(12.7), 20.0))
    assert o.deposition_ul_cm2 == pytest.approx(expect, rel=0.01)
    assert o.drift_frac < 0.01


def test_drift_increases_with_wind():
    sim = make_sim()
    d = [run(sim, cond(u)).drift_frac for u in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)]
    assert all(b > a for a, b in zip(d, d[1:]))
    assert run(sim, cond(7.0)).drift_frac >= run(sim, cond(2.0)).drift_frac


def test_drift_increases_with_height():
    sim = make_sim()
    d = [run(sim, cond(1.0, h=h)).drift_frac for h in (0.3, 0.45, 0.6, 0.75)]
    assert all(b > a for a, b in zip(d, d[1:]))


def test_drift_decreases_with_coarser_class():
    d = [run(make_sim(vmd=VMD_CENTER[c]), cond(1.5)).drift_frac for c in ("VF", "F", "M", "C", "VC", "XC", "UC")]
    assert all(b < a for a, b in zip(d, d[1:]))


def test_anchor_crosswind_drift_by_class():
    """Decision D10 targets: 15 GPA, 2 m/s crosswind, H 0.5 m, 25 C / 50 % RH, 40 psi."""
    def at(c):
        sim = make_sim(vmd=VMD_CENTER[c], flow_gpm=0.4)      # 15.1 GPA at 12.7 km/h
        return [run(sim, cond(2.0), s) for s in range(6)], sim
    o_f, _ = at("F")
    o_m, sim_m = at("M")
    o_c, _ = at("C")
    d = lambda o: float(np.mean([r.drift_frac for r in o]))  # noqa: E731
    assert 0.10 <= d(o_f) <= 0.35
    assert 0.02 <= d(o_m) <= 0.15
    assert d(o_c) < 0.05
    calm = np.mean([run(sim_m, cond(0.0), s).coverage_pct for s in range(6)])
    assert abs(np.mean([r.coverage_pct for r in o_m]) - calm) <= 5.0


def test_exit_velocity_reduces_drift_at_higher_pressure():
    sim = make_sim(vmd=VMD_CENTER["F"])
    d = [sim.simulate(cond(2.0), Setting("XR", "04", p), np.random.default_rng(1)).drift_frac
         for p in (20.0, 40.0, 60.0, 90.0)]
    assert all(b < a for a, b in zip(d, d[1:]))


def test_fine_droplet_drift_grows_with_T_down_RH_in_wind():
    sim = make_sim(vmd=VMD_CENTER["F"])
    hot = run(sim, cond(2.0, t=38.0, rh=15.0)).drift_frac
    cool = run(sim, cond(2.0, t=20.0, rh=70.0)).drift_frac
    assert hot > cool


def test_fall_with_exit_velocity_limits():
    from coverageai.physics.drift import fall_with_exit_velocity
    from coverageai.physics.droplets import terminal_velocity
    d = np.array([[30.0, 100.0, 300.0]])
    h, v0, lam = np.array([[0.5]]), np.array([[14.0]]), np.array([[0.0]])
    _, t, alive, xc, xa = fall_with_exit_velocity(d, h, v0, lam, np.array([[2.0]]), np.array([[0.0]]))
    assert alive.all() and np.all(xa == 0)
    assert np.all(np.diff(t[0]) < 0) and np.all(np.diff(xc[0]) < 0)      # coarser: faster, less drift
    # v0 = Vt: fall is plain settling t = H / Vt, lateral lag x = U (t - tau (1 - exp(-t/tau)))
    vt = terminal_velocity(d)
    _, t2, _, xc2, _ = fall_with_exit_velocity(d, h, vt[:, :1] * 0 + vt, lam, np.array([[2.0]]), np.array([[0.0]]))
    tau = vt / 9.81
    assert np.allclose(t2, 0.5 / vt, rtol=1e-6)
    assert np.allclose(xc2, 2.0 * (t2 - tau * (1 - np.exp(-t2 / tau))), rtol=2e-3)


def test_coverage_decreases_with_wind_and_displacement_grows():
    sim = make_sim(vmd=VMD_CENTER["C"])
    c = [run(sim, cond(u)).coverage_pct for u in (0.0, 1.0, 2.0, 4.0)]
    assert all(b < a for a, b in zip(c, c[1:]))
    disp = [run(sim, cond(u)).displacement_mm for u in (0.0, 0.5, 1.0)]
    assert disp[0] == 0.0 and disp[1] > 0.0 and disp[2] > disp[1]
    assert max(run(sim, cond(7.0)).displacement_mm, 0) <= 1000 * spec()["physics"]["drift_threshold_m"] * 1.5


def test_along_track_wind_adds_displacement_but_not_drift():
    sim = make_sim()
    along = run(sim, cond(2.0, wd=0.0, hd=0.0))   # wind toward heading: pure along-track
    cross = run(sim, cond(2.0, wd=90.0, hd=0.0))
    assert along.drift_frac == pytest.approx(run(sim, cond(0.0)).drift_frac, abs=1e-6)
    assert along.displacement_mm > 0 and cross.drift_frac > along.drift_frac


def test_min_detectable_cutoff():
    sim = make_sim(vmd=60.0)
    o = run(sim, cond(0.0))
    assert o.stains_per_cm2 > 0
    # a 8 um VMD spray has no droplets >= 50 um (max bin 8*exp(3.5 s) = 38 um)
    assert run(make_sim(vmd=8.0), cond(0.0)).stains_per_cm2 < 1e-6


# ------------------------------------------------------------------ mismatch
def test_none_deterministic_and_seeded():
    sim = make_sim()
    a, b = run(sim, cond(1.0), seed=5), run(sim, cond(1.0), seed=5)
    assert a == b
    assert run(sim, cond(1.0), seed=6).coverage_pct != a.coverage_pct   # SF differs by seed


def test_mismatch_reproducible_with_begin_pass():
    sim = make_sim()
    def seq():
        sim.begin_pass(3, "med")
        return [sim.simulate(cond(2.0), U0, np.random.default_rng([3, k]), "med").coverage_pct
                for k in range(6)]
    assert seq() == seq()


def test_mismatch_hurts_coverage():
    sim = make_sim()
    base = np.mean([run(sim, cond(0.5), s).coverage_pct for s in range(60)])
    high = np.mean([run(sim, cond(0.5), s, "high").coverage_pct for s in range(60)])
    assert high < base


def test_pure_without_begin_pass():
    """No pass: repeated calls with the same rng seed give identical results (no state)."""
    sim = make_sim()
    a = [run(sim, cond(1.0), 7, "high") for _ in range(3)]
    assert a[0] == a[1] == a[2]


def test_begin_pass_draws_canopy_and_wear_once():
    sim = make_sim()
    sim.begin_pass(5, "high")
    eta, ff, vf = sim._pass_factors
    lv = spec()["mismatch"]["levels"]["high"]
    assert lv["canopy_eta"][0] <= eta <= lv["canopy_eta"][1]
    assert 1 + lv["wear_flow"][0] <= ff <= 1 + lv["wear_flow"][1]
    assert 1 + lv["wear_vmd"][0] <= vf <= 1 + lv["wear_vmd"][1]
    # same factors regardless of the per-segment rng; deposit ratio to no-mismatch is stable
    hs = []
    for k in range(5):
        o = sim.simulate(cond(0.0), U0, np.random.default_rng([5, k]), "high")
        hs.append(o.deposition_ul_cm2)
    assert max(hs) / min(hs) == pytest.approx(1.0, abs=1e-9)   # no wind: drift constant, eta/flow fixed
    sim.begin_pass(5, "high")
    assert sim._pass_factors == (eta, ff, vf)                  # reproducible from seed
    sim.begin_pass(6, "high")
    assert sim._pass_factors != (eta, ff, vf)
    sim.begin_pass(5, "low")                                   # other level: pass factors replaced
    assert sim._pass_level == "low"


def test_pass_factors_apply_only_to_matching_level():
    sim = make_sim()
    sim.begin_pass(5, "high")
    pure = make_sim()
    a = sim.simulate(cond(0.0), U0, np.random.default_rng(9), "med")
    b = pure.simulate(cond(0.0), U0, np.random.default_rng(9), "med")
    assert a == b


def test_gust_ar1_in_pass_and_iid_otherwise():
    phi = spec()["mismatch"]["gust_ar1_phi"]
    def lag1(sim, n=6000):
        g = []
        for k in range(n):
            hid, _ = sim._draw_segment(np.random.default_rng([1, k]), "high")
            g.append(hid.gust_mult)
        g = np.array(g) - np.mean(g)
        return float((g[1:] * g[:-1]).mean() / g.var())
    in_pass = make_sim()
    in_pass.begin_pass(1, "high")
    assert lag1(in_pass) == pytest.approx(phi, abs=0.06)
    assert abs(lag1(make_sim())) < 0.06


def test_gust_state_idempotent_per_segment():
    """Oracle pattern: many candidate calls with the same per-segment rng do not advance
    the AR(1) state; the real outcome then sees the same gust."""
    sim = make_sim()
    sim.begin_pass(2, "high")
    ref = make_sim()
    ref.begin_pass(2, "high")
    for k in range(5):
        outs = [sim.simulate(cond(2.0), Setting("XR", str(i), 40.0), np.random.default_rng([2, k]), "high")
                for i in range(4)]
        assert all(o == outs[0] for o in outs)
        assert outs[0] == ref.simulate(cond(2.0), U0, np.random.default_rng([2, k]), "high")


def test_hidden_realization_independent_of_candidate():
    """Gust and SF are the first draws and do not depend on u: two settings with identical
    Derived see the same hidden realization even if the other setting draws differ."""
    sim = make_sim()
    base = sim.simulate(cond(1.5), Setting("A", "01", 30.0), np.random.default_rng([4, 2]), "high")
    other = sim.simulate(cond(1.5), Setting("B", "09", 30.0), np.random.default_rng([4, 2]), "high")
    assert base == other           # stub catalog ignores u: only u-independent draws could differ
    # and the draws themselves: hidden state from the same rng is identical across u
    h1, sf1 = sim._draw_segment(np.random.default_rng([4, 2]), "high")
    h2, sf2 = sim._draw_segment(np.random.default_rng([4, 2]), "high")
    assert (h1, sf1) == (h2, sf2)


def test_sampling_does_not_change_analytic_outcome():
    sim = make_sim()
    for mm in ("none", "med"):
        for seed in range(3):
            a = run(sim, cond(1.0), seed, mm, sample=False)
            b = run(sim, cond(1.0), seed, mm, sample=True)
            assert (a.coverage_pct, a.deposition_ul_cm2, a.drift_frac, a.displacement_mm,
                    a.stains_per_cm2) == (b.coverage_pct, b.deposition_ul_cm2, b.drift_frac,
                                          b.displacement_mm, b.stains_per_cm2)
            assert a.droplets is None and b.droplets is not None


# ------------------------------------------------------------------ MC vs analytic
def rasterize_coverage(sample, dpi=600) -> float:
    """Exact pixel-center rasterization of the stain disks (cv2.circle is biased high by
    its boundary pixels at these radii, so it is not used)."""
    ppm = dpi / 25.4
    w, h = int(round(CARD_W * ppm)), int(round(CARD_H * ppm))
    img = np.zeros((h, w), bool)
    rad = sample.spread_factor * sample.diameters_um * 1e-3 / 2.0 * ppm
    cx, cy = sample.x_mm * ppm - 0.5, sample.y_mm * ppm - 0.5
    for a, b, r in zip(cx, cy, rad):
        x0, x1 = max(int(a - r) - 1, 0), min(int(a + r) + 2, w)
        y0, y1 = max(int(b - r) - 1, 0), min(int(b + r) + 2, h)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        img[y0:y1, x0:x1] |= (xx - a) ** 2 + (yy - b) ** 2 <= r * r
    return 100.0 * float(img.mean())


def test_analytic_coverage_matches_monte_carlo_raster():
    rng = np.random.default_rng(123)
    errs, covs = [], []
    for i in range(20):
        sim = make_sim(vmd=float(rng.uniform(110, 420)), flow_gpm=float(rng.uniform(0.1, 0.55)))
        x = cond(float(rng.uniform(0, 1.0)), wd=float(rng.uniform(0, 360)), t=float(rng.uniform(15, 35)),
                 rh=float(rng.uniform(30, 80)), v=float(rng.uniform(8, 24)),
                 h=float(rng.uniform(0.3, 0.75)))
        o = sim.simulate(x, U0, np.random.default_rng(1000 + i), "none", sample_droplets=True)
        mc = rasterize_coverage(o.droplets)
        errs.append(abs(mc - o.coverage_pct))
        covs.append(o.coverage_pct)
    assert max(covs) > 25 and min(covs) < 25          # cases span the useful range
    assert np.mean(errs) <= 2.0, (np.mean(errs), np.round(errs, 1))


def test_droplet_sample_contract():
    sim = make_sim()
    o = run(sim, cond(1.0), sample=True)
    s = o.droplets
    assert s is not None and len(s.diameters_um) > 500
    assert len(s.x_mm) == len(s.y_mm) == len(s.diameters_um)
    assert s.diameters_um.min() >= 0.9 * spec()["physics"]["min_detectable_droplet_um"]
    assert spec()["physics"]["spread_factor"][0] <= s.spread_factor <= spec()["physics"]["spread_factor"][1]
    m = 1.1 * s.spread_factor * s.diameters_um * 1e-3 / 2
    assert np.all(s.x_mm >= -m - 1e-9) and np.all(s.x_mm <= CARD_W + m + 1e-9)
    assert np.all(s.y_mm >= -m - 1e-9) and np.all(s.y_mm <= CARD_H + m + 1e-9)
    # stains per cm2 matches the sampled count (Poisson noise, ignoring edge margin)
    n_sample = len(s.diameters_um) / (CARD_W * CARD_H / 100.0)
    assert n_sample == pytest.approx(o.stains_per_cm2, rel=0.2)


# ------------------------------------------------------------------ default derive + batch + timing
def _settings(n=200):
    try:
        from coverageai.catalog import candidates
        c = list(candidates())
    except ImportError:
        from coverageai.mocks import mock_candidates
        c = mock_candidates()
    return [c[i % len(c)] for i in range(n)]


def test_default_derive_batch_matches_scalar():
    sim = SpraySimulator()
    us = _settings(120)
    rng = np.random.default_rng(0)
    xs = [cond(float(rng.uniform(0, 5)), wd=float(rng.uniform(0, 360)), hd=float(rng.uniform(0, 360)),
               t=float(rng.uniform(10, 40)), rh=float(rng.uniform(10, 100)), v=float(rng.uniform(6.4, 32)),
               h=float(rng.uniform(0.3, 0.75))) for _ in us]
    batch = sim.simulate_batch(xs, us, np.random.default_rng(4), "none")
    # same SF draws are not reproducible one by one; compare drift/deposition exactly
    for x, u, b in zip(xs[:20], us[:20], batch[:20]):
        o = sim.simulate(x, u, np.random.default_rng(0), "none")
        assert o.drift_frac == pytest.approx(b.drift_frac, abs=1e-9)
        assert o.deposition_ul_cm2 == pytest.approx(b.deposition_ul_cm2, rel=1e-9)
        assert o.displacement_mm == pytest.approx(b.displacement_mm, abs=1e-6)
    assert all(0 <= b.coverage_pct <= 100 and 0 <= b.drift_frac <= 1 for b in batch)


def test_timing_single_and_sampling():
    sim = SpraySimulator()
    u = _settings(1)[0]
    x = cond(1.0)
    rng = np.random.default_rng(0)
    sim.simulate(x, u, rng)           # warm up (table build)
    t0 = time.perf_counter()
    for _ in range(50):
        sim.simulate(x, u, rng)
    per = (time.perf_counter() - t0) / 50
    assert per <= 0.05
    t0 = time.perf_counter()
    o = sim.simulate(cond(0.5), u, rng, sample_droplets=True)
    assert time.perf_counter() - t0 <= 0.3
    assert o.droplets is not None


def test_timing_batch_scales_to_50k_rows():
    sim = SpraySimulator()
    n = 5000
    us = _settings(n)
    rng = np.random.default_rng(0)
    xs = [cond(float(rng.uniform(0, 7)), wd=float(rng.uniform(0, 360))) for _ in range(n)]
    t0 = time.perf_counter()
    out = sim.simulate_batch(xs, us, rng, "med")
    per_row = (time.perf_counter() - t0) / n
    assert len(out) == n
    assert per_row * 50_000 < 120.0
