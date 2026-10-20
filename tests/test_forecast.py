"""Short-horizon wind forecaster (coverageai.forecast) and its closed-loop integration."""
import dataclasses
import json

import numpy as np
import pytest

from coverageai import catalog, scenarios
from coverageai.contracts import segment_log_to_json
from coverageai.forecast import (METHODS, GustOracleForecaster, WindForecaster,
                                 make_forecaster)
from coverageai.integration import build_modules
from coverageai.loop import (FairOracleModel, default_mock_modules, run_pass, run_passes_slow,
                             segment_rng)
from coverageai.simulator import SpraySimulator

C_REQ = 30.0
SEED = 7
N_SEG = 12


@pytest.fixture(scope="module")
def conds():
    return scenarios.generate(scenarios.get("gusty_afternoon"), 1)[:N_SEG]


@pytest.fixture(scope="module")
def mods():
    # nominal-physics model + perfect measurement: real plant, fast, no trained artifact needed
    return build_modules("oracle_nominal", "perfect")


def _decisions(logs):
    """Everything in a SegmentLog except the forecaster's extra['fc_*'] keys and timings."""
    out = []
    for lg in logs:
        d = dataclasses.asdict(lg)
        d["extra"] = {k: v for k, v in lg.extra.items()
                      if not k.startswith("fc_") and not k.startswith("t_")}
        d["outcome"]["droplets"] = None
        out.append(d)
    return out


# ------------------------------------------------------------------ off = unchanged

@pytest.mark.parametrize("policy", ["open", "fast", "oracle", "best_fixed"])
def test_forecaster_off_is_default_behaviour(mods, conds, policy):
    kw = {"fixed_setting": catalog.candidates()[0]} if policy == "best_fixed" else {}
    a = run_pass(conds, policy, mods, SEED, "high", C_REQ, "p", **kw)
    b = run_pass(conds, policy, mods, SEED, "high", C_REQ, "p", forecaster=None, **kw)
    assert _decisions(a) == _decisions(b)
    assert not any(k.startswith("fc_") for lg in a for k in lg.extra)


@pytest.mark.parametrize("policy", ["open", "fast"])
def test_nominal_forecaster_reproduces_off_and_only_adds_fc_extras(mods, conds, policy):
    """The 'nominal' forecaster plans with x_k exactly like forecaster=None: same settings,
    predictions, biases and outcomes; it only adds fc_* log entries."""
    off = run_pass(conds, policy, mods, SEED, "high", C_REQ, "p")
    nom = run_pass(conds, policy, mods, SEED, "high", C_REQ, "p", forecaster="nominal")
    assert _decisions(off) == _decisions(nom)
    assert all("fc_wind_ms" in lg.extra and "fc_realized_wind_ms" in lg.extra for lg in nom)


def test_nominal_forecaster_reproduces_off_in_slow_loop(mods, conds):
    off = run_passes_slow(conds, mods, SEED, "med", C_REQ, n_passes=2)
    nom = run_passes_slow(conds, mods, SEED, "med", C_REQ, n_passes=2, forecaster="nominal")
    assert _decisions(off) == _decisions(nom)


def test_forecaster_ignored_by_oracle_policies(mods, conds):
    a = run_pass(conds, "oracle", mods, SEED, "high", C_REQ, "p")
    b = run_pass(conds, "oracle", mods, SEED, "high", C_REQ, "p", forecaster="ar1+gust")
    assert _decisions(a) == _decisions(b)


def test_simulator_last_realized_is_additive(conds):
    """Recording the realized wind draws nothing: outcome and rng state are unchanged and
    the gust equals the fair oracle's hidden state for that segment."""
    sim = SpraySimulator(derive_fn=catalog.derive)
    u = catalog.candidates()[5]
    sim.begin_pass(SEED, "high")
    r1 = segment_rng(SEED, 0)
    o1 = sim.simulate(conds[0], u, r1, "high")
    g = sim.last_realized["gust_mult"]
    assert sim.last_realized["wind_speed_ms"] == pytest.approx(conds[0].wind_speed_ms * g)
    fo = FairOracleModel(sim, "high", SEED)
    fo.k = 0
    assert fo.hidden_state().gust_mult == pytest.approx(g)
    sim.end_pass()
    # simulate consumes exactly the hidden draws of _draw_segment (nothing extra)
    sim2 = SpraySimulator(derive_fn=catalog.derive)
    sim2.begin_pass(SEED, "high")
    r2 = segment_rng(SEED, 0)
    sim2._draw_segment(r2, "high")
    assert r1.random() == r2.random()
    sim3 = SpraySimulator(derive_fn=catalog.derive)
    sim3.begin_pass(SEED, "high")
    assert sim3.simulate(conds[0], u, segment_rng(SEED, 0), "high") == o1


# ------------------------------------------------------------------ estimation

def _feed(fc, ratios, wind=5.0):
    x = scenarios.generate(scenarios.get("calm_morning"), 0)[0]
    x = dataclasses.replace(x, wind_speed_ms=wind)
    for k, r in enumerate(ratios):
        fc.forecast(k, x)
        fc.observe(k, x, wind * r)
    return x


@pytest.mark.parametrize("phi", [0.0, 0.5, 0.7, 0.9])
def test_ar1_recovers_phi_on_synthetic_data(phi):
    rng = np.random.default_rng(3)
    n, sig = 3000, 0.2
    e = np.empty(n)
    e[0] = rng.standard_normal()
    for k in range(1, n):
        e[k] = phi * e[k - 1] + np.sqrt(1 - phi ** 2) * rng.standard_normal()
    fc = WindForecaster(method="ar1", obs_sd_ms=1e-6)
    _feed(fc, 1.0 + sig * e)
    assert fc.phi() == pytest.approx(phi, abs=0.04)
    st = fc.state()
    assert st["mu"] == pytest.approx(1.0, abs=0.03)
    # one-step forecast error sd of the ratio ~ sig * sqrt(1 - phi^2) (recent window: loose)
    assert st["sd_ratio"] == pytest.approx(sig * np.sqrt(1 - phi ** 2), rel=0.5)


def test_ar1_shrinks_to_mean_with_little_data():
    fc = WindForecaster(method="ar1")
    _feed(fc, [1.3, 1.25])                    # one pair -> phi strongly shrunk toward 0
    assert 0.0 <= fc.phi() < 0.2


def test_methods_and_gust_aware_plan():
    x = _feed(WindForecaster(method="nominal"), [])
    for m in METHODS:
        fc = WindForecaster(method=m)
        _feed(fc, [1.2, 1.3, 1.25, 1.4])
        f = fc.forecast(4, x)
        assert f.wind_ms == pytest.approx(x.wind_speed_ms * f.ratio)
        if m == "nominal":
            assert f.ratio == 1.0
        if m == "last":
            assert f.ratio == pytest.approx(1.4)
    ga = WindForecaster.from_name("ar1+gust")
    assert ga.gust_aware and ga.method == "ar1" and ga.name == "ar1+gust"
    _feed(ga, [1.2, 0.9, 1.1])
    f = ga.forecast(3, x)
    assert f.plan_wind_ms == pytest.approx(x.wind_speed_ms * (f.ratio + ga.z * f.sd_ratio))
    assert f.plan_wind_ms > f.wind_ms
    with pytest.raises(ValueError):
        WindForecaster(method="kalman")
    assert make_forecaster(None) is None
    assert isinstance(make_forecaster("gust_oracle"), GustOracleForecaster)


def test_begin_pass_keeps_fit_history_but_not_state():
    fc = WindForecaster(method="ar1")
    _feed(fc, [1.2, 1.3, 1.1, 1.25])
    n_pairs = fc.state()["n_pairs"]
    fc.begin_pass()
    assert fc.state()["n_pairs"] == n_pairs and fc._last_r is None
    fc.reset()
    assert fc.state()["n_obs"] == 0


def test_low_wind_observations_are_skipped():
    fc = WindForecaster(method="last")
    x = _feed(fc, [], wind=0.2)
    fc.forecast(0, x)
    assert fc.observe(0, x, 0.5)["fc_obs_used"] is False
    assert fc.state()["n_obs"] == 0


# ------------------------------------------------------------------ no leakage

class _Spy(WindForecaster):
    def __post_init__(self):
        super().__post_init__()
        self.events = []

    def forecast(self, k, x):
        self.events.append(("forecast", k))
        return super().forecast(k, x)

    def observe(self, k, x, w):
        self.events.append(("observe", k))
        return super().observe(k, x, w)


def test_observation_never_precedes_its_own_forecast(mods, conds):
    """Segment k's wind is observed only after k is sprayed: the forecast for k uses
    observations of segments < k only."""
    spy = _Spy(method="ar1")
    run_pass(conds, "fast", mods, SEED, "high", C_REQ, "p", forecaster=spy)
    seen = -1
    for kind, k in spy.events:
        if kind == "forecast":
            assert seen < k
        else:
            assert k == seen + 1
            seen = k
    assert seen == len(conds) - 1


def test_forecast_does_not_depend_on_current_gust(mods, conds):
    """Changing the hidden seed of segment k only (via its gust) cannot change the forecast
    for k: replaying the logged observations of segments < k into a fresh forecaster
    reproduces every logged forecast."""
    logs = run_pass(conds, "fast", mods, SEED, "high", C_REQ, "p", forecaster="ar1")
    fresh = WindForecaster(method="ar1")
    fresh.begin_pass()
    for lg in logs:
        f = fresh.forecast(lg.k, lg.x)
        assert f.wind_ms == pytest.approx(lg.extra["fc_wind_ms"])
        fresh.observe(lg.k, lg.x, lg.extra["fc_obs_wind_ms"])
    # anemometer noise is present and the realized wind differs from the nominal one
    assert any(abs(lg.extra["fc_obs_wind_ms"] - lg.extra["fc_realized_wind_ms"]) > 1e-6 for lg in logs)
    assert any(abs(lg.extra["fc_realized_wind_ms"] - lg.x.wind_speed_ms) > 1e-3 for lg in logs)


def test_gust_oracle_reference_is_exact(mods, conds):
    logs = run_pass(conds, "fast", mods, SEED, "high", C_REQ, "p", forecaster="gust_oracle")
    assert max(lg.extra["fc_abs_err_ms"] for lg in logs) < 1e-9


# ------------------------------------------------------------------ logging

def test_extras_are_json_serializable(mods, conds):
    logs = run_pass(conds, "fast", mods, SEED, "med", C_REQ, "p", forecaster="ar1+gust")
    for lg in logs:
        json.dumps(lg.extra)
        d = json.loads(segment_log_to_json(lg))
        assert d["extra"]["fc_method"] == "ar1+gust"
        assert d["x"]["wind_speed_ms"] == lg.x.wind_speed_ms      # log keeps nominal x


def test_mock_modules_without_realized_wind():
    """A plant without last_realized: the forecaster gets no readings and stays nominal."""
    m = default_mock_modules()
    c = scenarios.generate(scenarios.get("calm_morning"), 0)[:4]
    logs = run_pass(c, "fast", m, 1, "low", C_REQ, "p", render=False, forecaster="ar1")
    assert all(lg.extra["fc_obs_used"] is False for lg in logs)
    json.dumps([lg.extra for lg in logs])
