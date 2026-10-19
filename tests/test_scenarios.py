import numpy as np
import pytest

from coverageai import scenarios
from coverageai.config import spec
from coverageai.contracts import Conditions

R = spec()["ranges"]
LIMIT = spec()["wind_limit_ms"]
NAMES = scenarios.list_presets()
KEYS = ("wind_speed_ms", "wind_dir_deg", "heading_deg", "temp_c", "rh_pct",
        "travel_speed_kmh", "boom_height_m")


def _arr(xs, attr):
    return np.array([getattr(x, attr) for x in xs])


def _in_ranges(xs):
    for key in KEYS:
        a = _arr(xs, key)
        lo, hi = R[key]
        assert a.min() >= lo and a.max() <= hi, key


def test_expected_presets():
    for n in ("calm_morning", "gusty_afternoon", "hot_dry", "crosswind_shift", "marginal_limits"):
        assert n in NAMES
    with pytest.raises(KeyError):
        scenarios.get("nope")


@pytest.mark.parametrize("name", NAMES)
def test_reproducible_by_seed(name):
    sp = scenarios.get(name)
    a, b, c = scenarios.generate(sp, 3), scenarios.generate(sp, 3), scenarios.generate(sp, 4)
    assert a == b
    assert a != c
    assert len(a) == sp.segments == spec()["pass"]["segments"]
    assert all(isinstance(x, Conditions) for x in a)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("seed", range(10))
def test_inside_spec_ranges(name, seed):
    xs = scenarios.generate(scenarios.get(name), seed)
    _in_ranges(xs)
    if name != "marginal_limits":
        assert _arr(xs, "wind_speed_ms").max() <= LIMIT


def test_marginal_exceeds_limit_sometimes():
    peaks = [_arr(scenarios.generate(scenarios.get("marginal_limits"), s), "wind_speed_ms").max()
             for s in range(20)]
    assert max(peaks) > LIMIT


@pytest.mark.parametrize("name", NAMES)
def test_headings_alternate_per_lap(name):
    sp = scenarios.get(name)
    h = _arr(scenarios.generate(sp, 0), "heading_deg")
    n = sp.segments_per_lap
    for k in range(sp.segments):
        assert h[k] == sp.headings[(k // n) % len(sp.headings)]
    laps = h.reshape(-1, n)
    assert (laps == laps[:, :1]).all()
    assert (laps[:-1, 0] != laps[1:, 0]).all()


def test_gusty_has_higher_wind_variance_than_calm():
    g = np.mean([_arr(scenarios.generate(scenarios.get("gusty_afternoon"), s),
                      "wind_speed_ms").var() for s in range(20)])
    c = np.mean([_arr(scenarios.generate(scenarios.get("calm_morning"), s),
                      "wind_speed_ms").var() for s in range(20)])
    assert g > 2 * c


def test_hot_dry_is_hotter_and_drier():
    def mean(name, attr):
        return np.mean([_arr(scenarios.generate(scenarios.get(name), s), attr).mean()
                        for s in range(10)])
    for other in ("calm_morning", "gusty_afternoon", "crosswind_shift"):
        assert mean("hot_dry", "rh_pct") < mean(other, "rh_pct")
        assert mean("hot_dry", "temp_c") > mean(other, "temp_c")


def test_rh_falls_as_temp_rises_in_warming_preset():
    xs = scenarios.generate(scenarios.get("hot_dry"), 0)
    assert np.corrcoef(_arr(xs, "temp_c"), _arr(xs, "rh_pct"))[0, 1] < -0.5


def test_crosswind_flips_sign_between_laps():
    sp = scenarios.get("crosswind_shift")
    xs = scenarios.generate(sp, 0)
    n = sp.segments_per_lap
    first = np.mean([scenarios.crosswind_ms(x) for x in xs[:n]])
    second = np.mean([scenarios.crosswind_ms(x) for x in xs[n:2 * n]])
    assert first > 0 > second


def test_random_conditions_reproducible_and_cover_ranges():
    a = scenarios.random_conditions(200, 1)
    assert a == scenarios.random_conditions(200, 1)
    assert a != scenarios.random_conditions(200, 2)
    assert len(a) == 200
    _in_ranges(a)
    for key in KEYS:
        v = _arr(a, key)
        lo, hi = R[key]
        span = hi - lo
        assert v.min() < lo + 0.05 * span and v.max() > hi - 0.05 * span, key
        assert abs((v.mean() - lo) / span - 0.5) < 0.05, key
