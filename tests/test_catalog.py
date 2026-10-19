import math

import pytest

from coverageai import catalog, units
from coverageai.config import spec
from coverageai.contracts import Conditions, Derived, Setting
from coverageai.mocks import mock_derive


def test_load_cached_and_columns():
    df = catalog.load()
    assert catalog.load() is df
    assert list(df.columns) == ["family", "size", "pressure_psi", "flow_gpm", "droplet_class",
                                "spray_angle_deg", "source", "chart_spacing_in", "notes"]
    assert not df[["flow_gpm", "droplet_class"]].isna().any().any()


def test_families_sizes():
    assert set(catalog.families()) == {"TP65", "ULD", "3D", "XR", "DGX40"}
    assert catalog.sizes("DGX40") == ["03", "04", "05", "06"]
    xr = catalog.sizes("XR")
    assert xr[:3] == ["01", "015", "02"] and xr[-1] == "15" and "035" in xr
    assert catalog.sizes("ULD") == ["015", "02", "025", "03", "04", "05", "06", "08"]
    with pytest.raises(ValueError):
        catalog.sizes("NOPE")


@pytest.mark.parametrize("fam,size,p,flow,cls", [
    ("TP65", "04", 40, 0.40, "M"),
    ("TP65", "04", 50, 0.45, "F"),
    ("TP65", "08", 30, 0.69, "C"),
    ("XR", "04", 40, 0.40, "M"),
    ("XR", "04", 60, 0.49, "F"),
    ("XR", "06", 15, 0.37, "C"),
    ("DGX40", "04", 40, 0.40, "UC"),
    ("DGX40", "06", 70, 0.79, "VC"),
    ("ULD", "04", 40, 0.40, "UC"),
    ("ULD", "06", 115, 1.02, "M"),
    ("3D", "03", 40, 0.30, "M"),
    ("3D", "08", 90, 1.20, "M"),
])
def test_chart_rows_exact(fam, size, p, flow, cls):
    q, c = catalog.flow_and_class(fam, size, p)
    assert q == pytest.approx(flow, abs=1e-9)
    assert c == cls


def test_sqrt_scaling_between_rows():
    # TP6504: 0.35 gpm at 30 psi, 0.40 at 40 psi; 35 psi is also a chart row (0.37)
    q35, _ = catalog.flow_and_class("TP65", "04", 35)
    assert q35 == pytest.approx(0.37, abs=1e-9)
    q45, c45 = catalog.flow_and_class("TP65", "04", 45)
    assert 0.40 < q45 < 0.45
    assert q45 == pytest.approx(0.40 * math.sqrt(45 / 40), rel=0.02)
    assert c45 in ("M", "F")


def test_flow_monotone_in_pressure_everywhere():
    for fam in catalog.families():
        for size in catalog.sizes(fam):
            lo, hi = catalog.pressure_range(fam, size)
            qs = [catalog.flow_and_class(fam, size, lo + (hi - lo) * i / 40)[0] for i in range(41)]
            assert all(b >= a for a, b in zip(qs, qs[1:])), (fam, size)


@pytest.mark.parametrize("fam,size,p", [
    ("TP65", "04", 25), ("TP65", "04", 65), ("XR", "04", 10), ("XR", "04", 65),
    ("DGX40", "04", 15), ("ULD", "04", 118), ("3D", "04", 5),
])
def test_out_of_range_raises(fam, size, p):
    with pytest.raises(ValueError):
        catalog.flow_and_class(fam, size, p)


def test_unknown_nozzle_raises():
    with pytest.raises(ValueError):
        catalog.flow_and_class("TP65", "99", 40)
    with pytest.raises(ValueError):
        catalog.flow_and_class("ZZ", "04", 40)


def test_vmd_um():
    s = spec()["droplets"]
    assert catalog.vmd_um("M", 40, "XR") == pytest.approx(s["vmd_center_um"]["M"])
    assert catalog.vmd_um("M", 60, "XR") < catalog.vmd_um("M", 20, "XR")
    for cls, (lo, hi) in s["vmd_bounds_um"].items():
        for p in (10, 40, 115):
            assert lo <= catalog.vmd_um(cls, p, "ULD") <= hi
    # clipping engages: UC at 115 psi would be 700*(2.875)^-0.3 = 510 -> clipped to 650
    assert catalog.vmd_um("UC", 115, "ULD") == pytest.approx(650.0)


X = Conditions(wind_speed_ms=4.0, wind_dir_deg=100.0, heading_deg=10.0, temp_c=25.0, rh_pct=50.0,
               travel_speed_kmh=16.0, boom_height_m=0.5)


def test_derive_fields_sane():
    u = Setting("XR", "04", 40.0)
    d = catalog.derive(X, u)
    assert isinstance(d, Derived)
    assert d.flow_gpm == pytest.approx(0.40)
    mph = units.kmh_to_mph(16.0)
    assert d.app_rate_gpa == pytest.approx(5940 * 0.40 / (mph * 20.0))
    assert d.app_rate_lha == pytest.approx(units.gpa_to_lha(d.app_rate_gpa))
    assert 10 < d.app_rate_gpa < 20
    assert d.droplet_class == "M"
    assert d.vmd_um == pytest.approx(215.0)
    assert d.spray_angle_deg == 110
    assert d.rel_wind_deg == pytest.approx(90.0)
    assert d.u_cross_ms == pytest.approx(4.0)
    assert d.u_along_ms == pytest.approx(0.0, abs=1e-9)


def test_derive_wind_geometry_wraps():
    u = Setting("TP65", "04", 40.0)
    x = Conditions(5.0, 10.0, 350.0, 20.0, 60.0, 16.0, 0.5)   # wind-heading = -340 -> +20
    d = catalog.derive(x, u)
    assert d.rel_wind_deg == pytest.approx(20.0)
    assert d.u_cross_ms == pytest.approx(5 * math.sin(math.radians(20)))
    assert d.u_along_ms == pytest.approx(5 * math.cos(math.radians(20)))
    x2 = Conditions(5.0, 350.0, 10.0, 20.0, 60.0, 16.0, 0.5)
    assert catalog.derive(x2, u).rel_wind_deg == pytest.approx(-20.0)
    x3 = Conditions(5.0, 190.0, 10.0, 20.0, 60.0, 16.0, 0.5)
    assert catalog.derive(x3, u).rel_wind_deg == pytest.approx(180.0)


def test_derive_matches_mock_shape():
    # real derive agrees with the mock's formula where the stub catalog rows coincide with chart rows
    for u in (Setting("XR", "04", 40.0), Setting("TP65", "04", 40.0), Setting("DGX40", "04", 40.0)):
        r, m = catalog.derive(X, u), mock_derive(X, u)
        assert r.flow_gpm == pytest.approx(m.flow_gpm)
        assert r.app_rate_gpa == pytest.approx(m.app_rate_gpa)
        assert r.spray_angle_deg == m.spray_angle_deg
        assert r.u_cross_ms == pytest.approx(m.u_cross_ms)


def test_derive_gpa_rises_with_lower_speed():
    u = Setting("ULD", "04", 40.0)
    slow = catalog.derive(Conditions(2, 0, 0, 20, 60, 8.0, 0.5), u)
    fast = catalog.derive(Conditions(2, 0, 0, 20, 60, 24.0, 0.5), u)
    assert slow.app_rate_gpa == pytest.approx(3 * fast.app_rate_gpa)


def test_candidates():
    c = catalog.candidates()
    assert len(c) >= 200
    assert len(set(c)) == len(c)
    rng = spec()["pressure_range_psi"]
    for s in c:
        lo, hi = rng[s.family]
        assert lo <= s.pressure_psi <= hi
        catalog.flow_and_class(s.family, s.size, s.pressure_psi)   # must not raise
    # chart extremes are included
    assert Setting("XR", "04", 15.0) in c and Setting("XR", "04", 60.0) in c
    assert Setting("ULD", "04", 115.0) in c
    assert len(catalog.candidates(2.5)) > len(c)
    with pytest.raises(ValueError):
        catalog.candidates(0)
