import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from coverageai import app_logic as al  # noqa: E402
from coverageai.field_anim import STRATEGY_NAMES, field_animation_payload  # noqa: E402


@pytest.fixture(scope="module")
def res():
    return al.run_pass_wrapper(al.build_modules_adapter("oracle_nominal", "perfect"),
                               al.make_conditions("gusty_afternoon", 1, n_segments=8),
                               "fast", 1, "med", 30.0, 0.6)


def _walk(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)
    else:
        yield o


def test_payload(res):
    p = field_animation_payload(res, 30.0)
    assert p["n"] == len(p["sections"]) == res.n
    json.dumps(p, allow_nan=False)
    assert all(math.isfinite(v) for v in _walk(p) if isinstance(v, float))
    layout = al.lap_layout(res.conds)
    for s, (r, c) in zip(p["sections"], layout):
        assert (s["row"], s["col"]) == (r, c)
        assert s["label"] in {"Good", "Borderline", "Poor"}
        assert " psi" in s["nozzle"]
    assert p["strategy"] == "Model + card feedback"
    assert p["surprises"] == "Medium"


def test_strategy_names_plain():
    assert set(STRATEGY_NAMES) == set(al.APP_POLICIES)
    for v in STRATEGY_NAMES.values():
        assert v not in al.APP_POLICIES and " " in v


def _app():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd() / "app"))
    from coverageai import app_logic as al
    from field_run_view import render_field_animation
    from smooth_field_view import render_smooth_field
    r = al.run_pass_wrapper(al.build_modules_adapter("oracle_nominal", "perfect"),
                            al.make_conditions("gusty_afternoon", 1, n_segments=8),
                            "fast", 1, "med", 30.0, 0.6)
    render_smooth_field(r, 30.0)
    render_field_animation(r, 30.0)


def test_apptest_smoke():
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = AppTest.from_function(_app, default_timeout=120)
    at.run()
    assert not at.exception, at.exception


def test_scientific_fields(res):
    p = field_animation_payload(res, 30.0)
    assert 0 < p["bands"]["border_min"] < p["bands"]["good_min"]
    for s in p["sections"]:
        for key in ("vmd_um", "crosswind_ms", "feedback_pp", "error_pp", "wind_speed_ms",
                    "drift_pct", "predicted_drift_pct", "pressure_psi", "speed_mph", "boom_height_in"):
            assert math.isfinite(s[key])
        assert s["error_pp"] == pytest.approx(s["actual"] - s["after_feedback"], abs=0.2)


def test_smooth_field_payload_is_lightweight_and_tracks_lap_geometry(res):
    p = field_animation_payload(res, 30.0, cards=False, advice=False)
    assert p["n"] == res.n
    assert p["rows"] == max(s["row"] for s in p["sections"]) + 1
    assert p["cols"] == max(s["col"] for s in p["sections"]) + 1
    assert all(not s["img"] and not s["overlay"] and not s["advice"] for s in p["sections"])
    assert all(0 <= s["actual"] <= 100 for s in p["sections"])


def test_smooth_field_starts_at_one_x_and_shows_live_feedback_adjustments():
    from smooth_field_view import _HTML

    assert 'data-speed="1" class="on">1×' in _HTML
    assert "t=0,speed=1,playing=true" in _HTML
    assert "const s=S[Math.min(Math.floor(t),N-1)]" in _HTML
    assert "grid-template-columns:repeat(5,minmax(0,1fr))" in _HTML
    assert "LIVE FEEDBACK ADJUSTMENTS" in _HTML
    assert "put('reading-section','Section '+s.number+' / '+N)" in _HTML
    assert "el.classList.add('changed')" in _HTML
    for element, source in (
        ("rpressure", "s.pressure_psi"),
        ("rspeed", "s.speed_mph"),
        ("rgpa", "s.rate_gpa"),
        ("rboom", "s.boom_height_in"),
        ("rcoverage", "s.card"),
    ):
        assert f"putLive('{element}'" in _HTML
        assert source in _HTML
    for element, source, target in (
        ("apressure", "s.pressure_psi", "D.setup.pressure"),
        ("aspeed", "s.speed_mph", "D.setup.speed"),
        ("agpa", "s.rate_gpa", "D.setup.gpa"),
        ("aboom", "s.boom_height_in", "D.setup.boom"),
    ):
        assert f"putLive('{element}',adjustment({source},Number({target})" in _HTML
    assert "putLive('acoverage',adjustment(s.card,D.target,'pp',1))" in _HTML
    assert '<span class="label">Coverage</span>' in _HTML
    assert "Live coverage reading" in _HTML
    assert "Coverage vs goal" in _HTML
