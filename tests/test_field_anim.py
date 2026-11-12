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
    r = al.run_pass_wrapper(al.build_modules_adapter("oracle_nominal", "perfect"),
                            al.make_conditions("gusty_afternoon", 1, n_segments=8),
                            "fast", 1, "med", 30.0, 0.6)
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
        for key in ("vmd_um", "crosswind_ms", "feedback_pp", "error_pp", "wind_speed_ms", "drift_pct"):
            assert math.isfinite(s[key])
        assert s["error_pp"] == pytest.approx(s["actual"] - s["after_feedback"], abs=0.2)


def test_payload_carries_model_output_and_what_the_correction_rests_on(res):
    """The live panel under the card shows, per section, what the model predicted, the correction
    applied with its basis, and the error the next correction is built from."""
    p = field_animation_payload(res, 30.0)
    assert p["feedback"] is True
    basis = al.feedback_basis(res)
    for s, lg, b in zip(p["sections"], res.logs, basis):
        n = al.loop_numbers(lg)
        assert s["pred_lo"] <= s["predicted"] <= s["pred_hi"]
        assert s["pred_drift_pct"] == pytest.approx(100 * lg.pred.drift_frac, abs=0.06)
        assert s["card_minus_model_pp"] == pytest.approx(s["card"] - s["predicted"], abs=0.11)
        assert s["next_feedback_pp"] == pytest.approx(n["next_feedback_pp"], abs=0.006)
        assert s["basis"] == b["text"] and "crosswind" in s["basis"]
    from field_run_view import _HTML
    for shown in ("Model predicted", "based on", "Card &minus; model", "D.next_label"):
        assert shown in _HTML, shown
    assert p["next_label"] == "correction for this group now"      # the rule keeps one correction per group

