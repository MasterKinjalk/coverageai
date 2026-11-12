"""What card feedback changed (D33): the no-feedback choice of every section and its outcome.

``run_pass_wrapper(explain_feedback=True)`` records, for the feedback strategies, what the
controller would have done without card feedback. That is by construction the "Model only"
strategy on the same field: these tests pin that, and that the record changes no logged number.
"""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from coverageai import app_logic as al  # noqa: E402
from coverageai.field_anim import field_animation_payload  # noqa: E402

SEED, C_REQ, LAM = 3, 30.0, 0.6


@pytest.fixture(scope="module")
def stack():
    return al.build_modules_adapter("oracle_nominal", "perfect")


@pytest.fixture(scope="module")
def conds():
    return al.make_conditions("gusty_afternoon", SEED, n_segments=16)


def _run(stack, conds, policy, **kw):
    return al.run_pass_wrapper(stack, conds, policy, SEED, "med", C_REQ, LAM, same_field=True, **kw)


@pytest.fixture(scope="module")
def open_run(stack, conds):
    return _run(stack, conds, "open", fixed_nozzle="auto")


@pytest.mark.parametrize("policy", ["fast", "fast_slow"])
def test_no_feedback_is_model_only(stack, conds, open_run, policy):
    res = _run(stack, conds, policy, fixed_nozzle="auto", explain_feedback=True)
    assert res.no_feedback is not None and len(res.no_feedback) == res.n
    for o, lg in zip(res.no_feedback, open_run.logs):
        assert o["u"] == lg.u
        assert o["actual_cov"] == pytest.approx(lg.outcome.coverage_pct, abs=1e-9)
        assert o["pred_cov"] == pytest.approx(lg.pred.coverage_pct, abs=1e-9)


def test_same_setting_same_outcome(stack, conds):
    res = _run(stack, conds, "fast_slow", explain_feedback=True)
    eff = al.feedback_effect(res)
    for e, lg in zip(eff, res.logs):
        if not e["changed"]:
            assert e["gain_pp"] == pytest.approx(0.0, abs=1e-9)
            assert e["actual_without"] == pytest.approx(lg.outcome.coverage_pct, abs=1e-9)
            assert "only the expected coverage" in e["text"]
        else:
            assert e["changes"] and e["setting_without"] in e["text"]


def test_record_changes_no_logged_number(stack, conds):
    a = _run(stack, conds, "fast")
    b = _run(stack, conds, "fast", explain_feedback=True)
    assert a.no_feedback is None and al.feedback_effect(a) is None
    assert [(lg.u, lg.outcome, lg.bias) for lg in a.logs] == [(lg.u, lg.outcome, lg.bias) for lg in b.logs]


def test_selector_options_restored(stack, conds):
    inner = stack.selector.inner if isinstance(stack.selector, al.RecordingSelector) else stack.selector
    before = (inner.lam, inner.nozzle_lock)
    _run(stack, conds, "fast", fixed_nozzle="auto", explain_feedback=True)
    assert (inner.lam, inner.nozzle_lock) == before


@pytest.mark.parametrize("policy", ["open", "oracle", "best_fixed"])
def test_not_recorded_without_feedback(stack, conds, policy):
    res = _run(stack, conds, policy, explain_feedback=True)
    assert res.no_feedback is None
    assert al.feedback_effect_summary(res) is None
    assert field_animation_payload(res, C_REQ, cards=False, advice=False)["no_feedback"] is False


def test_summary_matches_kpis(stack, conds):
    res = _run(stack, conds, "fast_slow", explain_feedback=True)
    s = al.feedback_effect_summary(res)
    k = al.compute_kpis(res.logs, res.c_req)
    assert s["n"] == res.n and s["n_met"] == k["n_met"]
    assert s["mean_cov"] == pytest.approx(k["mean_cov"])
    assert s["n_changed"] == sum(e["changed"] for e in al.feedback_effect(res))
    frame = al.feedback_effect_frame(res)
    assert len(frame) == res.n
    assert frame["gain_pp"].mean() == pytest.approx(s["mean_gain_pp"])


def test_payload_carries_effect(stack, conds):
    res = _run(stack, conds, "fast_slow", explain_feedback=True)
    p = field_animation_payload(res, C_REQ, cards=False, advice=False)
    json.dumps(p, allow_nan=False)
    assert p["no_feedback"] is True
    for s, e in zip(p["sections"], al.feedback_effect(res)):
        assert s["nofb_changed"] == e["changed"]
        assert s["nofb_actual"] == pytest.approx(e["actual_without"], abs=0.051)
        assert math.isfinite(s["nofb_gain_pp"])
