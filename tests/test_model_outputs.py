"""Model outputs shown in the app: loop numbers, setting changes, run diffs, paper cards."""
from __future__ import annotations

import base64
import dataclasses

import cv2
import numpy as np
import pytest

from coverageai import app_logic as al
from coverageai import integration
from coverageai.config import px_per_mm
from coverageai.field_anim import field_animation_payload
from coverageai.vision.classical import ClassicalVerifier


@pytest.fixture(scope="module")
def modules():
    return al.build_modules_adapter("oracle_nominal", "perfect")


@pytest.fixture(scope="module")
def res(modules):
    conds = al.make_conditions("gusty_afternoon", 1, n_segments=12)
    return al.run_pass_wrapper(modules, conds, "fast", 1, "med", 30.0, 0.6)


# ----------------------------------------------------------------- loop numbers


def test_loop_numbers_match_log_fields(res):
    for lg in res.logs:
        n = al.loop_numbers(lg)
        assert n["predicted"] == lg.pred.coverage_pct
        assert n["feedback_pp"] == lg.bias
        assert n["after_feedback"] == lg.corrected_pred_cov
        assert n["card"] == lg.meas.coverage_pct
        assert n["actual"] == lg.outcome.coverage_pct
        assert n["error_pp"] == pytest.approx(lg.outcome.coverage_pct - lg.corrected_pred_cov)
        assert n["card_error_pp"] == lg.residual
        assert n["next_feedback_pp"] == lg.extra.get("bias_after", lg.bias)
        # eq. 3 learns from the card reading minus the RAW model prediction
        assert n["raw_error_pp"] == pytest.approx(lg.meas.coverage_pct - lg.pred.coverage_pct)
    assert any(abs(lg.bias) > 1e-9 for lg in res.logs)      # the feedback actually moves in a fast run


# ------------------------------------------------- what a correction is based on


def test_feedback_basis_agrees_with_the_corrections_the_loop_applied(modules):
    """The page says where each section's correction came from. That statement is rebuilt from
    the logs, so check it against what the fast loop really did, and against eq. 3 by hand."""
    from coverageai.config import spec
    fb = spec()["feedback"]
    alpha, warm = float(fb["alpha"]), int(fb["warmup_cards"])
    conds = al.make_conditions("gusty_afternoon", 1)
    res = al.run_pass_wrapper(modules, conds, "fast", 1, "med", 30.0, 0.6, fixed_nozzle="auto")
    basis = al.feedback_basis(res)
    assert len(basis) == res.n and all(b["uses_feedback"] for b in basis)
    assert {b["source"] for b in basis} >= {"group", "none"}
    own: dict[str, float] = {}          # each group's running correction, recomputed from the logs
    seen: dict[str, int] = {}
    for lg, b in zip(res.logs, basis):
        regime = lg.extra["regime"]
        assert b["cards_before"] == seen.get(regime, 0)
        assert b["group"] and b["group"] in b["text"]
        if b["source"] == "none":
            assert lg.bias == 0.0 and b["cards_before"] < warm
        elif b["source"] == "group":
            assert b["cards_before"] >= warm
            assert lg.bias == pytest.approx(own[regime], abs=1e-9)      # that group's own correction
            assert str(b["cards_before"]) in b["text"]
        else:
            assert b["source"] == "field" and b["cards_before"] == 0
        e = al.loop_numbers(lg)["raw_error_pp"]
        own[regime] = (1 - alpha) * own.get(regime, 0.0) + alpha * e
        seen[regime] = seen.get(regime, 0) + 1
        assert al.loop_numbers(lg)["next_feedback_pp"] == pytest.approx(
            own[regime] if seen[regime] >= warm else 0.0, abs=1e-9)


def test_feedback_basis_for_a_strategy_without_feedback(modules):
    conds = al.make_conditions("gusty_afternoon", 1, n_segments=6)
    for policy in ("open", "oracle", "best_fixed"):
        res = al.run_pass_wrapper(modules, conds, policy, 1, "med", 30.0, 0.6)
        basis = al.feedback_basis(res)
        assert len(basis) == res.n and not any(b["uses_feedback"] for b in basis), policy
        assert all(b["source"] == "none" and "does not use card feedback" in b["text"] for b in basis)
        assert field_animation_payload(res, 30.0)["feedback"] is False


def test_regime_text_and_rule_text_use_plain_words_and_the_spec_numbers():
    from coverageai.config import spec
    fb = spec()["feedback"]
    assert al.regime_text("u0-2|fine") == "crosswind 0 to 2 m/s, fine droplets"
    assert al.regime_text("u4-99|coarse") == "crosswind above 4 m/s, coarse droplets"
    assert al.regime_text("open") == al.regime_text("fixed") == al.regime_text(None) == ""
    rule = al.feedback_rule_text()
    assert f"{100 * fb['alpha']:.0f}% of (card reading minus model prediction)" in rule
    assert f"{fb['warmup_cards']} cards" in rule and f"{fb['bias_clip_pp']:g} pp" in rule
    for raw in ("alpha", "regime", "warmup", "u_cross", "bias"):
        assert raw not in rule


def test_payload_equals_timeseries_frame_rounded(res):
    """Animation panels and the charts under it show the same numbers."""
    p = field_animation_payload(res, 30.0, cards=False, advice=False)
    df = al.timeseries_frame(res.logs)
    assert len(p["sections"]) == len(df) == res.n
    for s, (_, row) in zip(p["sections"], df.iterrows()):
        assert s["number"] == row["k"]
        assert s["feedback_pp"] == pytest.approx(round(row["bias"], 2), abs=1e-9)
        assert s["error_pp"] == pytest.approx(round(row["error"], 2), abs=1e-9)
        assert s["predicted"] == pytest.approx(round(row["predicted"], 1), abs=1e-9)
        assert s["card"] == pytest.approx(round(row["measured"], 1), abs=1e-9)
        assert s["actual"] == pytest.approx(round(row["true"], 1), abs=1e-9)
        assert s["after_feedback"] == pytest.approx(round(row["corrected"], 1), abs=1e-9)


def test_timeseries_frame_columns_and_upto(res):
    df = al.timeseries_frame(res.logs, upto=3)
    assert len(df) == 4 and list(df["k"]) == [1, 2, 3, 4]
    for c in ("predicted", "corrected", "measured", "true", "bias", "bias_after", "residual", "error"):
        assert c in df.columns
    full = al.timeseries_frame(res.logs)
    assert full["error"].tolist() == pytest.approx(
        [lg.outcome.coverage_pct - lg.corrected_pred_cov for lg in res.logs])


def test_sel_n_candidates_is_the_real_count(res, modules):
    n = len(modules.selector.candidates)
    assert n > 100
    assert all(lg.extra["sel_n_candidates"] == n for lg in res.logs)


def test_logs_frame_row_count(res):
    f = al.logs_frame(res)
    assert len(f) == res.n
    assert f["extra_sel_n_candidates"].notna().all()
    assert f["k"].tolist() == list(range(res.n)) if "k" in f.columns else True


# ------------------------------------------------------------- setting changes


def _with(lg, **kw):
    return dataclasses.replace(lg, **kw)


def test_setting_changes_detects_each_controlled_change(res):
    a = res.logs[0]
    assert al.setting_changes(a, a) == []
    p = _with(a, u=dataclasses.replace(a.u, pressure_psi=a.u.pressure_psi + 10))
    ch = al.setting_changes(a, p)
    assert len(ch) == 1 and ch[0].startswith("pressure") and "+10" in ch[0]
    nz = _with(a, u=dataclasses.replace(a.u, family="TP65", size="08"))
    ch = al.setting_changes(a, nz)
    assert any(c.startswith("nozzle") and "TP65 08" in c for c in ch)
    sp = _with(a, u=dataclasses.replace(a.u, travel_speed_kmh=10.0))
    ch = al.setting_changes(a, sp)
    assert len(ch) == 1 and ch[0].startswith("speed") and "10" in ch[0]
    bm = _with(a, u=dataclasses.replace(a.u, boom_height_m=0.4))
    ch = al.setting_changes(a, bm)
    assert len(ch) == 1 and ch[0].startswith("boom") and "0.40" in ch[0]


def test_setting_changes_ignores_field_speed_and_height(res):
    a = res.logs[0]
    assert a.u.travel_speed_kmh is None and a.u.boom_height_m is None
    x2 = dataclasses.replace(a.x, travel_speed_kmh=a.x.travel_speed_kmh + 3.0,
                             boom_height_m=a.x.boom_height_m + 0.1)
    assert al.setting_changes(a, _with(a, x=x2)) == []


def test_controller_speed_equal_to_field_speed_is_no_change(res):
    a = res.logs[0]
    same = _with(a, u=dataclasses.replace(a.u, travel_speed_kmh=a.x.travel_speed_kmh))
    assert al.setting_changes(a, same) == []


def test_decision_frame_rows_and_changed_column(res):
    df = al.decision_frame(res)
    assert len(df) == res.n and list(df["section"]) == list(range(1, res.n + 1))
    assert df.loc[0, "changed"] == ""
    for k in range(1, res.n):
        same = res.logs[k].u == res.logs[k - 1].u
        assert (df.loc[k, "changed"] == "") == same, k
    assert (df["changed"] != "").any()
    assert df["predicted_cov"].tolist() == pytest.approx([lg.pred.coverage_pct for lg in res.logs])
    assert df["error_pp"].tolist() == pytest.approx(al.timeseries_frame(res.logs)["error"].tolist())


# ------------------------------------------------------------------ run changes


def test_run_changes_lists_inputs_results_and_changed_sections(modules):
    conds = al.make_conditions("gusty_afternoon", 1, n_segments=12)
    a = al.run_pass_wrapper(modules, conds, "open", 1, "med", 30.0, 0.6)
    b = al.run_pass_wrapper(modules, conds, "fast", 1, "med", 30.0, 0.6)
    a.meta["inputs"] = {"Control strategy": "Model only", "Target (%)": 30.0, "Same": 1}
    b.meta["inputs"] = {"Control strategy": "Model + card feedback", "Target (%)": 30.0, "Same": 1}
    d = al.run_changes(a, b)
    assert d["comparable"] and d["n"] == 12
    assert d["inputs"]["Input"].tolist() == ["Control strategy"]
    assert d["inputs"].iloc[0]["Previous run"] == "Model only"
    assert d["inputs"].iloc[0]["This run"] == "Model + card feedback"
    sa, sb = al.run_summary(a), al.run_summary(b)
    assert set(d["results"]["Result"]) == set(sb)
    for _, r in d["results"].iterrows():
        assert r["Change"] == pytest.approx(sb[r["Result"]] - sa[r["Result"]])
        assert r["Previous run"] == pytest.approx(sa[r["Result"]])
    expect = [k + 1 for k, (p, c) in enumerate(zip(a.logs, b.logs)) if al.setting_changes(p, c)]
    assert expect == [k + 1 for k, (p, c) in enumerate(zip(a.logs, b.logs)) if p.u != c.u]
    assert d["sections"]["Section"].tolist() == expect and d["n_changed"] == len(expect)
    for _, r in d["sections"].iterrows():
        k = int(r["Section"]) - 1
        assert r["This run"] == al.setting_text(b.logs[k].u)
        assert r["Predicted coverage change (pp)"] == pytest.approx(
            b.logs[k].pred.coverage_pct - a.logs[k].pred.coverage_pct)


def test_run_changes_identical_runs_and_different_lengths(modules):
    conds = al.make_conditions("gusty_afternoon", 1, n_segments=8)
    a = al.run_pass_wrapper(modules, conds, "fast", 1, "med", 30.0, 0.6)
    b = al.run_pass_wrapper(modules, conds, "fast", 1, "med", 30.0, 0.6)
    d = al.run_changes(a, b)
    assert d["n_changed"] == 0 and d["sections"].empty and d["inputs"].empty
    assert (d["results"]["Change"].abs() < 1e-9).all()
    c = al.run_pass_wrapper(modules, conds[:5], "fast", 1, "med", 30.0, 0.6)
    d2 = al.run_changes(a, c)
    assert d2["comparable"] is False and d2["sections"].empty


# ------------------------------------------------------------------ paper cards


def _decode(b64: str):
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def test_cards_captured_with_b0_and_numbers_unchanged():
    mods = integration.build_modules("oracle_nominal", "b0")
    conds = al.make_conditions("gusty_afternoon", 1, n_segments=6)
    with_cards = al.run_pass_wrapper(mods, conds, "fast", 1, "med", 30.0, 0.6, capture_cards=True)
    without = al.run_pass_wrapper(mods, conds, "fast", 1, "med", 30.0, 0.6, capture_cards=False)
    assert without.cards is None
    cards = with_cards.cards
    assert cards is not None and len(cards) == 6
    for c in cards:
        for key in ("img", "reader"):
            assert c[key]
            im = _decode(c[key])
            assert im is not None and im.ndim == 3 and im.shape[1] == al.CARD_THUMB_W
    # recording does not change a number the loop produced
    for p, q in zip(with_cards.logs, without.logs):
        assert p.u == q.u
        assert (p.outcome.coverage_pct, p.outcome.drift_frac, p.outcome.deposition_ul_cm2) == (
            q.outcome.coverage_pct, q.outcome.drift_frac, q.outcome.deposition_ul_cm2)
        a, b = dataclasses.asdict(p.meas), dataclasses.asdict(q.meas)
        a.pop("runtime_s"), b.pop("runtime_s")
        assert a == b
        assert p.corrected_pred_cov == q.corrected_pred_cov
    # the result carries the original objects, not the recording proxies
    assert with_cards.modules.verifier is mods.verifier
    assert with_cards.modules.renderer is mods.renderer
    assert type(with_cards.modules.verifier) is ClassicalVerifier
    # the reader overlay is drawn: reader thumbnail differs from the bare card
    assert any(c["reader"] != c["img"] for c in cards)


def test_perfect_reader_cards_are_rendered_on_demand(modules):
    conds = al.make_conditions("gusty_afternoon", 1, n_segments=6)
    r = al.run_pass_wrapper(modules, conds, "fast", 1, "med", 30.0, 0.6, capture_cards=True)
    assert r.cards is None                       # the perfect reader reads no image
    cards = al.ensure_cards(r)
    assert cards is r.cards and len(cards) == 6
    for c in cards:
        assert c["img"] and c["truth"] and "reader" not in c
        assert _decode(c["img"]) is not None and _decode(c["truth"]) is not None
    assert al.ensure_cards(r) is cards           # cached, not rendered again


def test_payload_card_fields(modules):
    conds = al.make_conditions("gusty_afternoon", 1, n_segments=6)
    r = al.run_pass_wrapper(modules, conds, "fast", 1, "med", 30.0, 0.6)
    p0 = field_animation_payload(r, 30.0, advice=False)
    assert p0["has_cards"] is False and all(s["img"] == "" for s in p0["sections"])
    al.ensure_cards(r)
    p1 = field_animation_payload(r, 30.0, advice=False)
    assert p1["has_cards"] and p1["overlay_name"] == "True stained area"
    assert all(s["img"] and s["overlay"] for s in p1["sections"])
    p2 = field_animation_payload(r, 30.0, cards=False, advice=False)
    assert p2["has_cards"] is False and all(s["img"] == "" for s in p2["sections"])


def test_measure_full_matches_measure_and_segment():
    mods = integration.build_modules("oracle_nominal", "b0")
    x = al.make_conditions("gusty_afternoon", 1, n_segments=3)[1]
    u = mods.selector.select(x, mods.model, 0.0, 30.0)[0]
    out = mods.simulator.simulate(x, u, np.random.default_rng(5), "med", sample_droplets=True)
    image, _ = mods.renderer.render(out, np.random.default_rng(6))
    v = mods.verifier
    ppm = px_per_mm()
    m1 = v.measure(image, ppm)
    m2, stain, roi = v.measure_full(image, ppm)
    a, b = dataclasses.asdict(m1), dataclasses.asdict(m2)
    a.pop("runtime_s"), b.pop("runtime_s")
    assert a == b
    s2, r2 = v.segment(image)
    assert stain.dtype == bool and roi.dtype == bool and stain.shape == image.shape[:2]
    assert np.array_equal(stain, s2) and np.array_equal(roi, r2)
    assert roi.any() and (stain & ~roi).sum() == 0
