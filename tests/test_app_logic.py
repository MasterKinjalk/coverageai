"""Unit tests for the non-UI helpers behind the Streamlit demo (coverageai.app_logic)."""
from __future__ import annotations

import numpy as np
import pytest

from coverageai import app_logic as al
from coverageai import scenarios
from coverageai.config import spec
from coverageai.contracts import Conditions, Setting


@pytest.fixture(scope="module")
def modules():
    return al.build_modules_adapter("oracle_nominal", "perfect")


@pytest.fixture(scope="module")
def short_pass(modules):
    conds = al.make_conditions("gusty_afternoon", 3, n_segments=12)
    return al.run_pass_wrapper(modules, conds, "fast", 3, "med", 30.0)


def test_make_conditions_reproducible_and_length():
    a = al.make_conditions("calm_morning", 5, n_segments=15)
    b = al.make_conditions("calm_morning", 5, n_segments=15)
    assert len(a) == 15 and a == b
    assert len(al.make_conditions("calm_morning", 5)) == spec()["pass"]["segments"]


def test_run_pass_wrapper_logs_and_records(short_pass):
    r = short_pass
    assert r.n == 12 and len(r.logs) == 12 and r.policy == "fast"
    assert [lg.k for lg in r.logs] == list(range(12))
    # selector info (top-5 table) is recoverable for every selected segment
    for k in range(r.n):
        info = r.info_for(k)
        assert info is not None and len(info["top"]) <= 5


def test_run_pass_wrapper_restores_lambda_and_is_deterministic(modules):
    conds = al.make_conditions("hot_dry", 2, n_segments=6)
    old = modules.selector.lam
    a = al.run_pass_wrapper(modules, conds, "open", 2, "high", 30.0, lam=0.1)
    b = al.run_pass_wrapper(modules, conds, "open", 2, "high", 30.0, lam=0.1)
    assert modules.selector.lam == old
    assert [lg.u for lg in a.logs] == [lg.u for lg in b.logs]
    assert [lg.outcome.coverage_pct for lg in a.logs] == [lg.outcome.coverage_pct for lg in b.logs]


def test_best_fixed_policy_uses_one_setting(modules):
    conds = al.make_conditions("calm_morning", 1, n_segments=6)
    r = al.run_pass_wrapper(modules, conds, "best_fixed", 1, "none", 30.0)
    assert len({lg.u for lg in r.logs}) == 1 and r.fixed_setting == r.logs[0].u


def test_compute_kpis_matches_logs(short_pass):
    r = short_pass
    kp = al.compute_kpis(r.logs, 30.0)
    cov = np.array([lg.outcome.coverage_pct for lg in r.logs])
    assert kp["n"] == 12 and kp["n_met"] == int((cov >= 30).sum())
    assert kp["mean_cov"] == pytest.approx(cov.mean())
    assert kp["mean_gpa"] == pytest.approx(np.mean([lg.derived.app_rate_gpa for lg in r.logs]))
    assert kp["n_retreat"] == sum(lg.flagged_retreat for lg in r.logs)
    part = al.compute_kpis(r.logs, 30.0, upto=3)          # only revealed segments count
    assert part["n"] == 4
    assert al.compute_kpis([], 30.0)["n"] == 0


def test_lap_layout_serpentine():
    base = dict(wind_speed_ms=1, wind_dir_deg=0, temp_c=20, rh_pct=50, travel_speed_kmh=15,
                boom_height_m=0.5)
    conds = [Conditions(heading_deg=0.0 if (k // 3) % 2 == 0 else 180.0, **base) for k in range(9)]
    lay = al.lap_layout(conds)
    assert lay[:3] == [(0, 0), (0, 1), (0, 2)]
    assert lay[3:6] == [(1, 2), (1, 1), (1, 0)]            # return lap runs right to left
    assert lay[6:] == [(2, 0), (2, 1), (2, 2)]
    assert len(set(lay)) == 9


def test_field_strip_hides_future_segments(short_pass):
    r = short_pass
    df = al.field_strip_data(r.conds, r.logs, cursor=4)
    assert len(df) == 12 and df["current"].sum() == 1 and df.loc[4, "current"]
    assert (df.loc[:4, "status"] != "pending").all()
    assert (df.loc[5:, "status"] == "pending").all()
    assert set(df["status"]) <= {"acceptable", "marginal", "critical", "pending"}
    # true basis classifies the simulator outcome
    dt = al.field_strip_data(r.conds, r.logs, cursor=11, basis="true")
    assert dt["true"].notna().all()
    flagged = {lg.k for lg in r.logs if lg.flagged_retreat}
    assert set(dt.loc[dt["retreat"], "k"]) == flagged


def test_timeseries_frame(short_pass):
    ts = al.timeseries_frame(short_pass.logs, upto=5)
    assert len(ts) == 6 and ts["k"].tolist() == [1, 2, 3, 4, 5, 6]
    assert np.allclose(ts["corrected"], ts["predicted"] + ts["bias"])
    assert np.allclose(ts["residual"], ts["measured"] - ts["corrected"])


def test_top_candidates_frame_and_explanation(short_pass):
    top = al.top_candidates_frame(short_pass.info_for(0))
    assert 1 <= len(top) <= 5 and "Setting" in top.columns
    assert al.top_candidates_frame(None).empty
    text = " ".join(al.explain_choice(short_pass, 0))
    assert "candidates meet" in text


def test_wind_arrow_angle_wraps():
    x = Conditions(3.0, 10.0, 350.0, 20, 50, 15, 0.5)
    assert al.wind_arrow_deg(x) == pytest.approx(20.0)
    x2 = Conditions(3.0, 350.0, 10.0, 20, 50, 15, 0.5)
    assert al.wind_arrow_deg(x2) == pytest.approx(-20.0)
    assert "<svg" in al.wind_arrow_svg(x)


def test_compare_policies_same_seed_oracle_best(modules):
    conds = al.make_conditions("gusty_afternoon", 1, n_segments=10)
    res = al.compare_policies(modules, conds, 1, "high", 30.0, policies=("open", "fast", "oracle"))
    tb = al.comparison_table(res)
    assert list(tb["policy"]) == ["open", "fast", "oracle"]
    met = dict(zip(tb["policy"], tb["requirement met (%)"]))
    assert met["oracle"] >= met["open"]
    # same conditions in every policy
    assert all(r.conds == res["open"].conds for r in res.values())


def test_replay_card_matches_logged_outcome(short_pass):
    cv = al.replay_card(short_pass, 5)
    assert cv.consistent
    assert cv.image.ndim == 3 and cv.gt_mask.dtype == bool
    ov = al.overlay_mask(cv.image, cv.gt_mask)
    assert ov.dtype == np.uint8 and ov.shape[1] <= 1100


def test_nozzle_helpers():
    x = Conditions(3.0, 90.0, 0.0, 25.0, 55.0, 16.0, 0.5)
    u = Setting("XR", "04", 40.0)
    pt = al.nozzle_point(u, x)
    assert pt["derived"].flow_gpm > 0 and 0 <= pt["pred"].coverage_pct <= 100
    cvp = al.coverage_vs_pressure("XR", "04", x, n=12)
    # volume rises with pressure (droplet class steps can cause small local dips)
    assert len(cvp) == 12 and cvp["coverage_pct"].iloc[-1] > cvp["coverage_pct"].iloc[0]
    dvw = al.drift_vs_wind(u, x, n=10)
    assert dvw["drift_frac"].iloc[-1] >= dvw["drift_frac"].iloc[0]
    g = al.coverage_grid("XR", "04", x, n_p=5, n_w=4)
    assert len(g) == 20
    pat = al.pattern_overlap(110.0, 0.5)
    assert 0.0 <= pat["uniformity"] <= 1.0
    # a wider fan overlaps more and is at least as uniform
    assert al.pattern_overlap(110.0, 0.6)["uniformity"] >= al.pattern_overlap(40.0, 0.4)["uniformity"]


def test_presets_available_for_app():
    assert set(al.APP_POLICIES) <= set(al.POLICIES)
    assert scenarios.list_presets()


def test_b0_verifier_pass_and_overlay():
    """Classical verifier path: cards are rendered, read, and the verifier mask is exposed."""
    mods = al.build_modules_adapter("oracle_nominal", "b0")
    conds = al.make_conditions("gusty_afternoon", 4, n_segments=3)
    r = al.run_pass_wrapper(mods, conds, "fast", 4, "med", 30.0)
    assert r.used_vision and all(lg.meas.runtime_s > 0 for lg in r.logs)
    cv = al.replay_card(r, 2)
    assert cv.meas is not None and cv.pred_mask is not None and cv.pred_mask.dtype == bool
    assert abs(cv.meas.coverage_pct - r.logs[2].meas.coverage_pct) < 1e-6


def test_noisy_verifier_replay_card():
    """The image-free noisy verifier can still read a replayed card (B0 under the hood)."""
    mods = al.build_modules_adapter("oracle_nominal", "noisy")
    conds = al.make_conditions("gusty_afternoon", 4, n_segments=3)
    r = al.run_pass_wrapper(mods, conds, "fast", 4, "med", 30.0)
    cv = al.replay_card(r, 1)
    assert cv.meas is not None and 0.0 <= cv.meas.coverage_pct <= 100.0
    assert cv.pred_mask is not None and cv.pred_mask.dtype == bool
    # the logged (image-free) reading and the image reading are both within a few pp of truth
    assert abs(cv.meas.coverage_pct - cv.outcome.coverage_pct) < 8.0


def test_fast_slow_runs_learning_passes_and_replays():
    mods = al.build_modules_adapter("oracle_nominal", "perfect")
    conds = al.make_conditions("gusty_afternoon", 3, n_segments=6)
    orig_model = mods.model
    r = al.run_pass_wrapper(mods, conds, "fast_slow", 3, "high", 30.0, n_learn=2)
    assert mods.model is orig_model                       # cached stack untouched
    assert r.policy == "fast_slow" and r.seed == 3 + 1009 * 2   # last pass is the one returned
    assert all(lg.policy == "fast_slow" and lg.seed == r.seed for lg in r.logs)
    assert type(r.modules.model).__name__ == "ResidualCorrectedModel"
    assert r.modules.model.n_records == 2 * len(conds)
    ps = r.meta["passes"]
    assert [p["pass"] for p in ps] == [1, 2, 3] and ps[-1]["displayed"] and not ps[0]["displayed"]
    assert all(0 <= p["met_pct"] <= 100 and p["mean_error_pp"] >= 0 for p in ps)
    for k in (0, 4):
        assert al.replay_card(r, k).consistent
    assert al.run_pass_wrapper(mods, conds, "fast_slow", 3, "high", 30.0, n_learn=0).seed == 3


def test_app_policies_all_run_and_oracle_card_in_compare(modules):
    assert {"fast_slow", "oracle", "oracle_card"} <= set(al.APP_POLICIES)
    conds = al.make_conditions("gusty_afternoon", 2, n_segments=5)
    res = al.compare_policies(modules, conds, 2, "med", 30.0, policies=al.APP_POLICIES, n_learn=1)
    assert list(res) == list(al.APP_POLICIES)
    assert len(al.comparison_table(res)) == len(al.APP_POLICIES)
