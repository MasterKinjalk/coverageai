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


@pytest.mark.parametrize("same_field", [False, True])
def test_fast_slow_runs_learning_passes_and_replays(same_field):
    mods = al.build_modules_adapter("oracle_nominal", "perfect")
    conds = al.make_conditions("gusty_afternoon", 3, n_segments=6)
    orig_model = mods.model
    r = al.run_pass_wrapper(mods, conds, "fast_slow", 3, "high", 30.0, n_learn=2,
                            same_field=same_field)
    assert mods.model is orig_model                       # cached stack untouched
    # same_field: the displayed pass is on the given seed, the learning passes on later seeds;
    # default (legacy): the learning passes come first and the last pass is the one returned
    assert r.policy == "fast_slow" and r.seed == (3 if same_field else 3 + 1009 * 2)
    assert all(lg.policy == "fast_slow" and lg.seed == r.seed for lg in r.logs)
    assert type(r.modules.model).__name__ == "ResidualCorrectedModel"
    assert r.modules.model.n_records == 2 * len(conds)
    ps = r.meta["passes"]
    assert [p["pass"] for p in ps] == [1, 2, 3] and ps[-1]["displayed"] and not ps[0]["displayed"]
    assert all(0 <= p["met_pct"] <= 100 and p["mean_error_pp"] >= 0 for p in ps)
    for k in (0, 4):
        assert al.replay_card(r, k).consistent
    assert al.run_pass_wrapper(mods, conds, "fast_slow", 3, "high", 30.0, n_learn=0,
                               same_field=same_field).seed == 3


def _realized_wind(res):
    return [lg.extra["fc_realized_wind_ms"] for lg in res.logs]


def test_fast_slow_same_field_sees_the_same_hidden_realization(modules):
    """With ``same_field`` the displayed fast_slow pass gets the same gusts, canopy and wear as
    the other policies on that seed. The realized wind depends only on the hidden draws, not on
    the chosen setting, so it must match the open pass section by section."""
    conds = al.make_conditions("gusty_afternoon", 5, n_segments=8)
    kw = dict(lam=0.6, forecaster="ar1", n_learn=1)
    op = al.run_pass_wrapper(modules, conds, "open", 5, "high", 30.0, **kw)
    fs = al.run_pass_wrapper(modules, conds, "fast_slow", 5, "high", 30.0, same_field=True, **kw)
    assert fs.seed == op.seed == 5
    assert _realized_wind(fs) == _realized_wind(op)
    old = al.run_pass_wrapper(modules, conds, "fast_slow", 5, "high", 30.0, **kw)
    assert _realized_wind(old) != _realized_wind(op)       # the check is not vacuous
    # the outcome of a forced-equal setting is the same too: best_fixed on that seed with the
    # displayed pass's first setting reproduces its first section's coverage
    bf = al.run_pass_wrapper(modules, conds[:1], "best_fixed", 5, "high", 30.0,
                             fixed_setting=fs.logs[0].u)
    assert bf.logs[0].outcome.coverage_pct == fs.logs[0].outcome.coverage_pct


def test_compare_policies_fast_slow_on_the_same_field(modules):
    conds = al.make_conditions("gusty_afternoon", 2, n_segments=5)
    res = al.compare_policies(modules, conds, 2, "med", 30.0, policies=("open", "fast_slow"),
                              n_learn=1)
    assert res["fast_slow"].seed == res["open"].seed == 2


# ------------------------------------- selector records with a wind forecast (info_for)

def _top_setting(info) -> Setting:
    import dataclasses
    return Setting(**{f.name: info["top"][0][f.name] for f in dataclasses.fields(Setting)})


@pytest.mark.parametrize("policy", ["open", "fast", "fast_slow"])
def test_info_for_with_forecaster_is_the_applied_selection(modules, policy):
    conds = al.make_conditions("gusty_afternoon", 4, n_segments=10)
    r = al.run_pass_wrapper(modules, conds, policy, 4, "high", 30.0, 0.6, forecaster="ar1",
                            n_learn=1)
    for k, lg in enumerate(r.logs):
        info = r.info_for(k)
        assert info is not None, k
        assert _top_setting(info) == lg.u
        if policy != "open":    # rule 7: the record of the final selection, not the first one
            assert info["bias"] == lg.extra["selection_bias"]


@pytest.mark.parametrize("policy", ["open", "fast", "fast_slow", "oracle"])
def test_info_for_without_forecaster_is_unchanged(modules, policy):
    conds = al.make_conditions("gusty_afternoon", 4, n_segments=10)
    r = al.run_pass_wrapper(modules, conds, policy, 4, "high", 30.0, 0.6, n_learn=1)
    for k, lg in enumerate(r.logs):
        assert r.info_for(k) is r.records[(lg.x, lg.u)]
    bf = al.run_pass_wrapper(modules, conds, "best_fixed", 4, "high", 30.0)
    assert all(bf.info_for(k) is None for k in range(bf.n))


# ---------------------------------------------------------------- progress hook

@pytest.mark.parametrize("policy, n_learn", [("fast", 0), ("best_fixed", 0), ("fast_slow", 2)])
def test_progress_hook_counts_sections_and_changes_nothing(modules, policy, n_learn):
    def digest(logs):
        return [(lg.u, lg.pred, lg.outcome.coverage_pct, lg.outcome.drift_frac,
                 lg.meas.coverage_pct, lg.bias, lg.flagged_retreat, lg.selector_infeasible)
                for lg in logs]

    conds = al.make_conditions("gusty_afternoon", 4, n_segments=6)
    calls = []
    a = al.run_pass_wrapper(modules, conds, policy, 4, "high", 30.0, 0.6, n_learn=n_learn)
    b = al.run_pass_wrapper(modules, conds, policy, 4, "high", 30.0, 0.6, n_learn=n_learn,
                            progress=lambda d, t: calls.append((d, t)))
    assert digest(a.logs) == digest(b.logs)
    total = len(conds) * (n_learn + 1 if policy == "fast_slow" else 1)
    assert calls[-1] == (total, total)
    assert len(calls) >= total and all(t == total for _, t in calls)
    assert [d for d, _ in calls] == sorted(d for d, _ in calls)


def test_progress_hook_error_propagates_and_selector_is_restored(modules):
    conds = al.make_conditions("gusty_afternoon", 4, n_segments=6)
    before = (*_selector_state(modules.selector), modules.selector.nozzle_lock)

    def boom(done, total):
        if done == 3:
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        al.run_pass_wrapper(modules, conds, "fast", 4, "high", 30.0, 0.1, speed_control=True,
                            fixed_nozzle="auto", progress=boom)
    assert (*_selector_state(modules.selector), modules.selector.nozzle_lock) == before


def test_stack_lock_is_reentrant():
    assert al._STACK_LOCK.acquire(blocking=False)
    try:
        assert al._STACK_LOCK.acquire(blocking=False)     # RLock: same thread may re-enter
        al._STACK_LOCK.release()
    finally:
        al._STACK_LOCK.release()


# ------------------------------------------------------- card thumbnails and replay

def test_card_thumb_size_and_type():
    import base64

    import cv2
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (900, 1400, 3), dtype=np.uint8)
    mask = rng.random((900, 1400)) < 0.3
    for m in (None, mask):
        s = al.card_thumb(img, m)
        assert isinstance(s, str)
        out = cv2.imdecode(np.frombuffer(base64.b64decode(s), np.uint8), cv2.IMREAD_COLOR)
        assert out.shape == (int(900 * al.CARD_THUMB_W / 1400), al.CARD_THUMB_W, 3)
    # a fully stained card is tinted toward the overlay color
    red = cv2.imdecode(np.frombuffer(base64.b64decode(
        al.card_thumb(np.zeros_like(img), np.ones(mask.shape, bool))), np.uint8), cv2.IMREAD_COLOR)
    assert red[..., 2].mean() > 100 and red[..., 0].mean() < 60       # BGR: red high, blue low


def test_replay_card_reads_once_with_measure_full():
    mods = al.build_modules_adapter("oracle_nominal", "b0")
    conds = al.make_conditions("gusty_afternoon", 4, n_segments=2)
    r = al.run_pass_wrapper(mods, conds, "fast", 4, "med", 30.0)
    calls = {"measure_full": 0, "measure": 0, "segment": 0}
    v = mods.verifier
    orig = {name: getattr(v, name) for name in calls}

    def spy(name):
        def f(*a, **k):
            calls[name] += 1
            return orig[name](*a, **k)
        return f

    for name in calls:
        setattr(v, name, spy(name))
    try:
        cv = al.replay_card(r, 1)
    finally:
        for name in calls:
            delattr(v, name)
    assert calls == {"measure_full": 1, "measure": 0, "segment": 0}
    assert cv.pred_mask is not None and cv.pred_mask.dtype == bool
    assert abs(cv.meas.coverage_pct - r.logs[1].meas.coverage_pct) < 1e-6


def test_replay_card_does_not_hide_reader_errors(short_pass):
    class BadReader:
        def measure(self, image, px_per_mm):
            return None

        def segment(self, image, px_per_mm=None):
            raise RuntimeError("reader broke")

    import dataclasses
    res = dataclasses.replace(short_pass, modules=dataclasses.replace(short_pass.modules,
                                                                      verifier=BadReader()))
    with pytest.raises(RuntimeError, match="reader broke"):
        al.replay_card(res, 0)


def test_ensure_cards_explains_why_there_are_none(short_pass):
    import dataclasses
    res = dataclasses.replace(short_pass, cards=None, meta={},
                              modules=dataclasses.replace(short_pass.modules, renderer=None))
    assert al.ensure_cards(res) is None
    note = res.meta["cards_note"]
    assert isinstance(note, str) and note and "_" not in note
    # a renderer that fails: None, a note for the page, and the error kept for debugging
    class NoDroplets:
        def render(self, outcome, rng, degrade=True):
            raise ValueError("outcome has no droplets")

    bad = dataclasses.replace(short_pass, cards=None, meta={},
                              modules=dataclasses.replace(short_pass.modules, renderer=NoDroplets()))
    assert al.ensure_cards(bad) is None
    assert bad.meta["cards_note"] and "_" not in bad.meta["cards_note"]
    assert "no droplets" in bad.meta["cards_error"]


def test_concurrent_runs_do_not_mix_selector_state(modules):
    """Two sessions sharing the cached stack at once: each run must equal the same run alone."""
    import threading
    conds = al.make_conditions("gusty_afternoon", 6, n_segments=6)
    jobs = {"a": dict(lam=0.05, speed_control=True), "b": dict(lam=0.95, fixed_nozzle="XR-04")}
    alone = {k: al.run_pass_wrapper(modules, conds, "fast", 6, "high", 30.0, **kw)
             for k, kw in jobs.items()}
    got, errors = {}, []

    def work(k):
        try:
            got[k] = al.run_pass_wrapper(modules, conds, "fast", 6, "high", 30.0, **jobs[k])
        except Exception as exc:     # surfaced below
            errors.append(exc)

    ts = [threading.Thread(target=work, args=(k,)) for k in jobs for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    for k in jobs:
        assert [lg.u for lg in got[k].logs] == [lg.u for lg in alone[k].logs]
    assert modules.selector.nozzle_lock is None and modules.selector.speed_options_kmh == []


def test_app_policies_all_run_and_oracle_card_in_compare(modules):
    assert {"fast_slow", "oracle", "oracle_card"} <= set(al.APP_POLICIES)
    conds = al.make_conditions("gusty_afternoon", 2, n_segments=5)
    res = al.compare_policies(modules, conds, 2, "med", 30.0, policies=al.APP_POLICIES, n_learn=1)
    assert list(res) == list(al.APP_POLICIES)
    assert len(al.comparison_table(res)) == len(al.APP_POLICIES)


def test_noisy_verifier_is_not_reported_as_image_analysis():
    """The image-free noisy sensor never reads a rendered card in the loop, so the pass must not
    be flagged as vision (the app would otherwise say 'image analysis')."""
    mods = al.build_modules_adapter("oracle_nominal", "noisy")
    conds = al.make_conditions("calm_morning", 2, n_segments=5)
    r = al.run_pass_wrapper(mods, conds, "fast", 2, "med", 30.0)
    assert hasattr(mods.verifier, "measure_outcome")
    assert r.used_vision is False
    # same numbers as an explicit render=False pass (nothing else changed)
    r0 = al.run_pass_wrapper(mods, conds, "fast", 2, "med", 30.0, render=False)
    assert [lg.meas.coverage_pct for lg in r.logs] == [lg.meas.coverage_pct for lg in r0.logs]


def test_perfect_and_image_verifiers_keep_their_vision_flag(modules):
    conds = al.make_conditions("calm_morning", 2, n_segments=3)
    assert al.run_pass_wrapper(modules, conds, "open", 2, "none", 30.0).used_vision is False


# ------------------------------------------------- opt-in controller options (sidebar "Advanced")

def test_options_off_are_bit_identical_to_default(modules):
    conds = al.make_conditions("gusty_afternoon", 4, n_segments=10)
    a = al.run_pass_wrapper(modules, conds, "fast", 4, "high", 30.0, 0.6)
    b = al.run_pass_wrapper(modules, conds, "fast", 4, "high", 30.0, 0.6, speed_control=False,
                            boom_control=False, forecaster="off")
    assert [lg.u for lg in a.logs] == [lg.u for lg in b.logs]
    assert [lg.outcome.coverage_pct for lg in a.logs] == [lg.outcome.coverage_pct for lg in b.logs]
    assert all(lg.u.travel_speed_kmh is None and lg.u.boom_height_m is None for lg in b.logs)
    assert not any("fc_wind_ms" in lg.extra for lg in b.logs)


def test_speed_and_boom_control_run_and_restore_selector(modules):
    conds = al.make_conditions("gusty_afternoon", 4, n_segments=10)
    r = al.run_pass_wrapper(modules, conds, "fast", 4, "med", 30.0, 0.6, speed_control=True,
                            boom_control=True)
    assert r.n == 10
    assert any(lg.u.travel_speed_kmh is not None for lg in r.logs)
    assert any(lg.u.boom_height_m is not None for lg in r.logs)
    # the shared, cached selector is back to its defaults (off) after the call
    assert modules.selector.speed_options_kmh == [] and modules.selector.boom_height_options_m == []
    # the card replay still reproduces the recorded outcome with the controlled setting
    assert all(al.replay_card(r, k).consistent for k in (0, 5, 9))
    # best_fixed keeps its single setting (it does not select)
    bf = al.run_pass_wrapper(modules, conds, "best_fixed", 4, "med", 30.0, speed_control=True)
    assert len({lg.u for lg in bf.logs}) == 1


def test_forecaster_option_runs_for_supported_policies_only(modules):
    conds = al.make_conditions("gusty_afternoon", 4, n_segments=10)
    on = al.run_pass_wrapper(modules, conds, "fast", 4, "high", 30.0, 0.6, forecaster="ar1")
    assert all("fc_plan_wind_ms" in lg.extra for lg in on.logs)
    orc = al.run_pass_wrapper(modules, conds, "oracle", 4, "high", 30.0, 0.6, forecaster="ar1")
    orc0 = al.run_pass_wrapper(modules, conds, "oracle", 4, "high", 30.0, 0.6)
    assert [lg.u for lg in orc.logs] == [lg.u for lg in orc0.logs]      # ignored by the oracle
    fs = al.run_pass_wrapper(modules, conds, "fast_slow", 4, "high", 30.0, 0.6, forecaster="ar1",
                             speed_control=True, n_learn=1)
    assert fs.n == 10 and len(fs.meta["passes"]) == 2
    with pytest.raises(ValueError):
        al.run_pass_wrapper(modules, conds, "fast", 4, "high", 30.0, forecaster="bogus")


def test_top_candidates_label_includes_controlled_speed_and_height(modules):
    conds = al.make_conditions("calm_morning", 1, n_segments=4)
    r = al.run_pass_wrapper(modules, conds, "open", 1, "none", 30.0, speed_control=True,
                            boom_control=True)
    txt = " ".join(al.top_candidates_frame(r.info_for(0))["Setting"])
    assert "km/h" in txt and "boom" in txt


# ---------------------------- review fixes: failed calls leave the cached selector untouched

def _selector_state(sel):
    return (sel.lam, list(sel.speed_options_kmh), list(sel.boom_height_options_m))


def test_failed_call_leaves_cached_selector_unmodified(modules):
    """Bad forecaster name: the ValueError must fire before lam / option grids are mutated."""
    conds = al.make_conditions("gusty_afternoon", 4, n_segments=6)
    before = _selector_state(modules.selector)
    with pytest.raises(ValueError):
        al.run_pass_wrapper(modules, conds, "fast", 1, "med", 30.0, 0.9, speed_control=True,
                            boom_control=True, forecaster="bogus")
    assert _selector_state(modules.selector) == before
    with pytest.raises(ValueError):
        al.compare_policies(modules, conds, 1, "med", 30.0, 0.9, policies=("open", "fast"),
                            speed_control=True, forecaster="bogus")
    assert _selector_state(modules.selector) == before
    # a later "off" run equals a fresh off run (no leaked lambda / speed grid)
    a = al.run_pass_wrapper(modules, conds, "fast", 4, "med", 30.0)
    fresh = al.build_modules_adapter("oracle_nominal", "perfect")
    b = al.run_pass_wrapper(fresh, conds, "fast", 4, "med", 30.0)
    assert [lg.u for lg in a.logs] == [lg.u for lg in b.logs]
    assert [lg.outcome.coverage_pct for lg in a.logs] == [lg.outcome.coverage_pct for lg in b.logs]


def test_unsupported_selector_error_leaves_selector_unmodified(modules):
    """Selector that has the speed grid but not the boom grid: nothing may stay modified."""
    import dataclasses

    class SpeedOnlySelector:
        def __init__(self):
            self.lam = 0.6
            self.speed_options_kmh = []

        def select(self, x, model, bias, c_req):
            raise AssertionError("must not run")

    stub = SpeedOnlySelector()
    mods = dataclasses.replace(modules, selector=stub)
    conds = al.make_conditions("calm_morning", 1, n_segments=4)
    with pytest.raises(ValueError, match="boom-height"):
        al.run_pass_wrapper(mods, conds, "open", 1, "none", 30.0, 0.9, speed_control=True,
                            boom_control=True)
    assert stub.lam == 0.6 and stub.speed_options_kmh == []


# ------------------- review fix: fast_slow + forecaster, final refit uses the forecast wind

def _spy_final_refit(monkeypatch, modules, forecaster):
    from coverageai import feedback
    fits, slow_logs = [], []
    orig_fit = feedback.ResidualCorrectedModel.fit
    orig_slow = al.run_passes_slow

    def fit_spy(self, records, *a, **k):
        fits.append(list(records))
        return orig_fit(self, records, *a, **k)

    def slow_spy(*a, **k):
        logs = orig_slow(*a, **k)
        slow_logs.append(logs)
        return logs

    monkeypatch.setattr(feedback.ResidualCorrectedModel, "fit", fit_spy)
    monkeypatch.setattr(al, "run_passes_slow", slow_spy)
    conds = al.make_conditions("gusty_afternoon", 4, n_segments=10)
    al.run_pass_wrapper(modules, conds, "fast_slow", 4, "high", 30.0, 0.6, forecaster=forecaster,
                        n_learn=1)
    return fits, slow_logs


def test_fast_slow_final_refit_uses_forecast_wind(modules, monkeypatch):
    fits, slow_logs = _spy_final_refit(monkeypatch, modules, "ar1")
    assert len(fits) == 1 and len(slow_logs) == 1        # n_learn=1: only the final refit fits
    logs = slow_logs[0]
    assert all("fc_wind_ms" in lg.extra for lg in logs)
    recs = fits[0]
    assert [r[0].wind_speed_ms for r in recs] == [lg.extra["fc_wind_ms"] for lg in logs]
    assert any(lg.extra["fc_wind_ms"] != lg.x.wind_speed_ms for lg in logs)   # not vacuous


def test_fast_slow_final_refit_forecaster_off_uses_nominal_wind(modules, monkeypatch):
    fits, slow_logs = _spy_final_refit(monkeypatch, modules, None)
    assert len(fits) == 1
    logs = slow_logs[0]
    assert [r[0] for r in fits[0]] == [lg.x for lg in logs]
