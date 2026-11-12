"""Headless UI tests for the Streamlit demo (app/main.py) using streamlit.testing.

The page has three views (top tabs). The sidebar holds the inputs of the view on screen with its
Run button on top; only that view is computed. A view is selected in a test by setting
``session_state["view"]``.

Regression kept from the first version: pressing "Run pass" a second time crashed with
StreamlitAPIException because the script reset a slider's session-state key after the slider had
been created in the same run (the "Section" slider, key "section", is reset in the button callback).
"""
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "main.py")
FIELD, COMPARE, LAB = "Field run", "Compare strategies", "Nozzle lab"


def _button(at, label):
    return next(b for b in at.sidebar.button if b.label == label)


def _fast_app(**params):
    """AppTest on the simple fast path: physics model, perfect reader, no card drawing."""
    at = AppTest.from_file(APP, default_timeout=240)
    at.query_params["verifier"] = "perfect"
    at.query_params["model"] = "oracle_nominal"
    at.query_params["cards"] = "0"
    for k, v in params.items():
        at.query_params[k] = v
    at.run()
    assert not at.exception, at.exception
    return at


def _slider(at, label):
    return next(s for s in at.slider if s.label == label)


def _show(at, view):
    at.session_state["view"] = view
    at.run()
    assert not at.exception, at.exception
    return at


def _sidebar_labels(at):
    sb = at.sidebar
    return [w.label for group in (sb.selectbox, sb.select_slider, sb.slider, sb.multiselect, sb.number_input,
                                  sb.checkbox, sb.toggle) for w in group]


def test_run_pass_twice_does_not_crash():
    at = _fast_app()
    sl = _slider(at, "Section")
    sl.set_value(min(3, sl.max)).run()          # move the section detail off section 1
    assert not at.exception, at.exception
    assert at.session_state["section"] == min(3, sl.max) > 1
    _button(at, "Run pass").click().run()
    assert not at.exception, at.exception
    assert at.session_state["section"] == 1
    for gone in ("cursor", "playing", "advance"):
        assert gone not in at.session_state
    assert not [b for b in at.sidebar.button if b.label in ("Back", "Step", "Play all")]


def test_default_nozzle_is_auto_and_run_uses_one_nozzle():
    at = _fast_app()
    sel = next(s for s in at.sidebar.selectbox if s.label == "Nozzle on the boom")
    assert sel.value == "auto"
    res = at.session_state["result"]
    assert res.meta["nozzle"]["mode"] == "auto"
    assert len({(lg.u.family, lg.u.size) for lg in res.logs}) == 1
    assert at.session_state["previous"] is None


def test_free_nozzle_query_param_runs_and_may_change_nozzle():
    # with the running-average feedback this field uses several nozzles when the choice is free;
    # with the trained feedback (the app default) the same field happens to keep one, which is
    # allowed, so the "several nozzles" evidence is taken on the rule
    at = _fast_app(nozzle="free", feedback="rule")
    sel = next(s for s in at.sidebar.selectbox if s.label == "Nozzle on the boom")
    assert sel.value == "free"
    res = at.session_state["result"]
    assert res.meta["nozzle"]["mode"] == "free"
    assert len({(lg.u.family, lg.u.size) for lg in res.logs}) > 1


def test_named_nozzle_is_held():
    at = _fast_app(nozzle="XR-04")
    res = at.session_state["result"]
    assert res.meta["nozzle"]["mode"] == "fixed"
    assert {(lg.u.family, lg.u.size) for lg in res.logs} == {("XR", "04")}


def test_second_run_fills_previous():
    at = _fast_app()
    first = at.session_state["result"]
    _button(at, "Run pass").click().run()
    assert not at.exception, at.exception
    assert at.session_state["previous"] is first
    assert at.session_state["result"] is not first


def test_field_detail_tabs():
    at = _fast_app()
    labels = [t.label for t in at.tabs]
    assert {FIELD, COMPARE, LAB} <= set(labels)
    for want in ("Section detail", "Recommendations (", "Setting changes (", "Previous run", "Feedback loop",
                 "All numbers"):
        assert any(l.startswith(want) for l in labels), (want, labels)
    n_rec = next(l for l in labels if l.startswith("Recommendations ("))
    n_chg = next(l for l in labels if l.startswith("Setting changes ("))
    from coverageai import app_logic as al
    res = at.session_state["result"]
    assert n_rec == f"Recommendations ({len(al.recommendation_frame(res))})"
    assert n_chg == f"Setting changes ({int((al.decision_frame(res)['changed'] != '').sum())})"


RAW = ("oracle_nominal", "best_fixed", "C_req", "gusty_afternoon", "acceptable", "fast_slow", "mock")


def test_no_raw_identifiers_in_widget_labels_or_options():
    at = AppTest.from_file(APP, default_timeout=180)
    at.query_params["verifier"] = "perfect"
    at.query_params["cards"] = "0"
    at.run()
    assert not at.exception, at.exception
    for view in (FIELD, COMPARE, LAB):
        _show(at, view)
        texts = []
        for w in (list(at.sidebar.selectbox) + list(at.sidebar.select_slider) + list(at.sidebar.slider)
                  + list(at.sidebar.multiselect) + list(at.main.selectbox) + list(at.main.multiselect)
                  + list(at.main.radio)):
            texts.append(w.label)
            texts.extend(getattr(w, "options", []))   # options are the format_func-rendered strings
        assert texts, view
        for t in texts:
            for raw in RAW:
                assert raw not in str(t), (view, raw, t)
            assert str(t) not in ("auto", "free"), (view, t)
    _show(at, FIELD)
    noz = next(s for s in at.sidebar.selectbox if s.label == "Nozzle on the boom")
    assert len(noz.options) > 3
    assert all(not o.startswith(("auto", "free")) and "-" not in o.split(" ")[0] for o in noz.options)
    models = next(s for s in at.sidebar.selectbox if s.label == "Coverage predictor")
    assert len(models.options) == 2            # the test stub is not offered in the app


def test_switching_policies_and_verifiers_does_not_crash():
    at = AppTest.from_file(APP, default_timeout=180)
    at.query_params["verifier"] = "perfect"     # b0 is slow; keep the strategy sweep fast
    at.query_params["cards"] = "0"
    at.run()
    assert not at.exception, at.exception
    pol = next(s for s in at.sidebar.selectbox if s.label == "Control strategy")
    assert len(pol.options) == 6
    for opt in list(pol.options):
        pol.select(opt)
        _button(at, "Run pass").click().run()
        assert not at.exception, (opt, at.exception)
        assert not at.error, (opt, [e.value for e in at.error])
        pol = next(s for s in at.sidebar.selectbox if s.label == "Control strategy")
    ver = next(s for s in at.sidebar.selectbox if s.label == "Card reader")
    for opt in ("Perfect reading (true value)", "Noisy reading (simulated sensor)",
                "Classic image analysis"):
        ver.select(opt)
        _button(at, "Run pass").click().run()
        assert not at.exception, (opt, at.exception)
        assert not at.error, (opt, [e.value for e in at.error])
        ver = next(s for s in at.sidebar.selectbox if s.label == "Card reader")


def test_fast_slow_shows_learning_tab():
    at = _fast_app()
    pol = next(s for s in at.sidebar.selectbox if s.label == "Control strategy")
    pol.select("Model + card feedback + learning between passes")
    _button(at, "Run pass").click().run()
    assert not at.exception, at.exception
    assert "Learning between passes" in [t.label for t in at.tabs]
    assert at.session_state["result"].seed == 1      # the replayed pass is the field of this seed


def test_policy_bars_show_every_strategy_name():
    """Compare view: all strategy names are y-axis categories (previously only the first and last
    labels of the vertical bars survived label overlap removal)."""
    import sys
    sys.path.insert(0, str(Path(APP).parent))
    import pandas as pd
    import charts
    names = ["Model only", "Model + card feedback", "Model + learning", "Knows hidden conditions"]
    df = pd.DataFrame({"policy": names, "v": [1.0, 2.0, 3.0, 4.0]})
    spec = charts.policy_bars(df, "v", "Value").to_dict()
    bars = spec["layer"][0]
    assert bars["encoding"]["y"]["field"] == "policy" and bars["encoding"]["y"]["sort"] == names
    assert bars["encoding"]["y"]["axis"].get("labelLimit", 0) >= 150
    hidden = charts.policy_bars(df, "v", "Value", show_labels=False).to_dict()["layer"][0]
    assert hidden["encoding"]["y"]["axis"]["labels"] is False
    # an interval is drawn only when every bound is known
    ci = df.assign(lo=df["v"] - 0.5, hi=df["v"] + 0.5)
    assert len(charts.policy_bars(ci, "v", "Value", lo="lo", hi="hi").to_dict()["layer"]) == 3
    one_seed = df.assign(lo=float("nan"), hi=float("nan"))
    assert len(charts.policy_bars(one_seed, "v", "Value", lo="lo", hi="hi").to_dict()["layer"]) == 2


def test_coverage_chart_bands_are_the_fixed_classes_not_the_target():
    """The Good / Borderline / Poor bands are the spec's class limits. The target is its own line."""
    import sys
    sys.path.insert(0, str(Path(APP).parent))
    import pandas as pd
    import charts
    ts = pd.DataFrame({"k": [1, 2], "predicted": [40.0, 41.0], "corrected": [40.0, 41.0],
                       "measured": [39.0, 42.0], "true": [39.5, 41.5]})
    spec = charts.coverage_timeseries(ts, 40.0, 15.0, 30.0, 2).to_dict()
    data = spec["datasets"]
    band = next(rows for rows in data.values() if rows and "cls" in rows[0])
    assert [(r["lo"], r["hi"]) for r in band][:2] == [(0, 15.0), (15.0, 30.0)]
    assert any(rows == [{"y": 40.0}] for rows in data.values())      # the target line


def test_noisy_reader_run_is_not_called_image_analysis():
    at = AppTest.from_file(APP, default_timeout=240)
    at.query_params["verifier"] = "noisy"
    at.query_params["model"] = "oracle_nominal"
    at.run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "image analysis" not in html and "simulated sensor" in html
    assert any("simulated sensor" in c.value for c in at.caption)


def test_advanced_options_default_off_and_run_when_on():
    at = AppTest.from_file(APP, default_timeout=240)
    at.query_params["verifier"] = "perfect"
    at.query_params["model"] = "oracle_nominal"
    at.run()
    assert not at.exception, at.exception
    boxes = {c.label: c for c in at.sidebar.checkbox}
    assert boxes["Let the controller change travel speed"].value is False
    assert boxes["Let the controller change boom height"].value is False
    fc = next(s for s in at.sidebar.selectbox if s.label == "Wind forecast")
    assert fc.value == "off" and "ar1" not in " ".join(fc.options)     # plain-English labels
    res0 = at.session_state["result"]
    assert all(lg.u.travel_speed_kmh is None for lg in res0.logs)
    boxes["Let the controller change travel speed"].check()
    boxes["Let the controller change boom height"].check()
    fc.select("ar1")
    _button(at, "Run pass").click().run()
    assert not at.exception, at.exception
    res = at.session_state["result"]
    assert any(lg.u.travel_speed_kmh is not None for lg in res.logs)
    assert all("fc_plan_wind_ms" in lg.extra for lg in res.logs)
    assert any("controller" in m.value and "km/h" in m.value for m in at.markdown)
    # with a wind forecast on, every section still has its selector record (top settings table)
    assert all(res.info_for(k) is not None for k in range(res.n))


# ----------------------------------------------------------- views and their sidebars

def test_each_view_has_its_own_inputs_and_one_run_button():
    at = _fast_app()
    assert [b.label for b in at.sidebar.button] == ["Run pass"]
    assert "Control strategy" in _sidebar_labels(at) and "Control strategies" not in _sidebar_labels(at)
    _show(at, COMPARE)
    assert [b.label for b in at.sidebar.button] == ["Run comparison"]
    labels = _sidebar_labels(at)
    assert "Control strategies" in labels and "Number of seeds" in labels and "Control strategy" not in labels
    ms = next(m for m in at.sidebar.multiselect if m.label == "Control strategies")
    assert ms.value == ["open", "fast", "fast_slow", "oracle"]
    assert at.session_state["compare"] is None and len(at.info) == 1     # nothing ran yet
    _show(at, LAB)
    assert not at.sidebar.button                 # the lab updates as its inputs change
    labels = _sidebar_labels(at)
    assert "Nozzle type" in labels and "Weather scenario" not in labels


def test_inputs_survive_a_change_of_view():
    at = _fast_app()
    next(n for n in at.sidebar.number_input if n.label == "Random seed").set_value(5).run()
    next(s for s in at.sidebar.selectbox if s.label == "Weather scenario").select("hot_dry").run()
    _show(at, LAB)
    next(s for s in at.sidebar.slider if s.label == "Wind speed (m/s)").set_value(5.0).run()
    _show(at, COMPARE)
    assert next(n for n in at.sidebar.number_input if n.label == "Random seed").value == 5
    _show(at, FIELD)
    assert next(n for n in at.sidebar.number_input if n.label == "Random seed").value == 5
    assert next(s for s in at.sidebar.selectbox if s.label == "Weather scenario").value == "hot_dry"
    assert at.session_state["result"] is not None        # the earlier run is still on screen
    _show(at, LAB)
    assert next(s for s in at.sidebar.slider if s.label == "Wind speed (m/s)").value == 5.0


def test_status_line_marks_changed_inputs_but_not_the_cards_switch():
    at = _fast_app()
    stale = lambda: [c.value for c in at.sidebar.caption if "Inputs changed" in c.value]   # noqa: E731
    assert not stale()
    next(t for t in at.sidebar.toggle if t.label == "Show paper cards").set_value(True).run()
    assert not at.exception, at.exception
    assert not stale()                           # a display choice, not an input of the run
    next(n for n in at.sidebar.number_input if n.label == "Random seed").set_value(7).run()
    assert stale()
    _button(at, "Run pass").click().run()
    assert not stale() and at.session_state["result"].seed == 7


def test_options_the_strategy_cannot_use_are_disabled_and_passed_as_off():
    at = _fast_app(speed="1", forecast="ar1")
    res = at.session_state["result"]
    assert any(lg.u.travel_speed_kmh is not None for lg in res.logs)
    pol = next(s for s in at.sidebar.selectbox if s.label == "Control strategy")
    pol.select("Single best nozzle for whole field").run()
    box = next(c for c in at.sidebar.checkbox if c.label == "Let the controller change travel speed")
    assert box.disabled
    assert next(s for s in at.sidebar.selectbox if s.label == "Wind forecast").disabled
    assert next(s for s in at.sidebar.selectbox if s.label == "Nozzle on the boom").disabled
    _button(at, "Run pass").click().run()
    assert not at.exception, at.exception
    res = at.session_state["result"]
    assert res.meta["inputs"]["Controller changes travel speed"] == "No"
    assert res.meta["inputs"]["Wind forecast"].startswith("Off")


def test_failed_run_is_reported_once_not_as_a_traceback(monkeypatch):
    from coverageai import app_logic as al

    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise ValueError("no such nozzle")

    monkeypatch.setattr(al, "run_pass_wrapper", boom)
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["verifier"] = "perfect"
    at.query_params["model"] = "oracle_nominal"
    at.run()
    assert not at.exception, at.exception
    assert any("no such nozzle" in e.value for e in at.error)
    assert len(calls) == 1                       # not retried on every rerun
    at.run()
    assert len(calls) == 1
    _button(at, "Run pass").click().run()
    assert len(calls) == 2


def test_compare_runs_one_seed_and_several_seeds():
    at = _fast_app()
    _show(at, COMPARE)
    ms = next(m for m in at.sidebar.multiselect if m.label == "Control strategies")
    ms.set_value(["open", "fast", "oracle"]).run()
    _button(at, "Run comparison").click().run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    out = at.session_state["compare"]["out"]
    assert out["seeds"] == [1] and list(out["policies"]) == ["open", "fast", "oracle"]
    table = at.dataframe[0].value
    assert list(table["Control strategy"]) == ["Model only (no feedback)", "Model + card feedback",
                                               "Knows hidden conditions (fair upper bound)"]
    assert "95% interval" not in table.columns   # one field: nothing to average
    # the same field for every strategy: the comparison says which card reading was used
    assert any("Card reading" in m.value for m in at.markdown)
    _slider(at, "Number of seeds").set_value(3).run()
    assert any("Inputs changed" in c.value for c in at.sidebar.caption)
    _button(at, "Run comparison").click().run()
    assert not at.exception, at.exception
    out = at.session_state["compare"]["out"]
    assert out["seeds"] == [1, 2, 3] and len(out["per_seed"]) == 9
    table = at.dataframe[0].value
    assert "95% interval" in table.columns
    diff = next(c for c in table.columns if c.startswith("Difference vs "))
    assert "Knows hidden conditions" in diff and list(table[diff])[-1] == "reference"
    assert not [c for c in at.sidebar.caption if "Inputs changed" in c.value]


def test_a_failed_comparison_stays_in_the_compare_view(monkeypatch):
    from coverageai import app_logic as al

    def boom(*a, **k):
        raise ValueError("bad compare")

    at = _fast_app()
    monkeypatch.setattr(al, "compare_seeds", boom)
    _show(at, COMPARE)
    _button(at, "Run comparison").click().run()
    assert not at.exception, at.exception
    assert any("bad compare" in e.value for e in at.error)
    _show(at, FIELD)                             # the field run is neither blocked nor marked failed
    assert not at.error and at.session_state["result"] is not None
    _show(at, LAB)
    assert not at.error


def test_compare_needs_a_strategy_and_ignores_the_reader_when_images_are_off():
    at = _fast_app(verifier="b0")
    _show(at, COMPARE)
    _button(at, "Run comparison").click().run()
    assert not at.exception, at.exception
    stale = lambda: [c.value for c in at.sidebar.caption if "Inputs changed" in c.value]   # noqa: E731
    assert not stale()
    next(s for s in at.sidebar.selectbox if s.label == "Card reader").select("b1").run()
    assert not stale()                           # card images are off: the readers are not used
    next(m for m in at.sidebar.multiselect if m.label == "Control strategies").set_value([]).run()
    assert _button(at, "Run comparison").disabled


def test_feedback_tab_says_what_a_correction_is_based_on():
    at = _fast_app(feedback="rule")
    md = " ".join(m.value for m in at.markdown)
    assert "What a correction is based on" in md and "card reading minus model prediction" in md
    assert "based on" in md and "crosswind" in md       # the step card of the picked section
    table = next(d.value for d in at.dataframe if "Correction based on" in d.value.columns)
    assert "Card minus model (pp)" in table.columns and len(table) == at.session_state["result"].n
    pol = next(s for s in at.sidebar.selectbox if s.label == "Control strategy")
    pol.select("Model only (no feedback)")
    _button(at, "Run pass").click().run()
    assert not at.exception, at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "No card feedback in this strategy" in md and "What a correction is based on" not in md



def test_trained_feedback_is_the_app_default_when_its_file_exists(tmp_path, monkeypatch):
    """D29 / D31: the trained feedback model is an optional file. Without it the sidebar offers
    nothing extra and the rule runs. With it, the app uses it by default, "Card feedback" lets you
    go back to the rule, and the page says what a trained correction is based on."""
    import numpy as np
    from coverageai import app_logic as al
    from coverageai import feedback_learned as fl

    monkeypatch.setattr(fl, "DEFAULT_PATHS", {"xgboost": tmp_path / "x.joblib", "lightgbm": tmp_path / "l.joblib"})
    monkeypatch.setattr(al, "_FEEDBACK_MODELS", {})
    at = _fast_app()
    assert "Card feedback" not in _sidebar_labels(at)
    assert at.session_state["result"].meta["feedback"] == {"kind": "rule"}

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, fl.N_FEATURES))
    fl.FeedbackModel.train(X, X[:, 0] - 3.0, backend="lightgbm", n_estimators=20, max_depth=3,
                           min_child_samples=5).save(tmp_path / "l.joblib")
    at = _fast_app()
    sel = next(s for s in at.sidebar.selectbox if s.label == "Card feedback")
    assert sel.value == "lightgbm" and len(sel.options) == 2 and not sel.disabled     # adopted default
    assert all("lightgbm" not in o and "rule" != o for o in sel.options)             # display names only
    res = at.session_state["result"]
    assert res.meta["feedback"]["kind"] == "learned" and res.meta["inputs"]["Card feedback"].startswith("Trained model")
    md = " ".join(m.value for m in at.markdown)
    assert "prediction of a trained LightGBM model" in md and "trained model, " in md
    assert not [c for c in at.sidebar.caption if "Inputs changed" in c.value]
    # back to the running average
    sel.select("rule").run()
    assert any("Inputs changed" in c.value for c in at.sidebar.caption)
    _button(at, "Run pass").click().run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    res = at.session_state["result"]
    assert res.meta["feedback"] == {"kind": "rule"} and res.meta["inputs"]["Card feedback"].startswith("Running average")
    assert "70% of the old one" in " ".join(m.value for m in at.markdown)
    # a strategy without card feedback cannot use either: greyed out, recorded as the rule
    pol = next(s for s in at.sidebar.selectbox if s.label == "Control strategy")
    pol.select("Model only (no feedback)").run()
    assert next(s for s in at.sidebar.selectbox if s.label == "Card feedback").disabled
    _button(at, "Run pass").click().run()
    assert not at.exception, at.exception
    res = at.session_state["result"]
    assert res.meta["feedback"] == {"kind": "none"} and res.meta["inputs"]["Card feedback"].startswith("Running average")
    # the URL can ask for the rule
    at = _fast_app(feedback="rule")
    assert at.session_state["result"].meta["feedback"] == {"kind": "rule"}


def test_app_corrects_the_card_readers_known_error_by_default():
    """D30: in the app the card reader's known mean error is removed unless switched off."""
    at = _fast_app(verifier="noisy")
    res = at.session_state["result"]
    assert type(res.modules.verifier).__name__ == "CalibratedVerifier"
    assert res.meta["inputs"]["Card reader error corrected"] == "Yes"
    box = next(c for c in at.sidebar.checkbox if c.label == "Correct the card reader's known error")
    assert box.value is True and not box.disabled
    box.uncheck().run()
    assert any("Inputs changed" in c.value for c in at.sidebar.caption)
    _button(at, "Run pass").click().run()
    assert not at.exception, at.exception
    res = at.session_state["result"]
    assert type(res.modules.verifier).__name__ == "NoisyVerifier"
    assert res.meta["inputs"]["Card reader error corrected"] == "No"
    # nothing to correct with perfect reading: greyed out and recorded as not applied
    at = _fast_app(verifier="perfect")
    box = next(c for c in at.sidebar.checkbox if c.label == "Correct the card reader's known error")
    assert box.disabled and at.session_state["result"].meta["inputs"]["Card reader error corrected"] == "No"
    assert _fast_app(verifier="noisy", calibrate="0").session_state["result"].meta["inputs"][
        "Card reader error corrected"] == "No"


def test_reader_correction_is_offered_only_when_a_table_exists(monkeypatch):
    """D30: a reader whose error has not been measured in the loop is not corrected."""
    from coverageai import app_logic as al
    monkeypatch.setattr(al, "reader_calibration_available", lambda verifier: False)
    at = _fast_app(verifier="noisy")
    box = next(c for c in at.sidebar.checkbox if c.label == "Correct the card reader's known error")
    assert box.disabled
    res = at.session_state["result"]
    assert type(res.modules.verifier).__name__ == "NoisyVerifier"
    assert res.meta["inputs"]["Card reader error corrected"] == "No"


def test_learning_strategy_keeps_the_running_average_in_the_app():
    at = _fast_app()
    if "Card feedback" not in _sidebar_labels(at):
        pytest.skip("no trained feedback model on disk")
    pol = next(s for s in at.sidebar.selectbox if s.label == "Control strategy")
    pol.select("Model + card feedback + learning between passes").run()
    assert next(s for s in at.sidebar.selectbox if s.label == "Card feedback").disabled
    _button(at, "Run pass").click().run()
    assert not at.exception, at.exception
    res = at.session_state["result"]
    assert res.meta["feedback"] == {"kind": "rule"} and res.meta["inputs"]["Card feedback"].startswith("Running average")
