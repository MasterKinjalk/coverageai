"""Headless UI test for the Streamlit demo (app/main.py) using streamlit.testing.

Regression: pressing "Run pass" a second time crashed with StreamlitAPIException because
the script reset a slider's session-state key after the slider had been created in the same run
(now the "Section to inspect" slider, key "section", reset in the button callback).
"""
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "main.py")


def _button(at, label):
    return next(b for b in at.sidebar.button if b.label == label)


def _fast_app(**params):
    """AppTest on the simple fast path: physics model, perfect reader, no card drawing."""
    at = AppTest.from_file(APP, default_timeout=240)
    at.query_params["verifier"] = "perfect"
    at.query_params["model"] = "oracle_nominal"
    at.query_params["cards"] = "0"
    for k, v in {"nozzle": "TP6506", **params}.items():
        at.query_params[k] = v
    at.run()
    assert not at.exception, at.exception
    return at


def _slider(at, label):
    return next(s for s in at.slider if s.label == label)


def test_run_pass_twice_does_not_crash():
    at = _fast_app()
    sl = _slider(at, "Section to inspect (details below)")
    sl.set_value(min(3, sl.max)).run()          # move the inspector off section 1
    assert not at.exception, at.exception
    assert at.session_state["section"] == min(3, sl.max) > 1
    _button(at, "Run pass").click().run()
    assert not at.exception, at.exception
    assert at.session_state["section"] == 1
    for gone in ("cursor", "playing", "advance"):
        assert gone not in at.session_state
    assert not [b for b in at.sidebar.button if b.label in ("Back", "Step", "Play all")]


def test_tp6501_cannot_meet_default_gpa_but_runs_closest_rate_with_warning():
    at = _fast_app(nozzle="TP6501")
    sel = next(s for s in at.sidebar.selectbox if s.label == "Nozzle")
    assert sel.value == "TP6501"
    result = at.session_state["result"]
    assert result is not None
    assert result.meta["gpa_unreachable_sections"]
    assert any("pass ran at the closest available rates" in warning.value.lower()
               for warning in at.warning)
    assert at.session_state["previous"] is None


def test_tp6505_holds_25_gpa_by_expanding_speed_range():
    at = _fast_app(nozzle="TP6505")
    _slider(at, "Target spray rate (gal/acre)").set_value(25.0).run()
    _button(at, "Run pass").click().run()
    assert not at.exception, at.exception
    result = at.session_state["result"]
    assert all(23.75 - 1e-9 <= lg.derived.app_rate_gpa <= 26.25 + 1e-9
               for lg in result.logs)


@pytest.mark.parametrize("params", [{"model": "mock"}, {"policy": "best_fixed"}])
def test_unsupported_modes_are_blocked_instead_of_skipping_gpa(params):
    at = _fast_app(**params)
    assert at.session_state["result"] is None
    assert any("cannot enforce the requested GPA target" in error.value for error in at.error)


def test_unreachable_run_uses_closest_rate_and_keeps_previous_result():
    at = _fast_app()
    previous = at.session_state["result"]
    nozzle = next(s for s in at.sidebar.selectbox if s.label == "Nozzle")
    nozzle.select("TP6501")
    _button(at, "Run pass").click().run()
    assert not at.exception, at.exception
    assert at.session_state["result"] is not previous
    assert at.session_state["previous"] is previous
    assert any("pass ran at the closest available rates" in warning.value.lower()
               for warning in at.warning)


def test_nozzle_codes_list_physical_nozzles_and_hold_the_selection():
    at = _fast_app()
    sel = next(s for s in at.sidebar.selectbox if s.label == "Nozzle")
    assert {"TP6501", "ULDQ2006", "PS3DQ0004", "XR11004", "DGX4003"} <= set(sel.options)
    sel.select("XR11006")
    _button(at, "Run pass").click().run()
    assert not at.exception, at.exception
    res = at.session_state["result"]
    assert res.meta["nozzle"]["mode"] == "fixed"
    assert {(lg.u.family, lg.u.size) for lg in res.logs} == {("XR", "06")}


def test_named_nozzle_is_held():
    at = _fast_app(nozzle="XR11006")
    res = at.session_state["result"]
    assert res.meta["nozzle"]["mode"] == "fixed"
    assert {(lg.u.family, lg.u.size) for lg in res.logs} == {("XR", "06")}


def test_unknown_nozzle_query_uses_a_valid_default():
    at = _fast_app(nozzle="not-a-catalog-nozzle")
    sel = next(s for s in at.sidebar.selectbox if s.label == "Nozzle")
    assert sel.value == "TP6501"
    result = at.session_state["result"]
    assert result is not None
    assert result.meta["gpa_unreachable_sections"]
    assert any("pass ran at the closest available rates" in warning.value.lower()
               for warning in at.warning)
    assert not at.exception, at.exception


def test_second_run_fills_previous():
    at = _fast_app()
    first = at.session_state["result"]
    _button(at, "Run pass").click().run()
    assert not at.exception, at.exception
    assert at.session_state["previous"] is first
    assert at.session_state["result"] is not first


def test_what_the_model_decided_tabs():
    at = _fast_app()
    assert any(h.value == "What the model decided" for h in at.subheader)
    assert any(h.value == "Section inspector" for h in at.subheader)
    assert any(h.value == "Feedback loop over the run" for h in at.subheader)
    labels = [t.label for t in at.tabs]
    for want in ("Operator recommendations (", "Changes along the field (", "Changes since the previous run",
                 "All model outputs"):
        assert any(l.startswith(want) for l in labels), (want, labels)
    n_rec = next(l for l in labels if l.startswith("Operator recommendations ("))
    n_chg = next(l for l in labels if l.startswith("Changes along the field ("))
    from coverageai import app_logic as al
    res = at.session_state["result"]
    assert n_rec == f"Operator recommendations ({len(al.recommendation_frame(res))})"
    assert n_chg == f"Changes along the field ({int((al.decision_frame(res)['changed'] != '').sum())})"


def test_live_dashboard_precedes_separate_replay_and_details_selector():
    at = _fast_app()
    headers = [h.value for h in at.subheader]
    assert headers.index("Field run dashboard") < headers.index("Field coverage replay")
    assert headers.index("Field coverage replay") < headers.index("Run summary")
    section = _slider(at, "Section to inspect (details below)")
    selected = min(3, section.max)
    section.set_value(selected).run()
    assert not at.exception, at.exception
    assert at.session_state["section"] == selected
    assert any(c.value.startswith("Set parameters in the sidebar.") for c in at.caption)


def test_drift_chart_distinguishes_prediction_from_simulated_outcome():
    import pandas as pd
    import sys
    sys.path.insert(0, str(Path(APP).parent))
    import charts

    chart = charts.drift_timeseries(
        pd.DataFrame({"k": [1, 2], "predicted": [2.0, 3.0], "simulated": [2.5, 2.8]}),
        n_total=2,
    ).to_dict()
    assert chart["encoding"]["color"]["scale"]["domain"] == [
        "Model prediction", "Simulated outcome"]


RAW = ("oracle_nominal", "best_fixed", "C_req", "gusty_afternoon", "acceptable")


def test_no_raw_identifiers_in_widget_labels_or_options():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    texts = []
    for w in list(at.sidebar.selectbox) + list(at.sidebar.select_slider) + list(at.sidebar.slider):
        texts.append(w.label)
        texts.extend(getattr(w, "options", []))   # options are the format_func-rendered strings
    for w in at.main.selectbox:
        texts.append(w.label)
        texts.extend(w.options)
    for w in list(at.main.multiselect) + list(at.main.radio):
        texts.append(w.label)
        texts.extend(w.options)
    for t in texts:
        for raw in RAW:
            assert raw not in str(t), (raw, t)
        assert str(t) not in ("auto", "free"), t
    noz = next(s for s in at.sidebar.selectbox if s.label == "Nozzle")
    assert len(noz.options) > 3
    assert all(o not in ("auto", "free") and "-" not in o for o in noz.options)


def test_switching_policies_and_verifiers_does_not_crash():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    ver = next(s for s in at.sidebar.selectbox if s.label == "Card reader")
    ver.select("Perfect reading (true value)")        # b0 is slow; keep the strategy sweep fast
    pol = next(s for s in at.sidebar.selectbox if s.label == "Control strategy")
    assert len(pol.options) == 6
    for opt in list(pol.options):
        pol.select(opt)
        _button(at, "Run pass").click().run()
        assert not at.exception, (opt, at.exception)
        pol = next(s for s in at.sidebar.selectbox if s.label == "Control strategy")
    ver = next(s for s in at.sidebar.selectbox if s.label == "Card reader")
    for opt in ("Perfect reading (true value)", "Noisy reading (simulated sensor)",
                "Classic image analysis"):
        ver.select(opt)
        _button(at, "Run pass").click().run()
        assert not at.exception, (opt, at.exception)
        ver = next(s for s in at.sidebar.selectbox if s.label == "Card reader")


def test_fast_slow_shows_learning_table_and_compare_defaults():
    at = AppTest.from_file(APP, default_timeout=240)
    at.query_params["nozzle"] = "TP6506"
    at.run()
    next(s for s in at.sidebar.selectbox if s.label == "Card reader").select("Perfect reading (true value)")
    pol = next(s for s in at.sidebar.selectbox if s.label == "Control strategy")
    pol.select("Model + card feedback + learning between passes")
    _button(at, "Run pass").click().run()
    assert not at.exception, at.exception
    assert any("Learning between passes" in h.value for h in at.subheader)
    ms = next(m for m in at.main.multiselect if m.label == "Control strategies")
    assert len(ms.value) == 4 and ms.value == ["open", "fast", "fast_slow", "oracle"]


def test_policy_bars_show_every_strategy_name():
    """Compare tab: all strategy names are y-axis categories (previously only the first and last
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


def test_noisy_reader_run_is_not_called_image_analysis():
    at = AppTest.from_file(APP, default_timeout=240)
    at.query_params["verifier"] = "noisy"
    at.query_params["model"] = "oracle_nominal"
    at.query_params["nozzle"] = "TP6506"
    at.run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "image analysis" not in html and "simulated sensor" in html
    assert any("simulated sensor" in c.value for c in at.caption)


def test_machine_setup_is_grouped_and_automatic_adjustment_defaults_on():
    from coverageai.units import mph_to_kmh

    at = AppTest.from_file(APP, default_timeout=240)
    at.query_params["verifier"] = "perfect"
    at.query_params["model"] = "oracle_nominal"
    at.query_params["nozzle"] = "TP6506"
    at.run()
    assert not at.exception, at.exception
    labels = [s.label for s in at.sidebar.slider]
    assert "Target spray rate (gal/acre)" in labels
    assert "Target travel speed (mph)" in labels
    assert "Target pressure (psi)" in labels
    assert "Starting boom height (m)" in labels
    assert "GPA target tolerance (%)" in labels
    boxes = {c.label: c for c in at.sidebar.checkbox}
    auto = boxes["Automatically adjust pressure, speed, and boom height"]
    assert auto.value is True and auto.disabled
    target_speed = next(s for s in at.sidebar.slider if s.label == "Target travel speed (mph)")
    assert at.session_state["result"].logs[0].x.travel_speed_kmh == pytest.approx(
        mph_to_kmh(target_speed.value))
    fc = next(s for s in at.sidebar.selectbox if s.label == "Wind forecast")
    assert fc.value == "off" and "ar1" not in " ".join(fc.options)
    next(s for s in at.sidebar.slider if s.label == "Target spray rate (gal/acre)").set_value(18.0).run()
    assert not at.exception, at.exception
    _button(at, "Run pass").click().run()
    assert not at.exception, at.exception
    res = at.session_state["result"]
    assert all(17.1 - 1e-9 <= lg.derived.app_rate_gpa <= 18.9 + 1e-9 for lg in res.logs)
