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
    for k, v in params.items():
        at.query_params[k] = v
    at.run()
    assert not at.exception, at.exception
    return at


def _slider(at, label):
    return next(s for s in at.slider if s.label == label)


def test_run_pass_twice_does_not_crash():
    at = _fast_app()
    sl = _slider(at, "Section to inspect")
    sl.set_value(min(3, sl.max)).run()          # move the inspector off section 1
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
    at = _fast_app(nozzle="free")
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
    noz = next(s for s in at.sidebar.selectbox if s.label == "Nozzle on the boom")
    assert len(noz.options) > 3
    assert all(not o.startswith(("auto", "free")) and "-" not in o.split(" ")[0] for o in noz.options)


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
