"""Headless UI test for the Streamlit demo (app/main.py) using streamlit.testing.

Regression: pressing "Run pass" a second time crashed with StreamlitAPIException because
the script reset st.session_state["cursor"] after the "Segment" slider (key="cursor")
had already been created in the same run.
"""
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "main.py")


def _button(at, label):
    return next(b for b in at.sidebar.button if b.label == label)


def test_run_pass_twice_does_not_crash():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    _button(at, "Step").click().run()          # move the cursor off 0
    assert not at.exception, at.exception
    _button(at, "Run pass").click().run()
    assert not at.exception, at.exception
    assert at.session_state["cursor"] == 0


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
