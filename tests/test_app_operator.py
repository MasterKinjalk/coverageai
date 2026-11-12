"""Headless UI tests for the "Operator console" view of app/main.py (streamlit.testing).

Fast path: physics model (``?model=oracle_nominal``) and perfect card reading. The view is opened
by setting ``session_state["view"]`` before the first run; it then runs once with its defaults.
"""
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "main.py")
OPERATOR, FIELD, LAB = "Operator console", "Field run", "Nozzle lab"
RUN = "Run console"
RAW = ("oracle_nominal", "best_fixed", "C_req", "gusty_afternoon", "acceptable", "fast_slow", "mock",
       "pressure_at_rate", "boom_height", "unreachable", "exact", "rate_tier", "rate_gpa", "tier")


def _console_app(**state):
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["model"] = "oracle_nominal"
    at.session_state["view"] = OPERATOR
    at.session_state["op_verifier"] = "perfect"
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, at.exception
    return at


def _show(at, view):
    at.session_state["view"] = view
    at.run()
    assert not at.exception, at.exception
    return at


def _button(at, label):
    return next(b for b in at.sidebar.button if b.label == label)


def _md(at) -> str:
    return " ".join(m.value for m in at.markdown)


def test_view_runs_once_with_its_defaults_and_has_one_run_button():
    at = _console_app()
    assert OPERATOR in [t.label for t in at.tabs]
    assert [b.label for b in at.sidebar.button] == [RUN]
    assert not at.error, [e.value for e in at.error]
    run = at.session_state["console"]
    assert run is not None and at.session_state["result"] is None      # only the open view ran
    assert run.nozzle == ("XR", "06") and run.setup.gpa == 20.0 and run.boom_control
    labels = [w.label for g in (at.sidebar.selectbox, at.sidebar.number_input, at.sidebar.checkbox,
                                at.sidebar.toggle, at.sidebar.select_slider) for w in g]
    for want in ("Weather scenario", "Hidden surprises", "Random seed", "Nozzle type", "Nozzle size",
                 "Target rate (gal/acre)", "Target speed (km/h)", "Let the controller change boom height",
                 "Card reader", "Show paper cards"):
        assert want in labels, (want, labels)
    assert "Control strategy" not in labels and "Pressure (psi)" not in labels   # pressure is not an input
    md = _md(at)
    assert f"Starting pressure **{run.setup.pressure_psi:.0f} psi**" in md
    assert "setting held:" in md and "Unreachable at rate" in md and "What to change" in md
    assert any("Showing the last run" in c.value for c in at.sidebar.caption)
    assert any("Simulation, not field data" in c.value for c in at.caption)
    assert {e.label for e in at.expander} >= {"Section by section", "Switch nozzle? (model prediction)"}
    table = next(d.value for d in at.dataframe if "Controller plan (model)" in d.value.columns)
    assert len(table) == run.n


def test_totals_are_the_app_kpis():
    from coverageai import app_logic as al
    at = _console_app()
    run = at.session_state["console"]
    kc, kh = al.compute_kpis(run.console.logs, 30.0), al.compute_kpis(run.held.logs, 30.0)
    md = _md(at)
    assert f"{kc['n_met']} / {run.n}" in md and f"setting held: {kh['n_met']} / {run.n}" in md
    assert f"{100 * kc['mean_drift']:.1f}" in md


def test_no_raw_identifiers_on_the_console():
    at = _console_app()
    texts = []
    for w in (list(at.sidebar.selectbox) + list(at.sidebar.select_slider) + list(at.sidebar.slider)
              + list(at.sidebar.number_input) + list(at.main.selectbox) + list(at.main.radio)):
        texts.append(w.label)
        texts.extend(getattr(w, "options", []))
    for d in at.dataframe:
        texts.extend(d.value.columns)
        for col in ("Controller plan (model)", "Card result", "What to change", "Nozzle"):
            if col in d.value.columns:
                texts.extend(d.value[col].astype(str))
    for t in texts:
        for raw in RAW:
            assert raw not in str(t), (raw, t)
    noz = next(s for s in at.sidebar.selectbox if s.label == "Nozzle type")
    assert all(" - " in o for o in noz.options)          # display names, not bare family codes


def test_unreachable_nozzle_says_so_and_suggests_another():
    """ULD 04 at 15 gal/acre in the gusty scenario (about 3 m/s crosswind)."""
    at = _console_app(op_family="ULD", op_size_ULD="04", op_gpa=15.0)
    run = at.session_state["console"]
    assert run.nozzle == ("ULD", "04")
    assert any("cannot reach 30%" in w.value and "Switch nozzle?" in w.value for w in at.warning)
    sw = next(e for e in at.expander if e.label.startswith("Switch nozzle?"))
    assert sw.proto.expanded
    alt = next(d.value for d in at.dataframe if "Mean predicted drift (%)" in d.value.columns)
    assert len(alt) >= 1 and "ULD 04" not in list(alt["Nozzle"])
    assert "For the next fill-up or field" in _md(at) and "Model predictions" in _md(at)


def test_setup_without_a_pressure_warns_and_disables_run():
    at = _console_app(op_family="XR", op_size_XR="04", op_gpa=15.0)
    assert at.session_state["console"] is None                       # nothing ran
    assert _button(at, RUN).disabled
    warn = [w.value for w in at.sidebar.warning] + [w.value for w in at.main.warning]
    assert any("No pressure from 15 to 60 psi" in w for w in warn), warn
    # fix the speed: the pressure appears and the console runs on its own (first visit)
    next(n for n in at.sidebar.number_input if n.label == "Target speed (km/h)").set_value(12.0).run()
    assert not at.exception, at.exception
    assert not _button(at, RUN).disabled
    assert at.session_state["console"] is not None


def test_inputs_survive_a_change_of_view_and_mark_the_run_stale():
    at = _console_app()
    stale = lambda: [c.value for c in at.sidebar.caption if "Inputs changed" in c.value]   # noqa: E731
    assert not stale()
    next(n for n in at.sidebar.number_input if n.label == "Target rate (gal/acre)").set_value(21.0).run()
    assert stale()
    _show(at, LAB)
    _show(at, OPERATOR)
    assert next(n for n in at.sidebar.number_input if n.label == "Target rate (gal/acre)").value == 21.0
    assert stale()
    _button(at, RUN).click().run()
    assert not at.exception, at.exception
    assert not stale() and at.session_state["console"].setup.gpa == 21.0
    # the field inputs are the Field run's: a seed set here is the seed there
    next(n for n in at.sidebar.number_input if n.label == "Random seed").set_value(4).run()
    _show(at, FIELD)
    assert next(n for n in at.sidebar.number_input if n.label == "Random seed").value == 4


def test_boom_switch_and_a_failed_run(monkeypatch):
    from coverageai import operator_console as oc
    at = _console_app(op_boom=False)
    run = at.session_state["console"]
    assert not run.boom_control and all(lg.u.boom_height_m is None for lg in run.console.logs)

    def boom(*a, **k):
        raise ValueError("bad console")

    monkeypatch.setattr(oc, "run_console", boom)
    _button(at, RUN).click().run()
    assert not at.exception, at.exception
    assert any("bad console" in e.value for e in at.error)
    assert at.session_state["console"] is run                          # the last run stays on screen
