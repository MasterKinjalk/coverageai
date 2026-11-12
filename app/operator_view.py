"""Operator console: the product as a spray operator would use it (simulation only).

The operator sets the nozzle on the boom, a label rate (gal/acre, held within ±5 %) and a target
speed; the starting pressure follows from those three. One press of "Run console" sprays the same
field twice on the same seed: once with that starting setting held, once with the console moving
pressure (speed follows, rate held) and, if allowed, boom height to keep cards at or above the
fixed 30 % minimum. The page then shows the replay of the console run, every section, what to
change where a card stayed short, and other nozzles the model expects to do better at this rate.
All logic is in ``coverageai.operator_console``; this file is widgets and layout only.

Usage from the main app (``app/`` is on sys.path, as main.py arranges):

    with st.sidebar:
        v = op_controls()                 # inputs; also returns model / reader choice
        op_run_button(modules, v)         # in the container above the inputs
    render_operator(modules, v)

Keys. The field inputs (weather scenario, hidden surprises, random seed) reuse the Field run's
``in_*`` widgets, so the console and a Field run look at the same field and seed. Everything that
belongs to the console has its own ``op_*`` key (re-assigned by ``main.init_state``, rule 10).
The coverage predictor is the Field run's choice (``in_model``); the card reader's known error
is removed whenever a table exists for the chosen reader (the app default, D30); the card
feedback is the app's default (``app_logic.default_feedback_choice``, D31). The run itself is
kept in ``console``, ``console_error`` and ``console_want_run`` (no ``op_`` prefix: not widgets).
"""
from __future__ import annotations

import streamlit as st

import labels as L
import ui
from field_run_view import render_field_animation
from coverageai import app_logic as al
from coverageai import catalog, scenarios
from coverageai import operator_console as oc
from coverageai import rate_control
from coverageai.config import spec
from coverageai.units import kmh_to_mph

SS = st.session_state
RUN_LABEL = "Run console"
REPLAY_HEIGHT = 740


def _default(key: str, value) -> str:
    """Seed a widget's value once and return its key (see ``spray_lab_view._default``)."""
    SS.setdefault(key, value)
    return key


def _size_label(fam: str, size: str) -> str:
    try:
        flow, _ = catalog.flow_and_class(fam, size, 40.0)
        return f"{size}  (about {flow:.2f} gal/min at 40 psi)"
    except ValueError:
        return size


def _family_note(fam: str) -> str:
    return L.family(fam).partition(" - ")[2] or fam


@st.cache_data(show_spinner=False)
def _conditions(preset: str, seed: int):
    return al.make_conditions(preset, seed)


def _cfg(v: dict) -> tuple:
    return oc.cfg(v["preset"], v["seed"], v["mismatch"], v["family"], v["size"], v["gpa"], v["speed"],
                  v["boom"], v["model"], v["verifier"], v["calibrate"], v["feedback"])


# ------------------------------------------------------------------ sidebar

def _url_defaults() -> None:
    """Seed the console inputs from the URL on the first visit, e.g.
    ``?op_nozzle=ULD-04&op_rate=15&op_speed=16&op_reader=noisy&op_boom=0``. Values that do not fit
    are ignored. An input already in the session keeps its value."""
    qp = st.query_params
    fam, _, size = str(qp.get("op_nozzle", "")).partition("-")
    if fam in catalog.families() and size in catalog.sizes(fam):
        SS.setdefault("op_family", fam)
        SS.setdefault(f"op_size_{fam}", size)
    lo_v, hi_v = (float(x) for x in spec()["ranges"]["travel_speed_kmh"])
    for name, key, lo, hi in (("op_rate", "op_gpa", 1.0, 60.0), ("op_speed", "op_speed", lo_v, hi_v)):
        try:
            val = float(qp.get(name, ""))
        except ValueError:
            continue
        if lo <= val <= hi:
            SS.setdefault(key, val)
    if qp.get("op_reader") in al.VERIFIER_CHOICES:
        SS.setdefault("op_verifier", qp.get("op_reader"))
    if qp.get("op_boom") in ("0", "1"):
        SS.setdefault("op_boom", qp.get("op_boom") == "1")


def op_controls() -> dict:
    """Draw the console's inputs into the current container and return their values, plus the
    setup they imply (``v["setup"]``, an ``operator_console.Setup``)."""
    for key, default in (("console", None), ("console_error", None), ("console_want_run", False)):
        SS.setdefault(key, default)
    _url_defaults()
    v: dict = {}
    ui.group("Field")
    presets = scenarios.list_presets()
    v["preset"] = st.selectbox("Weather scenario", presets, format_func=L.preset, key="in_preset",
                               help=scenarios.get(SS["in_preset"]).description)
    v["mismatch"] = st.select_slider(
        "Hidden surprises", al.MISMATCH_LEVELS, format_func=L.mismatch, key="in_mismatch",
        help="How far reality departs from what the model expects: sudden gusts, the crop canopy "
             "catching spray, worn nozzles. The model never sees these.")
    v["seed"] = int(st.number_input(
        "Random seed", min_value=0, max_value=10_000, step=1, key="in_seed",
        help="Same number = same simulated field and weather. Shared with the Field run."))

    ui.group("Sprayer")
    fams = catalog.families()
    fam = st.selectbox("Nozzle type", fams, format_func=L.family,
                       key=_default("op_family", "XR" if "XR" in fams else fams[0]),
                       help="The nozzle on the boom. It stays on for the whole field.")
    sizes = catalog.sizes(fam)
    size = st.selectbox("Nozzle size", sizes, format_func=lambda z: _size_label(fam, z),
                        key=_default(f"op_size_{fam}", "06" if "06" in sizes else sizes[len(sizes) // 2]),
                        help="A bigger number means more flow.")
    tol = 100 * rate_control.tolerance_frac()
    gpa = float(st.number_input(
        "Target rate (gal/acre)", min_value=1.0, max_value=60.0, step=0.5, format="%.1f",
        key=_default("op_gpa", 20.0),
        help=f"The label rate. The console holds it within ±{tol:g}%."))
    lo_v, hi_v = (float(x) for x in spec()["ranges"]["travel_speed_kmh"])
    speed = float(st.number_input(
        "Target speed (km/h)", min_value=lo_v, max_value=hi_v, step=0.5, format="%.1f",
        key=_default("op_speed", 16.0),
        help="The speed you plan to drive. With the rate fixed, it sets the starting pressure."))
    setup = oc.setup_for(fam, size, gpa, speed)
    _setup_line(setup)

    ui.group("Console")
    boom = st.checkbox(
        "Let the controller change boom height", key=_default("op_boom", True),
        help="Besides pressure, the console may raise or lower the boom a little (within 25% and the "
             "legal range). Lower booms drift less.")
    verifier = st.selectbox(
        "Card reader", list(al.VERIFIER_CHOICES), format_func=L.verifier, key=_default("op_verifier", "noisy"),
        help="How coverage is read from the card. Perfect and noisy reading are fast. The two image "
             "readers look at a picture of each card (slower).")
    cards = st.toggle(
        "Show paper cards", key=_default("op_cards", False),
        help="Shows each section's card in the replay. With a fast reader the cards are drawn after "
             "the run, which takes a few seconds.")
    v.update(family=fam, size=size, gpa=gpa, speed=speed, setup=setup, boom=bool(boom),
             verifier=verifier, cards=bool(cards), model=SS.get("in_model", "spray"),
             calibrate=al.reader_calibration_available(verifier),
             feedback=al.default_feedback_choice())
    return v


def _setup_line(s: oc.Setup) -> None:
    """Read-only: the starting pressure and droplet size that nozzle + rate + speed imply."""
    if s.ok:
        st.markdown(f"Starting pressure **{s.pressure_psi:.0f} psi** · "
                    f"{L.droplet(s.droplet_class).lower()} droplets")
        a, b = s.speed_range_kmh
        st.caption(f"Set by nozzle, rate and speed. This rate needs {a:.1f} to {b:.1f} km/h.")
    else:
        st.warning(s.problem())


def op_run_button(modules, v: dict) -> None:
    """The view's Run button and status line (main.py draws it at the top of the sidebar)."""
    if st.button(RUN_LABEL, type="primary", width="stretch",
                 disabled=modules is None or not v["setup"].ok,
                 help="Spray the field twice on the same seed: with the starting setting held, and "
                      "with the console adjusting."):
        SS["console_want_run"] = True
    run = SS.get("console")
    if run is not None and run.meta.get("cfg") != _cfg(v):
        st.caption(f":orange[Inputs changed since this run. Press {RUN_LABEL} to update.]")
    elif run is not None:
        st.caption(f"Showing the last run ({run.meta.get('elapsed_s', 0.0):.1f} s).")


# ------------------------------------------------------------------ run

def _run(modules, v: dict) -> None:
    bar = st.progress(0.0, text="Spraying the field twice ...")

    def tick(done: int, total: int) -> None:
        frac = min(1.0, done / max(total, 1))
        bar.progress(frac, text=f"Spraying the field twice ... {100 * frac:.0f}%")

    try:
        run = oc.run_console(modules, _conditions(v["preset"], v["seed"]), v["seed"], v["mismatch"],
                             v["family"], v["size"], v["gpa"], v["speed"], boom_control=v["boom"],
                             feedback_model=v["feedback"], capture_cards=v["cards"], progress=tick)
    except Exception as e:      # show the reason in the page, keep the last run
        bar.empty()
        SS["console_error"] = f"{type(e).__name__}: {e}"
        return
    bar.empty()
    run.meta["cfg"] = _cfg(v)
    run.meta["elapsed_s"] = run.held.elapsed_s + run.console.elapsed_s
    run.meta["chips"] = [("Scenario", L.preset(v["preset"])), ("Hidden surprises", L.mismatch(v["mismatch"])),
                         ("Seed", str(v["seed"])), ("Card reader", L.verifier(v["verifier"])),
                         ("Card feedback", L.feedback_choice(v["feedback"])),
                         ("Coverage predictor", L.model(v["model"])),
                         ("Boom height", "console may change it" if v["boom"] else "as in the field")]
    old = SS.get("console")
    if old is not None:
        old.console.cards = None             # only the newest run keeps its card pictures
    SS.update(console=run, console_error=None)


def render_operator(modules, v: dict) -> None:
    """Main area of the view: run if asked (or on the first visit), then show the last run."""
    if SS.get("console_error"):
        st.error(f"The console run failed: {SS['console_error']}")
    want = SS.get("console_want_run") or (SS.get("console") is None and not SS.get("console_error"))
    if modules is not None and v["setup"].ok and want:
        SS["console_want_run"] = False
        _run(modules, v)
        st.rerun()                   # redraw the sidebar status line for the new run
    run = SS.get("console")
    if run is None:
        if not v["setup"].ok:
            st.warning(v["setup"].problem())
        elif not SS.get("console_error"):
            st.info(f"Set the nozzle, rate and speed, then press **{RUN_LABEL}**.")
        return
    _render(run, v["cards"])


# ------------------------------------------------------------------ result

def _range(a: float, b: float, nd: int) -> str:
    return f"{a:.{nd}f}" if abs(a - b) < 10 ** -nd else f"{a:.{nd}f} to {b:.{nd}f}"


def _render(run: oc.ConsoleRun, show_cards: bool) -> None:
    s, n, c_min = run.setup, run.n, run.c_min
    fam, size = run.nozzle
    noz = f"{fam} {size}"
    lo, hi = s.rate_band
    tol = 100 * s.tol_frac
    ui.kpi_row([
        ("Nozzle on the boom", noz, _family_note(fam)),
        ("Set rate", f"{s.gpa:g}", "gal/acre", f"held within ±{tol:g}% ({lo:.1f} to {hi:.1f})"),
        ("Starting pressure", f"{s.pressure_psi:.0f}", "psi", f"{L.droplet(s.droplet_class)} droplets"),
        ("Target speed", f"{s.speed_kmh:.1f}", "km/h", f"{kmh_to_mph(s.speed_kmh):.1f} mph"),
        ("Minimum coverage", f"{c_min:g}", "%", "on every card"),
    ])
    ui.chips(run.meta.get("chips", []))

    sm = oc.summary(run)
    h, c = sm[oc.HELD], sm[oc.CONSOLE]
    rate = c["rate"]
    st.markdown(f"**Console adjusts vs setting held** (same field, same seed; "
                f"the held run keeps {noz} at {s.pressure_psi:.0f} psi and {s.speed_kmh:.1f} km/h)")
    ui.kpi_row([
        (f"At or above {c_min:g}%", f"{c['n_met']} / {n}", f"setting held: {h['n_met']} / {n}"),
        ("Average coverage", f"{c['mean_cov']:.1f}", "%", f"setting held: {h['mean_cov']:.1f}%"),
        ("Drift", f"{100 * c['mean_drift']:.1f}", "%", f"setting held: {100 * h['mean_drift']:.1f}%"),
        ("Applied rate", f"{rate['mean']:.1f}", "gal/acre",
         f"all sections within ±{tol:g}%" if rate["in_band"]
         else f"{rate['n_out_of_band']} sections off the band"),
        ("Unreachable at rate", f"{c['n_unreachable']}", f"of {n} sections (model)"),
        ("Needs re-spraying", f"{c['n_retreat']}", f"setting held: {h['n_retreat']}"),
    ])
    st.caption("Sections counted on actual coverage (simulator truth); the console sees only the "
               "card readings. Simulation, not field data.")
    if c["n_unreachable"]:
        st.warning(f"{noz} cannot reach {c_min:g}% at {s.gpa:g} gal/acre in {c['n_unreachable']} of {n} "
                   f"sections (model). The console used the highest coverage it could. "
                   f"See **Switch nozzle?** below.")
    else:
        st.success(f"The console found a setting predicted to reach {c_min:g}% at {s.gpa:g} gal/acre in "
                   f"every section.")

    res = run.console
    if show_cards and res.cards is None:
        with st.spinner("Drawing the paper cards ..."):
            al.ensure_cards(res)
    render_field_animation(res, c_min, height=REPLAY_HEIGHT, cards=show_cards)
    if show_cards and res.cards is None:
        st.caption(res.meta.get("cards_note") or "The paper cards could not be drawn for this run.")

    with st.expander("Section by section"):
        df = oc.section_frame(run)
        df = df.assign(card_class=df["card_class"].map(L.cov_class), tier=df["tier"].map(L.rate_tier))
        st.dataframe(df.rename(columns=L.OP_SECTION_COLS).style.format(
            {L.OP_SECTION_COLS[k]: f for k, f in L.OP_SECTION_FMT.items()}), hide_index=True,
            width="stretch", height=320)

    _what_to_change(run)
    _switch_nozzle(run, expanded=c["n_unreachable"] > 0)


def _what_to_change(run: oc.ConsoleRun) -> None:
    st.markdown("**What to change**")
    k = oc.advice_counts(run)
    c_min = run.c_min
    if not k["n_short"]:
        st.success(f"Every card read at or above {c_min:g}%. No change needed.")
        return
    parts = [f"{k['n_short']} of {k['n']} cards read below {c_min:g}%."]
    if k["n_fixable"]:
        parts.append(f"A pressure or boom change is predicted to bring {k['n_fixable']} of them to {c_min:g}%.")
    if k["n_unfixable"]:
        parts.append(f"For {k['n_unfixable']}, no change with this nozzle reaches {c_min:g}% at this rate.")
    if k["common_move"] is not None:
        parts.append(f"Most common advice: {L.rate_move(*k['common_move'])} ({k['n_common']} sections).")
    st.markdown(" ".join(parts))
    df = oc.advice_frame(run)
    cols = ["section", "card_cov", "short_pp", "advice"]
    st.dataframe(df[cols].rename(columns=L.OP_ADVICE_COLS).style.format(
        {L.OP_ADVICE_COLS["card_cov"]: "{:.1f}", L.OP_ADVICE_COLS["short_pp"]: "{:.1f}"}),
        hide_index=True, width="stretch", height=min(320, 38 + 35 * len(df)))
    st.caption("From the model, starting from each card reading. The rate stays inside its band.")


def _switch_nozzle(run: oc.ConsoleRun, expanded: bool) -> None:
    s, c_min = run.setup, run.c_min
    fam, size = run.nozzle
    with st.expander("Switch nozzle? (model prediction)", expanded=expanded):
        own = run.own_rank()
        df = oc.alternatives_frame(run, top=5)
        better = df[df["better"]]
        line = (f"For the next fill-up or field, at {s.gpa:g} gal/acre in the same weather. "
                "Model predictions before any card feedback; a section counts only if it reaches "
                f"{c_min:g}% with the controller's safety margin.")
        if own is not None:
            line += (f" Your {fam} {size}: {100 * own['share_meeting']:.0f}% of sections, mean "
                     f"{own['mean_cov']:.0f}% coverage.")
        st.markdown(line)
        if not len(better):
            st.info(f"No other nozzle is predicted to do better at {s.gpa:g} gal/acre in these conditions.")
            return
        show = better.assign(
            nozzle=[f"{f} {z}" for f, z in zip(better["family"], better["size"])],
            pressure=[_range(*r, 0) for r in better["pressure_psi"]],
            speed=[_range(*r, 1) for r in better["speed_kmh"]])
        cols = ["nozzle", "share_pct", "mean_cov", "drift_pct", "pressure", "speed"]
        st.dataframe(show[cols].rename(columns=L.OP_ALT_COLS).style.format(
            {L.OP_ALT_COLS["share_pct"]: "{:.0f}", L.OP_ALT_COLS["mean_cov"]: "{:.1f}",
             L.OP_ALT_COLS["drift_pct"]: "{:.1f}"}), hide_index=True, width="stretch")
