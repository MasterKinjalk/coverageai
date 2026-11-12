"""CoverageVision AI demo (WP-22).  Run:  uv run --no-sync streamlit run app/main.py

Closed-loop simulated pass: scenario -> selector -> simulator (hidden mismatch) -> WSP card
-> vision -> bias feedback.  All numbers come from the physics-informed simulator, so this
is simulation-based evidence, not field validation.

Page structure. Four views (the top tabs) follow one rule: the sidebar holds the inputs of the
view on screen, with that view's one Run button at the top; the main area holds results only.

- Field run: one run (``app_logic.PassResult`` in ``st.session_state``) feeds the KPI row, the
  animated replay and the detail tabs under it. Pausing, clicking or scrubbing in the replay
  selects that section in the "Section detail" tab.
- Compare strategies: every strategy on the same field, over one or more seeds.
- Nozzle lab: one nozzle setting in ideal conditions (``spray_lab_view``); it updates as you type.
- Operator console: the operator's nozzle, label rate and target speed; the same field sprayed with
  that setting held and with the console moving pressure (speed follows, rate held) and boom height
  (``operator_view``, logic in ``coverageai.operator_console``).

Only the view on screen is computed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st

import charts
import labels as L
import ui
from field_run_view import render_field_animation, section_picked
from spray_lab_view import lab_controls, render_nozzle_lab
from operator_view import op_controls, op_run_button, render_operator
from coverageai import app_logic as al
from coverageai import catalog, scenarios
from coverageai.config import spec
from coverageai.contracts import coverage_class, effective_boom_height_m, effective_speed_kmh
from coverageai.units import gpa_to_lha

st.set_page_config(page_title="CoverageVision AI", page_icon=None, layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()

FIELD, COMPARE, LAB, OPERATOR = "Field run", "Compare strategies", "Nozzle lab", "Operator console"
VIEWS = (FIELD, COMPARE, LAB, OPERATOR)
REPLAY_HEIGHT = 740
CARD_CACHE_SIZE = 6            # full-size cards kept for the section detail (about 10 MB each)

SS = st.session_state
QP = st.query_params


# ------------------------------------------------------------------ helpers

@st.cache_resource(show_spinner="Loading the model and the card reader ...")
def _build_modules(model: str, verifier: str, calibrate: bool):
    return al.build_modules_adapter(model, verifier, calibrate_reader=calibrate)


def get_modules(model: str, verifier: str, calibrate: bool = True):
    """The cached module stack (simulator, model, selector, verifier, renderer), or the reason
    it could not be built. A failure is not cached, so it is retried on the next rerun."""
    try:
        return _build_modules(model, verifier, calibrate), None
    except Exception as e:      # a missing model file or optional package: report it, keep the app up
        return None, f"{type(e).__name__}: {e}"


@st.cache_data(show_spinner=False)
def cached_conditions(preset: str, seed: int):
    return al.make_conditions(preset, seed)


def class_of(cov: float) -> str:
    cc = spec()["coverage_classes"]
    return coverage_class(cov, cc["acceptable_min"], cc["marginal_min"])


def pick_view():
    """Top-level navigation. Returns the view on screen and the container to draw it in."""
    try:
        tabs = st.tabs(list(VIEWS), key="view", on_change="rerun")
    except TypeError:           # older Streamlit: tabs carry no state, so use a radio row instead
        v = st.radio("View", VIEWS, horizontal=True, label_visibility="collapsed", key="view")
        return v, st.container()
    for v, t in zip(VIEWS, tabs):
        if t.open:
            return v, t
    return VIEWS[0], tabs[0]


def loop_overview_html() -> str:
    """The proposal's closed loop (Figure 1) in five plain-language steps."""
    return ui.loop_html([
        ("1 · Predict", "Predict coverage and drift",
         "The model predicts card coverage and drift for every nozzle setting in the catalog."),
        ("2 · Select", "Pick the setting",
         "The setting that reaches the target coverage at the lowest drift and spray volume."),
        ("3 · Spray", "Spray the section",
         "The simulator sprays it, with surprises the model never sees: gusts, canopy, worn nozzles."),
        ("4 · Verify", "Read the paper card",
         "Computer vision measures the coverage on a water-sensitive paper card."),
        ("5 · Feed back", "Correct the next decision",
         "The gap between card and prediction corrects the next sections. Poor cards are flagged."),
    ])


def loop_steps_html(res, k: int) -> str:
    """The five loop steps filled in with section k's actual numbers."""
    lg = res.logs[k]
    d = lg.derived
    after = lg.extra.get("bias_after", lg.bias)
    if res.policy in ("oracle", "oracle_card"):
        pred = ("Knows hidden conditions: the real gusts, canopy and nozzle wear"
                + (" and this card's random variation" if res.policy == "oracle_card" else "")
                + f" are known in advance; expects {lg.pred.coverage_pct:.1f}% coverage")
    elif res.policy == "best_fixed":
        pred = (f"Fixed setting chosen once for the whole field; model expects "
                f"{lg.pred.coverage_pct:.1f}% coverage here")
    else:
        pred = (f"Model predicts {lg.pred.coverage_pct:.1f}% coverage and "
                f"{100 * lg.pred.drift_frac:.1f}% drift for the chosen setting")
        if res.policy == "fast_slow":
            pred += f" (including {lg.extra.get('slow_corr', 0.0):+.1f} pp learned in earlier passes)"
    if "fc_plan_wind_ms" in lg.extra:
        pred += f"; planned with a forecast wind of {lg.extra['fc_plan_wind_ms']:.1f} m/s (reading {lg.x.wind_speed_ms:.1f})"
    if abs(lg.bias) > 1e-9:
        pred += f"; feedback shifts it by {lg.bias:+.1f} pp to {lg.corrected_pred_cov:.1f}%"
    sel = (f"{L.setting_of(lg.u)} · {L.droplet(d.droplet_class)} "
           f"droplets · {d.app_rate_gpa:.1f} gal/acre")
    if lg.selector_infeasible:
        sel += " · no setting reached the target, best effort used"
    spray = (f"Actual coverage {lg.outcome.coverage_pct:.1f}% · drift "
             f"{100 * lg.outcome.drift_frac:.1f}% (simulator truth, hidden from the controller)")
    reader = ("perfect reading" if res.modules.verifier is None
              else "image analysis" if res.used_vision else "simulated sensor")
    ver = (f"Card reads {lg.meas.coverage_pct:.1f}% ({reader}) · "
           f"{L.cov_class(lg.meas.coverage_class)}")
    basis = al.feedback_basis(res)[k]
    if basis["uses_feedback"]:
        raw = al.loop_numbers(lg)["raw_error_pp"]
        fb = (f"{lg.bias:+.1f} pp applied, based on {basis['text']}. Card minus model {raw:+.1f} pp, "
              f"so this group's correction is now {after:+.1f} pp")
    else:
        fb = (f"{basis['text']} Card minus prediction: "
              f"{lg.meas.coverage_pct - lg.corrected_pred_cov:+.1f} pp")
    if lg.flagged_retreat:
        fb += " · section flagged for re-spraying"
    return ui.loop_html([("1 · Predict", f"Section {k + 1}", pred), ("2 · Select", "Chosen setting", sel),
                         ("3 · Spray", "What really happened", spray), ("4 · Verify", "Card reading", ver),
                         ("5 · Feed back", "Correction", fb)])


# ------------------------------------------------------------------ inputs

PRESETS = scenarios.list_presets()
MODELS = [m for m in al.MODEL_CHOICES if m != "mock"]     # the test stub is not an app option
NOZZLES = ["auto", "free"] + [f"{f}-{z}" for f in catalog.families() for z in catalog.sizes(f)]
# widget key -> (URL parameter, default, allowed values); e.g. ?preset=hot_dry&policy=oracle
# (also ?seed=, ?seeds= for the comparison, and cards / speed / boom = 0 or 1)
CHOICES = {
    "in_preset": ("preset", "gusty_afternoon", PRESETS),
    "in_mismatch": ("mismatch", "med", al.MISMATCH_LEVELS),
    "in_policy": ("policy", "fast", al.APP_POLICIES),
    "in_model": ("model", "spray", MODELS),
    "in_verifier": ("verifier", "b0", al.VERIFIER_CHOICES),
    "in_nozzle": ("nozzle", "auto", NOZZLES),
    "in_forecast": ("forecast", "off", al.FORECAST_CHOICES),
    "in_feedback": ("feedback", al.default_feedback_choice(), al.FEEDBACK_CHOICES),   # D31: trained by default
}


def _url_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        return int(np.clip(int(QP.get(name, default)), lo, hi))
    except ValueError:
        return default


def init_state() -> None:
    """Seed every input once (from the URL where it has a parameter), then re-assign it.

    The sidebar draws different widgets for each view, and Streamlit drops the value of a widget
    that is not drawn in a run. Re-assigning a key keeps its value, so inputs survive a change of
    view."""
    for key, (name, default, allowed) in CHOICES.items():
        if key not in SS:
            v = QP.get(name, default)
            SS[key] = v if v in allowed else default
    SS.setdefault("in_seed", _url_int("seed", 1, 0, 10_000))
    SS.setdefault("in_c_req", float(spec()["c_req_pct"]))
    SS.setdefault("in_lam", float(spec()["selector"]["lambda"]))
    SS.setdefault("in_cards", QP.get("cards", "1") != "0")
    SS.setdefault("in_speed", QP.get("speed", "0") == "1")
    SS.setdefault("in_boom", QP.get("boom", "0") == "1")
    SS.setdefault("in_calibrate", QP.get("calibrate", "1") != "0")     # D30: on by default in the app
    SS.setdefault("in_cmp_policies", ["open", "fast", "fast_slow", "oracle"])
    SS.setdefault("in_cmp_seeds", _url_int("seeds", 1, 1, 20))
    SS.setdefault("in_cmp_images", False)
    SS.setdefault("section", 1)          # section shown in "Section detail" (1-based, slider key)
    for key in list(SS.keys()):
        if key == "section" or str(key).startswith(("in_", "lab_", "op_")):
            SS[key] = SS[key]
    for key, default in (("result", None), ("previous", None), ("run_id", 0), ("card_cache", {}),
                         ("compare", None), ("want_run", False), ("field_error", None),
                         ("compare_error", None)):
        SS.setdefault(key, default)


def keep_advanced_open() -> None:
    SS["adv_open"] = True        # an expander otherwise folds back up on the rerun a click causes


def run_pass_cb() -> None:
    # Callbacks run before widgets exist, so the "section" slider key may be reset here
    # (not later in the script, which raises StreamlitWidgetAlreadyInstantiatedError).
    SS["want_run"] = True
    SS["section"] = 1


def run_inputs(view: str) -> dict:
    """Sidebar inputs shared by Field run and Compare strategies. Returns their values."""
    v: dict = {}
    ui.group("Field")
    v["preset"] = st.selectbox("Weather scenario", PRESETS, format_func=L.preset, key="in_preset",
                               help=scenarios.get(SS["in_preset"]).description)
    v["mismatch"] = st.select_slider(
        "Hidden surprises", al.MISMATCH_LEVELS, format_func=L.mismatch, key="in_mismatch",
        help="How far reality departs from what the model expects: sudden gusts, the crop canopy "
             "catching spray, worn nozzles. The model never sees these.")
    v["seed"] = int(st.number_input(
        "Random seed", min_value=0, max_value=10_000, step=1, key="in_seed",
        help="Same number = same simulated field and weather, so runs can be repeated and compared."))
    if view == COMPARE:
        v["n_seeds"] = int(st.slider(
            "Number of seeds", 1, 20, key="in_cmp_seeds",
            help="Each seed is a different random field. With 2 or more, the results are averaged "
                 "and shown with a 95% interval. Seeds run from the random seed upward."))
    ui.group("Controller")
    if view == FIELD:
        v["policy"] = st.selectbox("Control strategy", al.APP_POLICIES, format_func=L.policy,
                                   key="in_policy", help=L.POLICY_HELP)
    else:
        v["policies"] = st.multiselect("Control strategies", list(al.APP_POLICIES),
                                       format_func=L.policy_short, key="in_cmp_policies", help=L.POLICY_HELP)
    one_setting = view == FIELD and v["policy"] == "best_fixed"
    poor_below = float(spec()["coverage_classes"]["marginal_min"])
    v["c_req"] = float(st.slider(
        "Target coverage (%)", poor_below, 50.0, step=0.5, format="%.1f", key="in_c_req",
        help="Share of the water-sensitive paper card that must be covered by droplets for a "
             f"field section to count as well sprayed. It starts at {poor_below:g}%: a card below "
             "that is classed Poor and flagged for re-spraying whatever the target."))
    v["lam"] = float(st.slider(
        "Priority: save water ↔ reduce drift", 0.0, 1.0, step=0.05, format="%.2f", key="in_lam",
        help="0 = mostly save water (less spray volume). 1 = mostly reduce drift. "
             "Values between balance the two."))
    v["nozzle"] = st.selectbox(
        "Nozzle on the boom", NOZZLES, format_func=L.nozzle_choice, key="in_nozzle", disabled=one_setting,
        help="A sprayer carries one nozzle for a whole field, so by default the model picks the "
             "single best nozzle for this field and then only adjusts pressure. You can also name "
             "the nozzle that is fitted, or let it change at every section (not possible on a real "
             "boom; kept for experiments). Not used by the single best nozzle strategy.")
    ui.group("Model and card reader")
    v["model"] = st.selectbox("Coverage predictor", MODELS, format_func=L.model, key="in_model",
                              help="The model that predicts how much of the card a nozzle setting will cover.")
    v["verifier"] = st.selectbox(
        "Card reader", list(al.VERIFIER_CHOICES), format_func=L.verifier, key="in_verifier",
        help="How coverage is read from the card. Perfect reading = the true value. Noisy reading = "
             "the true value plus simulated sensor error. Classic image analysis and the neural "
             "network look at a picture of the card, like a real camera would (slower).")
    image_reader = v["verifier"] not in ("perfect", "noisy")
    if view == FIELD:
        v["cards"] = bool(st.toggle(
            "Show paper cards", key="in_cards",
            help="Shows each section's card as the replay reaches it. With a reader that does not "
                 "look at pictures, the cards are drawn after the run, which takes a few seconds."))
    else:
        v["images"] = bool(st.toggle(
            "Read card images (slow)", key="in_cmp_images", disabled=not image_reader,
            help="Off: the strategies get the true coverage as the card reading, which is fast. "
                 "On: every card is drawn and read by the selected card reader.")) and image_reader
    uses_fc = view == COMPARE or v["policy"] in al.FORECAST_POLICIES
    with st.expander("Advanced controller options",
                     expanded=bool(SS.get("adv_open")) or (not one_setting and (SS["in_speed"] or SS["in_boom"]))
                     or (uses_fc and SS["in_forecast"] != "off")
                     or SS["in_feedback"] != al.default_feedback_choice() or not SS["in_calibrate"]):
        speed = st.checkbox(
            "Let the controller change travel speed", key="in_speed", on_change=keep_advanced_open,
            disabled=one_setting,
            help="Besides the nozzle setting, the controller may drive up to 25% slower or faster "
                 "(within limits). Not used by the single best nozzle strategy.")
        boom = st.checkbox(
            "Let the controller change boom height", key="in_boom", on_change=keep_advanced_open,
            disabled=one_setting,
            help="The controller may also raise or lower the boom (within 25% and the legal range, "
                 "never so low that the spray fans stop overlapping). Lower booms drift less.")
        fc = st.selectbox(
            "Wind forecast", list(al.FORECAST_CHOICES), format_func=L.forecast, key="in_forecast",
            on_change=keep_advanced_open, disabled=not uses_fc,
            help="Off: the controller plans with the wind read at the start of each section. "
                 "Otherwise it predicts the wind the spray will meet from the gusts measured so far. "
                 "Used by the three model strategies only.")
        # the trained feedback models are optional files: offer only what is on disk
        fb_opts = [c for c in al.FEEDBACK_CHOICES if al.feedback_model_available(c)]
        if SS["in_feedback"] not in fb_opts:
            SS["in_feedback"] = al.default_feedback_choice()
        uses_fb = view == COMPARE or v["policy"] in al.TRAINED_FEEDBACK_POLICIES
        can_calibrate = al.reader_calibration_available(v["verifier"])
        calibrate = st.checkbox(
            "Correct the card reader's known error", key="in_calibrate", on_change=keep_advanced_open,
            disabled=not can_calibrate,
            help="A card reader is a little off on average, and more so on dense cards (measured on "
                 "simulated cards with known coverage). On: that average error is removed from every "
                 "reading before it is used. Off: raw readings, as in the earlier experiments. Not "
                 "offered with perfect reading or for a reader whose error has not been measured.")
        fbk = "rule"
        if len(fb_opts) > 1:
            fbk = st.selectbox(
                "Card feedback", fb_opts, format_func=L.feedback_choice, key="in_feedback",
                on_change=keep_advanced_open, disabled=not uses_fb,
                help="How the correction is worked out from the cards. Trained model (the default): "
                     "a model that predicts the next card's error from the section and the cards so "
                     "far. Running average: 30% of each new card error, kept per crosswind range and "
                     "droplet group. Used by the Model + card feedback strategy; the learning "
                     "strategy keeps the running average.")
    # an option the strategy cannot use is passed as off, so the run record says what really ran
    v["speed"] = bool(speed) and not one_setting
    v["boom"] = bool(boom) and not one_setting
    v["forecast"] = fc if uses_fc else "off"
    v["feedback"] = fbk if uses_fb else "rule"
    v["calibrate"] = bool(calibrate) and can_calibrate
    return v


def field_inputs_table(v: dict) -> dict:
    """Display name -> value of every input of a field run, for the run-to-run comparison."""
    return {"Weather scenario": L.preset(v["preset"]), "Random seed": str(v["seed"]),
            "Hidden surprises": L.mismatch(v["mismatch"]), "Control strategy": L.policy(v["policy"]),
            "Target coverage (%)": f"{v['c_req']:g}", "Priority (0 water, 1 drift)": f"{v['lam']:g}",
            "Coverage predictor": L.model(v["model"]), "Card reader": L.verifier(v["verifier"]),
            "Card reader error corrected": "Yes" if v["calibrate"] else "No",
            "Nozzle on the boom": ("Not used by this strategy" if v["policy"] == "best_fixed"
                                   else L.nozzle_choice(v["nozzle"])),
            "Controller changes travel speed": "Yes" if v["speed"] else "No",
            "Controller changes boom height": "Yes" if v["boom"] else "No",
            "Wind forecast": L.forecast(v["forecast"]),
            "Card feedback": L.feedback_choice(v["feedback"])}


def controller_options(v: dict) -> dict:
    return dict(speed_control=v["speed"], boom_control=v["boom"], forecaster=v["forecast"],
                fixed_nozzle=v["nozzle"], feedback_model=v["feedback"])


# ------------------------------------------------------------------ field run

def run_field_pass(modules, v: dict) -> None:
    """Run one pass with the sidebar inputs and store it as the current result."""
    bar = st.progress(0.0, text="Running the pass ...")

    def tick(done: int, total: int) -> None:
        frac = min(1.0, done / max(total, 1))
        bar.progress(frac, text=f"Running the pass ... {100 * frac:.0f}%")

    try:
        # same_field: the replayed pass of the learning strategy runs on this seed's field, as in
        # the Compare view, so a run here and a row there are the same experiment
        new = al.run_pass_wrapper(modules, cached_conditions(v["preset"], v["seed"]), v["policy"], v["seed"],
                                  v["mismatch"], v["c_req"], v["lam"], capture_cards=v["cards"],
                                  progress=tick, same_field=True, **controller_options(v))
    except Exception as e:      # show the reason in the page instead of a traceback; keep the last run
        bar.empty()
        SS["field_error"] = f"{type(e).__name__}: {e}"
        return
    bar.empty()
    new.meta["cfg"] = field_cfg(v)
    new.meta["inputs"] = field_inputs_table(v)
    old = SS["result"]
    if old is not None:
        old.cards = None             # only the newest run keeps its card pictures
    SS.update(result=new, previous=old, run_id=SS["run_id"] + 1, card_cache={}, field_error=None)


def field_cfg(v: dict) -> tuple:
    """What a field run depends on. "Show paper cards" is a display choice and is not part of it."""
    return (v["preset"], v["seed"], v["mismatch"], v["policy"], v["c_req"], v["lam"], v["model"],
            v["verifier"], v["speed"], v["boom"], v["forecast"], v["nozzle"], v["feedback"], v["calibrate"])


def render_learning(passes: list[dict]) -> None:
    """fast_slow only: how the field result improves across passes."""
    import altair as alt
    df = pd.DataFrame(passes)
    df["Pass"] = [f"{r['pass']}" + (" (shown)" if r["displayed"] else "") for r in passes]
    show = df.rename(columns={"met_pct": "Sections on target (%)",
                              "mean_error_pp": "Average error (pp)",
                              "slow_offset_pp": "Learned correction (pp)"})
    a, b = st.columns([2, 1], gap="medium")
    a.dataframe(show[["Pass", "Sections on target (%)", "Average error (pp)",
                      "Learned correction (pp)"]].style.format(
        {"Sections on target (%)": "{:.0f}", "Average error (pp)": "{:.2f}",
         "Learned correction (pp)": "{:+.2f}"}), hide_index=True, width="stretch")
    b.altair_chart(alt.Chart(show).mark_bar(color="#6ea8fe").encode(
        x=alt.X("Pass:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Average error (pp):Q", title="Error (pp)"),
        tooltip=["Pass", "Average error (pp)"]).properties(height=140), width="stretch")
    st.caption("Each earlier pass is another simulated run over the same field with fresh gusts, canopy "
               "and wear. Its cards teach the model a correction. The last pass is the one replayed.")


def render_section_detail(res: al.PassResult) -> None:
    """One section of the run: the loop step by step, the conditions, the setting and the card."""
    st.slider("Section", 1, res.n, key="section",
              help="Pick a section here, or pause, click or scrub in the replay above.")
    k = int(SS["section"]) - 1
    lg = res.logs[k]
    x, d = lg.x, lg.derived
    info = res.info_for(k)
    st.markdown(loop_steps_html(res, k), unsafe_allow_html=True)
    left, right = st.columns([5, 6], gap="medium")
    with left:
        w1, w2 = st.columns([1, 3])
        w1.markdown(al.wind_arrow_svg(x, size=120), unsafe_allow_html=True)
        rel = al.wind_arrow_deg(x)
        w2.markdown(
            '<table class="cv-cond">'
            f'<tr><td>Wind</td><td>{x.wind_speed_ms:.1f} m/s, {rel:+.0f}° from the driving direction</td></tr>'
            f'<tr><td>Crosswind</td><td>{d.u_cross_ms:+.1f} m/s (headwind {d.u_along_ms:+.1f})</td></tr>'
            f'<tr><td>Air</td><td>{x.temp_c:.1f} °C, {x.rh_pct:.0f}% humidity</td></tr>'
            f'<tr><td>Travel speed</td><td>{effective_speed_kmh(x, lg.u):.1f} km/h'
            + (f' (field {x.travel_speed_kmh:.1f}, set by controller)' if lg.u.travel_speed_kmh is not None else '')
            + '</td></tr>'
            f'<tr><td>Boom height</td><td>{effective_boom_height_m(x, lg.u):.2f} m'
            + (f' (field {x.boom_height_m:.2f}, set by controller)' if lg.u.boom_height_m is not None else '')
            + '</td></tr></table>',
            unsafe_allow_html=True)
        # the badge states what happened (actual coverage against the target), then any warning
        met = lg.outcome.coverage_pct >= res.c_req
        badges = [ui.badge("Actual coverage on target", "acceptable") if met
                  else ui.badge("Actual coverage below target", "warn")]
        if lg.selector_infeasible:
            badges.append(ui.badge("No nozzle meets target (best effort used)", "warn"))
        if al.wind_limit_exceeded(x, info):
            badges.append(ui.badge("Wind above legal limit", "warn"))
        if lg.flagged_retreat:
            badges.append(ui.badge("Needs re-spraying", "warn"))
        st.markdown(
            f'<div class="cv-panel"><b>{L.setting_of(lg.u)}</b><br>'
            f'<span style="color:#8a94a6">{d.flow_gpm:.3f} gal/min, {d.app_rate_gpa:.1f} gal/acre, '
            f'{L.droplet(d.droplet_class).lower()} droplets ({d.vmd_um:.0f} µm), '
            f'spray fan {d.spray_angle_deg:.0f}°</span><br>' + " ".join(badges) + '</div>',
            unsafe_allow_html=True)
        rec = al.recommendations(res)[k]
        if rec.status != "ok":
            st.warning("**Recommendation.** " + "; ".join(rec.lines()))
        with st.expander("Why this setting"):
            for line in al.explain_choice(res, k):
                st.markdown(f"- {L.explain(line)}")
            if k:
                chg = al.setting_changes(res.logs[k - 1], lg)
                st.markdown("- Changed since the previous section: "
                            + ("; ".join(chg) if chg else "nothing") + ".")
            top = al.top_candidates_frame(info)
            if len(top):
                top = top.assign(Setting=top["Setting"].map(L.parse_setting_text)).rename(columns=L.TOP_COLS)
                st.dataframe(top, hide_index=True, width="stretch",
                             column_config={"Meets target": st.column_config.CheckboxColumn(
                                 "Meets target", help="Expected to reach the target coverage")})

    with right:
        cache = SS["card_cache"]
        key = (SS["run_id"], k)
        if key not in cache:
            with st.spinner("Drawing the card ..."):
                cache[key] = al.replay_card(res, k)
            while len(cache) > CARD_CACHE_SIZE:
                cache.pop(next(iter(cache)))
        card: al.CardView = cache[key]
        reads_picture = card.pred_mask is not None and not hasattr(res.modules.verifier, "measure_outcome")
        opts = ["none", "ground-truth mask"] + (["verifier mask"] if reads_picture else [])
        ov_names = {"none": "Card only", "ground-truth mask": "True stained area",
                    "verifier mask": "Area found by the card reader"}
        ov = st.radio("Card overlay", opts, horizontal=True, format_func=ov_names.get,
                      index=len(opts) - 1 if reads_picture else 0,
                      help=L.WSP_HELP + " The card is 76 x 26 mm, shown as photographed.")
        if ov == "ground-truth mask":
            img = al.overlay_mask(card.image, card.gt_mask, color=al.TRUTH_COLOR)
        elif ov == "verifier mask":
            img = al.overlay_mask(card.image, card.pred_mask, color=al.READER_COLOR)
        else:
            img = al.overlay_mask(card.image, np.zeros(card.image.shape[:2], bool))
        st.image(img, width="stretch")
        true_c = lg.outcome.coverage_pct
        st.markdown(
            ui.badge("Card reading: " + L.cov_class(lg.meas.coverage_class), lg.meas.coverage_class) +
            ui.badge("Actual: " + L.cov_class(class_of(true_c)), class_of(true_c)) +
            (ui.badge("Card fully saturated", "grey") if lg.meas.saturated else "") +
            ui.badge(f"{lg.outcome.stains_per_cm2:.0f} droplet stains/cm²", "grey") +
            ui.badge(f"spray shifted {lg.outcome.displacement_mm:.0f} mm", "grey"),
            unsafe_allow_html=True)
        if not card.consistent:
            st.caption("This card was redrawn and does not match the recorded outcome exactly.")
        if res.modules.verifier is None:
            st.caption("Perfect reading: the card reading is the actual coverage. The card is shown "
                       "for reference.")
        elif hasattr(res.modules.verifier, "measure_outcome"):
            st.caption("The reading comes from the simulated sensor, not from this picture.")


def render_field(res: al.PassResult, show_cards: bool) -> None:
    kp = al.compute_kpis(res.logs, res.c_req)
    recs = al.recommendation_frame(res)
    dec = al.decision_frame(res)
    ui.kpi_row([
        ("Sections on target", f"{kp['n_met']} / {kp['n']}", f"{100 * kp['frac_met']:.0f}% by actual coverage"),
        ("Average coverage", f"{kp['mean_cov']:.1f}", "%", f"card reading {kp['mean_meas']:.1f}%"),
        ("Spray volume", f"{kp['mean_gpa']:.1f}", "gal/acre", f"{gpa_to_lha(kp['mean_gpa']):.0f} L/ha"),
        ("Drift", f"{100 * kp['mean_drift']:.1f}", "%", "of sprayed volume"),
        ("Needs re-spraying", f"{kp['n_retreat']}", "sections with a poor card"),
        ("Recommendations", f"{len(recs)}", "cards below target"),
    ])
    if show_cards and res.cards is None:
        with st.spinner("Drawing the paper cards ..."):
            al.ensure_cards(res)
    SS["section"] = int(np.clip(SS["section"], 1, res.n))
    picked = section_picked()           # the replay reports a pause, click or scrub
    if picked is not None:
        SS["section"] = int(np.clip(picked, 1, res.n))
    render_field_animation(res, res.c_req, height=REPLAY_HEIGHT, cards=show_cards)
    if show_cards and res.cards is None:
        st.caption(res.meta.get("cards_note") or "The paper cards could not be drawn for this run.")

    n_chg = int((dec["changed"] != "").sum())
    passes = res.meta.get("passes") if res.policy == "fast_slow" else None
    names = ["Section detail", f"Recommendations ({len(recs)})", f"Setting changes ({n_chg})",
             "Previous run", "Feedback loop", "All numbers"] + (["Learning between passes"] if passes else [])
    t_sec, t_rec, t_chg, t_prev, t_loop, t_all, *t_more = st.tabs(names)
    with t_sec:
        render_section_detail(res)
    with t_rec:
        if len(recs):
            st.dataframe(recs.style.format({"Card reading (%)": "{:.1f}", "Short of target (pp)": "{:.1f}"}),
                         hide_index=True, width="stretch")
            st.caption("Sections whose card read below the target, with the change the model suggests "
                       "for the nozzle that is fitted.")
        else:
            st.success("Every card read at or above the target. No change is recommended.")
    with t_chg:
        chg = dec[dec["changed"] != ""]
        if len(chg):
            cols = ["section", "changed", "nozzle", "pressure_psi", "speed_kmh", "boom_height_m",
                    "wind_ms", "predicted_cov", "after_feedback_cov", "card_cov"]
            st.dataframe(chg[cols].rename(columns=L.DECISION_COLS).style.format(
                {L.DECISION_COLS[c]: f for c, f in L.DECISION_FMT.items() if c in cols}),
                hide_index=True, width="stretch")
        else:
            st.info("The controller kept the same setting for the whole field.")
    with t_prev:
        prev = SS["previous"]
        if prev is None:
            st.info("Run a second pass to see what changed between the two runs.")
        else:
            diff = al.run_changes(prev, res)
            a, b = st.columns([2, 3], gap="medium")
            with a:
                st.markdown("**Inputs you changed**")
                if len(diff["inputs"]):
                    st.dataframe(diff["inputs"], hide_index=True, width="stretch")
                else:
                    st.caption("None: the same inputs were run again.")
            with b:
                st.markdown("**Result of the run**")
                st.dataframe(diff["results"].style.format({"Previous run": "{:.2f}", "This run": "{:.2f}",
                                                           "Change": "{:+.2f}"}),
                             hide_index=True, width="stretch")
            if not diff["comparable"]:
                st.caption("The two runs have a different number of sections, so they are not compared "
                           "section by section.")
            elif diff["n_changed"]:
                st.markdown(f"**Settings the model changed: {diff['n_changed']} of {diff['n']} sections**")
                st.dataframe(diff["sections"].style.format({"Predicted coverage change (pp)": "{:+.1f}",
                                                            "Actual coverage change (pp)": "{:+.1f}"}),
                             hide_index=True, width="stretch")
            else:
                st.caption("The model chose the same setting in every section of both runs.")
    with t_loop:
        ts = al.timeseries_frame(res.logs)
        cc = spec()["coverage_classes"]
        k = int(SS["section"])
        t1, t2 = st.columns([3, 2], gap="medium")
        t1.altair_chart(charts.coverage_timeseries(ts, res.c_req, cc["marginal_min"], cc["acceptable_min"],
                                                   res.n, mark=k), width="stretch")
        t2.altair_chart(charts.bias_chart(ts, res.n, mark=k), width="stretch")
        basis = al.feedback_basis(res)
        if basis and basis[0]["uses_feedback"]:
            st.markdown("**What a correction is based on.** " + al.feedback_rule_text(res))
        else:
            st.markdown("**No card feedback in this strategy.** The correction stays at zero; the "
                        "charts show the model's error only.")
        st.caption("The same numbers as replay panels B and D. Vertical line: the section picked in "
                   "Section detail.")
        with st.expander("Numbers for every section"):
            table = ts[list(L.LOOP_COLS)].rename(columns=L.LOOP_COLS)
            table.insert(3, "Correction based on", [b["text"] for b in basis])
            st.dataframe(table.style.format(
                {c: "{:+.2f}" if "(pp)" in c else "{:.1f}" for c in L.LOOP_COLS.values() if c != "Section"}),
                hide_index=True, width="stretch")
    with t_all:
        st.dataframe(dec.assign(droplet_class=dec["droplet_class"].map(L.droplet))
                     .rename(columns=L.DECISION_COLS).style.format(
            {L.DECISION_COLS[c]: f for c, f in L.DECISION_FMT.items()}), hide_index=True, width="stretch",
            height=320)
        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button("Download this table (CSV)", dec.to_csv(index=False).encode("utf8"),
                           file_name=f"model_outputs_{res.policy}_seed{res.seed}.csv", mime="text/csv",
                           width="stretch")
        d2.download_button("Download the full run log (CSV)",
                           al.logs_frame(res).to_csv(index=False).encode("utf8"),
                           file_name=f"run_log_{res.policy}_seed{res.seed}.csv", mime="text/csv",
                           width="stretch")
    if passes:
        with t_more[0]:
            render_learning(passes)
    with st.expander("How the closed loop works"):
        st.markdown(loop_overview_html(), unsafe_allow_html=True)


# ------------------------------------------------------------------ compare

def compare_cfg(v: dict) -> tuple:
    """What a comparison depends on. With card images off, the image readers are not used (the
    strategies get the true coverage), so which of them is selected does not matter."""
    reader = v["verifier"] if v["images"] or v["verifier"] in ("perfect", "noisy") else "true coverage"
    return (v["preset"], v["seed"], v["n_seeds"], tuple(v["policies"]), v["mismatch"], v["c_req"], v["lam"],
            v["model"], reader, v["speed"], v["boom"], v["forecast"], v["nozzle"], v["feedback"],
            v["calibrate"] and reader != "true coverage")


def run_comparison(modules, v: dict) -> None:
    pols = list(v["policies"])
    seeds = list(range(v["seed"], v["seed"] + v["n_seeds"]))
    bar = st.progress(0.0, text="Running the strategies ...")

    def tick(done: int, total: int) -> None:
        bar.progress(min(1.0, done / max(total, 1)), text=f"Running the strategies ... {done} of {total} runs")

    try:
        out = al.compare_seeds(modules, v["preset"], seeds, v["mismatch"], v["c_req"], v["lam"], pols,
                               render=v["images"], progress=tick, **controller_options(v))
    except Exception as e:      # show the reason in the page instead of a traceback
        bar.empty()
        SS["compare_error"] = f"{type(e).__name__}: {e}"
        return
    bar.empty()
    reader = (L.verifier(v["verifier"]) if v["images"] or v["verifier"] in ("perfect", "noisy")
              else "True coverage (card images not read)")
    seed_txt = str(seeds[0]) if len(seeds) == 1 else f"{seeds[0]} to {seeds[-1]} ({len(seeds)} fields)"
    SS["compare"] = {"cfg": compare_cfg(v), "out": out, "c_req": v["c_req"], "chips": [
        ("Scenario", L.preset(v["preset"])), ("Hidden surprises", L.mismatch(v["mismatch"])),
        ("Seed", seed_txt), ("Target coverage", f"{v['c_req']:g}%"), ("Card reading", reader),
        ("Nozzle", L.nozzle_choice(v["nozzle"])), ("Card feedback", L.feedback_choice(v["feedback"]))]}
    SS["compare_error"] = None


def render_compare() -> None:
    cmp = SS["compare"]
    if cmp is None:
        st.info("Choose the strategies in the sidebar, then press **Run comparison**.")
        return
    out = cmp["out"]
    n_seeds = len(out["seeds"])
    ref = "oracle" if "oracle" in out["policies"] else out["policies"][0]
    stats = al.comparison_stats(out["per_seed"], reference=ref)
    stats = stats.assign(policy=stats["policy"].map(L.policy))
    ui.chips(cmp["chips"])
    a, b, c = st.columns([5, 3, 3])
    a.altair_chart(charts.policy_bars(stats, "met_pct", "Sections on target (%)", color="#3ddc97",
                                      lo="met_lo", hi="met_hi", fmt=".1f", percent=True), width="stretch")
    b.altair_chart(charts.policy_bars(stats, "volume_gpa", "Spray volume (gal/acre)", color="#6ea8fe",
                                      show_labels=False, lo="volume_lo", hi="volume_hi", fmt=".1f"),
                   width="stretch")
    c.altair_chart(charts.policy_bars(stats, "drift_pct", "Drift (% of sprayed volume)", color="#ffb84d",
                                      show_labels=False, lo="drift_lo", hi="drift_hi", fmt=".2f"),
                   width="stretch")
    st.dataframe(L.compare_table(stats, L.policy(ref), n_seeds), hide_index=True, width="stretch")
    if n_seeds == 1:
        frames = {p: al.timeseries_frame(r.logs) for p, r in out["first"].items()}
        st.altair_chart(charts.policy_series(frames, float(cmp["c_req"])), width="stretch")
        st.caption("One seed is one field. Raise **Number of seeds** for an average with a 95% interval.")
    else:
        st.caption(f"Mean over {n_seeds} fields. Whiskers and intervals: 95% bootstrap over the seeds. "
                   f"Differences are paired on the same seeds. Meeting the target is judged on actual "
                   f"coverage.")


# ------------------------------------------------------------------ page

init_state()
ui.header("CoverageVision AI", "Closed-loop spray coverage control · simulation, not field data")
view, body = pick_view()

modules = err = None
v: dict = {}
with st.sidebar:
    top = st.container()         # the view's Run button and status line sit above its inputs
    if view == LAB:
        lab_u, lab_x, lab_target = lab_controls()
    elif view == OPERATOR:
        v = op_controls()
        modules, err = get_modules(v["model"], v["verifier"], v["calibrate"])
    else:
        v = run_inputs(view)
        modules, err = get_modules(v["model"], v["verifier"], v["calibrate"])
    with top:
        if view == FIELD:
            st.button("Run pass", type="primary", on_click=run_pass_cb, width="stretch",
                      disabled=modules is None, help="Simulate driving the whole field with these inputs.")
            res0 = SS["result"]
            if res0 is not None and res0.meta.get("cfg") != field_cfg(v):
                st.caption(":orange[Inputs changed since this run. Press Run pass to update.]")
            elif res0 is not None:
                st.caption(f"Showing the last run ({res0.elapsed_s:.1f} s).")
        elif view == COMPARE:
            if st.button("Run comparison", type="primary", width="stretch",
                         disabled=modules is None or not v["policies"],
                         help="Run every selected strategy on the same field or fields."):
                SS["want_compare"] = True
            cmp0 = SS["compare"]
            if cmp0 is not None and cmp0["cfg"] != compare_cfg(v):
                st.caption(":orange[Inputs changed since this comparison. Press Run comparison to update.]")
        elif view == OPERATOR:
            op_run_button(modules, v)
        if err:
            st.error(f"{L.model(v['model'])} with {L.verifier(v['verifier'])} could not be loaded "
                     f"({err}). Choose another predictor or card reader.")

with body:
    failed = SS["field_error"] if view == FIELD else SS["compare_error"] if view == COMPARE else None
    if failed:
        st.error(f"The {'run' if view == FIELD else 'comparison'} failed: {failed}")
    if view == FIELD:
        # first visit: run once with the defaults; never retry a failed run without a click
        if modules is not None and (SS["want_run"] or (SS["result"] is None and not SS["field_error"])):
            SS["want_run"] = False
            run_field_pass(modules, v)
            st.rerun()               # redraw the sidebar status line for the new run
        if SS["result"] is not None:
            render_field(SS["result"], v["cards"])
        elif not SS["field_error"]:
            st.info("Choose a predictor and card reader that can be loaded, then press **Run pass**.")
    elif view == COMPARE:
        if modules is not None and SS.pop("want_compare", False):
            run_comparison(modules, v)
            st.rerun()
        render_compare()
    elif view == OPERATOR:
        render_operator(modules, v)
    else:
        render_nozzle_lab(lab_u, lab_x, lab_target)
