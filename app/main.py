"""CoverageVision AI demo (WP-22).  Run:  uv run streamlit run app/main.py

Closed-loop simulated pass: scenario -> selector -> simulator (hidden mismatch) -> WSP card
-> vision -> bias feedback.  All numbers come from the physics-informed simulator, so this
is simulation-based evidence, not field validation.

Page structure (Field run tab): one run (``app_logic.PassResult`` in ``st.session_state``)
feeds the animated replay, the "What the model decided" tables (operator recommendations,
changes along the field, changes since the previous run, export) and the section inspector.
The replay plays in the browser; the inspector below it is a separate, static look at one
section chosen with the "Section to inspect" slider.
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
from field_run_view import render_field_animation
from spray_lab_view import render_spray_lab
from coverageai import app_logic as al
from coverageai import catalog, scenarios
from coverageai.config import spec
from coverageai.field_anim import nozzle_note
from coverageai.contracts import (Conditions, Setting, coverage_class, effective_boom_height_m,
                                   effective_speed_kmh)

st.set_page_config(page_title="CoverageVision AI", page_icon=None, layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
h1, h2, h3 {letter-spacing: 0.2px;}
.kpi {background:#161b22; border:1px solid #2b3340; border-radius:10px; padding:10px 14px;}
.kpi .l {color:#8a94a6; font-size:0.74rem; text-transform:uppercase; letter-spacing:0.6px;}
.kpi .v {color:#e6edf3; font-size:1.65rem; font-weight:650; line-height:1.15;}
.kpi .s {color:#8a94a6; font-size:0.78rem;}
.panel {background:#161b22; border:1px solid #2b3340; border-radius:10px; padding:12px 16px;}
.badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.78rem;
        font-weight:600; margin-right:6px; color:#0e1117;}
.b-acceptable {background:#3ddc97;} .b-marginal {background:#f5b942;} .b-critical {background:#ef5b5b;}
.b-warn {background:#ef5b5b; color:#fff;} .b-info {background:#2f81f7; color:#fff;}
.b-grey {background:#39414f; color:#e6edf3;}
.cond td {padding:2px 12px 2px 0; color:#c9d1d9; font-size:0.92rem;}
.cond td:first-child {color:#8a94a6;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers

def badge(text: str, kind: str) -> str:
    return f'<span class="badge b-{kind}">{text}</span>'


def kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s">{sub}&nbsp;</div></div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner="Building simulation modules ...")
def get_modules(model: str, verifier: str):
    """Cached module stack (simulator, model, selector, verifier, renderer)."""
    try:
        return al.build_modules_adapter(model, verifier), None
    except Exception as e:  # unavailable option: report in the sidebar, keep the app alive
        return None, f"{type(e).__name__}: {e}"


@st.cache_data(show_spinner=False)
def cached_conditions(preset: str, seed: int):
    return al.make_conditions(preset, seed)


def _qp_index(name: str, options, default: str) -> int:
    """Initial widget choice from the URL (?preset=hot_dry&mismatch=high&policy=oracle ...)."""
    v = st.query_params.get(name, default)
    return list(options).index(v) if v in options else list(options).index(default)


def class_of(cov: float) -> str:
    cc = spec()["coverage_classes"]
    return coverage_class(cov, cc["acceptable_min"], cc["marginal_min"])


LOOP_CSS = """
<style>
.loop {display:flex; gap:6px; align-items:stretch; flex-wrap:wrap; margin:4px 0 10px 0;}
.loop .st {flex:1 1 150px; background:#161b22; border:1px solid #2b3340; border-radius:10px;
           padding:8px 10px; min-width:140px;}
.loop .st .n {color:#4da3ff; font-size:0.72rem; font-weight:700; text-transform:uppercase;
              letter-spacing:0.5px;}
.loop .st .t {color:#e6edf3; font-weight:650; font-size:0.95rem; margin:2px 0;}
.loop .st .d {color:#8a94a6; font-size:0.8rem; line-height:1.3;}
.loop .ar {align-self:center; color:#4da3ff; font-size:1.1rem;}
</style>
"""


def _loop_html(steps: list[tuple[str, str, str]]) -> str:
    parts = []
    for i, (n, t, d) in enumerate(steps):
        if i:
            parts.append('<div class="ar">&#8594;</div>')
        parts.append(f'<div class="st"><div class="n">{n}</div><div class="t">{t}</div>'
                     f'<div class="d">{d}</div></div>')
    return LOOP_CSS + '<div class="loop">' + "".join(parts) + "</div>"


def loop_overview_html() -> str:
    """The proposal's closed loop (Figure 1) in five plain-language steps."""
    return _loop_html([
        ("1 · Predict", "Predict coverage and drift",
         "From the weather and machine state, the model predicts card coverage and drift for every "
         "nozzle setting in the catalog."),
        ("2 · Select", "Pick the setting",
         "Choose the setting that reaches the target coverage (with a safety margin) at the lowest "
         "drift and spray volume."),
        ("3 · Spray", "Spray the section",
         "The simulator sprays it, including surprises the model never sees: gusts, crop canopy, "
         "worn nozzles."),
        ("4 · Verify", "Read the paper card",
         "Computer vision measures the coverage actually achieved on a water-sensitive paper card."),
        ("5 · Feed back", "Correct the next decision",
         "The gap between measured and predicted coverage corrects predictions for the next "
         "sections. Poor cards are flagged for re-spraying."),
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
    fb = (f"Error {lg.meas.coverage_pct - lg.corrected_pred_cov:+.1f} pp · correction for next "
          f"sections {after:+.1f} pp")
    if lg.flagged_retreat:
        fb += " · section flagged for re-spraying"
    return _loop_html([("1 · Predict", f"Section {k + 1}", pred), ("2 · Select", "Chosen setting", sel),
                       ("3 · Spray", "What really happened", spray), ("4 · Verify", "Card reading", ver),
                       ("5 · Feed back", "Correction", fb)])


def render_learning_table(passes: list[dict]) -> None:
    """fast_slow only: how the field result improves across passes."""
    import altair as alt
    st.subheader("Learning between passes")
    df = pd.DataFrame(passes)
    df["Pass"] = [f"{r['pass']}" + (" (shown)" if r["displayed"] else "") for r in passes]
    show = df.rename(columns={"met_pct": "Sections meeting target (%)",
                              "mean_error_pp": "Average error (pp)",
                              "slow_offset_pp": "Learned correction (pp)"})
    a, b = st.columns([2, 1], gap="medium")
    a.dataframe(show[["Pass", "Sections meeting target (%)", "Average error (pp)",
                      "Learned correction (pp)"]].style.format(
        {"Sections meeting target (%)": "{:.0f}", "Average error (pp)": "{:.2f}",
         "Learned correction (pp)": "{:+.2f}"}), hide_index=True, width="stretch")
    b.altair_chart(alt.Chart(show).mark_bar(color="#6ea8fe").encode(
        x=alt.X("Pass:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Average error (pp):Q", title="Error (pp)"),
        tooltip=["Pass", "Average error (pp)"]).properties(height=140), width="stretch")
    st.caption("Each earlier pass is a new simulated run over the same field with fresh gusts, canopy and "
               "wear; cards from it teach the model a correction. The last pass is the one shown below.")


# ------------------------------------------------------------------ sidebar

SS = st.session_state
SS.setdefault("result", None)
SS.setdefault("previous", None)    # the run before this one, for "changes since the previous run"
SS.setdefault("section", 1)        # section shown by the inspector (1-based, slider key)
SS.setdefault("run_id", 0)
SS.setdefault("card_cache", {})
SS.setdefault("compare", None)
SS.setdefault("want_run", False)

with st.sidebar:
    st.markdown("### CoverageVision AI")
    st.caption("Closed-loop spray coverage control: simulation demo")
    preset = st.selectbox("Weather scenario", scenarios.list_presets(), format_func=L.preset,
                          index=_qp_index("preset", scenarios.list_presets(), "gusty_afternoon"),
                          help="A ready-made set of wind, temperature and humidity conditions for a "
                               "simulated pass across the field.")
    st.caption(scenarios.get(preset).description)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, step=1,
                           value=int(st.query_params.get("seed", 1)),
                           help="Same number = same simulated field and weather, so runs can be repeated "
                                "and compared. Change it for a different random field.")
    mismatch = st.select_slider("Real-world surprises (unknown to the model)", al.MISMATCH_LEVELS,
                                format_func=L.mismatch,
                                value=al.MISMATCH_LEVELS[_qp_index("mismatch", al.MISMATCH_LEVELS, "med")],
                                help="How far reality departs from what the model expects: sudden gusts, "
                                     "the crop canopy catching spray, worn nozzles. The model never sees these.")
    policy = st.selectbox("Control strategy", al.APP_POLICIES, format_func=L.policy,
                          index=_qp_index("policy", al.APP_POLICIES, "fast"), help=(
        "Model only: pick nozzles from the model's prediction alone. "
        "Model + card feedback: also correct the model using what the paper card shows. "
        "Model + card feedback + learning between passes: the same, after learning a correction from "
        "three earlier passes over the field. "
        "Knows hidden conditions: picks knowing the real gusts, canopy and nozzle wear (a fair upper "
        "bound, not achievable in practice). "
        "Perfect knowledge incl. card noise: also knows each card's random variation (unreachable). "
        "Single best nozzle: one setting for the whole field."))
    c_req = st.slider("Target coverage (%)", 10.0, 50.0, float(spec()["c_req_pct"]), 0.5,
                      help="Share of the water-sensitive paper card area that must be covered by "
                           "droplets for a field section to count as well sprayed.")
    lam = st.slider("Priority: save water  <->  reduce drift", 0.0, 1.0,
                    float(spec()["selector"]["lambda"]), 0.05,
                    help="0 = mostly save water (use less spray volume). 1 = mostly reduce drift "
                         "(keep spray from blowing off target). Values between balance the two.")
    model_name = st.selectbox("Coverage predictor", al.MODEL_CHOICES, format_func=L.model,
                              index=_qp_index("model", al.MODEL_CHOICES, "spray"),
                              help="The model that predicts how much of the card a nozzle setting will cover.")
    reader_opts = list(al.VERIFIER_CHOICES)
    verifier_name = st.selectbox("Card reader", reader_opts, format_func=L.verifier,
                                 index=_qp_index("verifier", reader_opts, "b0"),
                                 help="How coverage is read from the water-sensitive paper card. "
                                      "Perfect reading = the true value. Noisy reading = the true value "
                                      "plus simulated sensor error. Classic image analysis and the "
                                      "neural network look at a picture of the card, like a real camera "
                                      "would (slower).")
    nozzle_opts = ["auto", "free"] + [f"{f}-{z}" for f in catalog.families() for z in catalog.sizes(f)]
    nozzle_choice = st.selectbox("Nozzle on the boom", nozzle_opts, format_func=L.nozzle_choice,
                                 index=_qp_index("nozzle", nozzle_opts, "auto"),
                                 help="A sprayer carries one nozzle for a whole field, so by default "
                                      "the model picks the single best nozzle for this field and then "
                                      "only adjusts pressure (and speed or boom height if allowed "
                                      "below). You can also name the nozzle that is fitted, or let "
                                      "the nozzle change at every section (not possible on a real "
                                      "boom; kept for experiments).")
    show_cards = st.checkbox("Show the paper cards in the replay",
                             value=st.query_params.get("cards", "1") != "0",
                             help="Shows each section's water-sensitive paper card as the replay "
                                  "reaches it. With a reader that does not look at pictures (perfect "
                                  "or noisy reading) the cards are drawn after the run, which takes a "
                                  "few seconds.")

    def keep_advanced_open() -> None:
        SS["adv_open"] = True        # an expander otherwise folds back up on the rerun a click causes

    with st.expander("Advanced controller options",
                     expanded=bool(SS.get("adv_open")) or any(
                         st.query_params.get(q) == "1" for q in ("speed", "boom"))
                     or st.query_params.get("forecast", "off") != "off"):
        speed_ctl = st.checkbox("Let the controller change travel speed",
                                value=st.query_params.get("speed", "0") == "1",
                                on_change=keep_advanced_open,
                                help="Off: speed is whatever the field conditions say. On: besides the nozzle "
                                     "setting, the controller may drive up to 25% slower or faster (within "
                                     "limits) to trade spray volume against coverage and drift.")
        boom_ctl = st.checkbox("Let the controller change boom height",
                               value=st.query_params.get("boom", "0") == "1",
                               on_change=keep_advanced_open,
                               help="Off: boom height is whatever the field conditions say. On: the controller "
                                    "may also raise or lower the boom (within 25% and the legal range, never "
                                    "so low that the spray fans stop overlapping). Lower booms drift less.")
        fc_opts = list(al.FORECAST_CHOICES)
        forecast_name = st.selectbox("Wind forecast", fc_opts, format_func=L.forecast,
                                     index=_qp_index("forecast", fc_opts, "off"),
                                     on_change=keep_advanced_open,
                                     help="Off: the controller plans with the wind read at the start of each "
                                          "section. Otherwise it predicts the wind the spray will meet from the "
                                          "gusts it measured in the sections already sprayed. Used by the "
                                          "first three strategies only (not by the knows-hidden-conditions "
                                          "or single-nozzle strategies).")
    modules, err = get_modules(model_name, verifier_name)
    if err:
        st.error(f"{L.model(model_name)} / {L.verifier(verifier_name)} unavailable: {err}")
        modules, err2 = get_modules("oracle_nominal", "perfect")
        st.caption("Falling back to Physics model (ideal) with Perfect reading.")
        model_name, verifier_name = "oracle_nominal", "perfect"
    st.divider()
    lockable = modules is not None and hasattr(modules.selector, "best_nozzle")
    if not lockable:
        nozzle_choice = "free"       # the simple test model has its own toy nozzle list
    cfg = (preset, int(seed), mismatch, policy, float(c_req), float(lam), model_name, verifier_name,
           bool(speed_ctl), bool(boom_ctl), forecast_name, nozzle_choice, bool(show_cards))
    ctl_opts = dict(speed_control=bool(speed_ctl), boom_control=bool(boom_ctl), forecaster=forecast_name,
                    fixed_nozzle=nozzle_choice)
    inputs = {"Weather scenario": L.preset(preset), "Random seed": str(int(seed)),
              "Real-world surprises": L.mismatch(mismatch), "Control strategy": L.policy(policy),
              "Target coverage (%)": f"{c_req:g}", "Priority (0 water, 1 drift)": f"{lam:g}",
              "Coverage predictor": L.model(model_name), "Card reader": L.verifier(verifier_name),
              "Nozzle on the boom": L.nozzle_choice(nozzle_choice),
              "Controller changes travel speed": "Yes" if speed_ctl else "No",
              "Controller changes boom height": "Yes" if boom_ctl else "No",
              "Wind forecast": L.forecast(forecast_name)}

    def run_pass_cb() -> None:
        # Callbacks run before widgets exist, so the "section" slider key may be reset here
        # (not later in the script, which raises StreamlitWidgetAlreadyInstantiatedError).
        SS["want_run"] = True
        SS["section"] = 1

    st.button("Run pass", type="primary", help="Simulate driving the whole field with these settings.",
              on_click=run_pass_cb, width="stretch")
    st.caption("Everything here comes from a physics-based simulation, not real field data. Wind "
               "direction is the direction the air moves toward.")

if SS["result"] is None or SS["want_run"]:
    SS["want_run"] = False
    conds0 = cached_conditions(preset, int(seed))
    with st.spinner("Running pass (selector, simulator, card, feedback) ..."):
        new = al.run_pass_wrapper(modules, conds0, policy, int(seed), mismatch, float(c_req), float(lam),
                                  capture_cards=bool(show_cards), **ctl_opts)
    if show_cards and new.cards is None:
        with st.spinner("Drawing the paper cards ..."):
            al.ensure_cards(new)
    new.meta["cfg"] = cfg
    new.meta["inputs"] = inputs
    old = SS["result"]
    if old is not None:
        old.cards = None             # only the newest run keeps its card pictures
    SS.update(result=new, previous=old, run_id=SS["run_id"] + 1, card_cache={})
    st.rerun()
res: al.PassResult = SS["result"]
stale = res.meta.get("cfg") != cfg
cursor = int(np.clip(SS["section"], 1, res.n)) - 1


def render_decisions(res: al.PassResult, prev: al.PassResult | None) -> None:
    """What the predictive model put out for this run, in four views."""
    st.subheader("What the model decided")
    recs = al.recommendation_frame(res)
    dec = al.decision_frame(res)
    n_chg = int((dec["changed"] != "").sum())
    t_rec, t_chg, t_prev, t_all = st.tabs([
        f"Operator recommendations ({len(recs)})", f"Changes along the field ({n_chg})",
        "Changes since the previous run", "All model outputs"])
    with t_rec:
        st.caption("For every section whose card reads below the target: which of pressure, travel "
                   "speed and boom height to change, and by how much, with the nozzle that is fitted. "
                   "The model predicts the effect of each change; the card gives the starting point.")
        if len(recs):
            st.dataframe(recs.style.format({"Card reading (%)": "{:.1f}", "Short of target (pp)": "{:.1f}"}),
                         hide_index=True, width="stretch")
        else:
            st.success("Every card read at or above the target. No change is recommended.")
    with t_chg:
        st.caption("Every section where the controller chose a different setting than in the section "
                   "before it. Nozzle: " + nozzle_note(res) + ".")
        chg = dec[dec["changed"] != ""]
        if len(chg):
            st.dataframe(chg[["section", "changed", "nozzle", "pressure_psi", "speed_kmh", "boom_height_m",
                              "wind_ms", "predicted_cov", "after_feedback_cov", "card_cov"]]
                         .rename(columns=L.DECISION_COLS)
                         .style.format({L.DECISION_COLS[c]: f for c, f in L.DECISION_FMT.items()
                                        if c in ("pressure_psi", "speed_kmh", "boom_height_m", "wind_ms",
                                                 "predicted_cov", "after_feedback_cov", "card_cov")}),
                         hide_index=True, width="stretch")
        else:
            st.info("The controller kept the same setting for the whole field.")
    with t_prev:
        if prev is None:
            st.info("Run a second pass (change a sidebar setting, then press **Run pass**) to see what "
                    "the model changed between the two runs.")
        else:
            d = al.run_changes(prev, res)
            a, b = st.columns([2, 3], gap="medium")
            with a:
                st.markdown("**Inputs you changed**")
                if len(d["inputs"]):
                    st.dataframe(d["inputs"], hide_index=True, width="stretch")
                else:
                    st.caption("None: the same inputs were run again.")
            with b:
                st.markdown("**Result of the run**")
                st.dataframe(d["results"].style.format({"Previous run": "{:.2f}", "This run": "{:.2f}",
                                                        "Change": "{:+.2f}"}),
                             hide_index=True, width="stretch")
            if not d["comparable"]:
                st.caption("The two runs have a different number of sections, so they are not compared "
                           "section by section.")
            elif d["n_changed"]:
                st.markdown(f"**Settings the model changed: {d['n_changed']} of {d['n']} sections**")
                st.dataframe(d["sections"].style.format({"Predicted coverage change (pp)": "{:+.1f}",
                                                         "Actual coverage change (pp)": "{:+.1f}"}),
                             hide_index=True, width="stretch")
            else:
                st.caption("The model chose the same setting in every section of both runs.")
    with t_all:
        st.caption("One row per section: the setting chosen, the model's coverage prediction with its "
                   "80% interval, the feedback correction, the card reading, the actual coverage and "
                   "the predicted and actual drift.")
        st.dataframe(dec.assign(droplet_class=dec["droplet_class"].map(L.droplet))
                     .rename(columns=L.DECISION_COLS).style.format(
            {L.DECISION_COLS[c]: f for c, f in L.DECISION_FMT.items()}), hide_index=True, width="stretch",
            height=320)
        d1, d2 = st.columns(2)
        d1.download_button("Download this table (CSV)", dec.to_csv(index=False).encode("utf8"),
                           file_name=f"model_outputs_{res.policy}_seed{res.seed}.csv", mime="text/csv",
                           width="stretch")
        d2.download_button("Download the full run log (CSV)",
                           al.logs_frame(res).to_csv(index=False).encode("utf8"),
                           file_name=f"run_log_{res.policy}_seed{res.seed}.csv", mime="text/csv",
                           width="stretch")


# ------------------------------------------------------------------ tabs

st.title("CoverageVision AI")
tab_live, tab_cmp, tab_lab, tab_noz = st.tabs(
    ["Field run (closed loop)", "Compare strategies", "How the simulator works", "Nozzle explorer"])

# ================================================================== spray lab
with tab_lab:
    st.caption("Supporting view: what the physics-informed simulator does to one nozzle setting. "
               "The closed-loop system itself is in the Field run tab.")
    render_spray_lab()

# ====================================================================== live
with tab_live:
    if stale:
        st.info("Sidebar settings changed since this run was computed. Press **Run pass** to refresh.")
    st.markdown(loop_overview_html(), unsafe_allow_html=True)
    st.subheader("Replay of the run")
    st.caption("Each field section goes once around the loop above. Press Play to watch the run "
               "section by section: the map, the paper card, the coverage and the feedback all move "
               "together. Hover or click a section for its values; the 4x speed plays the whole field "
               "in about ten seconds.")
    render_field_animation(res, res.c_req, cards=bool(show_cards))
    rc = res.c_req
    cc = spec()["coverage_classes"]
    kp = al.compute_kpis(res.logs, rc)
    n_adv = len(al.recommendation_frame(res))
    cols = st.columns(6)
    kpi(cols[0], "Sections meeting target", f"{kp['n_met']} / {kp['n']}",
        f"{100 * kp['frac_met']:.0f}% (actual coverage)")
    kpi(cols[1], "Average coverage", f"{kp['mean_cov']:.1f}%", f"card reading {kp['mean_meas']:.1f}%")
    kpi(cols[2], "Average spray volume", f"{kp['mean_gpa']:.1f} gal/acre", f"{kp['mean_gpa'] * 9.354:.0f} L/ha")
    kpi(cols[3], "Average drift", f"{100 * kp['mean_drift']:.1f}%", "of sprayed volume")
    kpi(cols[4], "Needs re-spraying", f"{kp['n_retreat']}", "sections with a poor card")
    kpi(cols[5], "Recommendations", f"{n_adv}", "sections with a card below target")
    st.caption(f"Whole run: {L.policy(res.policy)}; surprises: {L.mismatch(res.mismatch).lower()}; "
               f"nozzle: {nozzle_note(res)}. Computed in {res.elapsed_s:.1f} s.")
    render_decisions(res, SS["previous"])
    if res.policy == "fast_slow" and res.meta.get("passes"):
        render_learning_table(res.meta["passes"])
    st.divider()
    st.subheader("Section inspector")
    st.slider("Section to inspect", 1, res.n, key="section",
              help="A closer, static look at one section of the run above: the loop step by step, the "
                   "full-size card and the settings that were considered.")
    k = cursor
    lg = res.logs[k]
    x, d = lg.x, lg.derived
    st.markdown(loop_steps_html(res, k), unsafe_allow_html=True)

    # ---- 2 + 3. segment panel and card
    left, right = st.columns([5, 6], gap="medium")
    with left:
        st.subheader(f"Field section {k + 1}")
        info = res.info_for(k)
        w1, w2 = st.columns([1, 2])
        w1.markdown(al.wind_arrow_svg(x), unsafe_allow_html=True)
        rel = al.wind_arrow_deg(x)
        w2.markdown(
            '<table class="cond">'
            f'<tr><td>Wind speed</td><td>{x.wind_speed_ms:.1f} m/s</td></tr>'
            f'<tr><td>Wind blowing toward</td><td>{x.wind_dir_deg:.0f}° (relative to driving: {rel:+.0f}°)</td></tr>'
            f'<tr><td>Driving direction</td><td>{x.heading_deg:.0f}°</td></tr>'
            f'<tr><td>Crosswind / headwind</td><td>{d.u_cross_ms:+.1f} / {d.u_along_ms:+.1f} m/s</td></tr>'
            f'<tr><td>Temperature / humidity</td><td>{x.temp_c:.1f} °C / {x.rh_pct:.0f} %</td></tr>'
            f'<tr><td>Travel speed</td><td>{effective_speed_kmh(x, lg.u):.1f} km/h'
            + (f' (field {x.travel_speed_kmh:.1f}, set by controller)' if lg.u.travel_speed_kmh is not None else '')
            + '</td></tr>'
            f'<tr><td>Boom height</td><td>{effective_boom_height_m(x, lg.u):.2f} m'
            + (f' (field {x.boom_height_m:.2f}, set by controller)' if lg.u.boom_height_m is not None else '')
            + '</td></tr></table>',
            unsafe_allow_html=True)
        badges = []
        if lg.selector_infeasible:
            badges.append(badge("No nozzle meets target (best effort used)", "warn"))
        if al.wind_limit_exceeded(x, info):
            badges.append(badge("Wind above legal limit", "warn"))
        if lg.flagged_retreat:
            badges.append(badge("Needs re-spraying", "warn"))
        st.markdown(
            f'<div class="panel"><b>Chosen nozzle setting:</b> {L.setting_of(lg.u)}<br>'
            f'<span style="color:#8a94a6">flow {d.flow_gpm:.3f} gal/min, {d.app_rate_gpa:.1f} gal/acre, '
            f'{L.droplet(d.droplet_class).lower()} droplets, median droplet size {d.vmd_um:.0f} µm, '
            f'spray fan {d.spray_angle_deg:.0f}°</span><br>'
            + (" ".join(badges) if badges else badge("Meets target", "acceptable")) + '</div>',
            unsafe_allow_html=True)
        for line in al.explain_choice(res, k):
            st.markdown(f"- {L.explain(line)}")
        if k:
            chg = al.setting_changes(res.logs[k - 1], lg)
            st.markdown("- Changed since the previous section: "
                        + ("; ".join(chg) if chg else "nothing") + ".")
        rec = al.recommendations(res)[k]
        if rec.status != "ok":
            st.warning("**Recommendation.** " + "; ".join(rec.lines()))
        top = al.top_candidates_frame(info)
        if len(top):
            top = top.assign(Setting=top["Setting"].map(L.parse_setting_text)).rename(columns=L.TOP_COLS)
            st.caption("Top 5 nozzle settings considered for this section (best first)")
            st.dataframe(top, hide_index=True, width="stretch",
                         column_config={"Meets target": st.column_config.CheckboxColumn(
                             "Meets target", help="Expected to reach the target coverage")})
        elif res.policy == "best_fixed":
            st.caption("Single best nozzle strategy: the same setting is used everywhere, so there is "
                       "no per-section ranking.")

    with right:
        st.subheader("Water-sensitive paper card", help=L.WSP_HELP)
        cache = SS["card_cache"]
        key = (SS["run_id"], k)
        if key not in cache:
            with st.spinner("Rendering card ..."):
                cache[key] = al.replay_card(res, k)
        card: al.CardView = cache[key]
        opts = ["none", "ground-truth mask"] + (["verifier mask"] if card.pred_mask is not None else [])
        ov_names = {"none": "No overlay", "ground-truth mask": "True stained area",
                    "verifier mask": "Stained area found by card reader"}
        ov = st.radio("Overlay", opts, horizontal=True, format_func=ov_names.get, index=len(opts) - 1 if card.pred_mask is not None else 0,
                      label_visibility="collapsed")
        if ov == "ground-truth mask":
            img = al.overlay_mask(card.image, card.gt_mask, color=(0, 255, 200))
        elif ov == "verifier mask":
            img = al.overlay_mask(card.image, card.pred_mask, color=(255, 64, 64))
        else:
            img = al.overlay_mask(card.image, np.zeros(card.image.shape[:2], bool))
        st.image(img, width="stretch",
                 caption="76 x 26 mm card as photographed (with scanner/phone imperfections). The "
                         "overlay colors the stained (sprayed) pixels.")
        true_c, meas_c = lg.outcome.coverage_pct, lg.meas.coverage_pct
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(L.series("predicted"), f"{lg.pred.coverage_pct:.1f}%",
                  f"{lg.corrected_pred_cov - lg.pred.coverage_pct:+.1f} pp feedback correction"
                  if abs(lg.bias) > 1e-9 else None,
                  help="What the model expected before looking at any card.")
        m2.metric(L.series("corrected"), f"{lg.corrected_pred_cov:.1f}%",
                  help="The model prediction adjusted using what earlier cards showed.")
        m3.metric(L.series("measured"), f"{meas_c:.1f}%", f"{meas_c - true_c:+.1f} vs actual",
                  delta_color="off", help="Coverage as read from this card.")
        m4.metric(L.series("true"), f"{true_c:.1f}%", f"{true_c - rc:+.1f} vs target",
                  help="The real coverage in the simulation (not known in a real field).")
        st.markdown(
            badge("Card reading: " + L.cov_class(lg.meas.coverage_class), lg.meas.coverage_class) +
            badge("Actual: " + L.cov_class(class_of(true_c)), class_of(true_c)) +
            (badge("Card fully saturated", "grey") if lg.meas.saturated else "") +
            badge(f"{lg.outcome.stains_per_cm2:.0f} droplet stains/cm²", "grey") +
            badge(f"drift {100 * lg.outcome.drift_frac:.1f}%", "grey") +
            badge(f"spray shifted {lg.outcome.displacement_mm:.0f} mm", "grey"),
            unsafe_allow_html=True)
        if not card.consistent:
            st.caption("Note: the redrawn card differs slightly from the recorded outcome.")
        if res.modules.verifier is None:
            st.caption("Card reader is set to Perfect reading: the card reading equals the actual coverage; "
                       "the card is shown for inspection only.")
        elif hasattr(res.modules.verifier, "measure_outcome"):
            st.caption("Card reader is the simulated sensor: the card reading is the actual coverage plus "
                       "sensor error and no picture is read during the run. The card and outline shown "
                       "here are for illustration only (read afterwards with classic image analysis).")

    # ---- 4. the feedback loop over the whole run (the numbers of panels B and D above)
    st.subheader("Feedback loop over the run")
    ts = al.timeseries_frame(res.logs)
    t1, t2 = st.columns([3, 2], gap="medium")
    with t1:
        st.altair_chart(charts.coverage_timeseries(ts, rc, cc["marginal_min"], res.n, mark=k + 1),
                        width="stretch")
        st.caption("Coverage in every section (panel B of the replay). Dashed green line: target "
                   "coverage. Shaded bands: Poor / Borderline / Good. Prediction after feedback = model "
                   "prediction + feedback correction. Vertical line: the inspected section.")
    with t2:
        st.altair_chart(charts.bias_chart(ts, res.n, mark=k + 1), width="stretch")
        st.caption("The same numbers as panel D of the replay: the feedback correction applied to each "
                   "section's prediction (line) and the prediction error left after it, actual minus "
                   "prediction after feedback (bars), in percentage points.")
    with st.expander("Feedback-loop numbers for every section"):
        st.dataframe(ts[list(L.LOOP_COLS)].rename(columns=L.LOOP_COLS).style.format(
            {c: "{:+.2f}" if "(pp)" in c else "{:.1f}" for c in L.LOOP_COLS.values() if c != "Section"}),
            hide_index=True, width="stretch")
        st.caption("Error seen by the controller = card reading minus prediction after feedback; it is "
                   "what the next correction is computed from. Correction after this card applies to "
                   "the following sections with similar wind and droplet size.")

# ================================================================= comparison
with tab_cmp:
    st.subheader("Compare control strategies on the same field")
    st.caption("Same weather and same real-world surprises for every strategy; meeting the target is "
               "judged on actual coverage. 'Knows hidden conditions' is the fair upper bound; 'incl. card "
               "noise' also knows each card's random variation and cannot be reached.")
    pols = st.multiselect("Control strategies", list(al.APP_POLICIES), format_func=L.policy,
                          default=["open", "fast", "fast_slow", "oracle"])
    use_vis = st.checkbox("Read cards with the selected card reader (slow)", value=False,
                          disabled=verifier_name == "perfect")
    if st.button("Run comparison", type="primary"):
        conds = cached_conditions(preset, int(seed))
        with st.spinner("Running strategies ..."):
            fixed = al.best_fixed_for(modules, conds, float(c_req)) if "best_fixed" in pols else None
            SS["compare"] = (cfg, al.compare_policies(
                modules, conds, int(seed), mismatch, float(c_req), float(lam), pols or ["open"],
                render=use_vis, fixed_setting=fixed, **ctl_opts))
    if SS["compare"] is None:
        st.info("Press **Run comparison** to run the selected strategies on this scenario.")
    else:
        ccfg, results = SS["compare"]
        if ccfg != cfg:
            st.info("Sidebar settings changed since this comparison. Press **Run comparison** to refresh.")
        table = al.comparison_table(results)
        if "best_fixed" in results:
            fs_ = results["best_fixed"].fixed_setting
            st.caption(f"{L.policy('best_fixed')} uses {L.setting_of(fs_)}")
        table = table.assign(policy=table["policy"].map(L.policy))
        a, b, c = st.columns([5, 3, 3])
        a.altair_chart(charts.policy_bars(table, "requirement met (%)", "Sections meeting target (%)",
                                          color="#3ddc97"), width="stretch")
        b.altair_chart(charts.policy_bars(table, "mean volume (GPA)", "Average spray volume (gal/acre)",
                                          color="#6ea8fe", show_labels=False), width="stretch")
        c.altair_chart(charts.policy_bars(table, "mean drift (frac)", "Average drift (fraction)",
                                          color="#ffb84d", show_labels=False), width="stretch")
        CC = L.CMP_COLS
        st.dataframe(table.rename(columns=CC).style.format({
            CC["requirement met (%)"]: "{:.0f}", CC["mean shortfall (pp)"]: "{:.2f}",
            CC["mean volume (GPA)"]: "{:.2f}", CC["total volume (GPA)"]: "{:.0f}",
            CC["mean drift (frac)"]: "{:.4f}"}),
            hide_index=True, width="stretch")
        frames = {p: al.timeseries_frame(r.logs) for p, r in results.items()}
        st.altair_chart(charts.policy_series(frames, float(c_req)), width="stretch")
        st.caption("A single seed is an anecdote. Multi-seed results with confidence intervals come from "
                   "the closed-loop experiments in the project report.")

# ================================================================== nozzle
with tab_noz:
    st.subheader("Nozzle explorer")
    st.caption("Try different nozzles, pressures and weather and see the predicted spray, with no real-world "
               "surprises. Values come from the nozzle catalog plus the ideal physics model.")
    e1, e2 = st.columns([2, 3], gap="medium")
    with e1:
        fams = catalog.families()
        fam = st.selectbox("Nozzle type", fams, format_func=L.family,
                           index=fams.index("XR") if "XR" in fams else 0, key="nz_fam")
        szs = catalog.sizes(fam)
        size = st.selectbox("Nozzle size", szs, format_func=L.size,
                            help="Larger number = larger opening, more flow.", index=szs.index("04") if "04" in szs else len(szs) // 2,
                            key=f"nz_size_{fam}")
        plo, phi = al.nozzle_pressure_bounds(fam, size)
        pres = st.slider("Pressure (psi)", float(plo), float(phi), float(round((plo + phi) / 2)), 1.0,
                         key=f"nz_p_{fam}_{size}")
        r = spec()["ranges"]
        ws = st.slider("Wind speed (m/s)", *map(float, r["wind_speed_ms"]), 3.0, 0.1, key="nz_w")
        rel_ = st.slider("Wind direction relative to driving (degrees)", -180, 180, 90, 5, key="nz_rel",
                         help="Direction the wind blows toward, relative to the way the sprayer drives. "
                              "90 = pure crosswind blowing from the sprayer's left to its right.")
        tmp = st.slider("Temperature (°C)", *map(float, r["temp_c"]), 25.0, 0.5, key="nz_t")
        rh = st.slider("Relative humidity (%)", *map(float, r["rh_pct"]), 55.0, 1.0, key="nz_rh")
        spd = st.slider("Travel speed (km/h)", *map(float, r["travel_speed_kmh"]), 16.0, 0.5, key="nz_v")
        hgt = st.slider("Boom height (m)", *map(float, r["boom_height_m"]), 0.5, 0.01, key="nz_h")
    xn = Conditions(ws, float(rel_ % 360), 0.0, tmp, rh, spd, hgt)
    un = Setting(fam, size, float(pres))
    pt = al.nozzle_point(un, xn)
    dd, pp = pt["derived"], pt["pred"]
    with e2:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Flow (gal/min)", f"{dd.flow_gpm:.3f}", f"{dd.flow_gpm * 3.785:.2f} L/min", delta_color="off")
        k2.metric("Spray volume (gal/acre)", f"{dd.app_rate_gpa:.1f}", f"{dd.app_rate_lha:.0f} L/ha",
                  delta_color="off")
        k3.metric("Droplet size", L.droplet(dd.droplet_class), f"median {dd.vmd_um:.0f} µm", delta_color="off",
                  help="Median droplet size. Coarser droplets drift less but cover less.")
        k4.metric("Spray fan angle", f"{dd.spray_angle_deg:.0f}°")
        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Predicted coverage", f"{pp.coverage_pct:.1f}%",
                  f"{pp.coverage_pct - c_req:+.1f} vs target", delta_color="normal")
        k6.metric("Predicted drift", f"{100 * pp.drift_frac:.1f}%", help="Share of spray carried off target by wind.")
        k7.metric("Deposit (µL/cm²)", f"{pp.deposition_ul_cm2:.2f}", help="Spray liquid landing per square centimetre.")
        k8.metric("Sideways shift (mm)", f"{pp.displacement_mm:.0f}", help="How far wind pushes the spray sideways.")
        cvp = al.coverage_vs_pressure(fam, size, xn)
        ch1, ch2 = st.columns(2)
        ch1.altair_chart(charts.line_xy(cvp, "pressure_psi", "coverage_pct", "Pressure (psi)",
                                        "Coverage (% of card area)", extra_rule=float(c_req), y_fmt=".1f"),
                         width="stretch")
        ch1.caption("Coverage vs pressure at these conditions (dashed green: target coverage).")
        dvw = al.drift_vs_wind(un, xn)
        ch2.altair_chart(charts.line_xy(dvw, "wind_ms", "drift_frac", "Crosswind (m/s)",
                                        "Drift (fraction of spray)", color="#ffb84d"), width="stretch")
        ch2.caption("Drift vs crosswind at this setting.")
    g1, g2 = st.columns([3, 2], gap="medium")
    with g1:
        grid = al.coverage_grid(fam, size, xn)
        st.altair_chart(charts.heatmap(grid, "coverage_pct", "Coverage map: pressure vs crosswind",
                                       vtitle="Coverage (%)"), width="stretch")
    with g2:
        pat = al.pattern_overlap(dd.spray_angle_deg, hgt)
        st.altair_chart(charts.pattern_chart(pat), width="stretch")
        u1, u2, u3 = st.columns(3)
        u1.metric("Evenness of spray", f"{pat['uniformity']:.3f}")
        u2.metric("Spray width at the card", f"{100 * pat['fan_width_m']:.0f} cm")
        u3.metric("Peak spray intensity", f"{pat['peak']:.2f}")
        st.caption("Illustration of how neighbouring nozzles (20 in apart) overlap; not simulator output.")
