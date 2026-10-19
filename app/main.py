"""CoverageVision AI demo (WP-22).  Run:  uv run streamlit run app/main.py

Closed-loop simulated pass: scenario -> selector -> simulator (hidden mismatch) -> WSP card
-> vision -> bias feedback.  All numbers come from the physics-informed simulator, so this
is simulation-based evidence, not field validation.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st

import charts
from coverageai import app_logic as al
from coverageai import catalog, scenarios
from coverageai.config import spec
from coverageai.contracts import Conditions, Setting, coverage_class

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


# ------------------------------------------------------------------ sidebar

SS = st.session_state
SS.setdefault("result", None)
SS.setdefault("cursor", 0)
SS.setdefault("playing", False)
SS.setdefault("run_id", 0)
SS.setdefault("card_cache", {})
SS.setdefault("compare", None)
SS.setdefault("want_run", False)
if SS.pop("advance", False):       # autoplay: move on before the cursor slider exists
    SS["cursor"] += 1

with st.sidebar:
    st.markdown("### CoverageVision AI")
    st.caption("Closed-loop spray coverage control: simulation demo")
    preset = st.selectbox("Scenario preset", scenarios.list_presets(),
                          index=_qp_index("preset", scenarios.list_presets(), "gusty_afternoon"),
                          help="Field-pass generators from scenarios.py")
    st.caption(scenarios.get(preset).description)
    seed = st.number_input("Seed", min_value=0, max_value=10_000, step=1,
                           value=int(st.query_params.get("seed", 1)))
    mismatch = st.select_slider("Hidden mismatch level", al.MISMATCH_LEVELS,
                                value=al.MISMATCH_LEVELS[_qp_index("mismatch", al.MISMATCH_LEVELS, "med")],
                                help="Gusts, canopy interception and nozzle wear that the model never sees")
    policy = st.selectbox("Policy", al.APP_POLICIES, index=_qp_index("policy", al.APP_POLICIES, "fast"), help=(
        "open: model only. fast: model + card feedback bias. oracle: selector queries the true "
        "simulator (upper bound). best_fixed: one setting for the whole pass."))
    c_req = st.slider("C_req (% card coverage)", 10.0, 50.0, float(spec()["c_req_pct"]), 0.5)
    lam = st.slider("lambda (drift vs volume weight)", 0.0, 1.0, float(spec()["selector"]["lambda"]), 0.05,
                    help="Score = lambda x drift + (1 - lambda) x volume")
    model_name = st.selectbox("Predictive model", al.MODEL_CHOICES,
                              index=_qp_index("model", al.MODEL_CHOICES, "oracle_nominal"))
    verifier_name = st.selectbox("Verifier", al.VERIFIER_CHOICES,
                                 index=_qp_index("verifier", al.VERIFIER_CHOICES, "perfect"),
                                 help="perfect = true coverage; b0 = classical CV; b1 = U-Net. "
                                      "Non-perfect verifiers render and read a card for every segment (slower).")
    modules, err = get_modules(model_name, verifier_name)
    if err:
        st.error(f"{model_name} / {verifier_name} unavailable: {err}")
        modules, err2 = get_modules("oracle_nominal", "perfect")
        st.caption("Falling back to oracle_nominal / perfect.")
        model_name, verifier_name = "oracle_nominal", "perfect"
    st.divider()
    cfg = (preset, int(seed), mismatch, policy, float(c_req), float(lam), model_name, verifier_name)

    def run_pass_cb() -> None:
        SS["want_run"] = True

    def step(d: int) -> None:
        r = SS["result"]
        if r is not None:
            SS["cursor"] = int(np.clip(SS["cursor"] + d, 0, r.n - 1))

    def toggle_play() -> None:
        r = SS["result"]
        if r is None:
            return
        if SS["cursor"] >= r.n - 1:
            SS["cursor"] = 0
        SS["playing"] = not SS["playing"]

    st.button("Run pass", type="primary", on_click=run_pass_cb, width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.button("Back", on_click=step, args=(-1,), width="stretch")
    c2.button("Step", on_click=step, args=(1,), width="stretch")
    c3.button("Pause" if SS["playing"] else "Play all", on_click=toggle_play, width="stretch")
    if SS["result"] is not None:
        st.slider("Segment", 0, SS["result"].n - 1, key="cursor", format="%d",
                  help="0-based segment index; later segments stay hidden until reached")
    st.caption("Plant = physics-informed simulator, not field data. Wind direction is the "
               "direction the air moves toward.")

if SS["result"] is None or SS["want_run"]:
    SS["want_run"] = False
    conds0 = cached_conditions(preset, int(seed))
    with st.spinner("Running pass (selector, simulator, card, feedback) ..."):
        new = al.run_pass_wrapper(modules, conds0, policy, int(seed), mismatch, float(c_req), float(lam))
    new.meta["cfg"] = cfg
    SS.update(result=new, run_id=SS["run_id"] + 1, playing=False, card_cache={})
    SS["cursor"] = 0
    st.rerun()
res: al.PassResult = SS["result"]
stale = res.meta.get("cfg") != cfg
cursor = int(np.clip(SS["cursor"], 0, res.n - 1))

# ------------------------------------------------------------------ tabs

st.title("CoverageVision AI")
tab_live, tab_cmp, tab_noz = st.tabs(["Live pass", "Policy comparison", "Nozzle explorer"])

# ====================================================================== live
with tab_live:
    if stale:
        st.info("Sidebar settings changed since this pass was computed. Press **Run pass** to refresh.")
    k = cursor
    lg = res.logs[k]
    x, d = lg.x, lg.derived
    rc = res.c_req
    cc = spec()["coverage_classes"]

    # ---- 1. header KPIs
    kp = al.compute_kpis(res.logs, rc, upto=k)
    cols = st.columns(6)
    kpi(cols[0], "Meeting C_req", f"{kp['n_met']} / {kp['n']}",
        f"{100 * kp['frac_met']:.0f}% (true coverage)")
    kpi(cols[1], "Mean coverage", f"{kp['mean_cov']:.1f}%", f"measured {kp['mean_meas']:.1f}%")
    kpi(cols[2], "Mean volume", f"{kp['mean_gpa']:.1f} GPA", f"{kp['mean_gpa'] * 9.354:.0f} L/ha")
    kpi(cols[3], "Mean drift", f"{100 * kp['mean_drift']:.1f}%", "of emitted volume")
    kpi(cols[4], "Re-treatment flags", f"{kp['n_retreat']}", "critical cards")
    kpi(cols[5], "Segment", f"{k + 1} / {res.n}",
        f"{res.policy}, {res.mismatch} mismatch" + (f", {res.elapsed_s:.1f}s" if k == 0 else ""))
    st.write("")

    # ---- 2 + 3. segment panel and card
    left, right = st.columns([5, 6], gap="medium")
    with left:
        st.subheader(f"Segment {k + 1}")
        info = res.info_for(k)
        w1, w2 = st.columns([1, 2])
        w1.markdown(al.wind_arrow_svg(x), unsafe_allow_html=True)
        rel = al.wind_arrow_deg(x)
        w2.markdown(
            '<table class="cond">'
            f'<tr><td>Wind speed</td><td>{x.wind_speed_ms:.1f} m/s</td></tr>'
            f'<tr><td>Wind toward</td><td>{x.wind_dir_deg:.0f} deg (rel. {rel:+.0f} deg)</td></tr>'
            f'<tr><td>Heading</td><td>{x.heading_deg:.0f} deg</td></tr>'
            f'<tr><td>Crosswind / along</td><td>{d.u_cross_ms:+.1f} / {d.u_along_ms:+.1f} m/s</td></tr>'
            f'<tr><td>Temp / RH</td><td>{x.temp_c:.1f} C / {x.rh_pct:.0f} %</td></tr>'
            f'<tr><td>Travel speed</td><td>{x.travel_speed_kmh:.1f} km/h</td></tr>'
            f'<tr><td>Boom height</td><td>{x.boom_height_m:.2f} m</td></tr></table>',
            unsafe_allow_html=True)
        badges = []
        if lg.selector_infeasible:
            badges.append(badge("infeasible: fallback", "warn"))
        if al.wind_limit_exceeded(x, info):
            badges.append(badge("wind limit exceeded", "warn"))
        if lg.flagged_retreat:
            badges.append(badge("re-treatment", "warn"))
        st.markdown(
            f'<div class="panel"><b>Chosen setting:</b> {lg.u.family}-{lg.u.size} @ '
            f'{lg.u.pressure_psi:g} psi<br>'
            f'<span style="color:#8a94a6">flow {d.flow_gpm:.3f} gpm, {d.app_rate_gpa:.1f} GPA, '
            f'droplet class {d.droplet_class}, VMD {d.vmd_um:.0f} um, fan {d.spray_angle_deg:.0f} deg</span><br>'
            + (" ".join(badges) if badges else badge("feasible", "acceptable")) + '</div>',
            unsafe_allow_html=True)
        for line in al.explain_choice(res, k):
            st.markdown(f"- {line}")
        top = al.top_candidates_frame(info)
        if len(top):
            st.caption("Top-5 candidates (selector ranking)")
            st.dataframe(top, hide_index=True, width="stretch",
                         column_config={"Feasible": st.column_config.CheckboxColumn("Feasible")})
        elif res.policy == "best_fixed":
            st.caption("Best-fixed policy: no per-segment ranking.")

    with right:
        st.subheader("Water-sensitive paper card")
        cache = SS["card_cache"]
        key = (SS["run_id"], k)
        if key not in cache:
            with st.spinner("Rendering card ..."):
                cache[key] = al.replay_card(res, k)
        card: al.CardView = cache[key]
        opts = ["none", "ground-truth mask"] + (["verifier mask"] if card.pred_mask is not None else [])
        ov = st.radio("Overlay", opts, horizontal=True, index=len(opts) - 1 if card.pred_mask is not None else 0,
                      label_visibility="collapsed")
        if ov == "ground-truth mask":
            img = al.overlay_mask(card.image, card.gt_mask, color=(0, 255, 200))
        elif ov == "verifier mask":
            img = al.overlay_mask(card.image, card.pred_mask, color=(255, 64, 64))
        else:
            img = al.overlay_mask(card.image, np.zeros(card.image.shape[:2], bool))
        st.image(img, width="stretch",
                 caption="76 x 26 mm card as imaged (scanner/phone degradations); the overlay "
                         "marks stained pixels.")
        true_c, meas_c = lg.outcome.coverage_pct, lg.meas.coverage_pct
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted", f"{lg.pred.coverage_pct:.1f}%",
                  f"{lg.corrected_pred_cov - lg.pred.coverage_pct:+.1f} pp bias" if abs(lg.bias) > 1e-9 else None)
        m2.metric("Corrected", f"{lg.corrected_pred_cov:.1f}%")
        m3.metric("Measured", f"{meas_c:.1f}%", f"{meas_c - true_c:+.1f} vs true", delta_color="off")
        m4.metric("True", f"{true_c:.1f}%", f"{true_c - rc:+.1f} vs C_req")
        st.markdown(
            badge("measured: " + lg.meas.coverage_class, lg.meas.coverage_class) +
            badge("true: " + class_of(true_c), class_of(true_c)) +
            (badge("saturated", "grey") if lg.meas.saturated else "") +
            badge(f"{lg.outcome.stains_per_cm2:.0f} stains/cm2", "grey") +
            badge(f"drift {100 * lg.outcome.drift_frac:.1f}%", "grey") +
            badge(f"shift {lg.outcome.displacement_mm:.0f} mm", "grey"),
            unsafe_allow_html=True)
        if not card.consistent:
            st.caption("Note: replayed card differs slightly from the logged outcome.")
        if res.modules.verifier is None:
            st.caption("Verifier = perfect: measured coverage equals true coverage; the card is shown "
                       "for inspection only.")

    # ---- 4. time series
    st.subheader("Coverage along the pass")
    ts = al.timeseries_frame(res.logs, upto=k)
    t1, t2 = st.columns([3, 2], gap="medium")
    with t1:
        st.altair_chart(charts.coverage_timeseries(ts, rc, cc["marginal_min"], res.n), width="stretch")
        st.caption("Dashed green: C_req. Bands: acceptable / marginal / critical. "
                   "Corrected = predicted + feedback bias.")
    with t2:
        st.altair_chart(charts.bias_chart(ts, res.n), width="stretch")
        st.caption("Feedback bias (line) and the residual the controller sees after correction (bars).")

    # ---- 5. field strip
    st.subheader("Field strip")
    basis = st.radio("Tile color shows", ["measured", "true"], horizontal=True,
                     help="Measured = what the card reading says; true = simulator outcome")
    fs = al.field_strip_data(res.conds, res.logs, k, basis=basis)
    st.altair_chart(charts.field_strip_chart(fs), width="stretch")
    st.caption("Segments in driving order (laps alternate direction). White ring: current segment. "
               "✖: flagged for re-treatment (critical card).")

# ================================================================= comparison
with tab_cmp:
    st.subheader("Policy comparison on the same seed")
    st.caption("Same conditions, same hidden mismatch realization. Requirement met is judged on true "
               "coverage. Open = model only; fast = model + card feedback; oracle = upper bound.")
    pols = st.multiselect("Policies", list(al.APP_POLICIES), default=["open", "fast", "oracle"])
    use_vis = st.checkbox("Read cards with the selected verifier (slow)", value=False,
                          disabled=verifier_name == "perfect")
    if st.button("Run comparison", type="primary"):
        conds = cached_conditions(preset, int(seed))
        with st.spinner("Running policies ..."):
            fixed = al.best_fixed_for(modules, conds, float(c_req)) if "best_fixed" in pols else None
            SS["compare"] = (cfg, al.compare_policies(
                modules, conds, int(seed), mismatch, float(c_req), float(lam), pols or ["open"],
                render=use_vis, fixed_setting=fixed))
    if SS["compare"] is None:
        st.info("Press **Run comparison** to run the selected policies on this scenario and seed.")
    else:
        ccfg, results = SS["compare"]
        if ccfg != cfg:
            st.info("Sidebar settings changed since this comparison. Press **Run comparison**.")
        table = al.comparison_table(results)
        if "best_fixed" in results:
            fs_ = results["best_fixed"].fixed_setting
            st.caption(f"best_fixed uses {fs_.family}-{fs_.size} @ {fs_.pressure_psi:g} psi")
        a, b, c = st.columns(3)
        a.altair_chart(charts.policy_bars(table, "requirement met (%)", "requirement met (%)",
                                          color="#3ddc97"), width="stretch")
        b.altair_chart(charts.policy_bars(table, "mean volume (GPA)", "mean volume (GPA)",
                                          color="#6ea8fe"), width="stretch")
        c.altair_chart(charts.policy_bars(table, "mean drift (frac)", "mean drift (fraction)",
                                          color="#ffb84d"), width="stretch")
        st.dataframe(table.style.format({"requirement met (%)": "{:.0f}", "mean shortfall (pp)": "{:.2f}",
                                         "mean volume (GPA)": "{:.2f}", "total volume (GPA)": "{:.0f}",
                                         "mean drift (frac)": "{:.4f}"}),
                     hide_index=True, width="stretch")
        frames = {p: al.timeseries_frame(r.logs) for p, r in results.items()}
        st.altair_chart(charts.policy_series(frames, float(c_req)), width="stretch")
        st.caption("A single seed is an anecdote. Multi-seed results with confidence intervals come from "
                   "the WP-21 closed-loop experiments.")

# ================================================================== nozzle
with tab_noz:
    st.subheader("Nozzle explorer")
    st.caption("Catalog-derived quantities plus the nominal simulator (no hidden mismatch). "
               "Sensitivity style of the MATLAB spray-pattern model.")
    e1, e2 = st.columns([2, 3], gap="medium")
    with e1:
        fams = catalog.families()
        fam = st.selectbox("Family", fams, index=fams.index("XR") if "XR" in fams else 0, key="nz_fam")
        szs = catalog.sizes(fam)
        size = st.selectbox("Size", szs, index=szs.index("04") if "04" in szs else len(szs) // 2,
                            key=f"nz_size_{fam}")
        plo, phi = al.nozzle_pressure_bounds(fam, size)
        pres = st.slider("Pressure (psi)", float(plo), float(phi), float(round((plo + phi) / 2)), 1.0,
                         key=f"nz_p_{fam}_{size}")
        r = spec()["ranges"]
        ws = st.slider("Wind speed (m/s)", *map(float, r["wind_speed_ms"]), 3.0, 0.1, key="nz_w")
        rel_ = st.slider("Wind relative to heading (deg, toward)", -180, 180, 90, 5, key="nz_rel",
                         help="90 = pure crosswind from the vehicle's left to right")
        tmp = st.slider("Temperature (C)", *map(float, r["temp_c"]), 25.0, 0.5, key="nz_t")
        rh = st.slider("Relative humidity (%)", *map(float, r["rh_pct"]), 55.0, 1.0, key="nz_rh")
        spd = st.slider("Travel speed (km/h)", *map(float, r["travel_speed_kmh"]), 16.0, 0.5, key="nz_v")
        hgt = st.slider("Boom height (m)", *map(float, r["boom_height_m"]), 0.5, 0.01, key="nz_h")
    xn = Conditions(ws, float(rel_ % 360), 0.0, tmp, rh, spd, hgt)
    un = Setting(fam, size, float(pres))
    pt = al.nozzle_point(un, xn)
    dd, pp = pt["derived"], pt["pred"]
    with e2:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Flow (gpm)", f"{dd.flow_gpm:.3f}", f"{dd.flow_gpm * 3.785:.2f} L/min", delta_color="off")
        k2.metric("Rate (GPA)", f"{dd.app_rate_gpa:.1f}", f"{dd.app_rate_lha:.0f} L/ha", delta_color="off")
        k3.metric("Droplet class", dd.droplet_class, f"VMD {dd.vmd_um:.0f} um", delta_color="off")
        k4.metric("Fan angle", f"{dd.spray_angle_deg:.0f}°")
        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Predicted coverage", f"{pp.coverage_pct:.1f}%",
                  f"{pp.coverage_pct - c_req:+.1f} vs C_req", delta_color="normal")
        k6.metric("Predicted drift", f"{100 * pp.drift_frac:.1f}%")
        k7.metric("Deposit (uL/cm2)", f"{pp.deposition_ul_cm2:.2f}")
        k8.metric("Card shift (mm)", f"{pp.displacement_mm:.0f}")
        cvp = al.coverage_vs_pressure(fam, size, xn)
        ch1, ch2 = st.columns(2)
        ch1.altair_chart(charts.line_xy(cvp, "pressure_psi", "coverage_pct", "pressure (psi)",
                                        "coverage (%)", extra_rule=float(c_req), y_fmt=".1f"),
                         width="stretch")
        ch1.caption("Coverage vs pressure at these conditions (dashed: C_req).")
        dvw = al.drift_vs_wind(un, xn)
        ch2.altair_chart(charts.line_xy(dvw, "wind_ms", "drift_frac", "crosswind (m/s)",
                                        "drift fraction", color="#ffb84d"), width="stretch")
        ch2.caption("Drift vs crosswind at this setting.")
    g1, g2 = st.columns([3, 2], gap="medium")
    with g1:
        grid = al.coverage_grid(fam, size, xn)
        st.altair_chart(charts.heatmap(grid, "coverage_pct", "Coverage map: pressure x crosswind (%)",
                                       vtitle="coverage %"), width="stretch")
    with g2:
        pat = al.pattern_overlap(dd.spray_angle_deg, hgt)
        st.altair_chart(charts.pattern_chart(pat), width="stretch")
        u1, u2, u3 = st.columns(3)
        u1.metric("Uniformity coeff.", f"{pat['uniformity']:.3f}")
        u2.metric("Fan width at card", f"{100 * pat['fan_width_m']:.0f} cm")
        u3.metric("Peak intensity", f"{pat['peak']:.2f}")
        st.caption("Illustrative geometric overlap (raised-cosine fan, 20 in spacing); not simulator output.")

# ------------------------------------------------------------------ autoplay
if SS["playing"]:
    if SS["cursor"] < res.n - 1:
        time.sleep(0.2)
        SS["advance"] = True
    else:
        SS["playing"] = False
    st.rerun()
