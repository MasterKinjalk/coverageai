"""Spray lab tab: pick a nozzle and weather, watch the spray, read the results.

Usage from the main app (``app/`` must be on sys.path, as main.py already does):

    from spray_lab_view import render_spray_lab
    with tab_lab:
        render_spray_lab()

Every widget key starts with ``lab_``. The result is kept in ``st.session_state["lab_result"]``.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from field_run_view import embed_html
from coverageai import catalog
from coverageai import spray_lab as sl
from coverageai.app_logic import nozzle_pressure_bounds
from coverageai.config import spec
from coverageai.contracts import Setting
from coverageai.units import gpa_to_lha

_HTML_PATH = Path(__file__).resolve().parent / "static_spray_lab.html"
_BADGE = {"acceptable": "#3ddc97", "marginal": "#f5b942", "critical": "#ef5b5b"}
_SEED = 7

_CSS = """
<style>
.labkpi {background:#161b22; border:1px solid #2b3340; border-radius:10px; padding:10px 14px; height:100%;}
.labkpi .l {color:#8a94a6; font-size:0.74rem; text-transform:uppercase; letter-spacing:0.6px;}
.labkpi .v {color:#e6edf3; font-size:1.6rem; font-weight:650; line-height:1.2;}
.labkpi .s {color:#8a94a6; font-size:0.8rem;}
.labbadge {display:inline-block; padding:1px 9px; border-radius:999px; font-size:0.75rem;
           font-weight:650; color:#0e1117; margin-left:6px; vertical-align:middle;}
</style>
"""


def _kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(f'<div class="labkpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s">{sub}&nbsp;</div></div>', unsafe_allow_html=True)


def _pct(v: float) -> str:
    return "<1%" if 0 < v < 1 else f"{v:.0f}%"


def _size_label(fam: str, size: str) -> str:
    try:
        flow, _ = catalog.flow_and_class(fam, size, 40.0)
        return f"{size}  (about {flow:.2f} gal/min at 40 psi)"
    except ValueError:
        return size


# ------------------------------------------------------------------ controls

def _controls():
    r = spec()["ranges"]
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown("**Nozzle**")
        fams = catalog.families()
        fam = st.selectbox("Nozzle type", fams, index=fams.index("XR") if "XR" in fams else 0,
                           format_func=sl.family_label, key="lab_family")
        st.caption(sl.FAMILY_INFO.get(fam, {}).get("desc", ""))
        sizes = catalog.sizes(fam)
        size = st.selectbox("Nozzle size (bigger number = more flow)", sizes,
                            index=min(len(sizes) // 2, len(sizes) - 1),
                            format_func=lambda s: _size_label(fam, s), key=f"lab_size_{fam}")
        lo, hi = nozzle_pressure_bounds(fam, size)
        pressure = st.slider("Spray pressure (psi)", float(lo), float(hi),
                             float(min(max(40.0, lo), hi)), step=1.0,
                             key=f"lab_pressure_{fam}_{size}",
                             help="Higher pressure gives more flow and smaller droplets.")
    with c2:
        st.markdown("**Weather**")
        wind = st.slider("Wind speed (m/s)", float(r["wind_speed_ms"][0]), float(r["wind_speed_ms"][1]),
                         3.0, step=0.1, key="lab_wind",
                         help="1 m/s is about 2.2 mph. Spraying is not allowed above about 6.7 m/s (15 mph).")
        wdir = st.selectbox("Wind direction (relative to the sprayer)", list(sl.WIND_DIRECTIONS),
                            index=0, key="lab_wind_dir",
                            help="Only the sideways (crosswind) part of the wind carries droplets off "
                                 "the sprayed strip in this simulation.")
        temp = st.slider("Air temperature (°C)", float(r["temp_c"][0]), float(r["temp_c"][1]), 25.0,
                         step=0.5, key="lab_temp")
        rh = st.slider("Relative humidity (%)", float(r["rh_pct"][0]), float(r["rh_pct"][1]), 50.0,
                       step=1.0, key="lab_rh", help="Dry, hot air makes small droplets evaporate faster.")
    with c3:
        st.markdown("**Sprayer and goal**")
        speed = st.slider("Travel speed (km/h)", float(r["travel_speed_kmh"][0]),
                          float(r["travel_speed_kmh"][1]), 12.0, step=0.5, key="lab_speed",
                          help="Driving faster spreads the same flow over more ground (lower rate per acre).")
        height = st.slider("Boom height above the target (m)", float(r["boom_height_m"][0]),
                           float(r["boom_height_m"][1]), 0.50, step=0.01, key="lab_height",
                           help="Higher booms give droplets more time in the air to drift and evaporate.")
        target = st.slider("Target coverage (%)", 5.0, 60.0, float(spec()["c_req_pct"]), step=1.0,
                           key="lab_target",
                           help="Share of the water-sensitive paper card that should turn blue.")
    u = Setting(fam, size, float(pressure))
    x = sl.make_conditions(wind, sl.WIND_DIRECTIONS[wdir], temp, rh, speed, height)
    return u, x, float(target)


# ------------------------------------------------------------------ results

def _animation(res: sl.LabResult) -> None:
    payload = json.dumps(res.particles, separators=(",", ":")).replace("</", "<\\/")
    html = _HTML_PATH.read_text(encoding="utf8").replace("/*__PAYLOAD__*/null", payload, 1)
    embed_html(html, 575)


def _kpis(res: sl.LabResult) -> None:
    o, d = res.outcome, res.derived
    color = _BADGE[res.rating_key]
    cols = st.columns(5)
    _kpi(cols[0], "Card coverage", f"{o.coverage_pct:.0f}%"
         f'<span class="labbadge" style="background:{color}">{res.rating}</span>',
         f"target {res.target_pct:.0f}%")
    _kpi(cols[1], "Spray drift", _pct(res.drift_pct),
         f"blown off {_pct(100 * res.fate['blown'])} · evaporated {_pct(100 * res.fate['evaporated'])}")
    _kpi(cols[2], "Application rate", f"{d.app_rate_gpa:.1f} GPA",
         f"{gpa_to_lha(d.app_rate_gpa):.0f} L/ha · {d.flow_gpm:.2f} gal/min per nozzle")
    _kpi(cols[3], "Droplet size", res.droplet_class,
         f"half the spray volume is in droplets under {d.vmd_um:.0f} µm")
    _kpi(cols[4], "Spray deposit", f"{o.deposition_ul_cm2:.2f} µL/cm²",
         f"about {100 * o.deposition_ul_cm2:.0f} L/ha reaches the target")


_D_SCALE = alt.Scale(type="log", scheme="viridis", domain=list(sl.D_COLOR_RANGE))
_FATE3 = {"Deposited on target": "#3ddc97", "Drifts (displaced > threshold)": "#ef5b5b",
          "Evaporates before landing": "#9aa4b2"}
_SPEC_COLORS = {"Emitted": "#8a94a6", "Deposited on target": "#3ddc97",
                "Drifted (displaced > threshold)": "#ef5b5b", "Evaporated": "#c9d1d9"}
_RED, _BLUE, _AMBER = "#ef5b5b", "#4da3ff", "#f5b942"


def _style(chart, height: int = 300):
    """Dark, publication-style Altair config consistent with the app theme."""
    return (chart.properties(height=height)
            .configure(background="#0e1117", font="system-ui, -apple-system, Segoe UI, sans-serif")
            .configure_view(stroke="#2b3340")
            .configure_axis(labelColor="#c9d1d9", titleColor="#c9d1d9", gridColor="#1f2630",
                            domainColor="#5b6577", tickColor="#5b6577", labelFontSize=11,
                            titleFontSize=12, titleFontWeight="normal")
            .configure_legend(labelColor="#c9d1d9", titleColor="#c9d1d9", orient="top",
                              labelFontSize=11, titleFontSize=11, titleFontWeight="normal")
            .configure_text(color="#c9d1d9", fontSize=11))


def _show(chart, height: int = 300) -> None:
    st.altair_chart(_style(chart, height), width="stretch", theme=None)


def _vrule(value: float, color: str, label: str, dash=(5, 4), dy: int = 0, align: str = "left"):
    df = pd.DataFrame({"v": [value], "label": [label]})
    rule = alt.Chart(df).mark_rule(color=color, strokeDash=list(dash), strokeWidth=1.3).encode(x="v:Q")
    text = alt.Chart(df).mark_text(color=color, align=align, baseline="top", dx=4 if align == "left" else -4,
                                   dy=dy, fontSize=11).encode(x="v:Q", y=alt.value(4 + dy), text="label:N")
    return rule + text


# ---------------------------------------------------------------- (a) trajectories

def _fig_trajectories(res: sl.LabResult) -> None:
    F = res.figures
    lines, ends = F["trajectories"].copy(), F["trajectory_ends"].copy()
    U = res.derived.u_cross_ms
    sign = -1.0 if U < 0 else 1.0
    lines["xd"] = sign * lines["x_m"]
    ends["xd"] = sign * ends["x_m"]
    thr = float(spec()["physics"]["drift_threshold_m"])
    hi = max(1.6 * thr, 1.2)
    H = res.conditions.boom_height_m
    xs = alt.Scale(domain=[-0.05, hi], clamp=False)
    ys = alt.Scale(domain=[0, H * 1.05])
    inr = lines[(lines["xd"] <= hi) & (lines["xd"] >= -0.05)]
    lab = inr.sort_values("step").groupby("d_um").tail(1).copy()
    lab = lab[lab["xd"] > 0.12 * hi]          # label only curves that bend away from x = 0
    lab["label"] = lab["d_um"].map(lambda d: f"{d:g} µm")
    base = alt.Chart(lines).mark_line(clip=True, strokeWidth=2).encode(
        x=alt.X("xd:Q", scale=xs, title="Downwind displacement (m)"),
        y=alt.Y("h_m:Q", scale=ys, title="Height above target (m)"),
        color=alt.Color("d_um:Q", scale=_D_SCALE, title="Initial diameter (µm)",
                        legend=alt.Legend(orient="right", type="gradient", gradientLength=180,
                                          values=[50, 100, 200, 500, 1000])),
        detail="d_um:N", order="step:Q",
        tooltip=[alt.Tooltip("d_um:Q", title="Initial diameter (µm)"),
                 alt.Tooltip("t_s:Q", title="Time (s)", format=".3f"),
                 alt.Tooltip("xd:Q", title="Downwind (m)", format=".3f"),
                 alt.Tooltip("h_m:Q", title="Height (m)", format=".3f")])
    e_in = ends[(ends["xd"] <= hi) & (ends["xd"] >= -0.05)]
    land = alt.Chart(e_in[e_in["outcome"] == "lands"]).mark_point(filled=True, size=40, clip=True).encode(
        x=alt.X("xd:Q", scale=xs), y=alt.Y("h_m:Q", scale=ys), color=alt.Color("d_um:Q", scale=_D_SCALE, legend=None))
    evap = alt.Chart(e_in[e_in["outcome"] == "evaporates"]).mark_point(shape="cross", angle=45, size=90,
                                                                        strokeWidth=2, clip=True).encode(
        x=alt.X("xd:Q", scale=xs), y=alt.Y("h_m:Q", scale=ys), color=alt.value("#e6edf3"))
    text = alt.Chart(lab).mark_text(align="right", dx=-4, dy=-7, fontSize=11, clip=True).encode(
        x=alt.X("xd:Q", scale=xs), y=alt.Y("h_m:Q", scale=ys), text="label:N", color=alt.value("#c9d1d9"))
    _show(base + land + evap + text + _vrule(thr, _RED, f"drift threshold {thr:g} m"), 320)
    out = ends[ends["xd"] > hi]
    gone = ", ".join(f"{d:g} µm ({'lands' if o == 'lands' else 'evaporates'} at {x:.1f} m)"
                     for d, o, x in zip(out["d_um"], out["outcome"], out["xd"]))
    st.caption(
        "Path of single droplets released straight down from one nozzle (centre of the fan), from the "
        "physics model. Each line is one initial droplet size; a dot marks where it lands, a × where it "
        "has evaporated completely. Droplets leave the nozzle fast, slow down within a few centimetres, "
        "then settle at their own fall speed while the crosswind carries them sideways; the farther a line "
        "bends right, the more that size drifts. Sizes shown: "
        + ", ".join(f"{d:g}" for d in sl.TRAJ_DIAMETERS_UM) + " µm (colour scale on the right; "
        "nearly vertical lines are the large sizes). Axes in metres."
        + (f" Off the right edge: {gone}." if gone else "")
        + (" No crosswind: all paths are vertical." if abs(U) < 0.05 else ""))


# ---------------------------------------------------------------- (b) drift potential

def _fig_potential(res: sl.LabResult) -> None:
    F = res.figures
    df, m = F["potential"], F["potential_meta"]
    thr = m["thr_m"]
    dx = alt.X("d_um:Q", scale=alt.Scale(type="log", domain=[20, 1000], nice=False),
               title="Initial droplet diameter (µm, log scale)",
               axis=alt.Axis(values=[20, 50, 100, 200, 500, 1000]))
    fate_col = alt.Color("fate:N", title=None, scale=alt.Scale(domain=list(_FATE3), range=list(_FATE3.values())))
    marks = []
    if m["d_crit_um"] is not None:
        marks.append(_vrule(m["d_crit_um"], _AMBER, f"lost below ≈ {m['d_crit_um']:.0f} µm", dy=16))
    marks.append(_vrule(m["vmd_um"], _BLUE, f"VMD {m['vmd_um']:.0f} µm", dash=(1, 0)))
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        d1 = df.dropna(subset=["x_plot_m"])
        top = max(10.0, float(d1["x_plot_m"].max()) * 1.5) if len(d1) else 10.0
        ys = alt.Scale(type="log", domain=[1e-3, top])
        line = alt.Chart(d1).mark_line(color="#5b6577", strokeWidth=1).encode(x=dx, y=alt.Y("x_plot_m:Q", scale=ys))
        pts = alt.Chart(d1).mark_circle(size=22, opacity=0.95).encode(
            x=dx, y=alt.Y("x_plot_m:Q", scale=ys, title="Wind displacement at landing (m, log scale)"),
            color=fate_col,
            tooltip=[alt.Tooltip("d_um:Q", title="Diameter (µm)", format=".0f"),
                     alt.Tooltip("x_land_m:Q", title="Displacement (m)", format=".3f"),
                     alt.Tooltip("fate:N", title="Outcome")])
        thr_rule = alt.Chart(pd.DataFrame({"v": [thr]})).mark_rule(color=_RED, strokeDash=[5, 4]).encode(y="v:Q")
        thr_txt = alt.Chart(pd.DataFrame({"v": [thr], "t": [f"drift threshold {thr:g} m"]})).mark_text(
            color=_RED, align="left", baseline="bottom", dx=4, dy=-3, fontSize=11).encode(
            y="v:Q", x=alt.value(0), text="t:N")
        chart = line + pts + thr_rule + thr_txt
        for mk in marks:
            chart = chart + mk
        _show(chart, 300)
    with c2:
        ys2 = alt.Scale(type="log")
        line2 = alt.Chart(df).mark_line(color="#5b6577", strokeWidth=1).encode(x=dx, y=alt.Y("t_s:Q", scale=ys2))
        pts2 = alt.Chart(df).mark_circle(size=22, opacity=0.95).encode(
            x=dx, y=alt.Y("t_s:Q", scale=ys2, title="Time in the air (s, log scale)"), color=fate_col,
            tooltip=[alt.Tooltip("d_um:Q", title="Diameter (µm)", format=".0f"),
                     alt.Tooltip("t_s:Q", title="Time (s)", format=".3f"),
                     alt.Tooltip("fate:N", title="Outcome")])
        chart2 = line2 + pts2
        for mk in marks:
            chart2 = chart2 + mk
        _show(chart2, 300)
    crit = (f"Droplets smaller than about **{m['d_crit_um']:.0f} µm** drift or evaporate under these conditions; "
            if m["d_crit_um"] is not None else "No droplet size between 20 and 1000 µm is lost under these conditions; ")
    st.caption(
        "Drift potential by droplet size (one droplet per size, from the physics model). Left: how far "
        "the crosswind moves a droplet before it lands (log scale; values below 1 mm are drawn at 1 mm). "
        "Right: how long it stays in the air (fall time, or time until it has evaporated). Small droplets "
        "fall slowly, so they spend longer in the wind and evaporate more. Points are coloured by outcome. "
        + crit + f"half of this nozzle's spray volume is in droplets smaller than the VMD "
        f"({m['vmd_um']:.0f} µm, blue line).")


# ---------------------------------------------------------------- (c) spectrum

def _fig_spectrum(res: sl.LabResult) -> None:
    F = res.figures
    per, cum, m = F["spectrum"], F["cumulative"], F["spectrum_meta"]
    lo, hi = float(cum["d_um"].min()), float(cum["d_um"].max())
    dx = alt.X("d_um:Q", scale=alt.Scale(type="log", domain=[max(10.0, lo), hi], nice=False),
               title="Initial droplet diameter (µm, log scale)")
    colors = alt.Color("series:N", title=None, scale=alt.Scale(domain=list(_SPEC_COLORS),
                                                                 range=list(_SPEC_COLORS.values())))
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        emitted = per[per["series"] == "Emitted"]
        area = alt.Chart(emitted).mark_area(color="#8a94a6", opacity=0.18, interpolate="monotone").encode(
            x=dx, y=alt.Y("pct:Q", title="Share of emitted volume per size class (%)"))
        lines = alt.Chart(per).mark_line(strokeWidth=2, interpolate="monotone").encode(
            x=dx, y="pct:Q", color=colors,
            strokeDash=alt.condition(alt.datum.series == "Emitted", alt.value([4, 3]), alt.value([1, 0])),
            tooltip=[alt.Tooltip("d_um:Q", title="Diameter (µm)", format=".0f"),
                     alt.Tooltip("series:N", title="Series"),
                     alt.Tooltip("pct:Q", title="% of emitted volume", format=".2f")])
        _show(area + lines, 300)
    with c2:
        curve = alt.Chart(cum).mark_line(color="#e6edf3", strokeWidth=2).encode(
            x=dx, y=alt.Y("cum_pct:Q", title="Cumulative volume (%)", scale=alt.Scale(domain=[0, 100])),
            tooltip=[alt.Tooltip("d_um:Q", title="Diameter (µm)", format=".0f"),
                     alt.Tooltip("cum_pct:Q", title="Volume in smaller droplets (%)", format=".1f")])
        pts = pd.DataFrame({"d_um": [m["dv10_um"], m["dv50_um"], m["dv90_um"]], "cum_pct": [10.0, 50.0, 90.0],
                            "label": [f"DV10 {m['dv10_um']:.0f} µm", f"DV50 (VMD) {m['dv50_um']:.0f} µm",
                                      f"DV90 {m['dv90_um']:.0f} µm"]})
        pts["zero"] = 0.0
        guide = alt.Chart(pts).mark_rule(color=_BLUE, strokeDash=[3, 3]).encode(x="d_um:Q", y="zero:Q", y2="cum_pct:Q")
        dots = alt.Chart(pts).mark_point(color=_BLUE, filled=True, size=50).encode(x="d_um:Q", y="cum_pct:Q")
        text = alt.Chart(pts).mark_text(color=_BLUE, align="right", dx=-6, dy=-6, fontSize=11).encode(
            x="d_um:Q", y="cum_pct:Q", text="label:N")
        _show(curve + guide + dots + text, 300)
    st.caption(
        "Droplet size spectrum of this nozzle at this pressure (40 size classes of equal width on the "
        "log axis, by size when leaving the nozzle). Left: how the spray volume is spread over droplet sizes "
        "(dashed grey = everything emitted) and what happens to each size (green deposited on target, red "
        "drifted, light grey evaporated); where the green line sits below the grey one, that size is being "
        "lost. Right: cumulative share of the volume in droplets smaller than a given size; DV10, DV50 and "
        "DV90 are the sizes below which 10 %, 50 % and 90 % of the volume lies (DV50 = VMD, the volume "
        "median diameter used to name droplet-size classes).")


# ---------------------------------------------------------------- (d) deposition profile

def _fig_profile(res: sl.LabResult) -> None:
    F = res.figures
    df, m = F["profile"], F["profile_meta"]
    xs = alt.Scale(domain=[float(df["x_m"].min()), float(df["x_m"].max())], nice=False)
    colors = alt.Color("component:N", title=None,
                       scale=alt.Scale(domain=["Deposited on target", "Drifted (displaced > threshold)"],
                                       range=["#3ddc97", _RED]))
    sw = pd.DataFrame({"a": [-m["swath_half_m"]], "b": [m["swath_half_m"]]})
    band = alt.Chart(sw).mark_rect(color="#3ddc97", opacity=0.07).encode(x=alt.X("a:Q", scale=xs), x2="b:Q")
    area = alt.Chart(df).mark_area(opacity=0.85, interpolate="step").encode(
        x=alt.X("x_m:Q", scale=xs, title="Lateral position from boom centre (m)"),
        y=alt.Y("ul_cm2:Q", stack="zero", title="Deposit (µL/cm²)"), color=colors,
        order=alt.Order("component:N", sort="ascending"),
        tooltip=[alt.Tooltip("x_m:Q", title="Position (m)", format=".2f"),
                 alt.Tooltip("component:N", title="Part"),
                 alt.Tooltip("ul_cm2:Q", title="µL/cm²", format=".3f")])
    applied = pd.DataFrame({"v": [m["applied_ul_cm2"]], "t": [f"applied rate {m['applied_ul_cm2']:.2f} µL/cm²"]})
    app_rule = alt.Chart(applied).mark_rule(color=_BLUE, strokeDash=[4, 3]).encode(y="v:Q")
    app_txt = alt.Chart(applied).mark_text(color=_BLUE, align="left", baseline="bottom", dx=4, dy=-3,
                                           fontSize=11).encode(y="v:Q", x=alt.value(0), text="t:N")
    thr = pd.DataFrame({"v": [-m["thr_line_m"], m["thr_line_m"]]})
    thr_rule = alt.Chart(thr).mark_rule(color=_RED, strokeDash=[5, 4]).encode(x=alt.X("v:Q", scale=xs))
    _show(band + area + app_rule + app_txt + thr_rule, 300)
    cv = m["cv_pct"]
    cv_txt = (f" Uniformity in the middle of the boom (between the 2nd and 4th nozzle): coefficient of "
              f"variation **CV = {cv:.1f} %** (a common guideline is that below about 10 % counts as uniform)."
              if cv is not None else "")
    st.caption(
        "Where the liquid lands across the boom (sum of all nozzles, per unit of ground area). Each nozzle "
        f"spreads its liquid over a fan about {m['fan_width_m']:.2f} m wide (an illustrative bell-shaped "
        "pattern, not a measured patternator test), shifted downwind by how far the wind moves each droplet "
        "size. Green = liquid deposited on target, red = liquid displaced more than the drift threshold; "
        "liquid that evaporates is missing. Shaded band = target swath, red dashed lines = drift threshold "
        "beyond the swath, blue dashed line = applied rate (what would land with no losses)." + cv_txt)


# ---------------------------------------------------------------- (e) sensitivity

def _fig_sensitivity(res: sl.LabResult) -> None:
    F = res.figures
    df, m = F["sensitivity"], F["sensitivity_meta"]
    names = [m["nozzle"]] + ([m["alt"]] if m["alt"] else [])
    colors = alt.Color("nozzle:N", title=None, scale=alt.Scale(domain=names, range=["#4da3ff", "#f5b942"][:len(names)]))
    lines = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X("wind_ms:Q", title="Wind speed (m/s)"),
        y=alt.Y("drift_pct:Q", title="Spray drift (% of spray volume)"), color=colors,
        tooltip=[alt.Tooltip("nozzle:N", title="Nozzle"), alt.Tooltip("wind_ms:Q", title="Wind (m/s)", format=".1f"),
                 alt.Tooltip("drift_pct:Q", title="Spray drift (%)", format=".1f")])
    cur = pd.DataFrame({"wind_ms": [m["wind_ms"]], "drift_pct": [m["drift_pct"]],
                        "t": [f"now: {m['drift_pct']:.1f} % at {m['wind_ms']:.1f} m/s"]})
    dot = alt.Chart(cur).mark_point(filled=True, size=110, color="#e6edf3", shape="diamond").encode(
        x="wind_ms:Q", y="drift_pct:Q")
    txt = alt.Chart(cur).mark_text(color="#e6edf3", align="left", dx=8, dy=-8, fontSize=11).encode(
        x="wind_ms:Q", y="drift_pct:Q", text="t:N")
    lim = _vrule(m["wind_limit_ms"], _RED, f"{m['wind_limit_ms']:g} m/s (15 mph) limit", align="right")
    _show(lines + dot + txt + lim, 300)
    alt_txt = (f" The second line is {m['alt']}, a nozzle with about the same flow, for contrast."
               if m["alt"] else "")
    st.caption(
        "How spray drift grows with wind speed for this nozzle and pressure, keeping the other conditions "
        "and the wind direction you chose (only the sideways part of the wind causes drift here). The "
        "diamond marks the current wind. Spraying is not allowed above about 6.7 m/s (15 mph)." + alt_txt)


# ---------------------------------------------------------------- results layout

def _results(res: sl.LabResult) -> None:
    st.markdown("#### Results")
    _kpis(res)
    st.caption("Card coverage is the share of a water-sensitive paper card at the target that turns "
               "blue. Spray drift is the share of the sprayed liquid that never reaches the target: "
               "carried more than "
               f"{100 * spec()['physics']['drift_threshold_m']:.0f} cm sideways by the wind, or "
               "evaporated in the air.")
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown("**Water-sensitive paper card at the target**")
        if res.card_image is not None:
            st.image(res.card_image, width="stretch")
        st.caption("A simulated 76 × 26 mm water-sensitive paper card placed in the crop under this "
                   "spray: it is yellow, and turns blue wherever a droplet lands. "
                   f"About {res.outcome.stains_per_cm2:.0f} droplet stains per cm²; "
                   f"{res.outcome.coverage_pct:.0f}% of the card is covered.")
    with right:
        st.markdown("**Spray drift vs wind speed**")
        _fig_sensitivity(res)

    st.markdown("#### The physics behind the result")
    st.caption("All figures below come from the same physics model as the numbers above "
               "(no gusts, nozzle wear or canopy effects).")
    a, d = st.columns(2, gap="large")
    with a:
        st.markdown("**Droplet trajectories by size**")
        _fig_trajectories(res)
    with d:
        st.markdown("**Deposit across the boom**")
        _fig_profile(res)
    st.markdown("**Drift potential: displacement and time in the air vs droplet size**")
    _fig_potential(res)
    st.markdown("**Droplet size spectrum and where each size ends up**")
    _fig_spectrum(res)


def _explainer() -> None:
    with st.expander("How this simulation works (and its limits)"):
        st.markdown(
            "- **Droplet sizes** come from the nozzle chart: the nozzle type, size and pressure set "
            "the droplet size class, and the spray is split into 40 droplet sizes around it.\n"
            "- **Each droplet** leaves the nozzle fast (faster at higher pressure), is slowed by the "
            "air until it falls at its natural settling speed, and is pushed sideways by the "
            "crosswind. Small droplets settle slowly, so they spend longer in the wind.\n"
            "- **Evaporation**: droplets shrink as they fall, faster in hot, dry air. A droplet that "
            "shrinks to nothing before landing is lost.\n"
            "- **Drift**: a droplet that lands more than "
            f"{100 * spec()['physics']['drift_threshold_m']:.0f} cm sideways from where it was aimed "
            "counts as drift.\n"
            "- **The animation** shows a sample of about a thousand droplets. Large droplets are "
            "shown more often than their real share by count (a real spray is mostly tiny "
            "droplets), but each one is weighted by the liquid it carries, so the live percentages "
            "match the results. The fan shape is geometric (droplets spread over the fan width with "
            "a bell-shaped pattern); the drift decision depends only on how far the wind moves a "
            "droplet. The height axis of the animation is stretched by the factor shown on it.\n"
            "- This is a simplified, physics-based simulation with no gusts, nozzle wear or crop "
            "canopy effects. It shows trends; it is not field data and not a validated drift model.")


# ------------------------------------------------------------------ entry point

def render_spray_lab() -> None:
    """Render the Spray lab tab content."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown("### Spray lab")
    st.markdown("Choose a nozzle and the weather, then press **Start simulation** to watch the spray "
                "leave the nozzles, fall, evaporate and drift, and see how much reaches the target.")
    st.caption("Physics-based simulation, not field data.")

    u, x, target = _controls()
    inputs = (u, x, target)
    if st.button("Start simulation", type="primary", key="lab_start"):
        with st.spinner("Simulating the spray..."):
            st.session_state["lab_result"] = sl.run_lab(u, x, seed=_SEED, target_pct=target)
            st.session_state["lab_inputs"] = inputs

    res: sl.LabResult | None = st.session_state.get("lab_result")
    if res is None:
        st.info("Press **Start simulation** to run the spray.")
        _explainer()
        return
    if st.session_state.get("lab_inputs") != inputs:
        st.warning("You changed the settings. The animation and results below are for the previous "
                   "settings; press **Start simulation** to update them.")
    s, c = res.setting, res.conditions
    st.markdown(
        f"Showing **{sl.family_label(s.family)}**, size {s.size} at {s.pressure_psi:g} psi · wind "
        f"{c.wind_speed_ms:.1f} m/s · {c.temp_c:g} °C, {c.rh_pct:.0f}% humidity · boom "
        f"{c.boom_height_m:.2f} m · {c.travel_speed_kmh:g} km/h")
    _animation(res)
    st.caption(
        "Simulated droplets seen from behind the sprayer, on true axes in metres (the height axis is "
        "stretched by the stated factor so the fall is visible). Each point is a droplet coloured by its "
        "diameter when it leaves the nozzle (colour bar, log scale); points shrink as they evaporate and "
        "leave faint traces of their path. Short ticks on the ground line are droplets deposited on "
        "target, hollow circles are droplets the wind displaced by more than the drift threshold, and × "
        "marks where a droplet evaporated completely. Use the time slider to step through the spray; the "
        "bar below tallies the released spray volume by outcome. About a thousand droplets are shown, "
        "with large droplets over-represented by count but weighted by volume in the tally.")
    _results(res)
    _explainer()
