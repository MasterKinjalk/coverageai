"""Nozzle lab: one nozzle, one set of weather, one set of numbers.

Usage from the main app (``app/`` must be on sys.path, as main.py already does):

    from spray_lab_view import lab_controls, render_nozzle_lab
    with st.sidebar:
        u, x, target = lab_controls()
    render_nozzle_lab(u, x, target)

Every widget key starts with ``lab_``. The simulation takes about 0.2 s, so it runs on every
change and is cached on the inputs. The headline numbers, the curves and the animation tally
all come from the same noise-free simulator (``spray_lab.nominal_simulator``).
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import charts
import ui
from field_run_view import embed_html
from coverageai import catalog
from coverageai import spray_lab as sl
from coverageai.app_logic import (coverage_grid, coverage_vs_pressure, drift_vs_wind,
                                  nozzle_pressure_bounds, pattern_overlap)
from coverageai.config import spec
from coverageai.contracts import Conditions, Setting
from coverageai.units import gpa_to_lha

_HTML_PATH = Path(__file__).resolve().parent / "static_spray_lab.html"
_SEED = 7          # places droplets on the card and in the animation; it does not change any number


def _size_label(fam: str, size: str) -> str:
    try:
        flow, _ = catalog.flow_and_class(fam, size, 40.0)
        return f"{size}  (about {flow:.2f} gal/min at 40 psi)"
    except ValueError:
        return size


# ------------------------------------------------------------------ controls

def _default(key: str, value) -> str:
    """Seed a widget's value once and return its key. The app re-assigns every ``lab_`` key on each
    run so the lab's inputs survive a visit to another view; a widget whose key is set that way
    must not also pass a default value, so the default goes here."""
    st.session_state.setdefault(key, value)
    return key


def lab_controls() -> tuple[Setting, Conditions, float]:
    """Render the lab's input widgets into the current container (main.py calls this inside
    ``with st.sidebar:``) and return (setting, conditions, target coverage %)."""
    r = spec()["ranges"]
    ui.group("Nozzle")
    fams = catalog.families()
    fam = st.selectbox("Nozzle type", fams, key=_default("lab_family", "XR" if "XR" in fams else fams[0]),
                       format_func=sl.family_label,
                       help=" ".join(f"{sl.family_label(f)}: {sl.FAMILY_INFO.get(f, {}).get('desc', '')}"
                                     for f in fams))
    sizes = catalog.sizes(fam)
    size = st.selectbox("Nozzle size", sizes, format_func=lambda v: _size_label(fam, v),
                        key=_default(f"lab_size_{fam}", "04" if "04" in sizes else sizes[len(sizes) // 2]),
                        help="A bigger number means more flow.")
    lo, hi = nozzle_pressure_bounds(fam, size)
    pressure = st.slider("Pressure (psi)", float(lo), float(hi), step=1.0, format="%.0f",
                         key=_default(f"lab_pressure_{fam}_{size}", float(min(max(40.0, lo), hi))),
                         help="Higher pressure gives more flow and smaller droplets.")

    ui.group("Weather")
    wind = st.slider("Wind speed (m/s)", float(r["wind_speed_ms"][0]), float(r["wind_speed_ms"][1]),
                     step=0.1, format="%.1f", key=_default("lab_wind", 3.0),
                     help="1 m/s is about 2.2 mph. Spraying is not allowed above about 6.7 m/s (15 mph).")
    wdir = st.selectbox("Wind direction", list(sl.WIND_DIRECTIONS),
                        key=_default("lab_wind_dir", next(iter(sl.WIND_DIRECTIONS))),
                        help="Relative to the sprayer. Only the sideways (crosswind) part of the wind "
                             "carries droplets off the sprayed strip in this simulation.")
    temp = st.slider("Air temperature (°C)", float(r["temp_c"][0]), float(r["temp_c"][1]),
                     step=0.5, format="%.1f", key=_default("lab_temp", 25.0))
    rh = st.slider("Relative humidity (%)", float(r["rh_pct"][0]), float(r["rh_pct"][1]),
                   step=1.0, format="%.0f", key=_default("lab_rh", 55.0),
                   help="Dry, hot air makes small droplets evaporate faster.")

    ui.group("Sprayer and goal")
    speed = st.slider("Travel speed (km/h)", float(r["travel_speed_kmh"][0]),
                      float(r["travel_speed_kmh"][1]), step=0.5, format="%.1f",
                      key=_default("lab_speed", 16.0),
                      help="Driving faster spreads the same flow over more ground (lower rate per acre).")
    height = st.slider("Boom height (m)", float(r["boom_height_m"][0]), float(r["boom_height_m"][1]),
                       step=0.01, format="%.2f", key=_default("lab_height", 0.50),
                       help="Height above the target. Higher booms give droplets more time in the air "
                            "to drift and evaporate.")
    target = st.slider("Target coverage (%)", 5.0, 60.0, step=1.0, format="%.0f",
                       key=_default("lab_target", float(spec()["c_req_pct"])),
                       help="Share of the water-sensitive paper card that should turn blue. It does not "
                            "change the Good / Borderline / Poor class, which uses fixed limits.")
    u = Setting(fam, size, float(pressure))
    x = sl.make_conditions(wind, sl.WIND_DIRECTIONS[wdir], temp, rh, speed, height)
    return u, x, float(target)


# ------------------------------------------------------------------ cached run

@st.cache_data(show_spinner=False, max_entries=24)
def _compute(fam: str, size: str, psi: float, wind: float, rel_deg: float, temp: float, rh: float,
             speed: float, height: float, target: float) -> dict:
    """Everything the view needs for one input set, keyed on plain numbers."""
    u = Setting(fam, size, psi)
    x = sl.make_conditions(wind, rel_deg, temp, rh, speed, height)
    return {"res": sl.run_lab(u, x, seed=_SEED, target_pct=target),
            "cvp": coverage_vs_pressure(fam, size, x),
            "dvw": drift_vs_wind(u, x),
            "grid": coverage_grid(fam, size, x)}


def _run(u: Setting, x: Conditions, target: float) -> dict:
    return _compute(u.family, u.size, float(u.pressure_psi), float(x.wind_speed_ms),
                    float(x.wind_dir_deg), float(x.temp_c), float(x.rh_pct),
                    float(x.travel_speed_kmh), float(x.boom_height_m), float(target))


def _wind_name(deg: float) -> str:
    for name, d in sl.WIND_DIRECTIONS.items():
        if abs(d - deg) < 1e-6:
            return name.split(" (")[0].lower()
    return f"{deg:.0f}° from the travel direction"


# ------------------------------------------------------------------ result blocks

def _chips(res: sl.LabResult) -> None:
    s, c = res.setting, res.conditions
    ui.chips([("Nozzle", f"{sl.family_label(s.family)}, size {s.size}"),
              ("Pressure", f"{s.pressure_psi:.0f} psi"),
              ("Wind", f"{c.wind_speed_ms:.1f} m/s, {_wind_name(c.wind_dir_deg)}"),
              ("Air", f"{c.temp_c:.1f} °C, {c.rh_pct:.0f} % humidity"),
              ("Boom", f"{c.boom_height_m:.2f} m"),
              ("Speed", f"{c.travel_speed_kmh:.1f} km/h")])


def _kpis(res: sl.LabResult) -> None:
    o, d = res.outcome, res.derived
    met = "met" if res.target_met else "not met"
    ui.kpi_row([
        ("Card coverage", f"{o.coverage_pct:.1f}% " + ui.badge(res.rating, res.rating_key),
         f"target {res.target_pct:.0f}% {met}"),
        ("Spray drift", f"{res.drift_pct:.1f}%",
         f"{100 * res.fate['blown']:.1f}% blown, {100 * res.fate['evaporated']:.1f}% evaporated"),
        ("Spray volume", f"{d.app_rate_gpa:.1f}", "gal/acre",
         f"{gpa_to_lha(d.app_rate_gpa):.0f} L/ha · {d.flow_gpm:.2f} gal/min"),
        ("Droplet size", res.droplet_class, f"median {d.vmd_um:.0f} µm"),
        ("Spray deposit", f"{o.deposition_ul_cm2:.2f}", "µL/cm²",
         f"about {100 * o.deposition_ul_cm2:.0f} L/ha on target"),
    ])


def _animation(res: sl.LabResult) -> None:
    payload = json.dumps(res.particles, separators=(",", ":")).replace("</", "<\\/")
    html = _HTML_PATH.read_text(encoding="utf8").replace("/*__PAYLOAD__*/null", payload, 1)
    embed_html(html, 575)


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
        "Each line is one droplet size. Small droplets bend downwind; × marks full evaporation."
        + (f" Off the right edge: {gone}." if gone else "")
        + (" No crosswind, so all paths are vertical." if abs(U) < 0.05 else ""))


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
    crit = (f"Droplets below about {m['d_crit_um']:.0f} µm drift or evaporate. "
            if m["d_crit_um"] is not None else "No droplet size from 20 to 1000 µm is lost. ")
    st.caption(crit + f"The blue line is the VMD ({m['vmd_um']:.0f} µm).")


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
    st.caption("DV10, DV50 and DV90 are the sizes below which 10, 50 and 90% of the volume lies.")


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
    cv_txt = (f" Evenness in the middle of the boom: CV = {cv:.1f}%."
              if cv is not None else "")
    st.caption("Illustrative fan pattern, not a patternator test." + cv_txt)


# ---------------------------------------------------------------- operating-point curves

def _curve(df: pd.DataFrame, x: str, y: str, xtitle: str, ytitle: str, color: str,
           now: tuple[float, float], now_label: str, rule: tuple[float, str] | None = None):
    """Line with the current operating point marked by a dot, a label and a vertical guide."""
    line = alt.Chart(df).mark_line(color=color, strokeWidth=2.5).encode(
        x=alt.X(f"{x}:Q", title=xtitle), y=alt.Y(f"{y}:Q", title=ytitle),
        tooltip=[alt.Tooltip(f"{x}:Q", title=xtitle, format=".1f"),
                 alt.Tooltip(f"{y}:Q", title=ytitle, format=".1f")])
    cur = pd.DataFrame({"x": [now[0]], "y": [now[1]], "t": [now_label]})
    guide = alt.Chart(cur).mark_rule(color="#8a94a6", strokeDash=[2, 3], opacity=0.5).encode(x="x:Q")
    dot = alt.Chart(cur).mark_point(filled=True, size=110, color="#e6edf3", shape="diamond").encode(
        x="x:Q", y="y:Q")
    txt = alt.Chart(cur).mark_text(color="#e6edf3", align="left", baseline="bottom", dx=9, dy=-14,
                                   fontSize=11).encode(x="x:Q", y="y:Q", text="t:N")
    ch = line + guide + dot + txt
    if rule is not None:
        r = pd.DataFrame({"y": [rule[0]], "t": [rule[1]]})
        # the target is a white dotted line in every view (replay panel B, charts.py)
        ch = ch + alt.Chart(r).mark_rule(color="#e6edf3", strokeDash=[2, 3]).encode(y="y:Q")
        ch = ch + alt.Chart(r).mark_text(color="#e6edf3", align="left", baseline="bottom", dx=4,
                                         dy=-3, fontSize=11).encode(y="y:Q", x=alt.value(0), text="t:N")
    return ch


def _tab_pressure_wind(run: dict, res: sl.LabResult) -> None:
    o = res.outcome
    s, c = res.setting, res.conditions
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("**Coverage vs pressure**")
        ch = _curve(run["cvp"], "pressure_psi", "coverage_pct", "Pressure (psi)",
                    "Coverage (% of card area)", charts.C_PRED,
                    (s.pressure_psi, o.coverage_pct), f"{o.coverage_pct:.1f}% at {s.pressure_psi:.0f} psi",
                    rule=(res.target_pct, f"target {res.target_pct:.0f}%"))
        _show(ch, 260)
    with c2:
        st.markdown("**Drift vs crosswind**")
        dv = run["dvw"].assign(drift_pct=100.0 * run["dvw"]["drift_frac"])
        u_cross = abs(float(res.derived.u_cross_ms))
        ch = _curve(dv, "wind_ms", "drift_pct", "Crosswind (m/s)", "Spray drift (% of spray)",
                    "#ffb84d", (u_cross, res.drift_pct), f"{res.drift_pct:.1f}% at {u_cross:.1f} m/s")
        _show(ch, 260)
    g1, g2 = st.columns([3, 2], gap="medium")
    with g1:
        st.altair_chart(charts.heatmap(run["grid"], "coverage_pct", "Coverage map: pressure vs crosswind",
                                       vtitle="Coverage (%)"), width="stretch")
    with g2:
        pat = pattern_overlap(res.derived.spray_angle_deg, c.boom_height_m)
        st.altair_chart(charts.pattern_chart(pat), width="stretch")
        st.caption(f"Evenness {pat['uniformity']:.2f} · fan width at the card "
                   f"{100 * pat['fan_width_m']:.0f} cm · peak intensity {pat['peak']:.2f}. "
                   "Nozzles 20 in apart. An illustration, not simulator output.")


def _tab_droplets(res: sl.LabResult) -> None:
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown("**Water-sensitive paper card at the target**")
        if res.card_image is not None:
            st.image(res.card_image, width="stretch")
        st.caption(f"About {res.outcome.stains_per_cm2:.0f} droplet stains per cm². "
                   f"{res.outcome.coverage_pct:.1f}% of the card is covered.")
    with right:
        st.markdown("**Droplet trajectories by size**")
        _fig_trajectories(res)
    st.markdown("**Deposit across the boom**")
    _fig_profile(res)
    st.markdown("**Droplet size spectrum and where each size ends up**")
    _fig_spectrum(res)
    st.markdown("**Drift potential: displacement and time in the air vs droplet size**")
    _fig_potential(res)


def _explainer() -> None:
    thr_cm = 100 * spec()["physics"]["drift_threshold_m"]
    with st.expander("How this simulation works (and its limits)"):
        st.markdown(
            "- **Droplet sizes** come from the nozzle chart: the nozzle type, size and pressure set "
            "the droplet size class, and the spray is split into 40 droplet sizes around it.\n"
            "- **Each droplet** leaves the nozzle fast (faster at higher pressure), is slowed by the "
            "air until it falls at its natural settling speed, and is pushed sideways by the "
            "crosswind. Small droplets settle slowly, so they spend longer in the wind.\n"
            "- **Evaporation**: droplets shrink as they fall, faster in hot, dry air. A droplet that "
            "shrinks to nothing before landing is lost.\n"
            f"- **Drift** is the share of the spray that never reaches the target: droplets that land "
            f"more than {thr_cm:.0f} cm sideways from where they were aimed, plus droplets that "
            "evaporate.\n"
            "- **Card coverage** is the share of a simulated 76 × 26 mm water-sensitive paper card "
            f"that turns blue. Good is {spec()['coverage_classes']['acceptable_min']:g}% or more and "
            f"Borderline is {spec()['coverage_classes']['marginal_min']:g}% or more. These fixed limits are "
            "the same ones the card reader and the field run use. The target coverage is a separate "
            "goal and is reported as met or not met.\n"
            "- **One number per quantity.** The numbers use the average droplet spread on paper, with "
            "no random draw, so the coverage in the tiles, on the pressure curve and in the animation "
            "tally is the same number. Nothing here has gusts, nozzle wear or crop canopy; those "
            "appear only in the field run.\n"
            "- **The animation** shows a sample of about a thousand droplets seen from behind the "
            "sprayer, on true axes in metres (the height axis is stretched by the factor shown). Each "
            "point is coloured by its diameter when it leaves the nozzle and shrinks as it "
            "evaporates. Short ticks on the ground are deposited droplets, hollow circles are "
            "droplets displaced by more than the drift threshold, and × marks full evaporation. Large "
            "droplets are shown more often than their real share by count, but each is weighted by "
            "its liquid volume, so the tally matches the results. The fan shape is geometric; drift "
            "depends only on how far the wind moves a droplet.\n"
            "- **Charts.** The pressure and wind curves mark the current setting with a diamond. "
            "Trajectories follow single droplets released straight down from one nozzle. In the drift "
            "potential chart, small droplets fall slowly, spend longer in the wind and evaporate more. "
            "In the spectrum, DV10, DV50 and DV90 are the sizes below which 10, 50 and 90% of the "
            "volume lies; DV50 is the volume median diameter (VMD). The deposit profile uses an "
            "illustrative bell-shaped fan, not a measured patternator test, and CV is the "
            "coefficient of variation of the deposit in the middle of the boom (below about 10% "
            "counts as uniform).\n"
            "- This is a simplified, physics-based simulation. It shows trends. It is not field data "
            "and not a validated drift model.")


# ------------------------------------------------------------------ entry point

def render_nozzle_lab(u: Setting, x: Conditions, target: float) -> None:
    """Render the results for these inputs into the current container (the main area)."""
    run = _run(u, x, target)
    res: sl.LabResult = run["res"]
    _chips(res)
    _kpis(res)
    t_anim, t_curves, t_drops = st.tabs(["Spray animation", "Pressure and wind", "Droplets and card"])
    with t_anim:
        _animation(res)
    with t_curves:
        _tab_pressure_wind(run, res)
    with t_drops:
        _tab_droplets(res)
    _explainer()
