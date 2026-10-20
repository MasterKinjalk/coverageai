"""Altair chart builders for the demo (dark theme, class colors shared with app_logic)."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from coverageai.app_logic import CLASS_COLORS, CLASS_ORDER
import labels as L

BG = "#0e1117"
PANEL = "#161b22"
GRID = "#2b3340"
TEXT = "#c9d1d9"
C_PRED, C_CORR, C_MEAS, C_TRUE, C_BIAS = "#6ea8fe", "#c58bff", "#ffb84d", "#e6edf3", "#4dd0c8"


def _style(ch: alt.Chart | alt.LayerChart, height: int) -> alt.Chart:
    return (ch.properties(height=height, background=PANEL)
            .configure_axis(gridColor=GRID, domainColor=GRID, tickColor=GRID, labelColor=TEXT,
                            titleColor=TEXT)
            .configure_legend(labelColor=TEXT, titleColor=TEXT, orient="top")
            .configure_view(stroke=None))


def coverage_timeseries(df: pd.DataFrame, c_req: float, marginal_min: float, n_total: int,
                        height: int = 280):
    """Predicted (raw), corrected, measured and true coverage with class bands and C_req."""
    top = max(60.0, float(np.nanmax(df[["predicted", "corrected", "measured", "true"]].to_numpy())) * 1.1) \
        if len(df) else 60.0
    x = alt.X("k:Q", title="Field section", scale=alt.Scale(domain=[1, max(n_total, 2)]))
    y = alt.Y("v:Q", title="Coverage (% of card area)", scale=alt.Scale(domain=[0, top]))
    bands = pd.DataFrame({"lo": [0, marginal_min, c_req], "hi": [marginal_min, c_req, top],
                          "cls": [L.cov_class("critical"), L.cov_class("marginal"), L.cov_class("acceptable")]})
    band = alt.Chart(bands).mark_rect(opacity=0.10).encode(
        y=alt.Y("lo:Q", scale=alt.Scale(domain=[0, top])), y2="hi:Q",
        color=alt.Color("cls:N", scale=alt.Scale(domain=[L.cov_class(c) for c in list(CLASS_COLORS)[:3]],
                                                 range=list(CLASS_COLORS.values())[:3]), legend=None))
    long = df.melt(id_vars="k", value_vars=["predicted", "corrected", "measured", "true"],
                   var_name="series", value_name="v")
    long["series"] = long["series"].map(L.series)
    names = [L.series(s_) for s_ in ("predicted", "corrected", "measured", "true")]
    colors = alt.Scale(domain=names,
                       range=[C_PRED, C_CORR, C_MEAS, C_TRUE])
    dash = alt.StrokeDash("series:N", legend=None, scale=alt.Scale(
        domain=names,
        range=[[1, 0], [1, 0], [1, 0], [4, 3]]))
    lines = alt.Chart(long).mark_line(point=alt.OverlayMarkDef(size=28), strokeWidth=2).encode(
        x=x, y=y, color=alt.Color("series:N", scale=colors, title=None,
                        legend=alt.Legend(orient="top", labelColor=TEXT)),
        strokeDash=dash, tooltip=[alt.Tooltip("k:Q", title="Field section"), alt.Tooltip("series:N", title="Series"),
                 alt.Tooltip("v:Q", title="Coverage (%)", format=".1f")])
    rule = alt.Chart(pd.DataFrame({"y": [c_req]})).mark_rule(color="#3ddc97", strokeDash=[6, 4]).encode(y="y:Q")
    return _style((band + rule + lines).resolve_scale(color="independent"), height)


def bias_chart(df: pd.DataFrame, n_total: int, height: int = 280):
    """Applied feedback bias (line) and the residual the controller sees (bars), in pp."""
    x = alt.X("k:Q", title="Field section", scale=alt.Scale(domain=[1, max(n_total, 2)]))
    long = df.melt(id_vars="k", value_vars=["bias", "residual"], var_name="series", value_name="pp")
    long["series"] = long["series"].map(L.series)
    colors = alt.Scale(domain=[L.series("bias"), L.series("residual")], range=[C_BIAS, C_MEAS])
    col = alt.Color("series:N", scale=colors, title=None,
                    legend=alt.Legend(orient="top", labelColor=TEXT))
    y = alt.Y("pp:Q", title="Percentage points (pp)")
    bars = alt.Chart(long[long["series"] == L.series("residual")]).mark_bar(opacity=0.45, size=6).encode(
        x=x, y=y, color=col, tooltip=[alt.Tooltip("k:Q", title="Field section"), alt.Tooltip("series:N", title="Series"),
                 alt.Tooltip("pp:Q", title="pp", format="+.2f")])
    line = alt.Chart(long[long["series"] == L.series("bias")]).mark_line(
        strokeWidth=2.5, interpolate="step-after", point=alt.OverlayMarkDef(size=26)).encode(
        x=x, y=y, color=col, tooltip=[alt.Tooltip("k:Q", title="Field section"), alt.Tooltip("series:N", title="Series"),
                 alt.Tooltip("pp:Q", title="pp", format="+.2f")])
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=GRID).encode(y="y:Q")
    return _style(zero + bars + line, height)


def field_strip_chart(df: pd.DataFrame, height: int | None = None):
    """Serpentine tile map of segments colored by class, retreat flags and current ring."""
    df = df.assign(status=df["status"].map(L.cov_class), retreat_txt=df["retreat"].map(
        {True: "Needs re-spraying", False: "No"}))
    n_rows = int(df["row"].max()) + 1
    n_cols = int(df["col"].max()) + 1
    h = height or max(120, 46 * n_rows + 30)
    color = alt.Color("status:N", scale=alt.Scale(domain=[L.cov_class(c) for c in CLASS_COLORS],
                                                  range=list(CLASS_COLORS.values())),
                      legend=alt.Legend(title="Coverage", orient="top"))
    xs = alt.X("col:O", axis=None, scale=alt.Scale(paddingInner=0.08))
    ys = alt.Y("row:O", axis=alt.Axis(title="Pass (lap)", labelColor=TEXT, titleColor=TEXT),
               scale=alt.Scale(paddingInner=0.1))
    tip = [alt.Tooltip("seg:Q", title="Field section"), alt.Tooltip("status:N", title="Coverage"),
           alt.Tooltip("measured:Q", title="Card reading (%)", format=".1f"),
           alt.Tooltip("true:Q", title="Actual coverage (%)", format=".1f"),
           alt.Tooltip("retreat_txt:N", title="Re-spraying")]
    tiles = alt.Chart(df).mark_rect(cornerRadius=4).encode(x=xs, y=ys, color=color, tooltip=tip)
    labels = alt.Chart(df).mark_text(fontSize=10, color="#0e1117", fontWeight="bold").encode(
        x=xs, y=ys, text="seg:Q")
    flags = alt.Chart(df[df["retreat"]]).mark_text(text="✖", fontSize=18, color="#ffffff",
                                                   dy=-13, dx=12).encode(x=xs, y=ys)
    cur = alt.Chart(df[df["current"]]).mark_rect(fill=None, stroke="#ffffff", strokeWidth=3,
                                                 cornerRadius=4).encode(x=xs, y=ys)
    ch = (tiles + labels + flags + cur).properties(width="container", height=h, background=PANEL)
    return (ch.configure_axis(domainColor=GRID, labelColor=TEXT, titleColor=TEXT)
              .configure_legend(labelColor=TEXT, titleColor=TEXT).configure_view(stroke=None))


def line_xy(df: pd.DataFrame, x: str, y: str, xtitle: str, ytitle: str, color: str = C_PRED,
            height: int = 240, extra_rule: float | None = None, y_fmt: str = ".3f"):
    ch = alt.Chart(df).mark_line(color=color, strokeWidth=2.5).encode(
        x=alt.X(f"{x}:Q", title=xtitle), y=alt.Y(f"{y}:Q", title=ytitle),
        tooltip=[alt.Tooltip(f"{x}:Q", format=".2f"), alt.Tooltip(f"{y}:Q", format=y_fmt)])
    if extra_rule is not None:
        ch = ch + alt.Chart(pd.DataFrame({"y": [extra_rule]})).mark_rule(
            color="#3ddc97", strokeDash=[6, 4]).encode(y="y:Q")
    return _style(ch, height)


def heatmap(df: pd.DataFrame, value: str, title: str, scheme: str = "turbo", height: int = 300,
            vtitle: str = "") -> alt.Chart:
    """Pressure x wind map in the spirit of Emery's coverage-map screenshot."""
    ps, ws = np.sort(df["pressure_psi"].unique()), np.sort(df["wind_ms"].unique())
    dp, dw = (ps[1] - ps[0]) if len(ps) > 1 else 1.0, (ws[1] - ws[0]) if len(ws) > 1 else 1.0
    d = df.assign(p0=df["pressure_psi"] - dp / 2, p1=df["pressure_psi"] + dp / 2,
                  w0=df["wind_ms"] - dw / 2, w1=df["wind_ms"] + dw / 2)
    ch = alt.Chart(d).mark_rect().encode(
        x=alt.X("p0:Q", title="Pressure (psi)", scale=alt.Scale(zero=False, nice=False)), x2="p1:Q",
        y=alt.Y("w0:Q", title="Crosswind (m/s)", scale=alt.Scale(zero=True, nice=False)), y2="w1:Q",
        color=alt.Color(f"{value}:Q", scale=alt.Scale(scheme=scheme),
                        legend=alt.Legend(orient="right", title=vtitle or title, gradientLength=180)),
        tooltip=[alt.Tooltip("pressure_psi:Q", title="Pressure (psi)", format=".0f"),
                 alt.Tooltip("wind_ms:Q", title="Crosswind (m/s)", format=".1f"),
                 alt.Tooltip(f"{value}:Q", title=vtitle or "Value", format=".3f")])
    return _style(ch.properties(title=alt.TitleParams(title, color=TEXT)), height)


def pattern_chart(pat: dict, height: int = 220):
    x = pat["x_m"]
    df = pd.DataFrame({"x_cm": x * 100, "Single nozzle": pat["single"], "Whole boom (sum)": pat["total"]})
    long = df.melt("x_cm", var_name="profile", value_name="Relative spray intensity")
    ch = alt.Chart(long).mark_line(strokeWidth=2.5).encode(
        x=alt.X("x_cm:Q", title="Across the boom (cm)"), y=alt.Y("Relative spray intensity:Q"),
        color=alt.Color("profile:N", scale=alt.Scale(range=[C_PRED, C_MEAS]), title=None))
    return _style(ch, height)


def policy_bars(df: pd.DataFrame, col: str, title: str, color: str = C_PRED, height: int = 220):
    ch = alt.Chart(df).mark_bar(color=color, cornerRadiusEnd=4).encode(
        x=alt.X("policy:N", title=None, sort=list(df["policy"]),
                axis=alt.Axis(labelAngle=-20, labelLimit=140)),
        y=alt.Y(f"{col}:Q", title=title),
        tooltip=[alt.Tooltip("policy:N", title="Control strategy"),
                 alt.Tooltip(f"{col}:Q", title=title, format=".3f")])
    txt = alt.Chart(df).mark_text(dy=-8, color=TEXT).encode(
        x=alt.X("policy:N", sort=list(df["policy"])), y=f"{col}:Q",
        text=alt.Text(f"{col}:Q", format=".2f"))
    return _style(ch + txt, height)


def policy_series(frames: dict[str, pd.DataFrame], c_req: float, height: int = 260):
    """True coverage per segment for each policy on the same scenario."""
    long = pd.concat([f.assign(policy=L.policy(p)) for p, f in frames.items()])
    ch = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("k:Q", title="Field section"), y=alt.Y("true:Q", title="Actual coverage (%)"),
        color=alt.Color("policy:N", title=None),
        tooltip=[alt.Tooltip("policy:N", title="Control strategy"), alt.Tooltip("k:Q", title="Field section"),
                 alt.Tooltip("true:Q", title="Actual coverage (%)", format=".1f")])
    rule = alt.Chart(pd.DataFrame({"y": [c_req]})).mark_rule(color="#3ddc97", strokeDash=[6, 4]).encode(y="y:Q")
    return _style(ch + rule, height)
