"""Altair chart builders for the demo (dark theme, class colors shared with app_logic)."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from coverageai.app_logic import CLASS_COLORS
import labels as L

BG = "#0e1117"
PANEL = "#161b22"
GRID = "#2b3340"
TEXT = "#c9d1d9"
C_PRED, C_CORR, C_MEAS, C_TRUE, C_BIAS = "#6ea8fe", "#c58bff", "#ffb84d", "#e6edf3", "#4dd0c8"


def _style(ch: alt.Chart | alt.LayerChart, height: int, title: str | None = None) -> alt.Chart:
    if title:
        ch = ch.properties(title=alt.TitleParams(title, color=TEXT, anchor="start", fontSize=13))
    return (ch.properties(height=height, background=PANEL)
            .configure_axis(gridColor=GRID, domainColor=GRID, tickColor=GRID, labelColor=TEXT,
                            titleColor=TEXT)
            .configure_legend(labelColor=TEXT, titleColor=TEXT, orient="top")
            .configure_view(stroke=None))


def _mark(k: int | None):
    """Vertical rule at section ``k`` (the inspected section)."""
    return alt.Chart(pd.DataFrame({"k": [k]})).mark_rule(color="#4da3ff", strokeWidth=1.5).encode(x="k:Q")


def coverage_timeseries(df: pd.DataFrame, c_req: float, marginal_min: float, acceptable_min: float,
                        n_total: int, height: int = 280, mark: int | None = None):
    """Predicted (raw), corrected, measured and true coverage, the target (dashed) and the
    coverage-class bands. The bands are the fixed class limits of the spec (the same ones the card
    reader and replay panel B use); the target is a separate line and may differ from them."""
    top = max(60.0, float(np.nanmax(df[["predicted", "corrected", "measured", "true"]].to_numpy())) * 1.1) \
        if len(df) else 60.0
    x = alt.X("k:Q", title="Field section", scale=alt.Scale(domain=[1, max(n_total, 2)], nice=False))
    y = alt.Y("v:Q", title="Coverage (% of card area)", scale=alt.Scale(domain=[0, top]))
    bands = pd.DataFrame({"lo": [0, marginal_min, acceptable_min], "hi": [marginal_min, acceptable_min, top],
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
    # no legend for the dash: a second legend on the same field replaces the coloured one with
    # black symbols
    dash = alt.StrokeDash("series:N", legend=None,
                          scale=alt.Scale(
        domain=names,
        range=[[1, 0], [1, 0], [1, 0], [4, 3]]))
    lines = alt.Chart(long).mark_line(point=alt.OverlayMarkDef(size=28), strokeWidth=2).encode(
        x=x, y=y, color=alt.Color("series:N", scale=colors, title=None,
                        legend=alt.Legend(orient="top", labelColor=TEXT)),
        strokeDash=dash, tooltip=[alt.Tooltip("k:Q", title="Field section"), alt.Tooltip("series:N", title="Series"),
                 alt.Tooltip("v:Q", title="Coverage (%)", format=".1f")])
    rule = alt.Chart(pd.DataFrame({"y": [c_req]})).mark_rule(color=C_TRUE, strokeDash=[2, 3]).encode(y="y:Q")
    layers = band + rule + lines
    if mark is not None:
        layers = layers + _mark(mark)
    return _style(layers.resolve_scale(color="independent"), height,
                  f"Coverage by section (dashed: target {c_req:g}%)")


def bias_chart(df: pd.DataFrame, n_total: int, height: int = 280, mark: int | None = None):
    """The numbers of replay panel D: applied feedback correction (line) and the prediction
    error left after it, actual minus prediction after feedback (bars), in pp."""
    x = alt.X("k:Q", title="Field section", scale=alt.Scale(domain=[1, max(n_total, 2)], nice=False))
    long = df.melt(id_vars="k", value_vars=["bias", "error"], var_name="series", value_name="pp")
    long["series"] = long["series"].map(L.series)
    colors = alt.Scale(domain=[L.series("bias"), L.series("error")], range=[C_BIAS, C_TRUE])
    col = alt.Color("series:N", scale=colors, title=None,
                    legend=alt.Legend(orient="top", labelColor=TEXT))
    y = alt.Y("pp:Q", title="Percentage points (pp)")
    bars = alt.Chart(long[long["series"] == L.series("error")]).mark_bar(opacity=0.45, size=6).encode(
        x=x, y=y, color=col, tooltip=[alt.Tooltip("k:Q", title="Field section"), alt.Tooltip("series:N", title="Series"),
                 alt.Tooltip("pp:Q", title="pp", format="+.2f")])
    line = alt.Chart(long[long["series"] == L.series("bias")]).mark_line(
        strokeWidth=2.5, interpolate="step-after", point=alt.OverlayMarkDef(size=26)).encode(
        x=x, y=y, color=col, tooltip=[alt.Tooltip("k:Q", title="Field section"), alt.Tooltip("series:N", title="Series"),
                 alt.Tooltip("pp:Q", title="pp", format="+.2f")])
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=GRID).encode(y="y:Q")
    layers = zero + bars + line
    if mark is not None:
        layers = layers + _mark(mark)
    return _style(layers, height, "Feedback correction and the error left after it")


def line_xy(df: pd.DataFrame, x: str, y: str, xtitle: str, ytitle: str, color: str = C_PRED,
            height: int = 240, extra_rule: float | None = None, y_fmt: str = ".3f",
            point: tuple[float, float] | None = None, title: str | None = None):
    """A curve. ``extra_rule`` draws a dashed horizontal line (the target); ``point`` marks the
    current operating point (x, y) on the curve."""
    ch = alt.Chart(df).mark_line(color=color, strokeWidth=2.5).encode(
        x=alt.X(f"{x}:Q", title=xtitle), y=alt.Y(f"{y}:Q", title=ytitle),
        tooltip=[alt.Tooltip(f"{x}:Q", format=".2f"), alt.Tooltip(f"{y}:Q", format=y_fmt)])
    if extra_rule is not None:
        ch = ch + alt.Chart(pd.DataFrame({"y": [extra_rule]})).mark_rule(
            color=C_TRUE, strokeDash=[2, 3]).encode(y="y:Q")
    if point is not None:
        ch = ch + alt.Chart(pd.DataFrame({"x": [point[0]], "y": [point[1]]})).mark_point(
            color=C_TRUE, fill=C_TRUE, size=90, opacity=1).encode(x="x:Q", y="y:Q")
    return _style(ch, height, title)


def heatmap(df: pd.DataFrame, value: str, title: str, scheme: str = "viridis", height: int = 300,
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
    return _style(ch, height, title)


def pattern_chart(pat: dict, height: int = 220):
    x = pat["x_m"]
    df = pd.DataFrame({"x_cm": x * 100, "Single nozzle": pat["single"], "Whole boom (sum)": pat["total"]})
    long = df.melt("x_cm", var_name="profile", value_name="Relative spray intensity")
    ch = alt.Chart(long).mark_line(strokeWidth=2.5).encode(
        x=alt.X("x_cm:Q", title="Across the boom (cm)"), y=alt.Y("Relative spray intensity:Q"),
        color=alt.Color("profile:N", scale=alt.Scale(range=[C_PRED, C_MEAS]), title=None))
    return _style(ch, height)


def policy_bars(df: pd.DataFrame, col: str, title: str, color: str = C_PRED, height: int | None = None,
                show_labels: bool = True, lo: str | None = None, hi: str | None = None, fmt: str = ".2f",
                percent: bool = False):
    """Horizontal bars, one per strategy, so every long strategy name stays readable.
    ``show_labels=False`` hides the names (use it for the 2nd and 3rd chart in a row; the bar
    order is the same and the tooltip still names the strategy). ``lo`` / ``hi`` name the columns
    of an interval drawn as a whisker (left out when any bound is missing). ``fmt`` formats the
    value label, and should match the table next to the chart. ``percent=True`` keeps the axis
    ticks within 0 to 100 (the extra room on the right is only for the value labels)."""
    order = list(df["policy"])
    height = height or 46 + 34 * len(order)
    has_ci = bool(lo and hi and len(df) and df[[lo, hi]].notna().all(axis=None))
    vals = df[[col, hi] if has_ci else [col]].to_numpy(dtype=float) if len(df) else np.zeros(1)
    vmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else 0.0
    dom = [0, vmax * 1.22 if vmax > 0 else 1.0]
    ylab = alt.Axis(labelLimit=230, title=None) if show_labels else alt.Axis(labels=False, title=None, ticks=False)
    ch = alt.Chart(df).mark_bar(color=color, cornerRadiusEnd=4).encode(
        y=alt.Y("policy:N", title=None, sort=order, axis=ylab),
        x=alt.X(f"{col}:Q", title=title, scale=alt.Scale(domain=dom, nice=False),
                axis=alt.Axis(values=[0, 20, 40, 60, 80, 100]) if percent else alt.Axis()),
        tooltip=[alt.Tooltip("policy:N", title="Control strategy"),
                 alt.Tooltip(f"{col}:Q", title=title, format=fmt)])
    txt = alt.Chart(df).mark_text(align="left", dx=6, color=TEXT).encode(
        y=alt.Y("policy:N", sort=order), x=f"{hi if has_ci else col}:Q", text=alt.Text(f"{col}:Q", format=fmt))
    layers = ch + txt
    if has_ci:
        layers = layers + alt.Chart(df).mark_errorbar(color=C_TRUE, ticks=True, thickness=1.5).encode(
            y=alt.Y("policy:N", sort=order), x=alt.X(f"{lo}:Q", title=title), x2=f"{hi}:Q")
    return _style(layers, height)


def policy_series(frames: dict[str, pd.DataFrame], c_req: float, height: int = 260):
    """True coverage per segment for each policy on the same scenario."""
    long = pd.concat([f.assign(policy=L.policy(p)) for p, f in frames.items()])
    ch = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("k:Q", title="Field section"), y=alt.Y("true:Q", title="Actual coverage (%)"),
        color=alt.Color("policy:N", title=None, legend=alt.Legend(orient="top", labelColor=TEXT, labelLimit=420)),
        tooltip=[alt.Tooltip("policy:N", title="Control strategy"), alt.Tooltip("k:Q", title="Field section"),
                 alt.Tooltip("true:Q", title="Actual coverage (%)", format=".1f")])
    rule = alt.Chart(pd.DataFrame({"y": [c_req]})).mark_rule(color=C_TRUE, strokeDash=[2, 3]).encode(y="y:Q")
    return _style(ch + rule, height, f"Actual coverage by section (dashed: target {c_req:g}%)")
