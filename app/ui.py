"""Shared look for every view of the app: one stylesheet, one KPI tile, one badge, one group label.

Keep visual rules here so the three views (Field run, Compare strategies, Nozzle lab) cannot drift
apart. Colours match ``.streamlit/config.toml`` and the replay canvas (``field_run_view``).
"""
from __future__ import annotations

import html as _html

import streamlit as st

CSS = """
<style>
.block-container {padding-top: 2.6rem; padding-bottom: 2rem; max-width: 1500px;}
h1, h2, h3 {letter-spacing: 0.2px;}
/* page header: product name left, one-line scope note right */
.cv-hdr {display:flex; align-items:baseline; justify-content:space-between; gap:16px; margin:0 0 2px 0;}
.cv-hdr .t {color:#e6edf3; font-size:1.35rem; font-weight:700;}
.cv-hdr .s {color:#8a94a6; font-size:0.82rem;}
/* KPI tiles: one grid, so every tile in a row has the same height */
.cv-kpis {display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin:2px 0 10px 0;}
.cv-kpi {background:#161b22; border:1px solid #2b3340; border-radius:10px; padding:9px 14px;}
.cv-kpi .l {color:#8a94a6; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.6px;
            white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.cv-kpi .v {color:#e6edf3; font-size:1.5rem; font-weight:650; line-height:1.2; white-space:nowrap;}
.cv-kpi .v .u {color:#8a94a6; font-size:0.8rem; font-weight:500; margin-left:4px;}
.cv-kpi .s {color:#8a94a6; font-size:0.78rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.cv-panel {background:#161b22; border:1px solid #2b3340; border-radius:10px; padding:12px 16px;}
.cv-badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.78rem;
           font-weight:600; margin:0 6px 4px 0; color:#0e1117;}
.cv-b-acceptable {background:#3ddc97;} .cv-b-marginal {background:#f5b942;} .cv-b-critical {background:#ef5b5b;}
.cv-b-warn {background:#ef5b5b; color:#fff;} .cv-b-info {background:#2f81f7; color:#fff;}
.cv-b-grey {background:#39414f; color:#e6edf3;}
/* chips: the settings a result was computed with */
.cv-chips {display:flex; flex-wrap:wrap; gap:4px 14px; font-size:0.8rem; color:#8a94a6; margin:0 0 8px 0;}
.cv-chips b {color:#e6edf3; font-weight:600;}
.cv-cond, .cv-cond tr, .cv-cond td {border:none !important; background:transparent !important;}
.cv-cond td {padding:2px 12px 2px 0 !important; color:#c9d1d9; font-size:0.9rem; vertical-align:top;}
.cv-cond td:first-child {color:#8a94a6; white-space:nowrap;}
/* sidebar: group labels and a tighter stack */
.cv-grp {color:#8a94a6; font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.8px;
         margin:8px 0 0 0; padding-top:10px; border-top:1px solid #2b3340;}
[data-testid="stSidebarContent"] [data-testid="stVerticalBlock"] {gap:0.7rem;}
[data-testid="stSidebarHeader"] {height:1.5rem; padding-bottom:0;}
/* the five loop steps */
.cv-loop {display:flex; gap:6px; align-items:stretch; flex-wrap:wrap; margin:4px 0 10px 0;}
.cv-loop .st {flex:1 1 150px; background:#161b22; border:1px solid #2b3340; border-radius:10px;
              padding:8px 10px; min-width:140px;}
.cv-loop .st .n {color:#4da3ff; font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;}
.cv-loop .st .t {color:#e6edf3; font-weight:650; font-size:0.95rem; margin:2px 0;}
.cv-loop .st .d {color:#8a94a6; font-size:0.8rem; line-height:1.3;}
.cv-loop .ar {align-self:center; color:#4da3ff; font-size:1.1rem;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, note: str) -> None:
    st.markdown(f'<div class="cv-hdr"><span class="t">{_html.escape(title)}</span>'
                f'<span class="s">{_html.escape(note)}</span></div>', unsafe_allow_html=True)


def kpi_row(items: list[tuple]) -> None:
    """One row of equal-height tiles. Each item is ``(label, value, note)`` or
    ``(label, value, unit, note)``. ``value`` may contain HTML (for a badge)."""
    cells = []
    for it in items:
        label, value, unit, note = it if len(it) == 4 else (it[0], it[1], "", it[2])
        u = f'<span class="u">{unit}</span>' if unit else ""
        cells.append(f'<div class="cv-kpi"><div class="l">{label}</div><div class="v">{value}{u}</div>'
                     f'<div class="s">{note}&nbsp;</div></div>')
    st.markdown('<div class="cv-kpis">' + "".join(cells) + "</div>", unsafe_allow_html=True)


def badge(text: str, kind: str) -> str:
    """Pill. ``kind``: acceptable, marginal, critical, warn, info or grey."""
    return f'<span class="cv-badge cv-b-{kind}">{text}</span>'


def chips(pairs: list[tuple[str, str]]) -> None:
    """One muted line of ``name: value`` pairs: the settings a result was computed with."""
    st.markdown('<div class="cv-chips">' + "".join(
        f"<span>{_html.escape(k)}: <b>{_html.escape(str(v))}</b></span>" for k, v in pairs) + "</div>",
        unsafe_allow_html=True)


def group(label: str) -> None:
    """Small heading for a group of controls."""
    st.markdown(f'<div class="cv-grp">{_html.escape(label)}</div>', unsafe_allow_html=True)


def loop_html(steps: list[tuple[str, str, str]]) -> str:
    """The closed loop as a row of step cards: ``(step name, title, text)`` each."""
    parts = []
    for i, (n, t, d) in enumerate(steps):
        if i:
            parts.append('<div class="ar">&#8594;</div>')
        parts.append(f'<div class="st"><div class="n">{n}</div><div class="t">{t}</div>'
                     f'<div class="d">{d}</div></div>')
    return '<div class="cv-loop">' + "".join(parts) + "</div>"
