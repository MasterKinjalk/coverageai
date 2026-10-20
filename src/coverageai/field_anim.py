"""Payload builder for the animated "Field run" view (pure Python, no Streamlit).

Turns a ``PassResult`` into a JSON-serializable dict with plain-English labels that the
canvas animation in ``app/field_run_view.py`` consumes.
"""
from __future__ import annotations

import math
from typing import Any

from coverageai import app_logic as al
from coverageai.config import spec

STRATEGY_NAMES = {
    "open": "Model only (no feedback)",
    "fast": "Model + card feedback",
    "fast_slow": "Model + card feedback + learning between passes",
    "oracle": "Knows hidden conditions (fair upper bound)",
    "oracle_card": "Perfect knowledge incl. card noise (unreachable bound)",
    "best_fixed": "Single best nozzle for whole field",
}
MISMATCH_NAMES = {"none": "None", "low": "Low", "med": "Medium", "high": "High"}
DROPLET_NAMES = {"VF": "Very fine", "F": "Fine", "M": "Medium", "C": "Coarse",
                 "VC": "Very coarse", "XC": "Extremely coarse", "UC": "Ultra coarse"}
CLASS_LABELS = {"acceptable": "Good", "marginal": "Borderline", "critical": "Poor"}


def _f(v: Any, nd: int = 2) -> float:
    """Finite float (non-finite -> 0.0)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return round(v, nd) if math.isfinite(v) else 0.0


def class_label(cls: str) -> str:
    return CLASS_LABELS.get(cls, "Poor")


def field_animation_payload(res, c_req: float) -> dict:
    layout = al.lap_layout(res.conds)
    sections = []
    for k, lg in enumerate(res.logs):
        row, col = layout[k]
        x, u, d = lg.x, lg.u, lg.derived
        cls = lg.meas.coverage_class
        direction = 1 if row % 2 == 0 else -1
        rel = al.wind_arrow_deg(x)
        sections.append({
            "k": k, "number": k + 1, "row": int(row), "col": int(col), "dir": direction,
            "wind_speed_ms": _f(x.wind_speed_ms), "wind_rel_deg": _f(rel, 1),
            "wind_screen_deg": _f(rel + (180.0 if direction < 0 else 0.0), 1),
            "temp_c": _f(x.temp_c, 1), "rh_pct": _f(x.rh_pct, 0),
            "nozzle": f"{u.family} {u.size} at {u.pressure_psi:g} psi",
            "droplet": DROPLET_NAMES.get(d.droplet_class, str(d.droplet_class)),
            "rate_gpa": _f(d.app_rate_gpa),
            "vmd_um": _f(d.vmd_um, 0), "crosswind_ms": _f(abs(d.u_cross_ms)),
            "feedback_pp": _f(lg.bias), "error_pp": _f(lg.outcome.coverage_pct - lg.corrected_pred_cov),
            "predicted": _f(lg.pred.coverage_pct, 1),
            "after_feedback": _f(lg.corrected_pred_cov, 1),
            "card": _f(lg.meas.coverage_pct, 1),
            "actual": _f(lg.outcome.coverage_pct, 1),
            "label": class_label(cls),
            "color": al.CLASS_COLORS.get(cls, al.CLASS_COLORS["critical"]),
            "drift_pct": _f(lg.outcome.drift_frac * 100.0, 2),
            "respray": bool(lg.flagged_retreat),
            "no_nozzle": bool(lg.selector_infeasible),
            "on_target": bool(lg.outcome.coverage_pct >= c_req),
        })
    cc = spec()["coverage_classes"]
    return {
        "bands": {"good_min": _f(cc["acceptable_min"], 1), "border_min": _f(cc["marginal_min"], 1)},
        "strategy": STRATEGY_NAMES.get(res.policy, str(res.policy)),
        "surprises": MISMATCH_NAMES.get(res.mismatch, str(res.mismatch)),
        "target": _f(c_req, 1),
        "n": len(sections),
        "rows": (max((s["row"] for s in sections), default=-1) + 1),
        "cols": (max((s["col"] for s in sections), default=-1) + 1),
        "legend": [{"label": CLASS_LABELS[c], "color": al.CLASS_COLORS[c]}
                   for c in ("acceptable", "marginal", "critical")],
        "sections": sections,
    }
