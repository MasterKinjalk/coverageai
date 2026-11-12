"""Plain-English display names for the demo. Internal identifiers/values stay unchanged;
only what is DISPLAYED goes through these mappings (format_func, column renames, legends)."""
from __future__ import annotations

import re

POLICY = {
    "open": "Model only (no feedback)",
    "fast": "Model + card feedback",
    "fast_slow": "Model + card feedback + learning between passes",
    "oracle": "Knows hidden conditions (fair upper bound)",
    "oracle_card": "Perfect knowledge incl. card noise (unreachable bound)",
    "best_fixed": "Single best nozzle for whole field",
}
# Short names for places where several strategies sit side by side in a narrow space (the
# sidebar's strategy picker). Each must stay distinct when cut off after about 20 characters.
POLICY_SHORT = {
    "open": "Model only",
    "fast": "Card feedback",
    "fast_slow": "Feedback + learning",
    "oracle": "Knows hidden conditions",
    "oracle_card": "Perfect knowledge",
    "best_fixed": "Single best nozzle",
}
MISMATCH = {"none": "None", "low": "Low", "med": "Medium", "high": "High"}
MODEL = {"oracle_nominal": "Physics model (ideal)", "spray": "Trained ML model",
         "mock": "Simple test model"}
VERIFIER = {"perfect": "Perfect reading (true value)", "noisy": "Noisy reading (simulated sensor)",
            "b0": "Classic image analysis", "b1": "Neural network (U-Net)"}
COVERAGE_CLASS = {"acceptable": "Good", "marginal": "Borderline", "critical": "Poor",
                  "pending": "Not reached yet"}
DROPLET = {"VF": "Very fine", "F": "Fine", "M": "Medium", "C": "Coarse", "VC": "Very coarse",
           "XC": "Extremely coarse", "UC": "Ultra coarse"}
PRESET = {"gusty_afternoon": "Gusty afternoon", "hot_dry": "Hot and dry",
          "inversion_evening": "Calm evening (inversion)", "marginal_limits": "Wind near legal limit",
          "crosswind_shift": "Shifting crosswind", "calm_morning": "Calm morning"}
FAMILY = {"XR": "XR - Extended-range flat fan (110°)", "TP65": "TP65 - Standard flat fan (65°)",
          "ULD": "ULD - Air-induction, ultra-low drift (120°)",
          "3D": "3D - Angled flat fan (100°)",
          "DGX40": "DGX40 - Drift Guard pre-orifice flat fan (40°)"}
SERIES = {"predicted": "Model prediction", "corrected": "Prediction after feedback",
          "measured": "Card reading", "true": "Actual coverage",
          "bias": "Feedback correction (pp)", "residual": "Remaining error",
          "error": "Prediction error (pp)"}

WSP_HELP = ("A water-sensitive paper (WSP) card is a small yellow strip that turns blue where spray "
            "droplets land. The share of the card that turns blue is the 'coverage'.")


def _get(d: dict, key, fallback=None):
    return d.get(key, fallback if fallback is not None else str(key))


def policy(k): return _get(POLICY, k)
def policy_short(k): return _get(POLICY_SHORT, k)
def mismatch(k): return _get(MISMATCH, k)
def model(k): return _get(MODEL, k)
def verifier(k): return _get(VERIFIER, k)
def cov_class(k): return _get(COVERAGE_CLASS, k)
def droplet(k): return _get(DROPLET, k)
def series(k): return _get(SERIES, k)
def family(k): return _get(FAMILY, k)


def preset(k):
    return PRESET.get(k) or str(k).replace("_", " ").capitalize()


def size(s): return f"Size {s}"


def nozzle_choice(k) -> str:
    """Sidebar "Nozzle on the boom" options: auto, free, or 'FAMILY-SIZE'."""
    if k == "auto":
        return "One nozzle, picked by the model"
    if k == "free":
        return "Any nozzle, section by section"
    fam, _, sz = str(k).partition("-")
    return f"{fam} {sz} for the whole field"


def setting(fam: str, sz: str, psi: float) -> str:
    return f"{fam} {sz} at {psi:g} psi"


def setting_of(u) -> str:
    """Full text of a chosen Setting, including the controller-chosen travel speed / boom height
    when the opt-in speed or boom-height control picked them."""
    t = setting(u.family, u.size, u.pressure_psi)
    if getattr(u, "travel_speed_kmh", None) is not None:
        t += f", {u.travel_speed_kmh:g} km/h"
    if getattr(u, "boom_height_m", None) is not None:
        t += f", boom {u.boom_height_m:.2f} m"
    return t


FORECAST = {"off": "Off (plan with the wind read at the section start)",
            "last": "Last wind reading", "ewma": "Smoothed recent readings (EWMA)",
            "ar1": "Statistical forecast (AR(1))",
            "ar1+gust": "Statistical forecast + gust safety margin"}


def forecast(k): return _get(FORECAST, k)


FEEDBACK = {"rule": "Running average of card errors", "xgboost": "Trained model (XGBoost)",
            "lightgbm": "Trained model (LightGBM)"}


def feedback_choice(k): return _get(FEEDBACK, k)


def parse_setting_text(txt: str) -> str:
    """'XR-04 @ 40 psi' -> 'XR 04 at 40 psi'."""
    return re.sub(r"^(\w+)-(\w+) @ ", r"\1 \2 at ", str(txt))


def explain(line: str) -> str:
    """Rewrite code-ish words in app_logic.explain_choice sentences."""
    out = line
    for a, b in [
        ("Oracle: the selector queried the true simulator, including the hidden mismatch realization "
         "of this segment (gusts, canopy, nozzle wear) but not the random card-to-card spread. "
         "Fair upper bound, not deployable.",
         "Knows hidden conditions: the choice was made knowing the real gusts, crop canopy and nozzle "
         "wear in this section, but not the random card-to-card variation. A fair upper bound, not "
         "something a real sprayer can do."),
        ("Card-aware oracle: the selector knew the hidden mismatch AND this card's random spread. "
         "Unreachable bound, not deployable.",
         "Perfect knowledge: the choice also knew this card's random variation. An unreachable bound."),
        ("Oracle: the selector queried the true simulator, including the hidden mismatch realization "
         "of this segment. Upper bound, not deployable.",
         "Perfect knowledge: the choice was made knowing exactly what the real-world surprises "
         "would be in this section. This is an upper bound, not something a real sprayer can do."),
        ("Fixed setting for the whole pass: the single catalog setting with the highest predicted "
         "requirement-met rate over this scenario (ties: lowest drift).",
         "One nozzle setting for the whole field: the single catalog setting expected to meet the "
         "target coverage in the most sections (ties: least drift)."),
        ("candidates meet corrected coverage - margin >= C_req", "settings reach the target coverage"),
        ("No candidate is feasible, so the selector fell back to the highest corrected coverage and "
         "flagged the segment.",
         "No nozzle setting is expected to meet the target, so the one with the highest expected "
         "coverage was used and the section is flagged."),
        ("Among feasible candidates it picked the lowest score = lambda x drift + (1 - lambda) x volume",
         "Among the settings that meet the target, it picked the best trade-off between drift and "
         "water use (lowest score)"),
        ("(fast loop, eq. 3)", "(card feedback)"),
        ("Feedback bias", "Feedback correction"),
        ("the model's coverage", "the model's predicted coverage"),
        ("(lambda = ", "(priority = "),
        (" candidates ", " settings "),
        ("C_req", "target coverage"),
        ("selector", "optimizer"),
    ]:
        out = out.replace(a, b)
    return out


# Column renames for the selector top-5 table
TOP_COLS = {"Setting": "Nozzle setting", "Pred %": "Predicted (%)",
            "Corr. %": "After feedback (%)", "Drift": "Drift (fraction)",
            "GPA": "Spray volume (gal/acre)", "Score": "Score (lower is better)",
            "Feasible": "Meets target"}

POLICY_HELP = (
    "Model only: pick settings from the model's prediction alone. "
    "Model + card feedback: also correct the model using what the paper card shows. "
    "Model + card feedback + learning between passes: the same, after learning a correction from "
    "three earlier passes over the field. "
    "Knows hidden conditions: picks knowing the real gusts, canopy and nozzle wear (a fair upper "
    "bound, not achievable in practice). "
    "Perfect knowledge incl. card noise: also knows each card's random variation (unreachable). "
    "Single best nozzle: one setting for the whole field.")


def compare_table(stats, reference_name: str, n_seeds: int):
    """The "Compare strategies" table from ``app_logic.comparison_stats`` (one row per strategy,
    ``policy`` already a display name). With one seed there is nothing to average, so the interval
    and the paired difference are left out."""
    import pandas as pd
    many = n_seeds > 1
    count = "{:.1f}" if many else "{:.0f}"
    rows = []
    for r in stats.to_dict("records"):
        row = {"Control strategy": r["policy"], "Sections on target (%)": f"{r['met_pct']:.1f}"}
        if many:
            row["95% interval"] = f"{r['met_lo']:.1f} to {r['met_hi']:.1f}"
            d = r.get("d_met_pp")
            row[f"Difference vs {reference_name} (pp)"] = (
                "reference" if d is None or d != d
                else f"{d:+.1f} ({r['d_met_lo']:+.1f} to {r['d_met_hi']:+.1f})")
        row.update({"Spray volume (gal/acre)": f"{r['volume_gpa']:.1f}", "Drift (%)": f"{r['drift_pct']:.2f}",
                    "Average shortfall (pp)": f"{r['shortfall_pp']:.2f}",
                    "Needs re-spraying (sections)": count.format(r["n_retreat"]),
                    "No nozzle meets target (sections)": count.format(r["n_infeasible"])})
        rows.append(row)
    return pd.DataFrame(rows)

# "What the model decided" tables (app_logic.decision_frame) and their number formats
DECISION_COLS = {"section": "Section", "nozzle": "Nozzle", "pressure_psi": "Pressure (psi)",
                 "speed_kmh": "Travel speed (km/h)", "boom_height_m": "Boom height (m)",
                 "volume_gpa": "Spray volume (gal/acre)", "droplet_class": "Droplet size",
                 "wind_ms": "Wind (m/s)", "predicted_cov": "Model prediction (%)",
                 "predicted_lo": "Prediction low (%)", "predicted_hi": "Prediction high (%)",
                 "feedback_pp": "Feedback correction (pp)",
                 "after_feedback_cov": "Prediction after feedback (%)", "card_cov": "Card reading (%)",
                 "actual_cov": "Actual coverage (%)", "error_pp": "Prediction error (pp)",
                 "predicted_drift_pct": "Predicted drift (%)", "actual_drift_pct": "Actual drift (%)",
                 "changed": "Changed since the previous section"}
DECISION_FMT = {"pressure_psi": "{:g}", "speed_kmh": "{:.1f}", "boom_height_m": "{:.2f}",
                "volume_gpa": "{:.1f}", "wind_ms": "{:.1f}", "predicted_cov": "{:.1f}",
                "predicted_lo": "{:.1f}", "predicted_hi": "{:.1f}", "feedback_pp": "{:+.2f}",
                "after_feedback_cov": "{:.1f}", "card_cov": "{:.1f}", "actual_cov": "{:.1f}",
                "error_pp": "{:+.2f}", "predicted_drift_pct": "{:.2f}", "actual_drift_pct": "{:.2f}"}

# Feedback-loop table (app_logic.timeseries_frame), in reading order
LOOP_COLS = {"k": "Section", "predicted": "Model prediction (%)", "bias": "Feedback correction (pp)",
             "corrected": "Prediction after feedback (%)", "measured": "Card reading (%)",
             "true": "Actual coverage (%)", "error": "Prediction error (pp)",
             "residual": "Card minus prediction after feedback (pp)",
             "raw_error": "Card minus model (pp)",
             "bias_after": "Correction after this card (pp)"}


# Operator console (app/operator_view.py, coverageai.operator_console). The minimum is the fixed
# "Good" limit (spec coverage_classes.acceptable_min); a tier is the controller's plan from the model.
RATE_TIER = {"exact": "Minimum reached at the set rate",
             "band": "Minimum reached inside the rate band",
             "unreachable": "Minimum not reachable at this rate"}
RATE_KNOB = {"pressure_at_rate": "Pressure (speed follows, rate held)", "boom_height": "Boom height"}
RATE_MOVE = {("pressure_at_rate", 1): "raise pressure (speed follows)",
             ("pressure_at_rate", -1): "lower pressure (speed follows)",
             ("boom_height", 1): "raise the boom", ("boom_height", -1): "lower the boom"}


def rate_tier(k): return "No plan recorded" if k is None else _get(RATE_TIER, k)
def rate_knob(k): return _get(RATE_KNOB, k)
def rate_move(knob, sign): return RATE_MOVE.get((knob, 1 if sign > 0 else -1), str(knob))


OP_SECTION_COLS = {"section": "Section", "wind_ms": "Wind (m/s)", "pressure_psi": "Pressure (psi)",
                   "speed_kmh": "Travel speed (km/h)", "boom_height_m": "Boom height (m)",
                   "rate_gpa": "Rate (gal/acre)", "predicted_cov": "Model prediction (%)",
                   "card_cov": "Card reading (%)", "actual_cov": "Actual coverage (%)",
                   "card_class": "Card result", "tier": "Controller plan (model)"}
OP_SECTION_FMT = {"wind_ms": "{:.1f}", "pressure_psi": "{:.0f}", "speed_kmh": "{:.1f}",
                  "boom_height_m": "{:.2f}", "rate_gpa": "{:.2f}", "predicted_cov": "{:.1f}",
                  "card_cov": "{:.1f}", "actual_cov": "{:.1f}"}
OP_ADVICE_COLS = {"section": "Section", "card_cov": "Card reading (%)", "short_pp": "Short of the minimum (pp)",
                  "advice": "What to change"}
OP_ALT_COLS = {"nozzle": "Nozzle", "share_pct": "Sections predicted at the minimum or above (%)",
               "mean_cov": "Mean predicted coverage (%)", "drift_pct": "Mean predicted drift (%)",
               "pressure": "Pressure (psi)", "speed": "Travel speed (km/h)"}
