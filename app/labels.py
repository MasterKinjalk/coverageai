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
          "bias": "Feedback correction (pp)", "residual": "Remaining error"}

WSP_HELP = ("A water-sensitive paper (WSP) card is a small yellow strip that turns blue where spray "
            "droplets land. The share of the card that turns blue is the 'coverage'.")


def _get(d: dict, key, fallback=None):
    return d.get(key, fallback if fallback is not None else str(key))


def policy(k): return _get(POLICY, k)
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
TOP_COLS = {"Setting": "Nozzle setting", "Pred %": "Predicted coverage (%)",
            "Corr. %": "After feedback (%)", "Drift": "Drift (fraction)",
            "GPA": "Spray volume (gal/acre)", "Score": "Score (lower is better)",
            "Feasible": "Meets target"}

CMP_COLS = {"policy": "Control strategy", "requirement met (%)": "Sections meeting target (%)",
            "mean shortfall (pp)": "Average shortfall (pp)",
            "mean volume (GPA)": "Average spray volume (gal/acre)",
            "total volume (GPA)": "Total spray volume (gal/acre, summed)",
            "mean drift (frac)": "Average drift (fraction)",
            "re-treatment flags": "Needs re-spraying (sections)",
            "infeasible": "No nozzle meets target (sections)"}
