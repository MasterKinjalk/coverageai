"""Simulator trend figure for the midterm report (WP-02).

Writes experiments/figures/sim_trends.png: one-at-a-time sweeps of the physics core
with a physics-only stub catalog (Derived built from a fixed VMD and flow), so the
figure shows the encoded trends and not catalog details.

    uv run python experiments/sim_trends.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coverageai import units
from coverageai.config import spec
from coverageai.contracts import Conditions, Derived, Setting
from coverageai.simulator import SpraySimulator

OUT = Path(__file__).parent / "figures" / "sim_trends.png"
VMD = spec()["droplets"]["vmd_center_um"]
CLASSES = {"F (fine)": VMD["F"], "M (medium)": VMD["M"], "C (coarse)": VMD["C"]}
COLORS = {"F (fine)": "#c0392b", "M (medium)": "#2c7fb8", "C (coarse)": "#1b7837"}
U = Setting("stub", "00", 40.0)
N_SEED = 16


def make_sim(vmd: float, gpa_fixed: float | None = None, flow_gpm: float = 0.4) -> SpraySimulator:
    def derive(x: Conditions, u: Setting) -> Derived:
        g = gpa_fixed if gpa_fixed is not None else units.gpa(
            flow_gpm, units.kmh_to_mph(x.travel_speed_kmh), 20.0)
        rel = units.wrap_deg(x.wind_dir_deg - x.heading_deg)
        r = math.radians(rel)
        return Derived(flow_gpm, g, units.gpa_to_lha(g), "?", vmd, 110.0, rel,
                       x.wind_speed_ms * math.sin(r), x.wind_speed_ms * math.cos(r))
    return SpraySimulator(derive_fn=derive)


def cond(wind=0.0, t=25.0, rh=50.0, h=0.5, v=12.7) -> Conditions:
    return Conditions(wind, 90.0, 0.0, t, rh, v, h)    # wind toward 90 deg, heading 0: pure crosswind


def mean_of(sim, x, field) -> float:
    return float(np.mean([getattr(sim.simulate(x, U, np.random.default_rng(s)), field)
                          for s in range(N_SEED)]))


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)

    gpas = np.linspace(2, 30, 15)
    for name, vmd in CLASSES.items():
        ax[0, 0].plot(gpas, [mean_of(make_sim(vmd, g), cond(0.5), "coverage_pct") for g in gpas],
                      color=COLORS[name], label=f"{name}, VMD {vmd:.0f} um")
    ax[0, 0].axhline(spec()["coverage_classes"]["acceptable_min"], color="k", ls=":", lw=1)
    ax[0, 0].axhline(spec()["coverage_classes"]["marginal_min"], color="k", ls=":", lw=1)
    ax[0, 0].set(xlabel="application rate (GPA)", ylabel="card coverage (%)",
                 title="(a) Coverage vs application rate (wind 0.5 m/s, H 0.5 m)")
    ax[0, 0].legend(fontsize=8)

    winds = np.linspace(0, 7, 29)
    for name, vmd in CLASSES.items():
        sim = make_sim(vmd, 15.0)
        ax[0, 1].plot(winds, [mean_of(sim, cond(u), "drift_frac") for u in winds],
                      color=COLORS[name], label=name)
    ax[0, 1].set(xlabel="crosswind speed (m/s)", ylabel="drift fraction",
                 title="(b) Drift vs wind (H 0.5 m, 15 GPA)", ylim=(0, 1.02))
    ax[0, 1].legend(fontsize=8)

    heights = np.linspace(0.30, 0.75, 19)
    for name, vmd in CLASSES.items():
        sim = make_sim(vmd, 15.0)
        ax[1, 0].plot(heights, [mean_of(sim, cond(1.0, h=h), "drift_frac") for h in heights],
                      color=COLORS[name], label=name)
    ax[1, 0].set(xlabel="boom height (m)", ylabel="drift fraction",
                 title="(c) Drift vs boom height (crosswind 1 m/s, 15 GPA)", ylim=(0, 1.02))
    ax[1, 0].legend(fontsize=8)

    rhs = np.linspace(10, 95, 18)
    sim = make_sim(VMD["VF"], 15.0)
    for t, c in zip((15, 25, 35), ("#2c7fb8", "#7b6d1b", "#c0392b")):
        ax[1, 1].plot(rhs, [mean_of(sim, cond(0.0, t=t, rh=rh, h=0.75), "coverage_pct") for rh in rhs],
                      color=c, label=f"{t} C")
    ax[1, 1].set(xlabel="relative humidity (%)", ylabel="card coverage (%)",
                 title="(d) Evaporation: VMD 90 um, no wind, H 0.75 m")
    ax[1, 1].legend(title="air temperature", fontsize=8)

    fig.suptitle("Simulator trends (first-order physics; not a validated drift model)", fontsize=11)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
