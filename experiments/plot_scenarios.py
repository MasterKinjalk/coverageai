"""Plot every scenario preset: wind, crosswind, temperature, RH over segments.

Usage: uv run python experiments/plot_scenarios.py [--seed 0]
Writes experiments/figures/scenarios.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from coverageai import scenarios  # noqa: E402
from coverageai.config import spec  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures" / "scenarios.png"
COLS = [("Wind speed (m/s)", "wind"), ("Crosswind (m/s)", "cross"),
        ("Temperature (C)", "temp"), ("RH (%)", "rh")]
COLOR = "#2a6f97"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    names = scenarios.list_presets()
    r = spec()["ranges"]
    limit = spec()["wind_limit_ms"]
    fig, axes = plt.subplots(len(names), 4, figsize=(13, 2.1 * len(names)),
                             sharex=True, squeeze=False)
    for i, name in enumerate(names):
        sp = scenarios.get(name)
        xs = scenarios.generate(sp, args.seed)
        k = np.arange(len(xs))
        data = {
            "wind": [x.wind_speed_ms for x in xs],
            "cross": [scenarios.crosswind_ms(x) for x in xs],
            "temp": [x.temp_c for x in xs],
            "rh": [x.rh_pct for x in xs],
        }
        for j, (label, key) in enumerate(COLS):
            ax = axes[i, j]
            ax.plot(k, data[key], color=COLOR, lw=1.4)
            for b in range(sp.segments_per_lap, sp.segments, sp.segments_per_lap):
                ax.axvline(b - 0.5, color="0.8", lw=0.8)
            if key == "wind":
                ax.axhline(limit, color="#c0392b", ls="--", lw=0.9)
                ax.set_ylim(0, r["wind_speed_ms"][1])
            elif key == "cross":
                ax.axhline(0, color="0.5", lw=0.8)
                ax.set_ylim(-r["wind_speed_ms"][1], r["wind_speed_ms"][1])
            elif key == "temp":
                ax.set_ylim(*r["temp_c"])
            else:
                ax.set_ylim(*r["rh_pct"])
            if i == 0:
                ax.set_title(label, fontsize=10)
            if j == 0:
                ax.set_ylabel(name, fontsize=9)
            ax.grid(alpha=0.25)
        if i == len(names) - 1:
            for j in range(4):
                axes[i, j].set_xlabel("segment")
    fig.suptitle(f"Scenario presets, seed {args.seed} (grey lines: lap boundaries; "
                 "red dashed: wind limit)", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
