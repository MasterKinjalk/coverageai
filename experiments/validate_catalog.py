"""Validate data/catalog/nozzles.csv (WP-01). Prints violations; always exits 0.

Checks
  (a) flow follows sqrt(P) between adjacent chart rows within +-5% (a miss is
      excused, and counted separately, when it is explained by the printed
      rounding of the two flow values);
  (b) for 20-in charts, GPA = 5940*GPM/(MPH*20) reproduces the sampled chart GPA
      cells recorded in data/catalog/gpa_checks.csv (within rounding);
  (c) droplet class coarseness is non-increasing as pressure rises within a size.

Run: uv run python experiments/validate_catalog.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coverageai import catalog, units  # noqa: E402
from coverageai.config import ROOT, spec  # noqa: E402

SQRT_TOL = 0.05


def _flow_half_unit(q: float, family: str = "") -> float:
    """Half the last printed digit. TeeJet charts print 3 decimals below 0.1 gpm, else 2;
    Hypro charts (ULD, 3D) always print 2."""
    if family in ("ULD", "3D"):
        return 0.005
    return 0.0005 if q < 0.1 else 0.005


def check_sqrt(df: pd.DataFrame) -> tuple[list[str], int]:
    bad, excused = [], 0
    for (fam, size), g in df.groupby(["family", "size"], sort=False):
        g = g.sort_values("pressure_psi")
        p, q = g["pressure_psi"].to_numpy(), g["flow_gpm"].to_numpy()
        for i in range(len(p) - 1):
            r = math.sqrt(p[i + 1] / p[i])
            exp = q[i] * r
            dev = abs(q[i + 1] - exp)
            if dev <= SQRT_TOL * exp:
                continue
            slack = _flow_half_unit(q[i], fam) * r + _flow_half_unit(q[i + 1], fam)
            if dev <= SQRT_TOL * exp + slack:
                excused += 1
                continue
            bad.append(f"  SQRT  {fam}{size} {p[i]:g}->{p[i + 1]:g} psi: flow {q[i]:g}->{q[i + 1]:g}, "
                       f"sqrt law gives {exp:.3f} ({100 * (q[i + 1] / exp - 1):+.1f}%)")
    return bad, excused


def check_gpa(df: pd.DataFrame) -> tuple[list[str], int]:
    checks = pd.read_csv(ROOT / "data" / "catalog" / "gpa_checks.csv", dtype={"size": str})
    bad = []
    for r in checks.itertuples():
        row = df[(df.family == r.family) & (df["size"] == r.size) & (df.pressure_psi == r.pressure_psi)]
        if row.empty:
            bad.append(f"  GPA   {r.family}{r.size} {r.pressure_psi:g} psi: no such catalog row")
            continue
        if int(row.iloc[0].chart_spacing_in) != 20:
            bad.append(f"  GPA   {r.family}{r.size}: not a 20-in chart, check skipped")
            continue
        q = float(row.iloc[0].flow_gpm)
        calc = units.gpa(q, r.mph, 20.0)
        half = 0.05 if r.chart_gpa < 100 and abs(r.chart_gpa * 10 - round(r.chart_gpa * 10)) < 1e-9 \
            and r.chart_gpa != round(r.chart_gpa) else 0.5
        tol = 0.015 * r.chart_gpa + half + calc * _flow_half_unit(q, r.family) / q
        if abs(calc - r.chart_gpa) > tol:
            bad.append(f"  GPA   {r.family}{r.size} {r.pressure_psi:g} psi @ {r.mph:g} mph: "
                       f"calc {calc:.2f} vs chart {r.chart_gpa:g} (tol {tol:.2f})")
    return bad, len(checks)


def check_class_order(df: pd.DataFrame) -> list[str]:
    order = {c: i for i, c in enumerate(spec()["droplets"]["class_order"])}   # VF=0 ... UC=6
    bad = []
    for (fam, size), g in df.groupby(["family", "size"], sort=False):
        g = g.sort_values("pressure_psi")
        idx = [order[c] for c in g["droplet_class"]]
        for i in range(len(idx) - 1):
            if idx[i + 1] > idx[i]:
                bad.append(f"  CLASS {fam}{size}: {g.iloc[i].droplet_class} at {g.iloc[i].pressure_psi:g} psi "
                           f"-> {g.iloc[i + 1].droplet_class} at {g.iloc[i + 1].pressure_psi:g} psi (coarser at higher P)")
    return bad


def check_structure(df: pd.DataFrame) -> list[str]:
    bad = []
    s = spec()
    for fam, g in df.groupby("family"):
        lo, hi = s["pressure_range_psi"][fam]
        if g.pressure_psi.min() < lo or g.pressure_psi.max() > hi:
            bad.append(f"  RANGE {fam}: chart pressures [{g.pressure_psi.min():g}, {g.pressure_psi.max():g}] "
                       f"outside spec [{lo}, {hi}]")
        if (g.spray_angle_deg != s["spray_angle_deg"][fam]).any():
            bad.append(f"  ANGLE {fam}: catalog spray angle differs from spec.yaml")
    unknown = set(df.droplet_class) - set(s["droplets"]["class_order"])
    if unknown:
        bad.append(f"  CLASS unknown droplet classes {sorted(unknown)}")
    return bad


def main() -> int:
    df = catalog.load()
    print(f"catalog: {len(df)} rows, {df.groupby(['family', 'size']).ngroups} family/size combinations")
    print(df.groupby("family").size().to_string())
    sections = []
    sq, excused = check_sqrt(df)
    sections.append(("(a) sqrt(P) flow law, +-5% between adjacent rows", sq,
                     f"{excused} further deviations excused by chart rounding"))
    gp, n = check_gpa(df)
    sections.append((f"(b) GPA recompute at 20 in vs {n} sampled chart cells", gp, ""))
    sections.append(("(c) droplet class non-increasing with pressure", check_class_order(df), ""))
    sections.append(("(d) ranges / spray angle vs spec.yaml", check_structure(df), ""))
    total = 0
    for title, bad, extra in sections:
        print(f"\n{title}: {len(bad)} violation(s)" + (f"  [{extra}]" if extra else ""))
        for b in bad:
            print(b)
        total += len(bad)
    print(f"\nTOTAL violations: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
