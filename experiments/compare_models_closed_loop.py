"""Paired closed-loop comparison of two predictor files (D18: SprayModel v1 vs v2).

Run `closed_loop.py` once per model into its own --out-dir (same code, same seeds), then:
  uv run --no-sync python experiments/compare_models_closed_loop.py \
      --a results/model_v2_closed_loop/v1 --b results/model_v2_closed_loop/v2 \
      --out results/model_v2_closed_loop/compare
Runs are paired on (preset, mismatch, seed, policy): closed_loop.py fixes the scenario and the
hidden-mismatch realization by seed, so both models face identical fields. Reports B - A for
the share of segments meeting C_req, mean volume (GPA), mean drift and mean shortfall, with
paired-bootstrap 95 % CIs over the 320 cells (4 presets x 4 mismatch levels x 20 seeds), per
policy and per policy x mismatch. Oracle policies do not use the predictor and serve as a
pairing check (their difference must be exactly 0).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ("frac_met", "mean_volume_gpa", "mean_drift_frac", "mean_shortfall_pp")
MODEL_POLICIES = ("open", "fast", "fast_slow")
CHECK_POLICIES = ("oracle", "oracle_card")


def load(d: Path) -> pd.DataFrame:
    js = json.loads((d / "results" / "closed_loop_spray.json").read_text(encoding="utf8"))
    return pd.DataFrame(js["runs"])


def boot_ci(d: np.ndarray, B: int = 4000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
    return {"mean": float(d.mean()), "lo": float(np.percentile(m, 2.5)),
            "hi": float(np.percentile(m, 97.5)), "n": int(len(d))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="closed_loop.py --out-dir of model A (v1)")
    ap.add_argument("--b", required=True, help="closed_loop.py --out-dir of model B (v2)")
    ap.add_argument("--out", required=True, help="output path prefix (writes .json and .md)")
    ap.add_argument("--label-a", default="v1 (LightGBM)")
    ap.add_argument("--label-b", default="v2 (XGBoost, GPU)")
    a = ap.parse_args()
    on = ["preset", "mismatch", "seed", "policy"]
    m = load(Path(a.a)).merge(load(Path(a.b)), on=on, suffixes=("_a", "_b"), validate="one_to_one")
    res: dict = {"a": a.a, "b": a.b, "cells": {}, "by_mismatch": {}, "pairing_check": {}}
    for pol in CHECK_POLICIES:
        s = m[m.policy == pol]
        if len(s):
            res["pairing_check"][pol] = float(max(np.abs(s[f"{k}_b"] - s[f"{k}_a"]).max() for k in KEYS))
    for pol in MODEL_POLICIES:
        s = m[m.policy == pol]
        res["cells"][pol] = {k: {"a": float(s[f"{k}_a"].mean()), "b": float(s[f"{k}_b"].mean()),
                                 **boot_ci((s[f"{k}_b"] - s[f"{k}_a"]).to_numpy())} for k in KEYS}
        res["by_mismatch"][pol] = {mm: {k: boot_ci((g[f"{k}_b"] - g[f"{k}_a"]).to_numpy()) for k in KEYS}
                                   for mm, g in s.groupby("mismatch")}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=2), encoding="utf8")

    L = [f"# Closed loop: {a.label_b} minus {a.label_a}", "",
         f"Paired on preset x mismatch x seed (closed_loop.py protocol, true outcomes). "
         f"A = `{a.a}`, B = `{a.b}`. Brackets: paired-bootstrap 95 % CI over cells. "
         f"Pairing check (oracle policies, must be 0): "
         + ", ".join(f"{p} max |diff| {v:.3g}" for p, v in res["pairing_check"].items()), "",
         "| policy | cells | met A | met B | met B - A (pp) | GPA B - A | drift B - A | shortfall B - A (pp) |",
         "|---|---|---|---|---|---|---|---|"]
    for pol, c in res["cells"].items():
        f, v, dr, sh = (c[k] for k in KEYS)
        L.append(f"| {pol} | {f['n']} | {f['a']:.4f} | {f['b']:.4f} | "
                 f"{100 * f['mean']:+.2f} [{100 * f['lo']:+.2f}, {100 * f['hi']:+.2f}] | "
                 f"{v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] | "
                 f"{dr['mean']:+.5f} [{dr['lo']:+.5f}, {dr['hi']:+.5f}] | "
                 f"{sh['mean']:+.3f} [{sh['lo']:+.3f}, {sh['hi']:+.3f}] |")
    L += ["", "## Share meeting C_req, B - A (pp) by mismatch level", "",
          "| policy | " + " | ".join(next(iter(res["by_mismatch"].values())).keys()) + " |",
          "|---|" + "---|" * len(next(iter(res["by_mismatch"].values())))]
    for pol, g in res["by_mismatch"].items():
        L.append(f"| {pol} | " + " | ".join(
            f"{100 * x['frac_met']['mean']:+.2f} [{100 * x['frac_met']['lo']:+.2f}, {100 * x['frac_met']['hi']:+.2f}]"
            for x in g.values()) + " |")
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
