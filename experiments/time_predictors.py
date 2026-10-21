"""Idle-CPU timing of predict_batch over all catalog candidates for saved SprayModel files.

Usage: uv run --no-sync python experiments/time_predictors.py models/spray_model_v1.joblib ...
Writes results/predictor_timing.json. Measure on an idle machine (timings scale with load).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coverageai import catalog                         # noqa: E402
from coverageai.contracts import Conditions            # noqa: E402
from coverageai.model.predictor import SprayModel      # noqa: E402

X = Conditions(3.0, 90.0, 0.0, 25.0, 50.0, 19.0, 0.5)


def main() -> None:
    cands = catalog.candidates()
    out = {}
    for p in sys.argv[1:]:
        m = SprayModel.load(p)
        for _ in range(5):
            m.predict_batch([X] * len(cands), cands)
        ts = []
        for _ in range(300):
            t = time.perf_counter()
            m.predict_batch([X] * len(cands), cands)
            ts.append(time.perf_counter() - t)
        out[Path(p).name] = {"n_candidates": len(cands), "median_ms": float(np.median(ts) * 1e3),
                             "p95_ms": float(np.percentile(ts, 95) * 1e3), "reps": len(ts)}
        print(Path(p).name, {k: round(v, 2) for k, v in out[Path(p).name].items()})
    (ROOT / "results" / "predictor_timing.json").write_text(json.dumps(out, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
