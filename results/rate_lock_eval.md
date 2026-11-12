# Fixed application rate: evaluation (simulation only)

Everything here is simulated. Nothing is field validation.

## Setup

- Seeds 0 to 19 (evaluation seeds), 40 sections, mismatch `med`, target coverage 30 %.
- Scenario presets (all 6): calm_morning, gusty_afternoon, hot_dry, crosswind_shift, marginal_limits, inversion_evening. The paired unit is the seed; each seed value is the mean over presets, so the CIs resample seeds.
- Stack: `build_modules("spray", "noisy")` (LightGBM v1, noisy reader, rule feedback, library defaults).
- The six setups were chosen by hand after a small pilot scan (seeds 0 and 1, two presets, rate_fast only) to span families and difficulty. Those two seeds are also evaluation seeds, so the choice of setups is not independent of the evaluation data.
- Metrics use the TRUE outcome of the simulator, not the card reading.
- Total passes: 3600. Runtime: 0.7 min.

Operator setups. The held setting uses the start pressure that gives the rate at the operator speed.

| # | nozzle | rate gpa | speed km/h | start pressure psi |
|---|---|---|---|---|
| 1 | TP65-06 | 20 | 16 | 49.9 |
| 2 | 3D-04 | 20 | 12 | 63.0 |
| 3 | ULD-03 | 15 | 12 | 62.1 |
| 4 | XR-04 | 15 | 12 | 35.1 |
| 5 | ULD-04 | 15 | 16 | 63.0 |
| 6 | DGX40-04 | 10 | 20 | 43.6 |

Policies:

- `held`: the operator's setting for the whole field.
- `rate_open`: open loop, nozzle and rate locked, pressure (speed follows) chosen per section.
- `rate_fast`: as `rate_open` with the fast card-feedback loop.
- `rate_fast_boom`: as `rate_fast` plus boom-height control.
- `free_fast`: REFERENCE, not a peer. Free nozzle and free rate, the existing controller. It can raise the rate, so it wins on coverage by construction.

## Results

Mean over seeds with 95 % bootstrap CI in brackets. Shares in percent.

### Setup 1: TP65-06 20 gpa @ 16 km/h

| policy | true cov >= 30 % (%) | mean drift (%) | set rate gpa | set rate within 5 % (%) | true rate within 5 % (%) | unreachable (%) |
|---|---|---|---|---|---|---|
| held | 68.2 [62.8, 72.9] | 10.9 [10.6, 11.1] | 20.00 [20.00, 20.00] | 100.0 [100.0, 100.0] | 50.0 [25.0, 70.0] | 0.0 [0.0, 0.0] |
| rate_open | 72.9 [67.2, 78.0] | 10.9 [10.6, 11.1] | 20.51 [20.49, 20.52] | 100.0 [100.0, 100.0] | 24.2 [12.6, 34.9] | 19.9 [18.8, 21.0] |
| rate_fast | 77.2 [72.8, 80.9] | 11.0 [10.7, 11.3] | 20.74 [20.69, 20.80] | 100.0 [100.0, 100.0] | 9.7 [4.4, 15.4] | 54.0 [47.7, 61.1] |
| rate_fast_boom | 80.1 [75.7, 83.9] | 10.3 [10.1, 10.6] | 20.74 [20.69, 20.80] | 100.0 [100.0, 100.0] | 9.7 [4.4, 15.3] | 54.5 [48.2, 61.5] |
| free_fast | 95.4 [94.3, 96.5] | 1.8 [1.8, 1.9] | 25.88 [25.08, 26.81] | 5.0 [3.2, 7.0] | 1.1 [0.6, 1.6] | 0.0 [0.0, 0.0] |

### Setup 2: 3D-04 20 gpa @ 12 km/h

| policy | true cov >= 30 % (%) | mean drift (%) | set rate gpa | set rate within 5 % (%) | true rate within 5 % (%) | unreachable (%) |
|---|---|---|---|---|---|---|
| held | 76.8 [71.8, 81.1] | 11.2 [10.9, 11.4] | 20.00 [20.00, 20.00] | 100.0 [100.0, 100.0] | 50.0 [25.0, 70.0] | 0.0 [0.0, 0.0] |
| rate_open | 81.4 [76.0, 86.2] | 10.4 [10.2, 10.6] | 20.16 [20.15, 20.17] | 100.0 [100.0, 100.0] | 41.7 [21.2, 58.9] | 7.0 [5.9, 8.1] |
| rate_fast | 87.5 [84.1, 90.5] | 11.4 [10.9, 11.9] | 20.56 [20.49, 20.64] | 100.0 [100.0, 100.0] | 17.3 [8.4, 26.7] | 41.8 [33.1, 51.5] |
| rate_fast_boom | 93.5 [91.5, 95.3] | 9.8 [8.8, 10.8] | 20.35 [20.26, 20.45] | 100.0 [100.0, 100.0] | 26.9 [13.4, 41.0] | 22.4 [15.1, 31.0] |
| free_fast | 95.4 [94.3, 96.5] | 1.8 [1.8, 1.9] | 25.88 [25.08, 26.81] | 5.0 [3.2, 7.0] | 1.1 [0.6, 1.6] | 0.0 [0.0, 0.0] |

### Setup 3: ULD-03 15 gpa @ 12 km/h

| policy | true cov >= 30 % (%) | mean drift (%) | set rate gpa | set rate within 5 % (%) | true rate within 5 % (%) | unreachable (%) |
|---|---|---|---|---|---|---|
| held | 5.9 [3.4, 8.5] | 2.3 [2.3, 2.4] | 15.00 [15.00, 15.00] | 100.0 [100.0, 100.0] | 50.0 [25.0, 70.0] | 0.0 [0.0, 0.0] |
| rate_open | 62.7 [56.8, 68.0] | 8.1 [7.9, 8.3] | 15.75 [15.75, 15.75] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 84.5 [82.8, 86.1] |
| rate_fast | 62.7 [56.8, 67.9] | 8.1 [7.9, 8.3] | 15.74 [15.74, 15.75] | 100.0 [100.0, 100.0] | 0.2 [0.0, 0.7] | 94.2 [91.7, 96.5] |
| rate_fast_boom | 72.3 [66.3, 77.6] | 5.2 [5.0, 5.3] | 15.74 [15.73, 15.75] | 100.0 [100.0, 100.0] | 0.4 [0.0, 0.9] | 92.3 [88.9, 95.4] |
| free_fast | 95.4 [94.3, 96.5] | 1.8 [1.8, 1.9] | 25.88 [25.08, 26.81] | 0.1 [0.0, 0.3] | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.0] |

### Setup 4: XR-04 15 gpa @ 12 km/h

| policy | true cov >= 30 % (%) | mean drift (%) | set rate gpa | set rate within 5 % (%) | true rate within 5 % (%) | unreachable (%) |
|---|---|---|---|---|---|---|
| held | 14.1 [10.8, 17.3] | 10.6 [10.4, 10.9] | 15.00 [15.00, 15.00] | 100.0 [100.0, 100.0] | 50.0 [25.0, 70.0] | 0.0 [0.0, 0.0] |
| rate_open | 33.6 [29.4, 37.7] | 19.6 [19.1, 20.2] | 15.62 [15.61, 15.63] | 100.0 [100.0, 100.0] | 8.8 [4.7, 12.7] | 73.9 [72.1, 75.5] |
| rate_fast | 34.4 [30.3, 38.4] | 19.7 [19.1, 20.2] | 15.71 [15.70, 15.72] | 100.0 [100.0, 100.0] | 1.8 [0.7, 2.9] | 90.5 [87.9, 93.0] |
| rate_fast_boom | 48.4 [44.1, 52.4] | 22.8 [22.3, 23.3] | 15.67 [15.64, 15.69] | 100.0 [100.0, 100.0] | 4.1 [1.7, 6.7] | 82.1 [78.0, 86.2] |
| free_fast | 95.4 [94.3, 96.5] | 1.8 [1.8, 1.9] | 25.88 [25.08, 26.81] | 0.1 [0.0, 0.3] | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.0] |

### Setup 5: ULD-04 15 gpa @ 16 km/h

| policy | true cov >= 30 % (%) | mean drift (%) | set rate gpa | set rate within 5 % (%) | true rate within 5 % (%) | unreachable (%) |
|---|---|---|---|---|---|---|
| held | 0.0 [0.0, 0.0] | 0.1 [0.1, 0.1] | 15.00 [15.00, 15.00] | 100.0 [100.0, 100.0] | 50.0 [25.0, 70.0] | 0.0 [0.0, 0.0] |
| rate_open | 26.3 [20.3, 31.8] | 1.6 [1.5, 1.6] | 15.75 [15.75, 15.75] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] |
| rate_fast | 26.3 [20.3, 31.8] | 1.6 [1.5, 1.6] | 15.75 [15.75, 15.75] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] |
| rate_fast_boom | 28.8 [22.7, 34.3] | 1.0 [1.0, 1.1] | 15.75 [15.75, 15.75] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] |
| free_fast | 95.4 [94.3, 96.5] | 1.8 [1.8, 1.9] | 25.88 [25.08, 26.81] | 0.1 [0.0, 0.3] | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.0] |

### Setup 6: DGX40-04 10 gpa @ 20 km/h

| policy | true cov >= 30 % (%) | mean drift (%) | set rate gpa | set rate within 5 % (%) | true rate within 5 % (%) | unreachable (%) |
|---|---|---|---|---|---|---|
| held | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 10.00 [10.00, 10.00] | 100.0 [100.0, 100.0] | 50.0 [25.0, 70.0] | 0.0 [0.0, 0.0] |
| rate_open | 0.0 [0.0, 0.0] | 0.1 [0.1, 0.1] | 10.50 [10.50, 10.50] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] |
| rate_fast | 0.0 [0.0, 0.0] | 0.1 [0.1, 0.1] | 10.50 [10.50, 10.50] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] |
| rate_fast_boom | 0.0 [0.0, 0.0] | 0.1 [0.1, 0.1] | 10.50 [10.50, 10.50] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] |
| free_fast | 95.4 [94.3, 96.5] | 1.8 [1.8, 1.9] | 25.88 [25.08, 26.81] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

### Pooled over the 6 setups

| policy | true cov >= 30 % (%) | mean drift (%) | set rate gpa | set rate within 5 % (%) | true rate within 5 % (%) | unreachable (%) |
|---|---|---|---|---|---|---|
| held | 27.5 [24.9, 29.9] | 5.9 [5.7, 6.0] | 15.83 [15.83, 15.83] | 100.0 [100.0, 100.0] | 50.0 [25.0, 70.0] | 0.0 [0.0, 0.0] |
| rate_open | 46.2 [41.7, 50.2] | 8.4 [8.2, 8.6] | 16.38 [16.38, 16.39] | 100.0 [100.0, 100.0] | 12.4 [6.5, 17.8] | 64.2 [63.4, 64.9] |
| rate_fast | 48.0 [44.1, 51.5] | 8.6 [8.4, 8.8] | 16.50 [16.48, 16.53] | 100.0 [100.0, 100.0] | 4.9 [2.3, 7.5] | 80.1 [76.9, 83.6] |
| rate_fast_boom | 53.8 [50.2, 57.1] | 8.2 [7.9, 8.5] | 16.46 [16.43, 16.49] | 100.0 [100.0, 100.0] | 6.8 [3.3, 10.6] | 75.2 [71.9, 78.9] |
| free_fast | 95.4 [94.3, 96.5] | 1.8 [1.8, 1.9] | 25.88 [25.08, 26.81] | 1.7 [1.1, 2.4] | 0.4 [0.2, 0.6] | 0.0 [0.0, 0.0] |

## Paired differences

Mean paired difference over seeds with 95 % CI, in percentage points (drift in points of emitted volume).

| setup | comparison | d share >= 30 % | d drift | d set-rate-within-5 % |
|---|---|---|---|---|
| TP65-06 20 gpa @ 16 km/h | rate_fast - held | 9.0 [7.7, 10.3] | 0.1 [0.1, 0.1] | 0.0 [0.0, 0.0] |
| TP65-06 20 gpa @ 16 km/h | rate_fast_boom - rate_fast | 2.9 [2.3, 3.5] | -0.7 [-0.7, -0.6] | 0.0 [0.0, 0.0] |
| TP65-06 20 gpa @ 16 km/h | rate_open - rate_fast | -4.3 [-5.8, -2.8] | -0.2 [-0.2, -0.1] | 0.0 [0.0, 0.0] |
| 3D-04 20 gpa @ 12 km/h | rate_fast - held | 10.8 [9.1, 12.5] | 0.2 [-0.2, 0.7] | 0.0 [0.0, 0.0] |
| 3D-04 20 gpa @ 12 km/h | rate_fast_boom - rate_fast | 6.0 [4.5, 7.7] | -1.6 [-2.3, -0.9] | 0.0 [0.0, 0.0] |
| 3D-04 20 gpa @ 12 km/h | rate_open - rate_fast | -6.1 [-8.3, -4.2] | -1.0 [-1.4, -0.6] | 0.0 [0.0, 0.0] |
| ULD-03 15 gpa @ 12 km/h | rate_fast - held | 56.8 [52.7, 60.3] | 5.7 [5.6, 5.9] | 0.0 [0.0, 0.0] |
| ULD-03 15 gpa @ 12 km/h | rate_fast_boom - rate_fast | 9.7 [8.8, 10.5] | -2.9 [-3.0, -2.8] | 0.0 [0.0, 0.0] |
| ULD-03 15 gpa @ 12 km/h | rate_open - rate_fast | 0.1 [-0.1, 0.2] | -0.0 [-0.0, -0.0] | 0.0 [0.0, 0.0] |
| XR-04 15 gpa @ 12 km/h | rate_fast - held | 20.3 [19.2, 21.5] | 9.0 [8.6, 9.4] | 0.0 [0.0, 0.0] |
| XR-04 15 gpa @ 12 km/h | rate_fast_boom - rate_fast | 14.0 [13.1, 15.0] | 3.1 [2.8, 3.4] | 0.0 [0.0, 0.0] |
| XR-04 15 gpa @ 12 km/h | rate_open - rate_fast | -0.8 [-1.1, -0.5] | -0.1 [-0.1, -0.1] | 0.0 [0.0, 0.0] |
| ULD-04 15 gpa @ 16 km/h | rate_fast - held | 26.3 [20.3, 31.8] | 1.5 [1.4, 1.5] | 0.0 [0.0, 0.0] |
| ULD-04 15 gpa @ 16 km/h | rate_fast_boom - rate_fast | 2.4 [2.0, 2.9] | -0.6 [-0.6, -0.5] | 0.0 [0.0, 0.0] |
| ULD-04 15 gpa @ 16 km/h | rate_open - rate_fast | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| DGX40-04 10 gpa @ 20 km/h | rate_fast - held | 0.0 [0.0, 0.0] | 0.1 [0.1, 0.1] | 0.0 [0.0, 0.0] |
| DGX40-04 10 gpa @ 20 km/h | rate_fast_boom - rate_fast | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| DGX40-04 10 gpa @ 20 km/h | rate_open - rate_fast | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| pooled | rate_fast - held | 20.5 [19.2, 21.7] | 2.8 [2.7, 2.9] | 0.0 [0.0, 0.0] |
| pooled | rate_fast_boom - rate_fast | 5.8 [5.4, 6.2] | -0.4 [-0.6, -0.3] | 0.0 [0.0, 0.0] |
| pooled | rate_open - rate_fast | -1.9 [-2.5, -1.3] | -0.2 [-0.3, -0.1] | 0.0 [0.0, 0.0] |

## Hidden wear

The simulator multiplies the flow by a hidden wear factor drawn once per pass (here 1.002 to 1.096). The true rate is the set rate times this factor, read from a deepcopy of the simulator after the pass (the live plant is not touched). The column "true rate within 5 %" shows how often the delivered rate stays in the band. The controller cannot see this factor, so it cannot hold the true rate. The band tier picks rates up to +5 % above target, and wear adds more on top, so the delivered rate is often outside the 5 % band even though the set rate is inside it.

## Verdict

- Pooled share at target: held 27.5 %, rate_open 46.2 %, rate_fast 48.0 %, rate_fast_boom 53.8 %, free_fast (reference) 95.4 %.
- rate_fast vs held: +20.5 pp, CI [+19.2, +21.7], excludes 0.
- rate_fast_boom vs rate_fast: +5.8 pp, CI [+5.4, +6.2], excludes 0.
- rate_open vs rate_fast: -1.9 pp, CI [-2.5, -1.3], excludes 0.
- The locked modes hold the SET rate by construction. Where the nozzle at that rate cannot reach 30 % at any pressure, the selector flags it (unreachable column), and no policy that keeps the rate can fix that. It is a limit of the nozzle and rate, not of the controller.
- Where free_fast beats the locked modes, the gap is the cost of keeping the rate fixed.

## What was not tested

- Image readers (B0, B1); only the noisy reader was used.
- Trained card feedback (D29, D31) and the learning (multi-pass) strategy.
- Targets other than 30 % coverage, other mismatch levels, other models (XGBoost v2, hybrid).
- Wind forecast and speed-only control.
- Other nozzle and rate combinations than the six above; they were chosen by hand.
- Nothing here is field data.
