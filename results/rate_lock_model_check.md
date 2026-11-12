# Rate-lock model check

Question: is the default SprayModel v1 accurate along constant-application-rate lines?
Nozzle and rate are fixed, pressure moves, travel speed follows
(speed = s0 * GPA(P at s0) / target, s0 = 15 km/h, kept inside the spec speed range).
Simulation only.

## Setup
- Truth: `SpraySimulator.nominal_arrays` (mismatch-free, deterministic).
- Model: `models/spray_model_v1.joblib` through `integration.load_spray_model`.
- 300 conditions from `random_conditions(n, seed=30017)` (development seed).
- 155 lines (nozzle x target GPA on the 5 psi grid, at least 3 points).
  Each line uses 8 random conditions. 13200 predictions.
- Target GPA levels: 5, 10, 15, 20, 30.

## Error against the nominal truth
| group | rows | cov MAE (pp) | cov bias (pp) | drift MAE | drift bias |
|---|---|---|---|---|---|
| all | 13200 | 0.669 | +0.039 | 0.0027 | +0.0001 |

| GPA | rows | cov MAE (pp) | cov bias (pp) | drift MAE | drift bias |
|---|---|---|---|---|---|
| 5 | 1648 | 0.415 | +0.138 | 0.0037 | -0.0001 |
| 10 | 3112 | 0.534 | +0.088 | 0.0029 | +0.0003 |
| 15 | 3312 | 0.591 | -0.196 | 0.0025 | -0.0000 |
| 20 | 3000 | 0.741 | +0.233 | 0.0025 | -0.0000 |
| 30 | 2128 | 1.084 | -0.016 | 0.0020 | +0.0001 |

| family | rows | cov MAE (pp) | cov bias (pp) | drift MAE | drift bias |
|---|---|---|---|---|---|
| 3D | 3800 | 0.769 | +0.029 | 0.0034 | +0.0001 |
| DGX40 | 1256 | 0.310 | +0.041 | 0.0004 | +0.0003 |
| TP65 | 1488 | 0.905 | +0.025 | 0.0035 | -0.0001 |
| ULD | 4040 | 0.514 | +0.073 | 0.0016 | +0.0002 |
| XR | 2616 | 0.800 | +0.008 | 0.0037 | -0.0003 |

## Reference (general test set)
Source: `results/model_v1.json`, LightGBM v1, random split (found).
- Coverage MAE vs sampled outcomes: 2.844 pp (noise floor about 2.77 pp, so this is mostly sampling noise).
- Coverage MAE vs the expected (mismatch-free) value: 0.643 pp. This is the fair reference, because the truth here is also the expected value.
- Drift MAE: 0.0030.
- Rate-lock coverage MAE is 1.04 times the vs-expected reference.

## Ranking along pressure (same condition, nozzle and GPA)
- Spearman correlation, predicted vs true coverage: mean 0.873, median 0.977, 10th percentile 0.640 (1240 lines with variation).
- Among 595 lines with a predicted-feasible point (cov >= 30), the predicted-lowest-drift point is truly feasible in 89.9%.
- When it is feasible, its true drift exceeds the best truly feasible point by 0.0012 on average.

## Verdict
Retrain not needed. Coverage MAE on rate-lock lines is 0.67 pp against 0.64 pp on the general test set (vs expected), a ratio of 1.04. Bias is +0.04 pp. Within a line, pressure ranking is weaker than hoped (mean Spearman 0.87), and the drift-minimising feasible pick is truly feasible 90% of the time. Not tested: closed-loop performance with a rate-lock selector, hidden mismatch (gust, canopy, wear) on these lines, other s0 values, other seeds, boom-height changes, the XGBoost v2 or hybrid models, and any field data.
