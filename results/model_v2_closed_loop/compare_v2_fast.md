# Closed loop: v2_fast (XGBoost, GPU, cost cap 1800) minus v1 (LightGBM)

Paired on preset x mismatch x seed (closed_loop.py protocol, true outcomes). A = `results/model_v2_closed_loop/v1`, B = `results/model_v2_closed_loop/v2_fast`. Brackets: paired-bootstrap 95 % CI over cells. Pairing check (oracle policies, must be 0): oracle max |diff| 0, oracle_card max |diff| 0

| policy | cells | met A | met B | met B - A (pp) | GPA B - A | drift B - A | shortfall B - A (pp) |
|---|---|---|---|---|---|---|---|
| open | 320 | 0.8967 | 0.9009 | +0.42 [+0.09, +0.74] | -0.101 [-0.141, -0.061] | +0.00027 [+0.00017, +0.00039] | -0.011 [-0.017, -0.004] |
| fast | 320 | 0.9366 | 0.9438 | +0.73 [+0.42, +1.03] | +0.067 [+0.011, +0.128] | +0.00049 [+0.00040, +0.00059] | -0.019 [-0.025, -0.014] |
| fast_slow | 320 | 0.9534 | 0.9574 | +0.41 [+0.12, +0.68] | +0.047 [-0.014, +0.109] | +0.00053 [+0.00044, +0.00063] | -0.010 [-0.015, -0.005] |

## Share meeting C_req, B - A (pp) by mismatch level

| policy | high | low | med | none |
|---|---|---|---|---|
| open | -0.06 [-0.94, +0.81] | +0.37 [-0.09, +0.81] | +0.25 [-0.31, +0.81] | +1.13 [+0.53, +1.75] |
| fast | +0.41 [-0.25, +1.06] | +1.00 [+0.37, +1.59] | +0.72 [+0.16, +1.28] | +0.78 [+0.19, +1.41] |
| fast_slow | +0.37 [-0.22, +0.91] | +0.84 [+0.22, +1.47] | -0.41 [-0.94, +0.09] | +0.81 [+0.28, +1.37] |
