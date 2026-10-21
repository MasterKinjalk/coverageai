# Closed loop: v2 (XGBoost, GPU) minus v1 (LightGBM)

Paired on preset x mismatch x seed (closed_loop.py protocol, true outcomes). A = `results/model_v2_closed_loop/v1`, B = `results/model_v2_closed_loop/v2`. Brackets: paired-bootstrap 95 % CI over cells. Pairing check (oracle policies, must be 0): oracle max |diff| 0, oracle_card max |diff| 0

| policy | cells | met A | met B | met B - A (pp) | GPA B - A | drift B - A | shortfall B - A (pp) |
|---|---|---|---|---|---|---|---|
| open | 320 | 0.8967 | 0.8970 | +0.02 [-0.30, +0.33] | -0.141 [-0.176, -0.107] | +0.00011 [+0.00001, +0.00022] | -0.004 [-0.011, +0.003] |
| fast | 320 | 0.9366 | 0.9412 | +0.46 [+0.16, +0.77] | -0.031 [-0.084, +0.024] | +0.00033 [+0.00024, +0.00043] | -0.015 [-0.021, -0.009] |
| fast_slow | 320 | 0.9534 | 0.9556 | +0.23 [-0.07, +0.51] | -0.066 [-0.122, -0.006] | +0.00039 [+0.00031, +0.00048] | -0.006 [-0.011, -0.001] |

## Share meeting C_req, B - A (pp) by mismatch level

| policy | high | low | med | none |
|---|---|---|---|---|
| open | -0.53 [-1.37, +0.31] | +0.00 [-0.47, +0.47] | +0.03 [-0.56, +0.63] | +0.59 [+0.03, +1.22] |
| fast | +0.22 [-0.50, +0.91] | +0.62 [+0.00, +1.22] | +0.37 [-0.19, +0.97] | +0.62 [+0.03, +1.22] |
| fast_slow | +0.03 [-0.59, +0.59] | +0.50 [-0.13, +1.12] | -0.41 [-0.97, +0.13] | +0.78 [+0.25, +1.37] |
