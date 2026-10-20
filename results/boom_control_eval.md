# Boom-height control evaluation

N=200 random no-mismatch scenarios, C_req=30%. Heights [0.3, 0.4, 0.5, 0.6, 0.75] m and speeds [8.0, 12.0, 16.0, 20.0, 24.0, 28.0] km/h, each within +-25% of the scenario value, spec ranges, and (heights) above the family minimum overlap height H >= 1.3 * spacing / (2 tan(angle/2)): TP65 0.52 m, ULD 0.19 m, 3D 0.28 m, XR 0.23 m, DGX40 0.91 m. Scoring is paired: each variant's choice for scenario i is simulated with rng [seed, i]; CIs are paired bootstrap over scenarios of the per-scenario difference vs 'off'.

## Open loop (selector model: nominal_physics)

| variant | success | mean drift | mean GPA | height changed | h/h0 | speed changed | d drift vs off [95% CI] | d success vs off [95% CI] |
|---|---|---|---|---|---|---|---|---|
| off | 0.945 | 0.0133 | 24.83 | 0.00 | 1.00 | 0.00 | - | - |
| height | 0.945 | 0.0166 | 21.42 | 1.00 | 0.86 | 0.00 | +0.0033 [+0.0010, +0.0057] | +0.0000 [-0.0200, +0.0200] |
| speed | 0.940 | 0.0118 | 23.67 | 0.00 | 1.00 | 0.63 | -0.0015 [-0.0027, -0.0005] | -0.0050 [-0.0200, +0.0100] |
| speed+height | 0.925 | 0.0104 | 21.66 | 1.00 | 0.86 | 0.70 | -0.0029 [-0.0049, -0.0011] | -0.0200 [-0.0400, -0.0050] |

## Open loop (selector model: spray)

| variant | success | mean drift | mean GPA | height changed | h/h0 | speed changed | d drift vs off [95% CI] | d success vs off [95% CI] |
|---|---|---|---|---|---|---|---|---|
| off | 0.940 | 0.0131 | 24.78 | 0.00 | 1.00 | 0.00 | - | - |
| height | 0.950 | 0.0158 | 22.17 | 0.90 | 0.87 | 0.00 | +0.0027 [+0.0005, +0.0049] | +0.0100 [-0.0150, +0.0350] |
| speed | 0.940 | 0.0122 | 23.71 | 0.00 | 1.00 | 0.65 | -0.0010 [-0.0018, -0.0001] | +0.0000 [-0.0200, +0.0200] |
| speed+height | 0.935 | 0.0102 | 22.15 | 0.92 | 0.87 | 0.66 | -0.0029 [-0.0042, -0.0015] | -0.0050 [-0.0250, +0.0150] |

## Closed loop (oracle_nominal model, perfect verifier, 4 presets x 10 seeds, paired by preset/seed; 95% paired-bootstrap CI of the difference vs off; 'oracle' = fair oracle)

| policy/mismatch/variant | frac_met | shortfall pp | GPA | drift | height set | d frac_met | d drift | d GPA |
|---|---|---|---|---|---|---|---|---|
| fast/med/off | 0.934 | 0.09 | 24.13 | 0.0158 | 0.00 | - | - | - |
| fast/med/height | 0.918 | 0.13 | 21.93 | 0.0146 | 1.00 | -0.0162 [-0.0244, -0.0087] | -0.0012 [-0.0039, +0.0015] | -2.2026 [-2.7893, -1.6413] |
| fast/med/speed+height | 0.905 | 0.15 | 21.54 | 0.0137 | 1.00 | -0.0287 [-0.0362, -0.0206] | -0.0021 [-0.0045, +0.0003] | -2.5927 [-3.0943, -2.1093] |
| fast/high/off | 0.924 | 0.12 | 25.71 | 0.0198 | 0.00 | - | - | - |
| fast/high/height | 0.907 | 0.16 | 23.67 | 0.0165 | 1.00 | -0.0162 [-0.0262, -0.0062] | -0.0033 [-0.0062, -0.0005] | -2.0419 [-2.4612, -1.6292] |
| fast/high/speed+height | 0.889 | 0.19 | 22.90 | 0.0163 | 1.00 | -0.0344 [-0.0444, -0.0244] | -0.0035 [-0.0066, -0.0005] | -2.8117 [-3.3392, -2.3181] |
| oracle/med/off | 0.956 | 0.03 | 24.15 | 0.0153 | 0.00 | - | - | - |
| oracle/med/height | 0.951 | 0.04 | 22.66 | 0.0117 | 1.00 | -0.0050 [-0.0125, +0.0031] | -0.0036 [-0.0052, -0.0019] | -1.4932 [-1.8344, -1.1782] |
| oracle/med/speed+height | 0.934 | 0.05 | 22.08 | 0.0121 | 1.00 | -0.0213 [-0.0313, -0.0113] | -0.0032 [-0.0049, -0.0014] | -2.0698 [-2.4444, -1.7341] |
| oracle/high/off | 0.949 | 0.03 | 25.69 | 0.0190 | 0.00 | - | - | - |
| oracle/high/height | 0.948 | 0.04 | 24.39 | 0.0130 | 1.00 | -0.0019 [-0.0087, +0.0050] | -0.0060 [-0.0075, -0.0046] | -1.2959 [-1.5151, -1.0848] |
| oracle/high/speed+height | 0.922 | 0.06 | 23.85 | 0.0124 | 1.00 | -0.0269 [-0.0369, -0.0175] | -0.0066 [-0.0082, -0.0051] | -1.8317 [-2.0603, -1.6110] |
