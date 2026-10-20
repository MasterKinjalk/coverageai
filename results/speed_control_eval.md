# Travel-speed control evaluation

N=200 random no-mismatch scenarios, C_req=30%, speed grid 8-30 km/h step 2, clipped to the spec range and to +-band around the scenario speed. 'off' = v1.0 behaviour (speed uncontrolled). Selector model: oracle simulator (isolates the selector); 'spray': learned SprayModel selects, simulator scores.

## Open loop (oracle)

| config | success | mean drift | mean GPA | infeasible | speed changed | mean v/v0 |
|---|---|---|---|---|---|---|
| off | 0.385 | 0.0106 | 17.27 | 0.000 | 0.00 | 1.00 |
| on_band_0.15 | 0.305 | 0.0087 | 16.56 | 0.000 | 0.69 | 1.00 |
| on_band_0.25 | 0.340 | 0.0085 | 16.28 | 0.000 | 0.79 | 1.02 |
| on_band_0.4 | 0.275 | 0.0074 | 16.32 | 0.000 | 0.84 | 1.07 |

## Open loop (spray)

| config | success | mean drift | mean GPA | infeasible | speed changed | mean v/v0 |
|---|---|---|---|---|---|---|
| off | 0.935 | 0.0131 | 24.78 | 0.000 | 0.00 | 1.00 |
| on_band_0.15 | 0.940 | 0.0125 | 23.83 | 0.000 | 0.67 | 0.99 |
| on_band_0.25 | 0.900 | 0.0116 | 23.28 | 0.000 | 0.86 | 0.98 |
| on_band_0.4 | 0.915 | 0.0111 | 22.83 | 0.000 | 0.91 | 0.95 |

## Closed loop (oracle_nominal model, perfect verifier, mismatch med, 4 presets, 3 seeds, band 0.25)

| policy | frac_met | shortfall pp | GPA | drift |
|---|---|---|---|---|
| off/oracle | 1.000 | 0.00 | 21.67 | 0.0129 |
| off/fast | 0.954 | 0.07 | 24.19 | 0.0153 |
| on_band_0.25/oracle | 1.000 | 0.00 | 19.93 | 0.0135 |
| on_band_0.25/fast | 0.917 | 0.13 | 23.02 | 0.0141 |
