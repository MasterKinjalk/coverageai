## Verdict

**Circularity.** The data, the plant and the exact physics base are the same simulator equations (the base lacks only the hidden mismatch factors and uses the mean spread factor). The `physics (exact base)` and `hybrid (exact base)` rows are therefore an upper bound and say nothing about real fields. The informative rows are the misspecified bases (evaporation, drift threshold, spread factor and VMD each off by 10 to 30 %).

**Offline.** With a misspecified base, the learned residual removes 94 to 99 % of the excess coverage and drift error (all four splits, both presets). Coverage MAE vs the expected coverage falls from 4.3 / 4.8 pp (physics alone, optimistic / pessimistic) to 0.31 / 0.30 pp (hybrid), against 0.64 pp for SprayModel. The 80 % interval holds (0.803 in band, vs 0.553 for misspecified physics alone).

**Selector (200 scenarios, no mismatch).** Misspecified physics alone meets C_req in 44.0 % of scenarios (sampled; 49.0 % expected), the hybrid on the same wrong base in 93.5 % (94.6 % expected), SprayModel in 92.5 % (93.9 %). Hybrid vs SprayModel: +0.7 pp [0.0, +1.4] expected success, +0.5 % [-0.9, +1.9] volume.

**Closed loop (20 seeds, mismatch med + high, noisy verifier; fair oracle met 0.953 ± 0.014 at 25.9 GPA).** Differences are paired bootstrap 95 % CIs against SprayModel on the same seeds.

| model | policy | met | gap recovery | volume vs oracle | Δ met vs spray (pp) |
|---|---|---|---|---|---|
| SprayModel | open | 0.833 | - | -9.1 % [-12.5, -5.2] | - |
| SprayModel | fast_slow | 0.957 | +104 % [+93, +113] | +5.1 % [+3.8, +6.7] | - |
| hybrid, misspecified base | open | 0.822 | - | -10.4 % [-13.8, -6.6] | -1.1 [-1.7, -0.5] |
| hybrid, misspecified base | fast | 0.921 | +76 % [+58, +85] | -0.2 % [-1.4, +1.0] | -0.6 [-1.2, -0.2] |
| hybrid, misspecified base | fast_slow | 0.950 | +98 % [+88, +106] | +3.3 % [+2.1, +4.5] | -0.7 [-1.1, -0.3] |
| physics only, misspecified | open | 0.296 | - | -14.3 % [-17.7, -10.6] | -53.7 [-57.8, -48.6] |
| physics only, misspecified | fast | 0.854 | +85 % [+82, +87] | -2.6 % [-4.8, -0.3] | -7.3 [-9.1, -5.6] |
| physics only, misspecified | fast_slow | 0.978 | +104 % [+102, +105] | +9.7 % [+8.5, +10.9] | +2.1 [+1.7, +2.5] |

1. The expected recommendation holds in the open loop and the fast loop. With a wrong physics base, the hybrid meets the target in 0.822 of sections (open) against 0.296 for physics alone, and 0.921 against 0.854 with the fast loop. Pure ML and the hybrid are close: on med + high pooled the hybrid is 0.6 to 1.1 pp lower in every policy (CIs exclude zero; within the high-mismatch subgroup the fast and fast_slow CIs touch zero), at 1.4 to 1.8 % less volume. This is a small deficit, not parity, and not an advantage.
2. The recommendation does not hold after the slow loop. With `fast_slow`, misspecified physics alone reaches 0.978 met, above SprayModel (+2.1 pp [+1.7, +2.5]) and above the hybrid. It gets there by spraying more: +9.7 % volume vs the fair oracle (+4.3 % [+3.8, +4.9] vs SprayModel), against +3.3 % for the hybrid. The per-regime offset of the slow loop absorbs a large share of a physics error that is mostly a smooth bias. The hybrid advantage over physics alone is therefore in the quality of the first pass and in volume, not in final success after five passes.
3. Gap recovery of the hybrid on a wrong base (98 % [88, 106]) is statistically equal to SprayModel (104 % [93, 113]) and meets the proposal target (>= 50 % recovery, <= +10 % volume vs the oracle).
4. The exact-base hybrid is not better than SprayModel either (-1.4 pp [-1.8, -1.0] met, -2.9 % volume with `fast_slow`). The physics prior does not add closed-loop success over a model trained on 50,000 simulator rows.
5. Under hidden mismatch, every model except misspecified physics alone loses 12 to 14 pp of sections in the open loop (0.81 to 0.83 vs 0.953 for the oracle). The residual learned on the no-mismatch dataset does not address this; the feedback loops do.

**Recommendation.** Keep SprayModel as the default. The hybrid is a valid alternative (same gap recovery, slightly less volume, 0.6 to 1.1 pp fewer sections on target) and is the choice when a trusted physics model exists but is wrong by 10 to 30 %: it removes 94 to 99 % of that error from the data and needs no slow loop to be usable. It is not a closed-loop success gain over pure ML, and the circularity caveat applies: whether the physics-first prior helps on real fields is untested.
