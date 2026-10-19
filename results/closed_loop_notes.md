## Headline (read this first)

Setup: 4 presets x 4 mismatch levels x 20 seeds x 5 policies, 40 segments per pass, C_req = 30%,
simulator-internal evaluation with the "noisy" verifier (error calibrated per coverage band to the B0
synthetic-card results). Headline model is the learned `SprayModel` (WP-10); `oracle_nominal` (the
mismatch-free simulator physics) is the secondary comparison. Numbers are for mismatch low+med+high pooled.

| Proposal criterion (Table 1, "Closed loop") | Policy | Measured (95% CI) | Verdict |
|---|---|---|---|
| Recovers >= 50% of the open-to-oracle gap in requirement-met segments | fast loop | 58% [38, 69] | met on the point estimate; the CI includes values below 50% |
| | fast + slow loop | -6% [-44, 17] | missed |
| <= 10% more volume than oracle | fast loop | +15.2% [+13.7, +16.9] | missed |
| | fast + slow loop | +6.2% [+5.2, +7.2] | met |
| 20 seeds, 95% CI | | 20 seeds, paired-bootstrap CIs | done |
| **Both conditions together** | none of the policies | fast loop meets recovery but not volume; fast + slow meets volume but not recovery | **MISSED** |

With `oracle_nominal` the fast loop gives the same picture (60% [43, 71] recovery, +13.8% volume), and
`fast_slow` is identical to `fast` by construction because that model has nothing to refit.

What drives the result:

* The fast loop works as designed. Requirement-met fraction under mismatch stays at 0.94 (low 0.947,
  med 0.943, high 0.932) while open loop falls from 0.954 (low) to 0.864 (med) and 0.762 (high), and the best
  fixed setting falls to 0.61 at high. Per preset the recovery is 69% (calm_morning), 54% (crosswind_shift),
  54% (gusty_afternoon) and 48% (hot_dry, CI [20, 66]).
* The volume criterion is missed mostly for a reason that is not the feedback. With no mismatch at all the
  fast loop already uses +12.2% more volume than the oracle, because the oracle sees the per-segment random
  spread factor of the card (irreducible noise, sd about 5 pp of coverage) and needs no safety margin,
  while every model-based policy must keep a margin from its prediction interval. Only about 3 percentage
  points of the +15.2% are attributable to correcting the hidden mismatch. The same irreducible
  noise bounds how much of the residual any bias correction can remove (raw residual variance stays about
  27-29 pp^2 inside a pass).
* The open-loop policy is already fairly robust at low mismatch (0.954 met) because of that margin, so the
  open-to-oracle gap at low mismatch is only 4.6 pp and gap recovery there is not statistically
  distinguishable from zero. The criterion is only informative at med and high mismatch (13.6 and 23.8 pp gap).
* The slow loop as implemented (refit `SprayModel` on the measured cards of the previous passes, record weight
  30) removes the average model error (mean raw residual at high mismatch -6.0 pp in pass 1, -1.8 pp in
  pass 3) and cuts volume, but it hurts reliability: requirement-met drops to 0.85 (vs 0.94 for the fast
  loop) and the raw residual variance rises from about 27 to about 33 pp^2. Diagnostic
  (`experiments/slow_loop_diagnostic.py`, 6 seeds, no mismatch): after the refit the 80% prediction
  interval narrows (mean half-width 8.0 to 5.2 pp) and the true coverage falls below its lower bound in 21 to
  27% of segments instead of 7.5%, and MAE rises from 4.7 to 5.0-5.1 pp. The most plausible cause is that a few
  dozen noisy, heavily weighted records are fitted too closely (and the conformal calibration is not
  re-done on them); this is a hypothesis, not a tested mechanism. The weight ablation is monotone:
  record weight 1 gives 36% recovery, 5 gives 10%, 30 gives -20% (preset gusty_afternoon, med). The
  suggested weight of 30 was used in the grid as instructed and is the worst of the three; the slow loop in this
  form should be reported as not helping, and it would need a better update rule (shrinkage,
  recalibration of the interval, or a bias-only update) to be useful.
* Ablations (gusty_afternoon, med, 20 seeds, fast loop): alpha 0.1 / 0.3 / 0.6 gives 45% / 59% / 54%
  recovery at +13.0% / +17.9% / +20.5% volume, so a smaller alpha is cheaper at a small loss of reliability. A
  single global bias recovers more (68%) than per-regime biases (59%) at +19.8% volume vs +17.9%; with
  four regimes and 40 cards per pass, each regime sees few cards before the warm-up gate opens. Warm-up 0 / 2 / 4
  differ by less than the CI. Verifier: perfect 51%, noisy 59%, rendered B0 44% (only 5 seeds, CI [-200, 90]);
  the differences are inside the seed-to-seed spread, so measurement error is not what limits the loop here.
* Retreat flags were never raised (0 in all 1,600 spray-model passes; a critical card, coverage < 15%, does
  not occur because the selector keeps a margin), so re-treatment is reported but not a discriminating metric
  in this grid.
