## Headline (read this first)

Setup: 4 presets x 4 mismatch levels x 20 seeds x 6 policies, 40 segments per pass, C_req = 30%,
simulator-internal evaluation with the "noisy" verifier (error calibrated per coverage band to the B0
synthetic-card results). Headline model is the learned `SprayModel` (WP-10); `oracle_nominal` (the
mismatch-free simulator physics) is the secondary comparison. `fast_slow` runs 5 consecutive passes and
the 5th is scored; every policy is scored on the same hidden realization (seed s + 4036), so all
comparisons are paired. Criterion numbers pool low, med and high mismatch; brackets are 95% paired-bootstrap
CIs over the 20 seeds.

### What changed since the midterm

1. **Fair oracle (now the reference of the criterion).** Policy `oracle` knows the hidden mismatch of each
   segment exactly (AR(1) gust state, canopy interception, nozzle wear, read from the same random stream as
   the plant), but not the per-card WSP spread factor. That factor is drawn when the card is sprayed, so no
   controller can know it beforehand. The fair oracle predicts coverage at the mean spread factor, and
   its lower bound is the exact 10% quantile of the spread factor. The selector then applies the
   unchanged margin rule, which targets the same one-sided level as the learned model's 80% CQR interval.
   It therefore measures *perfect knowledge of the mismatch with the same risk policy*, which is what the
   criterion is about. The midterm oracle is kept as `oracle_card` (it also sees the spread factor and
   needs no margin) and is reported alongside. The fair oracle itself uses +12.6% [+11.2, +13.9] more
   volume than `oracle_card` (+9.8% with no mismatch). That is the cost of the irreducible card noise,
   and it is why the midterm volume criterion could not be met.
2. **Slow loop redesigned** (`feedback.ResidualCorrectedModel`, `loop.run_passes_slow(slow_mode="residual")`).
   The offline model is frozen. After every pass, all verified cards so far (x, u, measured C) are used to
   refit an additive coverage offset: residual = measured minus the frozen base prediction. The offset is
   fitted per fast-loop regime and shrunk with empirical Bayes toward a global offset and toward 0 (prior
   sd 5 pp). The selector and the fast loop see base + offset. The prediction interval is shifted, not
   narrowed. The fast-loop bias is reset at the start of each pass because it is now measured against
   the corrected model; the slow offset holds the cross-pass part and the bias only the pass-specific
   remainder. Refits are always against the frozen base, so corrections never compound. The midterm slow
   loop (LightGBM refit with record weight 30) is kept as `slow_mode="refit"` for the ablation.

### Criterion table (proposal Table 1, "Closed loop"; learned model)

| Criterion | Reference | Policy | Measured (95% CI) | Verdict | Cause / comment |
|---|---|---|---|---|---|
| Recovers >= 50% of the open-to-oracle gap in requirement-met segments | fair oracle | fast | 69% [44, 82] | met on the point estimate; CI crosses 50% | warm-up segments of each pass are uncorrected; hot_dry only 55% [4, 73] |
| | fair oracle | fast + slow | **96% [80, 107]** | **met (CI clears 50%)** | slow offset corrects the mean mismatch from segment 1 of each pass |
| <= 10% more volume than the oracle | fair oracle | fast | +1.4% [+0.2, +2.7] | met | |
| | fair oracle | fast + slow | **+4.1% [+2.8, +5.5]** | **met** | the learned model's interval is wider than the oracle's card-noise-only interval, so it keeps a slightly larger margin |
| **Both together, fair oracle** | | fast + slow | | **MET (both CIs clear the targets)** | fast loop alone: met on point estimates only |
| Recovery, card-aware oracle | oracle_card | fast / fast + slow | 46% [24, 57] / 63% [46, 73] | missed / met on point estimate | gap includes card noise that no controller can know |
| Volume, card-aware oracle | oracle_card | fast / fast + slow | +14.2% [+12.4, +15.9] / +17.2% [+15.4, +19.0] | missed | even the fair oracle is +12.6% over `oracle_card` |
| 20 seeds, 95% CI | | | 20 seeds, paired-bootstrap CIs | done | |

With `oracle_nominal` as the model the picture is the same. Against the fair oracle: fast 71% [48, 82] at
+0.2% [-1.0, +1.4], fast + slow 93% [80, 102] at +2.8% [+1.6, +4.0]. Against `oracle_card`: 49% / 64%
recovery at +12.8% / +15.7% volume. The slow loop now helps this model too: the frozen-physics model
cannot be refit, but it can carry an offset.

**Verdict: the closed-loop criterion is met against the fair oracle by the fast + slow loop, with both
confidence intervals clearing their targets. The fast loop alone meets it on point estimates, but its
recovery CI crosses 50%. Against the card-aware (midterm) oracle the criterion stays missed on volume,
for the structural reason in item 1.** The fair-oracle definition and the slow-loop design were fixed
before the evaluation grid was run. The slow-loop variants were compared only on dev seeds 100-109
(`experiments/slow_loop_diagnostic.py`), disjoint from the evaluation seeds 0-19. Nothing was tuned on
the criterion.

### Caveats that bound this result

* **The gap to recover is small.** Pooled open-to-fair-oracle gap is 8.3 pp (none 0.2, low 0.9, med 7.5,
  high 16.6 pp). The open loop is already quite reliable at none/low mismatch because of its interval margin,
  so the criterion is informative only at med and high. There, fast recovers 70% [36, 83] and 83% [72, 91],
  and fast + slow recovers 102% [84, 116] and 104% [96, 112].
* **Recovery above 100% does not mean the policy beat perfect information.** At med and high mismatch, fast + slow
  meets C_req slightly more often than the fair oracle (0.961 vs 0.959; 0.954 vs 0.947) because the learned
  model's interval is wider than pure card noise. It pays for that with volume (+3.8% and +6.4%). The fair
  oracle limits *information*, not reliability; the volume half of the criterion guards against buying
  reliability with volume.
* **The verdict depends on the oracle's risk level.** With a more conservative fair oracle (q = 0.05: met
  0.983, 25.9 GPA) recovery drops: fast 53% [30, 65], fast + slow 74% [56, 83]. Volume then favours the
  policies (-1.9% and +0.7%), so the criterion is still met, but with less margin on recovery. With q = 0.2 the
  oracle (0.888 met) barely beats the open loop and the ratio is not informative. q = 0.1 is the
  pre-registered choice because it matches the model's interval; the q = 0.05 row is the stricter check.
* Per preset, fast + slow recovers 89% to 102% of the gap at +2.9% to +5.6% volume. The fast loop alone ranges from 55% (hot_dry, CI
  [4, 73]) to 87% (calm_morning).
* The evaluation seed changed from s + 2018 (3 passes) to s + 4036 (5 passes), so policies that did not change
  moved: for example, fast-loop recovery against the card-aware oracle is now 46% [24, 57] (midterm 58% [38, 69]).
  The CIs overlap; this is how much a different hidden realization moves a 20-seed estimate.

### Slow loop: does the residual fall over passes?

Learned model, all presets, mean over 20 seeds. Residual = measured minus model in use (base + offset),
before the fast bias. Pass 1 has no offset yet.

| mismatch | metric | pass 1 | pass 2 | pass 3 | pass 4 | pass 5 |
|---|---|---|---|---|---|---|
| high | residual variance (pp^2) | 27.4 | 31.9 | 29.9 | 28.8 | 29.6 |
| high | residual MSE (pp^2) | 73.3 | 45.2 | 50.7 | 50.0 | 54.5 |
| high | mean residual (pp) | -6.02 | 0.03 | -0.13 | -2.00 | 1.11 |
| high | requirement met | 0.938 | 0.962 | 0.963 | 0.953 | 0.954 |
| med | residual variance | 27.0 | 31.7 | 29.8 | 28.2 | 29.2 |
| med | residual MSE | 43.1 | 38.7 | 40.6 | 38.3 | 42.5 |
| med | requirement met | 0.952 | 0.960 | 0.955 | 0.952 | 0.961 |
| none | residual variance | 28.7 | 30.2 | 28.8 | 29.0 | 29.7 |
| none | residual MSE | 28.9 | 30.6 | 29.7 | 29.4 | 30.4 |

* **The systematic error falls; the residual variance does not.** The offset removes the mean residual after
  one pass (high: -6.0 to about 0 pp) and the MSE drops by 26-38% at high mismatch. Within-pass residual variance
  stays at 27-32 pp^2 in every pass and at every mismatch level, including none. It is set by the card spread
  factor (and, under mismatch, the per-segment gust), which an offset cannot remove. The proposal expected
  the variance to fall; in this simulator it does not, and we report that as a negative result.
* Most of the pass-to-pass movement in the table (e.g. 27.4 to 31.9 at high) comes from each pass having a
  different hidden realization: the same swings appear in the dev-seed diagnostic with no slow loop at all.
  Paired on the same passes, the slow loop raises the variance by 0.4-1.7 pp^2. The global-only variant, a
  pure shift that cannot change the spread of a fixed set of segments, shows the same increase. So it
  comes from the selector choosing different, higher-coverage settings, where the card noise is larger,
  not from fitting noise.
* Reliability improves: in the evaluation pass fast + slow meets C_req in 0.961 (med) and 0.954 (high) of
  segments, against 0.937 and 0.918 for the fast loop, at +2.4% and +4.6% more volume than the fast loop.
  Dev-seed diagnostic (`results/closed_loop_slow_diagnostic.md`): the share of segments whose true coverage
  falls below the lower bound used by the selector goes from 0.10-0.13 (fast loop only) to 0.06-0.08 at high
  mismatch.
* **Why the midterm slow loop hurt, and the check that it is fixed.** Refitting the 50,000-row LightGBM model
  with 40 to 160 heavily weighted noisy cards narrowed the interval (lower half-width 8.0-8.8 to 4.6-5.4 pp in
  the dev-seed diagnostic) while the boosters fitted card noise. True coverage then fell below the lower bound
  in 0.19-0.26 of segments, and the residual variance rose to 31-41 pp^2. Same cells (ablation, gusty_afternoon, med, 20
  evaluation seeds): legacy refit weight 30 meets 0.832 with -89% [-308, -16] recovery; weight 1 meets 0.905
  with 12% [-107, 53]; the residual offset meets 0.966 with 98% [80, 125]. The per-regime and global-only
  offsets are indistinguishable (95% vs 98%): the empirical-Bayes step finds no between-regime difference
  larger than noise and pools them.

### Other observations (unchanged in direction from the midterm)

* Ablations of the fast loop (gusty_afternoon, med, reference = fair oracle): alpha 0.1 / 0.3 / 0.6 recover
  47% / 63% / 39%. A global bias recovers more than per-regime biases (79% vs 63%) at +3.4% vs +2.1%
  volume. Warm-up 0 / 2 / 4 recovers 70% / 63% / 53%. Rendered-card verifiers B0 and B1 (5 seeds) give
  46% with CIs from -67 to 78 and -100 to 76: too few seeds to separate them from the noisy verifier.
* Retreat flags were never raised (no card below 15% coverage), so re-treatment is not a discriminating
  metric in this grid.
* All of this tests the control logic against a known, injected mismatch inside our own simulator; it is not
  evidence of field efficacy (see Limitations).
