# Next steps (handoff, 2026-10-19)

Written for the next Claude Code session (or a teammate) picking this up on a new machine.
Read `README.md` for setup, then this file, then the proposal's success criteria
(`Docs/CoverageVisionAI_Project_Proposal (1).pdf`, Table 1) and `Docs/decisions.md`.

The last laptop session ran four background agents. Their code is all in this commit and the
test suite passes (apart from the known Windows parquet failures), but two of the long
experiment runs were cut off when the session ended, and the final integration check never
happened. Work through the items below in order.

## Status of each work stream

| Stream | Code | Results | What is left |
|---|---|---|---|
| Fair oracle + redesigned slow loop (`loop.FairOracleModel`, `feedback.ResidualCorrectedModel`) | done | `results/closed_loop_summary.md`: fast+slow recovers 96% [80, 107] of the gap at +4.1% volume vs the fair oracle, criterion met | nothing |
| Travel speed as an optional setting (D11, schema 1.1) | done | `results/speed_control_eval.md`: drift and volume dip, success flat | nothing |
| U-Net B1, full GPU training | done, weights in `models/unet_v1.pt` | `results/verification_summary.md`, `results/b1_synthetic.json` | re-time per-card speed on an idle machine (target <= 2 s on a laptop CPU; measured 1.09 s while the CPU was busy) |
| Boom height as an optional setting (D12, schema 1.2) | done (`tests/test_boom_control.py`) | `results/boom_control_eval.md` complete | agent was stopped before its final report; review the D12 row in `Docs/decisions.md`, re-run `uv run pytest tests/test_boom_control.py tests/test_speed_control.py`, write the verdict (see 3) |
| Wind forecaster (`src/coverageai/forecast.py`, `tests/test_forecast.py`) | done, 22 tests pass | only the pilot: `results/forecast_eval_pilot.md` (3 seeds, 2 presets) | **full 20-seed grid never finished** (see 1) |
| Physics-first hybrid (`src/coverageai/model/hybrid.py`, `tests/test_hybrid.py`) | done | `results/hybrid_eval.md`: offline + selector stages complete, closed loop is a 1-seed PILOT | **full closed-loop stage never finished** (see 2) |
| App catch-up (new strategies `fast_slow`, `oracle_card`, noisy reader back) | done, 56 app tests pass | - | never checked in a browser (see 4) |

## 1. Finish the wind-forecaster evaluation

```bash
uv run python experiments/eval_forecast.py --workers 16
```

* Grid: mismatch {none, low, med, high} x presets {gusty_afternoon, crosswind_shift, marginal_limits, calm_morning} x 20 seeds, arms off / last / ewma / ar1 / ar1+gust / nominal+gust / gust_oracle, fast and fast+slow. 320 cells; it was at 180/320 after ~55 min with 4 workers on the laptop.
* Writes `results/forecast_eval.json` and `.md`. Then add the hand-written verdict with `--combine --headline "..."`.
* Pilot signal (`results/forecast_eval_pilot.md`): the forecasters cut wind-forecast error (AR(1) by ~0.18 m/s) but do not raise the share of sections meeting the target; even the `gust_oracle` upper bound barely helps. Expect a negative or "no significant benefit" result; report it as such, with the gust-oracle bound as the explanation (the selector's margin already absorbs most gust risk).
* Add a decisions row (D13) for the forecaster design: anemometer observes the realized wind of completed sections with sd 0.3 m/s noise, never the upcoming one; forecaster off = bit-identical to before.

## 2. Finish the hybrid closed-loop stage

```bash
uv run python experiments/eval_hybrid.py --stage closed --workers 16 --seeds 20
uv run python experiments/eval_hybrid.py --stage report
```

* It was at 660/960 cells after ~40 min with 4 workers. The stage replaces the `## Closed loop (PILOT)` section of `results/hybrid_eval.md`.
* Key framing (keep it in the report): the exact physics base is the plant's own equations, so `physics` and `hybrid` rows are an upper bound (circular). The informative rows are the misspecified bases: offline, the learned residual removes 96 to 99% of the misspecification error; in the selector test, misspecified physics alone meets the target in 44% of scenarios vs 93.5% for the hybrid on the same wrong base.
* If the closed loop confirms the pilot, the recommendation is: hybrid with a misspecified base performs like the pure-ML SprayModel, and much better than physics alone when the physics is wrong.

## 3. Write the boom-height verdict

From `results/boom_control_eval.md` (deterministic paired scoring, 200 scenarios + 10-seed closed loop):

* Height control alone (open loop) *raises* drift slightly (+0.003) because the selector trades height for lower volume; success unchanged.
* Speed + height together lower drift (-0.003) and volume (about -2.6 GPA) but cost 2 to 3 pp of sections meeting the target in the closed loop.
* With the fair oracle, height control cuts drift clearly (-0.004 to -0.006) without losing success, so the limit is the learned/nominal model's prediction of the height effect, not the physics.
* Default stays off. Add these findings to `Docs/decisions.md` D12 and to the summary table in 5.

## 4. Check the app in a browser

```bash
uv run streamlit run app/main.py
```

* Field run tab: run each control strategy (open, fast, fast_slow, oracle, oracle_card, best_fixed) and each card reader (classic, U-Net, noisy). Press Play in the animated figures, test the scrubber, hover and click-to-seek, the end-of-run summary.
* `fast_slow` should show the "Learning between passes" table, and the card replay should be consistent (`card.consistent` true).
* Compare strategies tab: default selection open / fast / fast_slow / oracle, caption explaining fair vs card-aware oracle.
* "How the simulator works" tab (Spray lab): figures render.
* Label to check: the oracle is now the *fair* oracle ("Knows hidden conditions (fair upper bound)"); `oracle_card` is the unreachable one.
* Not yet exposed in the app: forecaster, boom-height and speed options, hybrid model. Add them to the sidebar only if time allows (all are opt-in in `config/spec.yaml` / loop arguments).
* First load takes 30 to 40 s (40 rendered cards per run).

## 5. Final scorecard and report

Build one table of every proposal success criterion plus the extensions, each marked met / missed with its number and CI. Current state:

| Target | Result | Status |
|---|---|---|
| Closed loop: >= 50% gap recovery, <= +10% volume vs oracle | fast+slow 96% [80, 107], +4.1% [+2.8, +5.5] vs fair oracle | met |
| Same vs card-aware oracle | 63% recovery, +17% volume | missed by design: that oracle uses 12.6% less volume than any real controller could |
| Card reading, own cards | B0 1.44 pp, B1 0.20 pp MAE | met |
| Card reading, Simões cards | B0 2.65 pp / 86% class agreement, B1 1.74 pp / 88% | MAE met by B1; class agreement (>= 90%) missed by both, underread at 30 to 50% coverage (only 16 such cards) |
| Card reading speed | B0 0.26 s, B1 1.09 s per card | met, re-time on an idle CPU |
| Real annotated cards | evaluation code ready | blocked: Zenodo record is restricted, someone must request access (`reports/email_draft_simoes.md`) |
| Speed as a setting | small drift/volume gain, no success gain | done, off by default |
| Boom height as a setting | see 3 | done, off by default |
| Wind forecaster | see 1 | pending full run |
| Hybrid predictor | see 2 | pending closed-loop run |

Negative results to keep in the report: prediction error variance does not shrink pass after pass (per-card noise and gusts are irreducible), and B1 is less safe outside the generator than on our own cards.

Then update the report (`reports/midterm/build.py` is the template for the final one) and commit.

## Notes for running on the new machine

* Raise `--workers` (the laptop used 4 to 7). LightGBM threads per worker: `--n-jobs`.
* Keep `uv run --no-sync` if you installed CUDA torch (see README).
* Commit the regenerated `results/*.json/.md` and figures; data and old checkpoints stay ignored.
