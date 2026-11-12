# Status and handoff (2026-11-11)

For the next Claude Code session or teammate. Read `README.md` for setup, then this file, then the
final scorecard `results/scorecard.md` (every proposal criterion and extension with its number,
95% CI and source file) and `Docs/decisions.md` (D1-D33).

All planned work streams are finished and the final report is built (`reports/final/`), with no
pending inputs left. The only open criterion is the real-card one, blocked by data access.

## Finished on 2026-11-11 (what feedback changed, D33)

* The Field run now shows what card feedback changed and what that did: for each section, the setting the Model only strategy would have picked on the same field and its actual coverage (simulator truth), against the setting used. Replay row Without feedback and grey rings in panel B, step 6 of Section detail, run totals with a bar chart and a table in the Feedback loop tab. Opt-in `run_pass_wrapper(explain_feedback=True)`, on in the app only; golden digests unchanged.

## Finished on 2026-11-11 (fixed application rate, Operator console, D32)

* Teammate review: the operator sets the nozzle and the label rate (gal/acre, held within +-5 %); the loop moves only pressure (speed follows to hold the rate) and boom height to keep coverage at or above 30 %. Opt-in `rate_lock` on the selector and in `recommend.py`, helpers in `src/coverageai/rate_control.py`, golden digests unchanged.
* Fourth app view, Operator console (`operator_console.py`, `app/operator_view.py`): the operator's setting held against the console adjusting it on the same field, the field replay, what to change, and which nozzle to switch to when 30 % cannot be reached at the rate.
* The coverage model needs no retrain for this (`results/rate_lock_model_check.md`). 20-seed paired evaluation in `results/rate_lock_eval.md`: +20.5 pp sections at or above 30 % against the setting held, +5.8 pp more with boom control, still far below the free controller, which raises the rate.

## Finished earlier on 2026-11-11 (app rework after the owner's review)

* One layout rule for the app: three views, the sidebar holds the inputs of the view on screen with its one Run button on top, the main area holds results, and only the open view is computed (D23).
* Field run is half as tall and has half the captions. The replay's controls are above the canvas. Picking a section in the replay (pause, click, scrub) selects it in Section detail.
* "Nozzle explorer" and "How the simulator works" are one view, Nozzle lab, with one set of controls and the same number for the same quantity (D26).
* Compare strategies puts the learning strategy on the same field as the others (D24) and can average over 1 to 20 seeds with a bootstrap interval (D25).
* Logic fixes from the audit: selector records with a wind forecast on, a lock on the shared module stack, a progress bar, failed runs reported in the page (D27).
* The replay shows the section's numbers round the loop again, now with what each feedback correction is based on: which group of cards, how many, and the update rule (D28). The first version of the rework had removed this readout.
* Trained card feedback (D29, opt-in): an XGBoost model (GPU) and a LightGBM model that predict the next card's error from the section and the cards read so far. On the evaluation seeds it is a little more accurate than the rule and puts 2 pp more sections on target, with slightly more volume and drift. A bigger model gave no further gain. Default stays the rule.
* Card reader calibration (D30, opt-in in the library, on by default in the app): the reader's known mean error is removed per coverage band, but only with a table measured on closed-loop cards. The table from the reader's validation cards reversed the classic reader's bias in the loop (negative result kept as `calibrate_reader="synthetic"`). The loop-measured b0 table was checked on held-out cards only, not in a closed-loop run.
* Trained card feedback v2 is the app default for the "Model + card feedback" strategy (D31; `fast` only, the learning strategy keeps the rule). On seeds 100 to 119: +1.5 pp sections on target [+0.8, +2.3] against the old default at c_req 30, at the cost of a little more spray and drift. The library default stays the rule.

## Finished on 2026-11-10

* One nozzle per field in the app: by default the model picks one nozzle and only pressure varies (D19).
* Operator recommendations: which of pressure, travel speed and boom height to change for sections below target (D20).
* Model outputs and run-to-run changes shown in the app (D21).
* Live paper cards and one timeline in the replay, plus the feedback charts tied to panel D (D22).
* Dead code and orphan outputs removed.
* README brought up to date.

## Finished in the earlier session (2026-10-19)

* Wind forecaster: full 20-seed grid (320 cells) run; negative result, default off (`results/forecast_eval.md`, headline in `results/forecast_eval_headline.md`, figure `experiments/figures/forecast_eval.png`, decision D13). Three committed syntax errors in `experiments/eval_forecast.py` fixed.
* Hybrid predictor: full 20-seed closed-loop stage run; not better than `SprayModel`, default stays `spray` (`results/hybrid_eval.md`, verdict from `results/hybrid_eval_notes.md`, D14).
* Boom height: verdict written at the top of `results/boom_control_eval.md` (notes in `results/boom_control_eval_notes.md`) and appended to D12; default off.
* Travel speed: scorecard status "inconclusive, not adopted". The paired 200-scenario `speed` variant of `results/boom_control_eval.md` (4 km/h grid) shows success unchanged and a small drift cut, but with the 2 km/h grid that D11 ships the open-loop learned-model success falls 0.935 to 0.900 (`results/speed_control_eval.md`, no CI), and the only closed loop is a 3-seed pilot.
* Decisions: D15 (fair oracle, `loop.FairOracleModel`), D17 (app opt-in controller options) and D16 (residual-offset slow loop, `feedback.ResidualCorrectedModel`) added; D14's offline figure corrected to 94-99 %.
* Final scorecard and report: `results/scorecard.md`, `reports/final/build.py` + `template.html` -> `final_report.html` / `.pdf`. Every number is read from a results file at build time.
* README: experiments table, layout, known issues updated.
* Test suite on the new machine: 406 tests pass, 1 skipped (Zenodo data not downloaded; run on commit c050f75 after the review fixes), including the three parquet tests that failed on the old laptop.
* App browser check (headless Chrome over CDP): all six strategies and the perfect / classic / noisy readers run, animation controls and end-of-run summary work, card replay consistent; five display bugs fixed; new opt-in sidebar expander "Advanced controller options" (travel speed, boom height, wind forecast), default off and bit-identical when off (D17). App tests: 65 pass.
* Reader timing re-measured with the same protocol on the desktop CPU (`results/b1_idle_retime.json`): B1 p95 0.52 s, B0 p95 0.15 s. The <= 2 s laptop criterion is met by the earlier busy-laptop measurement (B1 p95 1.09 s, an upper bound for that laptop); the desktop number is not a laptop number.
* Final report filled with both inputs (`APP_CHECK`, `B1_IDLE_RETIME` in `reports/final/build.py`); no PENDING boxes remain.

## Still open

0. **Fixed-rate mode, open checks (D32).**
    * Hidden nozzle wear raises the true flow by up to about 10 %, so the delivered rate leaves the +-5 % band even when the set rate is inside it. A real rate controller reads a flow meter and corrects; the simulation has none. Decide whether the console should hold the rate at the centre rather than at the +5 % band end when 30 % is unreachable.
    * The selector's margin flags 75 to 80 % of sections unreachable while 48 to 54 % truly reach 30 %. A smaller margin in rate mode was not tried.
    * Not evaluated across seeds: the console with image readers, with the trained feedback it uses in the app, and its nozzle suggestions. The six evaluation setups were picked after a pilot on evaluation seeds 0 and 1; re-pick them on development seeds before quoting the numbers in the report.
    * The final report and scorecard do not mention D32.
1. **Real annotated cards (blocked).** Zenodo record 14010262 (2 annotated real cards, 2,754 droplets, plus pre-annotated real squares) is restricted. Someone on the team must request access through the Zenodo page and/or send `reports/email_draft_simoes.md`. When the files arrive, put them under `data/external/simoes2025/real/image/*.png` and `real/label/*.txt` and run `uv run --no-sync python experiments/eval_verifiers.py --merge`; then rebuild the report. Until then the "<= 5 pp on real cards" criterion is blocked.
2. **Team checks before submission.** The report keeps `TEAM TO CONFIRM` marks: contributions of Emery and Junjie, whether the Zenodo request was sent, references [9]-[11] and [15]-[17]. `Docs/ai_usage_log.md` must have a line for every session (the post-midterm sessions are logged by the orchestrator).
3. **Known misses that stay misses** (do not tune them away): out-of-generator class agreement on the Simões synthetic cards (B0 86.3 %, B1 88.3 % vs 90 %); volume against the card-aware oracle (unreachable by construction); residual variance does not fall across passes.
4. Optional, not required by the proposal: a paired 20-seed closed-loop run of travel-speed control on the shipped 2 km/h grid (would settle the inconclusive D11 verdict); expose the hybrid model in the app (speed, boom height and forecaster are exposed since D17); rerun the rendered-card verifier ablation (`closed_loop.py`, B0/B1 verifiers) with 20 seeds instead of 5 now that the machine is fast. A full 40-card app pass with the U-Net reader takes about 30 s on the idle desktop; under heavy CPU load from parallel experiments it may not finish for minutes.
5. **SprayModel v2 (D18), team decision.** A GPU-trained XGBoost predictor (1M extra simulator rows) cuts the coverage error against the noise-free coverage from 0.64 to 0.24 pp and drift/displacement errors by about 80 %. In the closed loop `spray_model_v2_fast` meets C_req in +0.4 to +0.7 pp more segments than v1 (CIs exclude 0), with volume unchanged and drift +2-3 % relative. The default stays v1; the report and scorecard are built on v1 and were not regenerated. Decide whether to adopt v2_fast (then rerun `closed_loop.py` and rebuild the report). Install xgboost with `uv pip install xgboost==3.4.1` (keeps the CUDA torch; `uv sync --extra vision --extra gpu` also works but restores the CPU torch); files are gitignored, regenerate with `uv run --no-sync python experiments/train_spray_v2.py --cost-cap 1800 --tag fast --skip-small`. Results: `results/model_v2_fast.md`, `results/model_v2_closed_loop/compare_v2_fast.md`. README now covers the `gpu` extra (done 2026-11-10).

6. **Known clutter and risks from the 2026-11-10 audit** (none blocks the submission):
   * The mock module stack is still reachable from experiments (the app no longer offers it, D23). With it, `best_fixed` raises `KeyError` and speed or boom control raises `ValueError`.
   * Nominal physics is implemented in three places: `simulator.nominal_arrays`, `loop.FairOracleModel.predict_arrays` and `model/hybrid.PhysicsBase.arrays`.
   * Bootstrap CI helpers are copied across five experiment scripts, with no shared `stats` module.
   * `reports/midterm/build.py` is a near copy of `reports/final/build.py` and reads live results, so the midterm is frozen only as the committed HTML and PDF.
   * Model weights (about 50 MB) are plain git blobs, not LFS.
   * `experiments/demo_screenshots.py` live mode still clicks controls that no longer exist (the Step button and old tab names); the report uses the `--static` path.

7. **New app features are not evaluated across seeds (D19, D20).** The one-nozzle-per-field default and the operator recommendations were checked on single app runs only. Two paired 20-seed runs would settle them: closed loop with `nozzle_lock` from `best_nozzle` against free nozzle choice (share of sections meeting C_req, volume, drift), and a re-spray experiment that applies each recommendation and measures whether the section then meets the target. Until then, describe both as app behaviour, not as results.
8. **The final report predates the 2026-11-10 and 2026-11-11 app.** `reports/final/` was not rebuilt. Its app paragraph (`APP_CHECK` in `reports/final/build.py`) describes the earlier layout (sidebar step controls, free nozzle choice, four tabs). Update that text before the next rebuild. The report's numbers are unaffected: no experiment was rerun.
9. Done on 2026-11-11 (D23): Section detail follows the replay. The other direction is still open: moving the "Section" slider does not move the replay.
11. **Trained feedback, team decision and open checks (D29).**
    * Decide whether to adopt it. For: +2.1 pp sections on target [+1.3, +2.9] and a slightly more accurate correction. Against: +0.36 pp drift with one nozzle held, about half of the gain is caution that a constant margin would also give, and it has learned this simulator's hidden losses.
    * A plain average of the pass's card errors with a starting value did as well under mismatch. That is a simpler candidate to replace eq. 3 than a boosted model; it was only run as a post-hoc control.
    * Not run: the trained feedback with travel-speed or boom-height control, with the wind forecast, with the learning strategy, at targets other than 30 %, and with the image readers beyond 5 seeds.
    * The report and scorecard do not mention D29. Add it to `reports/final/build.py` before the next rebuild if it is adopted or reported.
10. **App rework, not yet checked (D23-D28).**
    * Only Streamlit 1.66 was run. The fallbacks for older versions (a radio row for the views, no replay-to-detail link) were not run.
    * The multi-seed comparison has unit tests only. Its interval is rough with 2 to 5 seeds, and app seeds can overlap evaluation seeds 0 to 19. Do not quote it in the report.
    * `same_field` is off outside the app. Making it the default needs a logged decision and new golden digests.
    * The README media (`Docs/media/`: `field_run.gif`, `spray_lab.gif`, `nozzle_compare.gif`, `app_field_run.png`) were regenerated for the new layout. `wsp_cards.gif` was not: the cards and the classical reader did not change.
    * `experiments/demo_screenshots.py` live mode clicks tabs and buttons that no longer exist.
    * For the learning strategy, the basis shown for a correction counts the cards of the replayed pass only. The correction learned in earlier passes is inside the model prediction and is not itemised per section.
    * Display names are still defined in three places (`app/labels.py`, `field_anim.py`, `spray_lab.py`).
    * With the noisy reader, a replayed card is read twice (`integration.NoisyVerifier` has no `measure_full`). `replay_card` and `render_cards` copy the simulator without taking the stack lock.
    * With "Read card images" off, Compare strategies gives the strategies the true coverage as the card reading. The page says so, but those rows are then not the Field run's experiment for an image reader.

## Rebuilding the final report

```bash
uv run --no-sync python reports/final/build.py            # HTML + PDF + results/scorecard.md
uv run --no-sync python reports/final/build.py --no-pdf   # HTML + scorecard only
```

* The PDF is printed by headless Edge (`C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe`); on another OS change `EDGE` in `build.py` or use `--no-pdf` and print the HTML.
* Figures are copied (and some cropped) from `experiments/figures/` into `reports/final/figures/` at build time.
* `results/scorecard.md` is generated; edit `build.py` (`scorecard()`, `negative_results()`), not the md.
* `reports/midterm/` is the frozen midterm pipeline; do not rebuild it against the new results (its prose quotes midterm numbers).

## Notes for running on the new machine

* 32 logical cores, RTX 5070 Ti, CUDA torch 2.11+cu128. Always `uv run --no-sync ...` (or `export UV_NO_SYNC=1`); a plain `uv run` re-syncs to the lock file and puts the CPU torch back.
* Raise `--workers` for the experiment grids (the laptop used 4 to 7); LightGBM threads per worker: `--n-jobs`.
* Timing tests and reader runtimes are sensitive to CPU load; measure on an idle machine (protocol in `results/b1_idle_retime.json`).
* Commit the regenerated `results/*.json/.md`, figures and `reports/final/` outputs; data and old checkpoints stay ignored.
