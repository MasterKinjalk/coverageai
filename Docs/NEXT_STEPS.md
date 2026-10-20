# Status and handoff (2026-10-19)

For the next Claude Code session or teammate. Read `README.md` for setup, then this file, then the
final scorecard `results/scorecard.md` (every proposal criterion and extension with its number,
95% CI and source file) and `Docs/decisions.md` (D1-D17).

All planned work streams are finished and the final report is built (`reports/final/`), with no
pending inputs left. The only open criterion is the real-card one, blocked by data access.

## Finished in this session

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

1. **Real annotated cards (blocked).** Zenodo record 14010262 (2 annotated real cards, 2,754 droplets, plus pre-annotated real squares) is restricted. Someone on the team must request access through the Zenodo page and/or send `reports/email_draft_simoes.md`. When the files arrive, put them under `data/external/simoes2025/real/image/*.png` and `real/label/*.txt` and run `uv run --no-sync python experiments/eval_verifiers.py --merge`; then rebuild the report. Until then the "<= 5 pp on real cards" criterion is blocked.
2. **Team checks before submission.** The report keeps `TEAM TO CONFIRM` marks: contributions of Emery and Junjie, whether the Zenodo request was sent, references [9]-[11] and [15]-[17]. `Docs/ai_usage_log.md` must have a line for every session (the post-midterm sessions are logged by the orchestrator).
3. **Known misses that stay misses** (do not tune them away): out-of-generator class agreement on the Simões synthetic cards (B0 86.3 %, B1 88.3 % vs 90 %); volume against the card-aware oracle (unreachable by construction); residual variance does not fall across passes.
4. Optional, not required by the proposal: a paired 20-seed closed-loop run of travel-speed control on the shipped 2 km/h grid (would settle the inconclusive D11 verdict); expose the hybrid model in the app (speed, boom height and forecaster are exposed since D17); rerun the rendered-card verifier ablation (`closed_loop.py`, B0/B1 verifiers) with 20 seeds instead of 5 now that the machine is fast. A full 40-card app pass with the U-Net reader takes about 30 s on the idle desktop; under heavy CPU load from parallel experiments it may not finish for minutes.

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
