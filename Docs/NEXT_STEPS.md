# Status and handoff (2026-10-19)

For the next Claude Code session or teammate. Read `README.md` for setup, then this file, then the
final scorecard `results/scorecard.md` (every proposal criterion and extension with its number,
95% CI and source file) and `Docs/decisions.md` (D1-D16).

All planned work streams are finished. The final report is built (`reports/final/`). Two inputs
from this session were still open when the report was first built (see "Still open" below).

## Finished in this session

* Wind forecaster: full 20-seed grid (320 cells) run; negative result, default off (`results/forecast_eval.md`, headline in `results/forecast_eval_headline.md`, figure `experiments/figures/forecast_eval.png`, decision D13). Three committed syntax errors in `experiments/eval_forecast.py` fixed.
* Hybrid predictor: full 20-seed closed-loop stage run; not better than `SprayModel`, default stays `spray` (`results/hybrid_eval.md`, verdict from `results/hybrid_eval_notes.md`, D14).
* Boom height: verdict written at the top of `results/boom_control_eval.md` (notes in `results/boom_control_eval_notes.md`) and appended to D12; default off.
* Travel speed: the scorecard cites the paired 200-scenario `speed` variant of `results/boom_control_eval.md`; the 3-seed closed loop in `results/speed_control_eval.md` is a pilot only.
* Decisions: D15 (fair oracle, `loop.FairOracleModel`) and D16 (residual-offset slow loop, `feedback.ResidualCorrectedModel`) added; D14's offline figure corrected to 94-99 %.
* Final scorecard and report: `results/scorecard.md`, `reports/final/build.py` + `template.html` -> `final_report.html` / `.pdf`. Every number is read from a results file at build time.
* README: experiments table, layout, known issues updated.
* Test suite on the new machine: 394 tests pass, including the three parquet tests that failed on the old laptop.

## Still open

1. **Real annotated cards (blocked).** Zenodo record 14010262 (2 annotated real cards, 2,754 droplets, plus pre-annotated real squares) is restricted. Someone on the team must request access through the Zenodo page and/or send `reports/email_draft_simoes.md`. When the files arrive, put them under `data/external/simoes2025/real/image/*.png` and `real/label/*.txt` and run `uv run --no-sync python experiments/eval_verifiers.py --merge`; then rebuild the report. Until then the "<= 5 pp on real cards" criterion is blocked.
2. **Inputs for the report** (if not yet filled): in `reports/final/build.py`, set `APP_CHECK` (result of the Streamlit browser check) and `B1_IDLE_RETIME` (B1 per-card CPU time on an idle CPU; target <= 2 s, previous p95 1.09 s on a busy CPU). While they are `None` the report shows a dashed PENDING box and the scorecard says `pending`.
3. **Team checks before submission.** The report keeps `TEAM TO CONFIRM` marks: contributions of Emery and Junjie, whether the Zenodo request was sent, references [9]-[11] and [15]-[17]. `Docs/ai_usage_log.md` must have a line for every session (the post-midterm sessions are logged by the orchestrator).
4. **Known misses that stay misses** (do not tune them away): out-of-generator class agreement on the Simões synthetic cards (B0 86.3 %, B1 88.3 % vs 90 %); volume against the card-aware oracle (unreachable by construction); residual variance does not fall across passes.
5. Optional, not required by the proposal: expose the speed / boom height / forecaster / hybrid options in the app sidebar (all opt-in in `config/spec.yaml` and the loop arguments); rerun the rendered-card verifier ablation (`closed_loop.py`, B0/B1 verifiers) with 20 seeds instead of 5 now that the machine is fast.

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
* Timing tests and the B1 runtime are sensitive to CPU load; measure on an idle machine.
* Commit the regenerated `results/*.json/.md`, figures and `reports/final/` outputs; data and old checkpoints stay ignored.
