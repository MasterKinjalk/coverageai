# CoverageVision AI

Closed-loop precision spraying, evaluated in a physics-informed simulator (ABE 598, Fall 2026).

For each field section the system:

1. predicts coverage and drift risk from the measured conditions (wind, temperature, humidity, speed, boom height) for every catalog setting,
2. picks the setting (nozzle, pressure, optionally travel speed and boom height) that meets the coverage requirement at the lowest predicted drift and volume,
3. sprays (in the simulator),
4. reads the coverage actually achieved from a rendered water-sensitive-paper (WSP) card with computer vision (classical B0 or U-Net B1),
5. feeds the error back: a fast loop corrects the next section, a slow loop corrects the model between passes.

Planning and design history: `PLAN.md`, `Docs/decisions.md`. Open work for the next session: **`Docs/NEXT_STEPS.md`**.

## Setup

Needs Python 3.12+ (the repo pins 3.14 in `.python-version`) and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/MasterKinjalk/coverageai.git
cd coverageai
bash scripts/setup.sh            # Linux / macOS / Git Bash
# or, in PowerShell:  ./scripts/setup.ps1
```

The script installs uv if missing, runs `uv sync --extra vision` (numpy, LightGBM, OpenCV, Streamlit, torch, segmentation-models-pytorch, pytest), swaps in a CUDA build of torch when `nvidia-smi` is found, regenerates the 50k-row training table `data/datasets/v1.parquet`, and runs the smoke test.

Options (both scripts): `--cuda cu128` picks the CUDA wheel index (default `cu128`, use `none` to keep the CPU torch), `--with-cards` also renders the 1200 synthetic WSP cards (~3 GB, needed for U-Net training/eval), `--with-simoes` downloads the open Simões 2025 synthetic cards (~850 MB, needed for the out-of-generator check).

Manual install, if you prefer:

```bash
uv sync --extra vision
# CUDA torch (pick the index that matches your driver; see https://pytorch.org)
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv run --no-sync python experiments/make_dataset.py
uv run --no-sync python -m coverageai.e2e --smoke
```

**After installing CUDA torch, use `uv run --no-sync ...`** (or `export UV_NO_SYNC=1`). A plain `uv run` re-syncs to the lock file and puts the CPU torch back.

Without uv: `pip install -e ".[vision]" pytest`.

## What is in git and what is regenerated

| Item | In git? | How to get it |
|---|---|---|
| `models/unet_v1.pt` (B1 U-Net, full GPU training, 16 MB) | yes | retrain: `Docs/gpu_training.md` |
| `models/spray_model_v1.joblib` (LightGBM + CQR) | yes | `uv run python experiments/train_eval_model.py` |
| `models/hybrid_v1.joblib`, `models/hybrid_misspec_v1.joblib` | yes | `uv run python experiments/eval_hybrid.py --stage train` |
| `data/catalog/*.csv` (nozzle catalog) | yes | - |
| `data/datasets/v1.parquet` (training table) | no | `uv run python experiments/make_dataset.py` |
| `data/synthetic/v1` (rendered WSP cards, ~3 GB) | no | `uv run python -m coverageai.vision.render --out data/synthetic/v1 --n 1200 --seed 0` |
| `data/external/simoes2025` (~850 MB) | no | `uv run python -m coverageai.vision.real_data --download --summary` |
| `models/unet_v1_cpu_partial.pt`, `models/unet_v1.last.pt` | no | old checkpoints, not needed |

## Running things

```bash
uv run streamlit run app/main.py          # demo app, http://localhost:8501
uv run pytest -q                          # tests
uv run python -m coverageai.e2e --smoke   # end-to-end smoke pass
uv run python -m coverageai.e2e --real --model spray --verifier b1
uv run --no-sync python reports/final/build.py   # final report (html + pdf) and results/scorecard.md
```

Experiments (each writes `results/<name>.json` + `.md`, figures in `experiments/figures/`):

| Script | What |
|---|---|
| `experiments/closed_loop.py` | main closed-loop result: open / fast / fast+slow vs fair and card-aware oracles, 20 seeds (`results/closed_loop_summary.md`) |
| `experiments/eval_selector.py` | selector success / drift / volume on no-mismatch scenarios |
| `experiments/eval_verifiers.py` | B0 vs B1 card reading on renderer and Simões cards (`results/verification_summary.md`) |
| `experiments/eval_speed_control.py`, `eval_boom_control.py` | travel speed and boom height as optional controlled settings (`results/boom_control_eval.md`, verdict at the top) |
| `experiments/eval_forecast.py` | short-horizon wind forecaster, 20 seeds (`results/forecast_eval.md`; negative result) |
| `experiments/eval_hybrid.py` | physics + learned residual predictor, offline / selector / 20-seed closed loop (`results/hybrid_eval.md`) |
| `experiments/slow_loop_diagnostic.py` | slow-loop residuals per pass |

All criteria and extensions in one table: `results/scorecard.md` (written by `reports/final/build.py`).

Most experiments take `--workers N`; on a big machine raise it (the laptop runs used 4 to 7). U-Net training on GPU: `Docs/gpu_training.md` or `notebooks/train_unet_gpu.ipynb`.

## Layout

```
src/coverageai/      library
  contracts.py       shared data types (Setting, Conditions, SegmentLog, ...), schema version
  catalog.py         nozzle catalog, derived quantities (GPA, droplet size), speed/height grids
  simulator.py       physics plant with hidden mismatch (gusts, canopy, wear)
  scenarios.py       weather presets
  selector.py        constrained setting selector
  loop.py            closed loop: policies, run_pass, run_passes_slow, fair oracle
  feedback.py        fast-loop bias, slow-loop ResidualCorrectedModel
  forecast.py        wind forecasters (persistence, EWMA, AR(1), gust-aware)
  model/             SprayModel (LightGBM + CQR), PhysicsResidualModel (hybrid), features
  vision/            WSP card renderer, classical reader (B0), U-Net (B1), Simões data loader
  app_logic.py, field_anim.py, spray_lab.py   back end of the Streamlit app
app/                 Streamlit app (main.py, field run and spray lab views)
config/spec.yaml     all parameters, ranges and selector options
experiments/         evaluation scripts and figures
results/             experiment outputs (json, md, csv)
reports/midterm/     midterm report (build.py -> html/pdf)
reports/final/       final report (build.py -> html/pdf, also writes results/scorecard.md)
Docs/                proposal, specs, decisions log, GPU training guide, NEXT_STEPS.md
tests/               pytest suite
```

## Known issues

* On the old Windows laptop, three tests that write parquet failed with a pyarrow DLL error (`test_smoke_pass`, `test_e2e_run_real_writes_logs`, `test_logs_frame_and_save`). This was laptop-specific: on the new Windows 11 machine the full suite (394 tests) passes.
* Timing tests (`test_render_time_under_one_second_worst_case`) can fail when the machine is busy.
