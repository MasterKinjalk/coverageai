# Training the B1 U-Net on a GPU

The U-Net verifier (`src/coverageai/vision/unet.py`) trains on a laptop CPU (about one hour gives a usable
model, see `results/unet_train_log.csv`), but the full run is meant for a GPU. Two supported places:

* Google Colab (free T4, or L4/A100 on Pro)
* A Jupyter GPU node, for example NCSA with an H100

Both use the same notebook, `notebooks/train_unet_gpu.ipynb`. It detects the environment itself.

## Before you start

1. Make sure the branch you clone (`BRANCH` in cell 1, default `main`) contains `vision/unet.py` and `vision/render.py`. Both are on `main` (WP-13 is merged); push any local commits first, the notebook clones from GitHub, not from your working copy.
2. Create a fine-grained GitHub personal access token with read-only "Contents" access to
   `MasterKinjalk/coverageai`. Never paste it into a notebook cell and never commit it. The notebook reads
   it from the `GITHUB_TOKEN` environment variable or asks for it with a hidden prompt, then removes it from
   `.git/config` after cloning.
3. No repo access from the GPU machine? Zip the repo (without `.venv`, `data`, `models`) and set `REPO_ZIP`
   in cell 1 of the notebook to the uploaded zip path.

## Option A: Google Colab

1. Open Colab, upload `notebooks/train_unet_gpu.ipynb` (File > Upload notebook).
2. Runtime > Change runtime type > GPU.
3. Run all cells. When asked, paste the GitHub token. Colab's image already has a CUDA torch; the install cell keeps
   it and adds `segmentation-models-pytorch` and the repo (`pip install -e . --no-deps`).
4. The dataset is generated in the VM (1200 cards took about 165 s on 7 workers locally; expect roughly 8 to 10
   minutes on Colab's 2 CPU cores). Alternatively zip the local `data/synthetic/v1` and upload it to `WORKDIR` unpacked.
   For 3000 cards set `N_CARDS = 3000` in the data cell.
5. On a T4 change the training cell to `batch=16`, `crop=320` (the defaults assume an H100). `amp="auto"` selects
   fp16 with a grad scaler there and bf16 on A100/L4.
6. The last cell zips `models/unet_v1.pt`, `results/b1_synthetic.json`, `results/unet_train_log.csv` and the parity
   plot and triggers a browser download. Colab VMs are ephemeral, so download before the session ends.

## Option B: NCSA (or any Jupyter GPU node)

1. Start a Jupyter session on a GPU node (H100). Open a terminal or a notebook cell and decide on a work directory
   with a few GB free, for example `/projects/<allocation>/<user>/coverageai`.
2. Set it before launching Jupyter (`export WORKDIR=/projects/.../coverageai`) or edit `WORKDIR` in cell 1.
3. Upload or open `notebooks/train_unet_gpu.ipynb`. Optionally `export GITHUB_TOKEN=...` in the shell that starts
   Jupyter so no prompt is needed.
4. Run all cells. If the node image already has a CUDA torch, it is kept. If not, the cell runs
   `pip install -e ".[vision]"`. If the kernel Python is older than 3.12, the editable install fails and the notebook
   falls back to `PYTHONPATH=src`.
5. Training uses `batch=48`, `crop=512`, bf16 and `os.cpu_count()` data-loader workers (max 16), capped at 45 minutes.
6. Get the zip from the Jupyter file browser or `scp` it; the path is printed in the last cell.

## After training: refresh the verification results (local, CPU)

The notebook only evaluates the renderer test split (`results/b1_synthetic.json`). The full O3 evidence
(coverage MAE on renderer and Simoes cards, IoU, count and Dv errors, stratification by coverage band and stain
size, runtime on your CPU, criterion table) comes from `experiments/eval_verifiers.py`:

1. Unzip `b1_unet_results.zip` into the repo root (this replaces `models/unet_v1.pt`; keep the CPU-partial one
   under another name if you want to compare).
2. `uv run python experiments/eval_verifiers.py --verifiers b1 --merge` (about 30 minutes on a laptop CPU; B0 rows
   already in `results/verification_eval.json` are kept). Needs the open Simoes synthetic cards under
   `data/external/simoes2025` (`uv run python -m coverageai.vision.real_data --download`).
3. Files updated: `models/unet_v1.pt`, `results/b1_synthetic.json` (from Colab), `results/verification_eval.json`
   and `results/verification_summary.md` (B1 rows). `results/b1_synthetic_cpu_partial.json` stays as the record of
   the 6-epoch CPU run.

## Same thing from a terminal

```bash
git clone https://github.com/MasterKinjalk/coverageai.git && cd coverageai
pip install -e ".[vision]"          # or: pip install segmentation-models-pytorch scikit-image opencv-python-headless pyyaml matplotlib timm && pip install -e . --no-deps
python -m coverageai.vision.render --out data/synthetic/v1 --n 1200 --seed 0
python -m coverageai.vision.unet train --data data/synthetic/v1 --device cuda --amp bf16 --batch 48 --crop 512 \
    --workers 16 --max-minutes 45 --epochs 40 --crops-per-card 6 --max-steps-per-epoch 300
python -m coverageai.vision.unet eval --data data/synthetic/v1 --split test --device cuda
```

CPU fallback (same CLI, no GPU): `--device cpu --batch 8 --crop 256 --workers 0 --max-minutes 60`.

## What the training does

* Crops (256 to 512 px) are sampled from the full cards with a random rotation (+-12 deg), scale (0.85 to 1.2), flips,
  brightness/contrast/channel gain, hue/saturation, illumination gradient, blur, noise and JPEG, all in numpy/cv2.
* Loss is BCE plus soft Dice; optimizer AdamW; cosine learning-rate schedule that is fitted to the wall-clock cap
  after the first two epochs; early stopping on validation Dice (`--patience`).
* `models/unet_v1.pt` holds the best-validation-Dice weights, `models/unet_v1.last.pt` the latest epoch.
  Both are gitignored. Copy `unet_v1.pt` to `models/` in the main repo; `UNetVerifier()` loads it by default.
* `eval` writes `results/b1_synthetic.json` (coverage MAE in pp, IoU, 3-class agreement, runtime, overall and by
  coverage band) and `experiments/figures/b1_parity.png`.

## Troubleshooting

* `CUDA out of memory`: lower `--batch` or `--crop`.
* Loader is the bottleneck (GPU utilization low): raise `--workers`. With workers the RAM image cache is off.
* `ImageNet weights unavailable` warning: the node has no internet to the Hugging Face hub; training falls back to
  random init (slower to converge). Pre-download once with a connected machine or use `--no-pretrained` on purpose.
* Time cap hit before the cosine schedule finished: increase `--max-minutes` or reduce `--epochs`.
