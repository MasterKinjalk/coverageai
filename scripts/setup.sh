#!/usr/bin/env bash
# One-shot environment setup for CoverageVision AI.
# Usage: bash scripts/setup.sh [--cuda cu128|cu126|none] [--with-cards] [--with-simoes]
set -euo pipefail

CUDA="cu128"
WITH_CARDS=0
WITH_SIMOES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda) CUDA="$2"; shift 2 ;;
    --with-cards) WITH_CARDS=1; shift ;;
    --with-simoes) WITH_SIMOES=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing dependencies (uv sync --extra vision)..."
uv sync --extra vision

if [[ "$CUDA" != "none" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU found; installing CUDA torch from the $CUDA index..."
  uv pip install --reinstall torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA"
fi
# From here on --no-sync keeps the CUDA torch (plain `uv run` would restore the locked CPU build).
RUN="uv run --no-sync"
$RUN python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

if [[ ! -f data/datasets/v1.parquet ]]; then
  echo "Generating training table data/datasets/v1.parquet..."
  $RUN python experiments/make_dataset.py
fi

if [[ $WITH_CARDS -eq 1 && ! -d data/synthetic/v1 ]]; then
  echo "Rendering 1200 synthetic WSP cards (~3 GB)..."
  $RUN python -m coverageai.vision.render --out data/synthetic/v1 --n 1200 --seed 0
fi

if [[ $WITH_SIMOES -eq 1 ]]; then
  echo "Downloading Simoes 2025 synthetic cards (~850 MB)..."
  $RUN python -m coverageai.vision.real_data --download --summary
fi

echo "Smoke test..."
$RUN python -m coverageai.e2e --smoke

echo
echo "Done. Next: uv run --no-sync pytest -q   |   uv run --no-sync streamlit run app/main.py"
echo "Open work is listed in Docs/NEXT_STEPS.md."
