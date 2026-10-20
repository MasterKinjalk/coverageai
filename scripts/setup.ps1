# One-shot environment setup for CoverageVision AI (Windows PowerShell).
# Usage: ./scripts/setup.ps1 [-Cuda cu128|cu126|none] [-WithCards] [-WithSimoes]
param(
    [string]$Cuda = "cu128",
    [switch]$WithCards,
    [switch]$WithSimoes
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "Installing dependencies (uv sync --extra vision)..."
uv sync --extra vision
if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

if ($Cuda -ne "none" -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host "GPU found; installing CUDA torch from the $Cuda index..."
    uv pip install --reinstall torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"
    if ($LASTEXITCODE -ne 0) { throw "CUDA torch install failed" }
}
# From here on --no-sync keeps the CUDA torch (plain `uv run` would restore the locked CPU build).
uv run --no-sync python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

if (-not (Test-Path data/datasets/v1.parquet)) {
    Write-Host "Generating training table data/datasets/v1.parquet..."
    uv run --no-sync python experiments/make_dataset.py
}

if ($WithCards -and -not (Test-Path data/synthetic/v1)) {
    Write-Host "Rendering 1200 synthetic WSP cards (~3 GB)..."
    uv run --no-sync python -m coverageai.vision.render --out data/synthetic/v1 --n 1200 --seed 0
}

if ($WithSimoes) {
    Write-Host "Downloading Simoes 2025 synthetic cards (~850 MB)..."
    uv run --no-sync python -m coverageai.vision.real_data --download --summary
}

Write-Host "Smoke test..."
uv run --no-sync python -m coverageai.e2e --smoke

Write-Host ""
Write-Host "Done. Next: uv run --no-sync pytest -q   |   uv run --no-sync streamlit run app/main.py"
Write-Host "Open work is listed in Docs/NEXT_STEPS.md."
