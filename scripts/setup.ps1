# One-time setup: fetch the prebuilt ONNX Runtime and export the phoneme model.
# Safe to re-run; each step is skipped if its output already exists.
#
#   .\scripts\setup.ps1                 # CPU ONNX Runtime + FP32/INT8 model
#   .\scripts\setup.ps1 -OrtVersion 1.26.0
param(
  [string]$OrtVersion = "1.26.0"
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

# 1) ONNX Runtime (prebuilt Windows x64) ------------------------------------
$ortDir = Join-Path $root "third_party\onnxruntime"
if (-not (Test-Path "$ortDir\include\onnxruntime_cxx_api.h")) {
  Write-Host "==> Downloading ONNX Runtime $OrtVersion ..." -ForegroundColor Cyan
  $tp = Join-Path $root "third_party"; New-Item -ItemType Directory -Force -Path $tp | Out-Null
  $zip = Join-Path $tp "onnxruntime.zip"
  $url = "https://github.com/microsoft/onnxruntime/releases/download/v$OrtVersion/onnxruntime-win-x64-$OrtVersion.zip"
  curl.exe -L --fail -o $zip $url
  Expand-Archive -Path $zip -DestinationPath $tp -Force
  Rename-Item "$tp\onnxruntime-win-x64-$OrtVersion" $ortDir -Force
  Remove-Item $zip
} else { Write-Host "==> ONNX Runtime already present." -ForegroundColor DarkGray }

# 2) espeak-ng for grapheme-to-phoneme (text targets) -----------------------
# The release ships only an .msi; an administrative install (/a) extracts the
# dll + exe + data without elevation or registering anything.
$espeakExe = Join-Path $root "third_party\espeak\eSpeak NG\espeak-ng.exe"
if (-not (Test-Path $espeakExe)) {
  Write-Host "==> Fetching espeak-ng (G2P for arbitrary words/sentences) ..." -ForegroundColor Cyan
  $tp = Join-Path $root "third_party"; New-Item -ItemType Directory -Force -Path $tp | Out-Null
  $msi = Join-Path $tp "espeak-ng.msi"
  curl.exe -L --fail -o $msi "https://github.com/espeak-ng/espeak-ng/releases/download/1.52.0/espeak-ng.msi"
  $dst = Join-Path $tp "espeak"; New-Item -ItemType Directory -Force -Path $dst | Out-Null
  Start-Process msiexec -ArgumentList "/a `"$msi`" /qn TARGETDIR=`"$dst`"" -Wait
  Remove-Item $msi -ErrorAction SilentlyContinue
} else { Write-Host "==> espeak-ng already present." -ForegroundColor DarkGray }

# 3) Python venv for model export + evaluation ------------------------------
$venvPy = Join-Path $root "tools\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
  Write-Host "==> Creating Python venv + installing torch/transformers/onnx ..." -ForegroundColor Cyan
  python -m venv tools\.venv
  & $venvPy -m pip install --upgrade pip
  & $venvPy -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  & $venvPy -m pip install transformers onnx onnxruntime numpy huggingface_hub
  & $venvPy -m pip install datasets soundfile   # speechocean762 evaluation harness
} else { Write-Host "==> Python venv already present." -ForegroundColor DarkGray }

# 4) Export the phoneme model -----------------------------------------------
if (-not (Test-Path "$root\models\phoneme.onnx")) {
  Write-Host "==> Exporting phoneme model to ONNX (downloads ~1.2 GB once) ..." -ForegroundColor Cyan
  $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
  & $venvPy tools\export_model.py
} else { Write-Host "==> Model already exported." -ForegroundColor DarkGray }

Write-Host ""
Write-Host "Footprint for YOUR APP (ship to learners):" -ForegroundColor Cyan
Write-Host "  INT8 model + engine + espeak  ~ 350 MB total (recommended)" -ForegroundColor White
Write-Host "  FP32 model                      ~ 1.2 GB (dev / max accuracy only)" -ForegroundColor DarkGray
Write-Host ""
Write-Host "==> Setup complete. Now run: .\build.ps1" -ForegroundColor Green
Write-Host "     Demo UI uses INT8 by default: .\run_ui.ps1" -ForegroundColor Green
