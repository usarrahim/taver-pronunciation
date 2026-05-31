# Bundle everything needed to run the public demo on a Windows VPS.
# Output: dist\taver-server\  (~350 MB with INT8 model)
param(
  [string]$Out = "dist\taver-server"
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$need = @(
  "build\bin\taver_server.exe",
  "build\bin\taver.dll",
  "build\bin\onnxruntime.dll",
  "models\phoneme.int8.onnx",
  "models\phoneme_vocab.txt",
  "third_party\espeak\eSpeak NG\espeak-ng.exe",
  "web\index.html",
  "web\logo.svg",
  "config\calibration.json",
  "data\lexicon.txt"
)
foreach ($f in $need) {
  if (-not (Test-Path $f)) { throw "Missing: $f  (run .\build.ps1 and .\scripts\setup.ps1 first)" }
}

if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force -Path $Out | Out-Null

# Binaries
New-Item -Force -ItemType Directory "$Out\bin" | Out-Null
Copy-Item build\bin\taver_server.exe, build\bin\taver.dll, build\bin\onnxruntime.dll "$Out\bin\"
$prov = "build\bin\onnxruntime_providers_shared.dll"
if (Test-Path $prov) { Copy-Item $prov "$Out\bin\" }

# Model + config
New-Item -Force -ItemType Directory "$Out\models" | Out-Null
Copy-Item models\phoneme.int8.onnx, models\phoneme_vocab.txt "$Out\models\"

# espeak-ng (full tree for data files)
Copy-Item -Recurse "third_party\espeak" "$Out\third_party\"

# Web UI
Copy-Item -Recurse web "$Out\web"
Copy-Item -Recurse config "$Out\config"
Copy-Item -Recurse data "$Out\data"

# Remote start script
@'
# Run Taver demo on VPS (listen on all interfaces, port 8080)
$root = $PSScriptRoot
$bin = Join-Path $root "bin"
$espeak = Join-Path $root "third_party\espeak\eSpeak NG\espeak-ng.exe"
& (Join-Path $bin "taver_server.exe") `
  --host 0.0.0.0 `
  --port 8080 `
  --model (Join-Path $root "models\phoneme.int8.onnx") `
  --vocab (Join-Path $root "models\phoneme_vocab.txt") `
  --espeak $espeak `
  --root (Join-Path $root "web")
'@ | Set-Content -Encoding UTF8 "$Out\run-demo.ps1"

Write-Host "Packed -> $Out" -ForegroundColor Green
$mb = [math]::Round((Get-ChildItem $Out -Recurse | Measure-Object Length -Sum).Sum / 1MB)
Write-Host "Size: ~$mb MB (upload this folder to the server)" -ForegroundColor Cyan
